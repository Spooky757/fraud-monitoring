# Monitoring and retraining design

This document explains *why* the system is built the way it is. `README.md` covers
how to run it.

The model being monitored is the tuned XGBoost fraud detector produced by
[Credit-Card-Fraud-testing](https://github.com/Koneko1625/Credit-Card-Fraud-testing).
That repo's job ends when it writes `model.pkl`, `scaler.pkl`, and `threshold.json`.
This repo's job starts there and answers one question on a schedule: **is that model
still doing its job, and if not, what should we do about it?**

---

## 1. Why a separate repository

Training and monitoring look like the same project and behave like different ones.

| | Training repo | This repo |
|---|---|---|
| Runs | On demand, when someone decides to build a model | On a schedule, forever, whether or not anyone is watching |
| Fails by | Crashing loudly | Going quiet — which looks identical to "everything is fine" |
| Changes when | The modelling approach changes | The alert policy changes, which is far more often |
| Reviewed by | Whoever owns the model | Whoever is on call |
| Depends on | The dataset | The model artifact contract, and nothing else |

Keeping them together means every threshold tweak re-runs model training in CI, and
every model experiment risks breaking the thing that watches production. Splitting
them means the coupling is one narrow, versioned contract — three files — instead of
a shared Python package.

The trade-off is real: the contract can drift silently. That is mitigated by
`artifacts.py` refusing to score a batch whose columns don't match, and by the
reference profile recording which `model_version` it was built for, so a mismatched
champion produces a loud warning rather than quietly wrong drift numbers.

---

## 2. What can actually go wrong with this model

Monitoring design starts from failure modes, not from metrics. For a card-fraud
model there are five that matter, and they need different detectors.

**Covariate drift** — the input distribution moves while the relationship between
inputs and fraud stays the same. A new acquirer onboards, a merchant category grows,
the upstream feature pipeline changes a transform. Detectable immediately, without
labels, from the features alone.

**Prior drift** — the fraud base rate itself changes. A campaign starts, or a card-BIN
range gets breached. Shows up as a flag-rate jump long before labels confirm it.

**Concept drift** — the inputs look completely normal and the mapping from inputs to
fraud has changed underneath. Fraudsters adapt to the model; the same feature values
now mean something different. **This is the one that matters most and the one no
unsupervised detector can see.** It is invisible in every PSI chart and only appears
when labels mature. The `concept` scenario in `make demo` demonstrates exactly this:
0/29 features alerting, recall collapsing from 0.98 to 0.27.

**Upstream data quality failure** — a column arrives null, a join drops rows, a batch
lands truncated. Fastest to detect and most damaging to act on blindly: scoring
garbage produces confident garbage.

**Threshold staleness** — nothing is wrong with the model at all. Its scores shifted
slightly and the fixed operating threshold now sits in the wrong place, so the alert
volume doubles. Retraining does not fix this and wastes days finding that out.

The design consequence: **two independent monitoring layers, plus an explicit
distinction between "the model is wrong" and "the operating point is wrong."**

---

## 3. The two layers

### Layer 1 — unsupervised, runs on every batch, no labels needed

Per feature: **PSI** against bin edges frozen at training time, plus a **KS test**
for corroboration. PSI is the trigger; KS is diagnostic. PSI bands follow the risk
convention: `<0.10` stable, `0.10–0.25` moderate, `>0.25` significant.

Two decisions here are worth defending:

*Bin edges come from the reference profile, not from the previous batch.* Comparing
each batch to the last one makes slow drift invisible — every consecutive pair looks
fine while the population walks steadily away from what the model learned.

*KS p-values are Bonferroni-corrected across the ~29 simultaneous tests.* At α=0.01
with 29 features, roughly one feature flags spuriously every three runs. Uncorrected,
the monitor cries wolf about once a week forever.

*Escalation is on breadth, not on any single feature.* One twitchy feature is not
dataset drift. The trigger is the **fraction** of monitored features alerting
(`drifted_fraction_warn: 0.10`, `alert: 0.25`), with a secondary WARN if any single
feature's PSI is extreme — in a PCA feature space that usually means one upstream
source changed.

On the model's own output: **score PSI** and **flag rate** (the share of the batch
above the operating threshold). Flag rate is the single most useful early signal in
the whole system, because it is the analyst queue size — a metric the business
already feels, available before any label exists.

> **A false-alarm trap this design hit and fixed.** Score PSI was originally computed
> with quantile bins, exactly like the features. It reported PSI ≈ 0.58 — deep in the
> "significant drift" band — on two honest samples of the *same* population. The
> cause: a well-separated fraud model puts ~99% of its mass near zero, so equal-mass
> quantile bins spend nineteen of twenty bins resolving p=1.5e-5 against p=4e-5.
> Nobody makes a decision in that range. The fix is **fixed decision-relevant score
> bands** (`reference.py: SCORE_BANDS`) which collapse that noise into one bucket and
> reserve resolution for probabilities a human would act on. Healthy-batch PSI dropped
> from 0.58 to 0.035, and a 5× prior-drift batch still reads 0.10. A monitor that
> alarms every day is a monitor everyone turns off; this was worth catching.

### Layer 2 — supervised, runs only on matured batches

Fraud labels arrive on chargeback lag — typically 30–90 days. Two guards follow
directly:

*Maturity gate.* A batch is graded only once `label_delay_days` have passed since its
newest transaction. Grading fresh data systematically undercounts fraud, because the
not-yet-disputed frauds still look legitimate. That produces a recall cliff which is
an artefact of the calendar, and a retrain triggered on it is pure waste.

*Volume floor.* At a 0.17% base rate, a 10,000-row batch holds ~17 frauds. Recall
computed on 17 positives swings ±12 points from luck alone. Batches below
`min_labeled_frauds` are reported `INSUFFICIENT` and are structurally incapable of
triggering a retrain.

**PR-AUC is the headline metric, not ROC-AUC.** At a 0.17% positive rate, ROC-AUC
stays flatteringly high while the model degrades, because the enormous true-negative
count dominates the false-positive rate. PR-AUC tracks what the fraud team feels.

Metric roles are deliberate: PR-AUC measures *ranking quality* and is threshold-free
— it is the signal that says the model itself is worse. Recall and precision are
measured *at the operating threshold* and can move purely from a distribution shift.
Only the first justifies a retrain; that distinction is the basis of the escalation
ladder below.

---

## 4. The escalation ladder

Everything upstream produces measurements. `rules.py` is the only place that turns
measurements into a verdict, so the entire retraining policy is readable in one file
and unit-testable without a model, a dataset, or a network.

```
                 ┌──────────────────────────────────────────────┐
   batch ───────►│ data quality? ── fail ──► HALT                │
                 └──────┬───────────────────────────────────────┘
                        │ pass
                        ▼
                 ┌──────────────────────────────────────────────┐
                 │ any ALERT signal this window?                 │
                 └──────┬───────────────────────┬───────────────┘
                    no  │                       │ yes
                        ▼                       ▼
              WARN present? ──► WATCH    streak += 1
                  else NO_ACTION              │
                                              ▼
                                 streak < required? ──► WATCH
                                              │ no
                                              ▼
                              drift alert but PR-AUC intact?
                                   ├── yes ──► RECALIBRATE_THRESHOLD
                                   └── no  ──► RETRAIN ──► cooldown? ──► WATCH
```

**`HALT` short-circuits everything.** A batch that is too small or missing features
cannot produce trustworthy drift statistics, so it must not be allowed to contribute
to a retrain decision at all. It resets the streak rather than incrementing it.

**Confirmation before action.** A signal must alert in
`consecutive_windows_to_confirm` windows in a row (2 in dev, 3 in prod) before it can
trigger a retrain. One bad window is noise, and a system that retrains on noise
retrains weekly on nothing. A clean window resets the streak to zero — the streak
means *consecutive*, and letting it accumulate across gaps would defeat the purpose.
A `WARN` never increments the streak; only `ALERT` does.

**Cheapest remedy first.** If the score distribution moved but PR-AUC held up, the
model still separates fraud from not-fraud — the operating point is just in the wrong
place. Recalibration is minutes of compute against a labelled window; retraining is
hours plus a new model to validate and a promotion decision to make. `RETRAIN` is
reserved for degradation that recalibration provably cannot fix, which means an
alerting PR-AUC.

**Cooldown.** A minimum gap (14 days in prod) between automated retrain triggers,
because a challenger takes days to validate and the monitor would otherwise trigger
three more retrains while the first is still in review.

**The state file is the monitor's memory.** `monitoring/state/monitor_state.json`
holds the streak and the last-retrain timestamp, and the workflow commits it back to
the repo after every run. That commit is what makes "two consecutive windows" mean
anything across separate GitHub Actions runs, which share nothing else. It also
doubles as an audit log: the last 50 decisions, in git history.

---

## 5. Retraining

### When

Three independent paths, deliberately:

1. **Scheduled floor** — monthly, regardless of signal. Drift detection has false
   negatives, and "no alert" is not the same as "verified healthy."
2. **Triggered** — a confirmed `RETRAIN` decision from the monitor.
3. **Manual** — a human with a reason the system cannot see.

### On what data

Time-ordered, never random. Fraud is overwhelmingly a temporal problem, and a random
split lets the model peek at the future — the resulting metrics are fiction. The
training repo already splits this way; `build_training_window` preserves it.

A **rolling window** (180 days) rather than all history. Old fraud patterns are not
just unhelpful, they are actively misleading — they teach the model to spend capacity
on attacks that no longer happen. The trade-off is losing rare-but-recurring patterns
(seasonal fraud), which is why the window is configurable rather than fixed.

The most recent 30 days are held out from training entirely and used as the **shared
holdout** both models are judged on.

### The promotion gate

A challenger ships only if it clears every check in `retrain.py: evaluate_gate`:

| Check | Why |
|---|---|
| PR-AUC improves by ≥ `min_pr_auc_improvement` | A challenger that is merely equal is not worth the deployment risk. Deploying costs review time and introduces a new unknown; parity buys nothing. |
| Recall regresses by ≤ `max_recall_regression` at the operating threshold | The trap this gate exists for. A challenger can win on PR-AUC while behaving *worse* at the specific operating point the fraud team actually runs — better average ranking, worse decisions. `test_gate_rejects_a_challenger_that_wins_on_auc_but_loses_recall` pins this. |
| Precision ≥ `min_precision` | Precision below the floor floods the review queue. A model nobody can keep up with is not deployed, it is ignored. |
| Pre-flight: enough rows, enough frauds | Refuses to train on a window that cannot support a good model, before spending the compute. |

**Both models are scored on the same untouched, most-recent window.** Comparing a
fresh challenger against the champion's *historical* metrics would compare across two
different populations — which is precisely the drift being measured. That comparison
would make every challenger look like an improvement.

### Promotion is never automatic

A passing gate produces a **pull request**, not a deployment. A fraud model is a
money-and-customers decision surface: the failure mode of a bad automatic promotion
— silently letting fraud through, or freezing thousands of legitimate cards — is far
worse than a day of staleness. A PR makes the change reviewed, reversible, and
attributable.

Every gate decision, including rejections, is written to
`monitoring/reports/promotions/` and committed. Six months later, "why is *this*
model live?" has an answer.

---

## 6. Known limitations

These are real and not designed away. Naming them is more useful than hiding them.

**The feedback loop / selection bias.** Transactions the model blocks never produce a
label — a blocked fraud is never charged back, and a blocked legitimate transaction is
never completed. So the labelled data flowing back is conditioned on the current
model's decisions, and each retraining generation narrows what the next one can learn.
Proper mitigations are a small randomised hold-out of unblocked traffic, or
propensity weighting; both require product decisions this repo cannot make alone. The
monitor tracks `flag_rate` partly as a proxy for how aggressively the loop is being
tightened.

**The `Time` feature.** The training pipeline feeds `Time` — seconds elapsed since the
dataset's first transaction — to the model as a feature. On any new batch this is
guaranteed to "drift" by construction, and it is not a real signal: a model that
learned anything from it learned an artefact of the dataset's collection window.
`Time` is therefore excluded from drift monitoring (`data.drift_exclude`). **The
better fix belongs upstream: drop `Time` from `feature_columns` in the training
repo, or replace it with a cyclical hour-of-day encoding, which is the thing that
actually carries fraud signal.** Until then, the exclusion is a workaround, not a fix.

**PCA features are undiagnosable.** V1–V28 are anonymised principal components. When
V14 drifts, nobody can say *what changed in the world* — only that something did.
This limits root-cause analysis to "look at the raw upstream feed," and it is inherent
to the dataset rather than to this design.

**Simulated batches are not production traffic.** `simulate.py` exists so the loop is
demonstrable and the detectors are provably capable of firing. Real thresholds should
be re-tuned by replaying several months of real historical batches through
`monitor --dry-run` and looking at the false-positive rate before trusting the
defaults in `configs/prod.yaml`.

**Cost is not modelled.** Precision and recall are proxies. The real objective is
expected loss: fraud value missed, plus review cost, plus the cost of declining a
good customer. A cost-weighted threshold — `Amount` is right there in the data —
would be the highest-value next addition to this repo.

---

## 7. What runs where

| Trigger | Workflow | Does | Human involvement |
|---|---|---|---|
| Daily 02:00 UTC | `monitor.yml` | Score a batch, measure, decide, commit state, open an issue if it alerts | Read the issue |
| Monitor says RETRAIN | `retrain.yml` (called) | Build a challenger, gate it | Review the gate record |
| Monthly, 1st 03:00 UTC | `retrain.yml` (scheduled) | Same, as a staleness floor | Same |
| Gate passes | PR opened automatically | Nothing is live yet | **Merge = deploy** |
| Push / PR | `ci.yml` | Lint, tests on 3.10–3.12, CLI smoke test | Fix what breaks |

Environments are config overlays (`configs/{dev,staging,prod}.yaml`), selected by
`FRAUD_MONITORING_ENV`, not code branches. Prod is strictly more conservative on
every axis: larger minimum batches, three confirmation windows instead of two, a
14-day cooldown, and a stricter gate. Staging deliberately sets
`fail_workflow_on_alert: true` — in staging, a broken monitor should be loud; in
prod, the issue is the signal and a red schedule badge trains people to ignore it.
