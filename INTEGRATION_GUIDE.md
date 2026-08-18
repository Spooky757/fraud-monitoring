# Integration Guide: Connecting the Training and Monitoring Repos

This guide explains how to make the monitoring repo (`fraud-monitoring`) automatically pull artifacts from the training repo (`Credit-Card-Fraud-testing`) — instead of manually copying `model.pkl`, `scaler.pkl`, and `threshold.json`.

Two methods are provided. Use **Option A** for local demos and development. Use **Option B** for production automation on GitHub.

---

## Option A: Local Demo Script (`run_full_pipeline.sh`)

A single bash script that trains the model and runs monitoring end-to-end.

### What it does (in order)

1. **Trains the model** — runs `python -m src.fraud_pipeline.pipeline` in the training repo
2. **Copies artifacts** — moves `model.pkl`, `scaler.pkl`, `threshold.json` from `training-repo/models/` to `monitoring-repo/artifacts/champion/`
3. **Builds the reference profile** — freezes the baseline distribution the model was trained on
4. **Generates test batches** — creates healthy, covariate-drift, and concept-drift batches
5. **Runs monitoring** — scores each batch and prints the escalation decision

### Setup

```bash
# Your folder structure should look like this:
# some-folder/
# ├── Credit-Card-Fraud-testing/    ← training repo
# │   └── data/raw/creditcard.csv   ← Kaggle dataset here
# └── fraud-monitoring/             ← monitoring repo

# 1. Install dependencies in both repos
cd Credit-Card-Fraud-testing && pip install -r requirements.txt && cd ..
cd fraud-monitoring && pip install -r requirements.txt && cd ..

# 2. Copy the script into the monitoring repo
cp run_full_pipeline.sh fraud-monitoring/

# 3. Run it
cd fraud-monitoring
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

If you don't have `creditcard.csv`, the script will automatically generate synthetic data so the demo still works.

### Customising

| Variable | Default | What it does |
|---|---|---|
| `TRAINING_REPO` | `../Credit-Card-Fraud-testing` | Path to the training repo |
| `MONITORING_REPO` | `.` (current directory) | Path to the monitoring repo |
| `ENV` | `dev` | Config overlay: dev, staging, or prod |

Example:
```bash
TRAINING_REPO=~/repos/Credit-Card-Fraud-testing ENV=staging ./run_full_pipeline.sh
```

---

## Option B: GitHub Actions Automation

This is the production setup. Three workflows work together so that the monitoring repo pulls artifacts automatically — no manual copying.

### How it works

```
Training Repo                         Monitoring Repo
─────────────                         ───────────────
Push to main
  → release.yml runs
  → Creates GitHub Release
    with model.pkl, scaler.pkl,       monitor.yml runs daily at 02:00 UTC
    threshold.json attached             → Downloads artifacts from the
                                          training repo's latest release
                                        → Scores the batch
                                        → If RETRAIN → triggers retrain.yml
                                            → Clones training repo
                                            → Trains challenger
                                            → Gates it vs champion
                                            → Opens a PR (human merges)
```

### Step-by-step setup

#### 1. Add `release.yml` to the training repo

Copy `training-repo-release.yml` into the training repo:

```bash
# In the training repo
mkdir -p .github/workflows
cp training-repo-release.yml .github/workflows/release.yml
git add .github/workflows/release.yml
git commit -m "Add release workflow for model artifacts"
git push
```

This workflow runs whenever model files change on `main`. It creates a GitHub Release with the three artifact files attached, which the monitoring repo's workflows download.

#### 2. Set repo variables on the monitoring repo

Go to the monitoring repo on GitHub → **Settings** → **Secrets and variables** → **Actions** → **Variables** tab:

| Variable | Value | Required? |
|---|---|---|
| `TRAINING_REPO` | `YourGitHubUsername/Credit-Card-Fraud-testing` | Yes |
| `TRAINING_REF` | `main` | Optional (pin a branch/tag for reproducible retrains) |

#### 3. Set secrets (only if training repo is private)

If the training repo is **private**, go to **Secrets** tab and add:

| Secret | Value |
|---|---|
| `TRAINING_REPO_TOKEN` | A GitHub Personal Access Token with `repo` scope |

If the training repo is **public**, you can skip this — `GITHUB_TOKEN` works.

#### 4. Create the first release

Push the model artifacts to trigger the release workflow:

```bash
# In the training repo
python -m src.fraud_pipeline.pipeline      # trains and saves to models/
git add models/model.pkl models/scaler.pkl models/threshold.json
git commit -m "Initial trained model"
git push
```

Or trigger it manually: Go to **Actions** → **Release Model Artifacts** → **Run workflow**.

#### 5. The monitoring repo workflows are already set up

The existing `monitor.yml` and `retrain.yml` already have the logic to download from the training repo's releases. With `TRAINING_REPO` set, they will:

- **`monitor.yml`** (daily): Downloads the champion, scores the batch, escalates if needed
- **`retrain.yml`** (monthly or triggered): Clones the training repo, trains a challenger, runs the promotion gate, opens a PR

No changes needed to these workflows — they already handle the release download.

#### 6. Build the reference profile once

After the first release, build the reference profile in the monitoring repo:

```bash
cd fraud-monitoring
# Download the champion artifacts
mkdir -p artifacts/champion
gh release download --repo YourUsername/Credit-Card-Fraud-testing \
    --pattern 'model.pkl' --pattern 'scaler.pkl' --pattern 'threshold.json' \
    --dir artifacts/champion --clobber

# Build and commit the reference profile
PYTHONPATH=src python -m fraud_monitoring.cli --env prod build-reference \
    --training-data /path/to/creditcard.csv
git add monitoring/reference/
git commit -m "Freeze reference profile for initial champion"
git push
```

### Verifying it works

1. **Check the release**: Go to the training repo → **Releases** → you should see the model files
2. **Trigger a manual monitor run**: Go to the monitoring repo → **Actions** → **Monitor** → **Run workflow**
3. **Check the output**: The job summary shows the drift metrics and the escalation decision

---

## How the two repos talk to each other

The entire coupling is these three files — nothing else is shared:

```
Training Repo                    Monitoring Repo
─────────────                    ───────────────
models/model.pkl        →        artifacts/champion/model.pkl
models/scaler.pkl       →        artifacts/champion/scaler.pkl
models/threshold.json   →        artifacts/champion/threshold.json
```

- **Local demo**: the script copies them directly
- **GitHub Actions**: `release.yml` publishes them, `monitor.yml` and `retrain.yml` download them via `gh release download`

The scaler is only ever **loaded and applied** in the monitoring repo — never re-fit. Re-fitting it would silently break accuracy and look like drift.
