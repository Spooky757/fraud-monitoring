"""The decision engine: signals in, one action out.

Everything upstream produces *measurements*. This module is the only place that
turns measurements into a verdict, which keeps the policy auditable — you can read
the whole retraining policy in one file and unit-test it without a model, a dataset,
or a network.

Escalation ladder, cheapest remedy first:

  NO_ACTION            nothing moved
  WATCH                something moved once; log it, do not act (a single bad window
                       is noise, and acting on noise is how teams end up retraining
                       weekly on nothing)
  RECALIBRATE_THRESHOLD the score distribution shifted but ranking quality (PR-AUC)
                       held: the model still separates fraud from not-fraud, the
                       operating point just sits in the wrong place. Re-cutting the
                       threshold is minutes of compute; retraining is hours and a new
                       model to validate.
  RETRAIN              confirmed degradation that recalibration cannot fix
  HALT                 data quality failure — schema breach, dead feature, batch too
                       small. Scoring on this is worse than not scoring; page a human.

Two dampers stop the ladder from thrashing: a signal must repeat across
`consecutive_windows_to_confirm` windows before it can trigger RETRAIN, and a
`cooldown_days` gap is enforced between automated retrains so the system cannot chase
its own tail while a challenger is still being validated.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

LEVEL_ORDER = {"OK": 0, "WARN": 1, "ALERT": 2}

ACTIONS = ["NO_ACTION", "WATCH", "RECALIBRATE_THRESHOLD", "RETRAIN", "HALT"]


@dataclass
class Signal:
    name: str
    level: str            # OK | WARN | ALERT
    value: float | None
    threshold: float | None
    message: str
    category: str = "drift"   # drift | prediction | performance | data_quality

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    action: str
    severity: str
    reasons: list[str] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    confirmed_streak: int = 0
    cooldown_active: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _max_level(signals: list[Signal]) -> str:
    return max((s.level for s in signals), key=lambda lvl: LEVEL_ORDER.get(lvl, 0), default="OK")


# --------------------------------------------------------------------------- signals


def data_quality_signals(batch_rows: int, drift_block: dict, config: dict) -> list[Signal]:
    signals: list[Signal] = []
    min_rows = config["drift"]["min_batch_rows"]
    if batch_rows < min_rows:
        signals.append(
            Signal(
                name="batch_size",
                level="ALERT",
                value=float(batch_rows),
                threshold=float(min_rows),
                message=(
                    f"Batch has {batch_rows} rows, below min_batch_rows={min_rows}. "
                    "Drift statistics on this batch are not trustworthy."
                ),
                category="data_quality",
            )
        )
    missing = [f["feature"] for f in drift_block.get("features", []) if f.get("missing_from_batch")]
    if missing:
        signals.append(
            Signal(
                name="missing_features",
                level="ALERT",
                value=float(len(missing)),
                threshold=0.0,
                message=f"Features absent from batch: {', '.join(missing[:8])}",
                category="data_quality",
            )
        )
    return signals


def drift_signals(drift_block: dict, config: dict) -> list[Signal]:
    dataset_cfg = config["drift"]["dataset"]
    fraction = drift_block.get("drifted_fraction", 0.0)

    if fraction >= dataset_cfg["drifted_fraction_alert"]:
        level = "ALERT"
    elif fraction >= dataset_cfg["drifted_fraction_warn"]:
        level = "WARN"
    else:
        level = "OK"

    signals = [
        Signal(
            name="dataset_drift_breadth",
            level=level,
            value=float(fraction),
            threshold=float(dataset_cfg["drifted_fraction_warn"]),
            message=(
                f"{drift_block.get('n_features_alert', 0)}/"
                f"{drift_block.get('n_features_tested', 0)} monitored features exceeded "
                f"the PSI alert band ({fraction:.1%} of features). "
                f"Largest movers: {', '.join(drift_block.get('top_drifted', [])[:3]) or 'none'}"
            ),
            category="drift",
        )
    ]

    # A single feature moving violently still deserves a look even when breadth is low
    # — in a PCA feature space that usually means one upstream source changed.
    max_psi = drift_block.get("max_psi", float("nan"))
    if np.isfinite(max_psi) and max_psi >= config["drift"]["psi"]["alert"]:
        top = (drift_block.get("top_drifted") or ["?"])[0]
        signals.append(
            Signal(
                name="max_feature_psi",
                level="WARN",
                value=float(max_psi),
                threshold=float(config["drift"]["psi"]["alert"]),
                message=f"Feature {top} has PSI {max_psi:.3f} against the reference profile",
                category="drift",
            )
        )
    return signals


def prediction_signals(prediction_block: dict, config: dict) -> list[Signal]:
    cfg = config["drift"]["prediction"]
    signals: list[Signal] = []

    score_psi = prediction_block.get("score_psi", float("nan"))
    level = (
        "ALERT" if score_psi >= cfg["score_psi_alert"]
        else "WARN" if score_psi >= cfg["score_psi_warn"]
        else "OK"
    )
    signals.append(
        Signal(
            name="score_distribution_psi",
            level=level if np.isfinite(score_psi) else "OK",
            value=float(score_psi),
            threshold=float(cfg["score_psi_warn"]),
            message=(
                f"Model score distribution PSI {_fmt(score_psi)} "
                f"(mean score {_fmt(prediction_block.get('mean_score'))} vs reference "
                f"{_fmt(prediction_block.get('reference_mean_score'))})"
            ),
            category="prediction",
        )
    )

    rel = prediction_block.get("flag_rate_relative_change", float("nan"))
    if np.isfinite(rel):
        flag_level = (
            "ALERT" if rel >= cfg["flag_rate_rel_alert"]
            else "WARN" if rel >= cfg["flag_rate_rel_warn"]
            else "OK"
        )
        signals.append(
            Signal(
                name="flag_rate_shift",
                level=flag_level,
                value=float(rel),
                threshold=float(cfg["flag_rate_rel_warn"]),
                message=(
                    f"Flag rate {prediction_block.get('flag_rate', float('nan')):.4%} vs reference "
                    f"{prediction_block.get('reference_flag_rate', float('nan')):.4%} "
                    f"({rel:+.0%} relative). This is the analyst queue size."
                ),
                category="prediction",
            )
        )
    return signals


def performance_signals(performance_block: dict, config: dict) -> list[Signal]:
    if performance_block.get("status") != "OK":
        return [
            Signal(
                name="performance_availability",
                level="OK",
                value=None,
                threshold=None,
                message=(
                    f"Performance not evaluated ({performance_block.get('status')}): "
                    f"{performance_block.get('reason', 'labels pending')}"
                ),
                category="performance",
            )
        ]

    cfg = config["performance"]["degradation"]
    comparison = performance_block.get("comparison", {})
    metrics = performance_block.get("metrics", {})
    signals: list[Signal] = []

    pr_drop = comparison.get("pr_auc_relative_drop")
    if pr_drop is not None:
        level = (
            "ALERT" if pr_drop >= cfg["pr_auc_rel_alert"]
            else "WARN" if pr_drop >= cfg["pr_auc_rel_warn"]
            else "OK"
        )
        signals.append(
            Signal(
                name="pr_auc_degradation",
                level=level,
                value=float(pr_drop),
                threshold=float(cfg["pr_auc_rel_warn"]),
                message=(
                    f"PR-AUC {_fmt(metrics.get('pr_auc'))} vs baseline "
                    f"{_fmt(comparison.get('baseline', {}).get('pr_auc'))} "
                    f"({pr_drop:+.1%} relative drop). This is ranking quality — "
                    "a threshold change cannot recover it."
                ),
                category="performance",
            )
        )

    recall_drop = comparison.get("recall_absolute_drop")
    if recall_drop is not None:
        level = (
            "ALERT" if recall_drop >= cfg["recall_abs_alert"]
            else "WARN" if recall_drop >= cfg["recall_abs_warn"]
            else "OK"
        )
        signals.append(
            Signal(
                name="recall_degradation",
                level=level,
                value=float(recall_drop),
                threshold=float(cfg["recall_abs_warn"]),
                message=(
                    f"Recall {_fmt(metrics.get('recall'))} vs baseline "
                    f"{_fmt(comparison.get('baseline', {}).get('recall'))} "
                    f"({recall_drop:+.3f} absolute). {metrics.get('missed_fraud_count', '?')} "
                    "frauds went through undetected in this window."
                ),
                category="performance",
            )
        )

    precision_drop = comparison.get("precision_absolute_drop")
    if precision_drop is not None and precision_drop >= cfg["precision_abs_alert"]:
        signals.append(
            Signal(
                name="precision_collapse",
                level="ALERT",
                value=float(precision_drop),
                threshold=float(cfg["precision_abs_alert"]),
                message=(
                    f"Precision {_fmt(metrics.get('precision'))} vs baseline "
                    f"{_fmt(comparison.get('baseline', {}).get('precision'))}. "
                    f"{metrics.get('false_positives', '?')} false positives are sitting in "
                    "the review queue."
                ),
                category="performance",
            )
        )
    return signals


# ---------------------------------------------------------------------------- state


def load_state(path: Path) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {"streak": 0, "last_action": "NO_ACTION", "last_retrain_at": None, "history": []}


def save_state(state: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state["history"] = state.get("history", [])[-50:]  # bounded; this file lives in git
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _in_cooldown(state: dict, cooldown_days: int, now: datetime) -> bool:
    last = state.get("last_retrain_at")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return False
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    return (now - last_dt) < timedelta(days=cooldown_days)


# ------------------------------------------------------------------------- decision


def decide(signals: list[Signal], config: dict, state: dict,
           now: datetime | None = None) -> tuple[Decision, dict]:
    """Pure-ish policy function: signals + previous state -> action + next state.

    Deliberately takes `state` and `now` as arguments rather than reading the clock
    and the filesystem, so every branch of the escalation ladder is unit-testable.
    """
    now = now or datetime.now(timezone.utc)
    state = dict(state)
    decision_cfg = config["decision"]

    quality = [s for s in signals if s.category == "data_quality" and s.level == "ALERT"]
    if quality:
        decision = Decision(
            action="HALT",
            severity="ALERT",
            reasons=[s.message for s in quality],
            signals=[s.to_dict() for s in signals],
        )
        state["streak"] = 0
        state["last_action"] = decision.action
        return decision, state

    perf = [s for s in signals if s.category == "performance"]
    drift = [s for s in signals if s.category in ("drift", "prediction")]
    severity = _max_level(signals)

    perf_alert = [s for s in perf if s.level == "ALERT"]
    drift_alert = [s for s in drift if s.level == "ALERT"]
    any_warn = [s for s in signals if s.level == "WARN"]

    # A window "counts" toward the retrain streak only if something actually alerted.
    triggering = perf_alert or drift_alert
    streak = state.get("streak", 0) + 1 if triggering else 0
    required = decision_cfg["consecutive_windows_to_confirm"]
    cooldown = _in_cooldown(state, decision_cfg["cooldown_days"], now)

    reasons = [s.message for s in signals if s.level in ("WARN", "ALERT")]

    if not triggering:
        action = "WATCH" if any_warn else "NO_ACTION"
    elif streak < required:
        action = "WATCH"
        reasons.insert(
            0,
            f"Alert observed in {streak}/{required} consecutive windows — holding for "
            "confirmation before acting.",
        )
    else:
        # Confirmed. Now choose the cheapest remedy that can actually fix it.
        ranking_intact = not any(
            s.name == "pr_auc_degradation" and s.level == "ALERT" for s in perf
        )
        distribution_only = bool(drift_alert) and not perf_alert
        if decision_cfg.get("prefer_recalibration", True) and distribution_only and ranking_intact:
            action = "RECALIBRATE_THRESHOLD"
            reasons.insert(
                0,
                "Score distribution moved but ranking quality is intact (no PR-AUC alert) — "
                "recalibrating the operating threshold is the cheaper fix; retrain only if "
                "the next window still alerts.",
            )
        else:
            action = "RETRAIN"
            reasons.insert(0, f"Degradation confirmed across {streak} consecutive windows.")

        if cooldown:
            action = "WATCH"
            reasons.insert(
                0,
                f"Retrain suppressed: last retrain was inside the "
                f"{decision_cfg['cooldown_days']}-day cooldown window.",
            )

    decision = Decision(
        action=action,
        severity=severity,
        reasons=reasons,
        signals=[s.to_dict() for s in signals],
        confirmed_streak=streak,
        cooldown_active=cooldown,
    )

    state["streak"] = streak
    state["last_action"] = action
    state.setdefault("history", []).append(
        {"at": now.isoformat(), "action": action, "severity": severity, "streak": streak}
    )
    if action == "RETRAIN":
        state["last_retrain_at"] = now.isoformat()
    return decision, state


def collect_signals(report: dict, config: dict) -> list[Signal]:
    """Assemble every signal for one monitoring window from a raw report dict."""
    signals: list[Signal] = []
    signals += data_quality_signals(report.get("n_rows", 0), report.get("drift", {}), config)
    signals += drift_signals(report.get("drift", {}), config)
    signals += prediction_signals(report.get("prediction", {}), config)
    signals += performance_signals(report.get("performance", {}), config)
    return signals
