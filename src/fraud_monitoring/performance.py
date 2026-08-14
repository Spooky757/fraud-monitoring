"""Supervised monitoring — the half that needs labels, and therefore always lags.

Fraud labels are not available at scoring time. They arrive when a chargeback or a
confirmed dispute lands, typically weeks later. So this module grades *matured*
batches only: a batch is eligible once `label_delay_days` have passed since its
newest transaction. Grading a fresh batch would systematically undercount fraud
(the not-yet-disputed frauds still look like legitimate transactions) and produce a
recall cliff that is an artefact of the calendar, not the model.

The other trap this guards against is small-sample noise. At a ~0.17% base rate a
10k-row batch holds ~17 frauds; recall computed on 17 positives swings +/-12 points
from luck alone. Batches below `min_labeled_frauds` are reported as INSUFFICIENT
rather than being allowed to trigger a retrain.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def is_batch_matured(batch: pd.DataFrame, timestamp_column: str, label_delay_days: int,
                     now: pd.Timestamp | None = None) -> bool:
    """True when every transaction in the batch is older than the label delay."""
    if timestamp_column not in batch.columns:
        return True  # no wall-clock timestamp available; caller vouches for maturity
    stamps = pd.to_datetime(batch[timestamp_column], errors="coerce", utc=True)
    if stamps.isna().all():
        return True
    now = now or pd.Timestamp.now(tz="UTC")
    return bool((now - stamps.max()).days >= label_delay_days)


def compute_performance(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float
) -> dict:
    """Metrics at the operating threshold, plus threshold-free ranking quality.

    PR-AUC is the headline: with a 0.17% positive rate, ROC-AUC stays flattering even
    as the model degrades, while PR-AUC tracks what the fraud team actually feels.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_positives = int(y_true.sum())
    metrics: dict[str, float | int | None] = {
        "n_rows": int(y_true.size),
        "n_frauds": n_positives,
        "fraud_rate": float(y_true.mean()) if y_true.size else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "alert_volume": int(y_pred.sum()),
    }
    precision, recall = metrics["precision"], metrics["recall"]
    metrics["f1"] = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    # Ranking metrics are undefined on a single-class batch; report None, not a crash.
    both_classes = 0 < n_positives < y_true.size
    metrics["pr_auc"] = float(average_precision_score(y_true, y_prob)) if both_classes else None
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if both_classes else None

    if y_true.size:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update(
            {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                # What a reviewer actually experiences: how many alerts are junk.
                "false_discovery_rate": float(fp / (fp + tp)) if (fp + tp) else 0.0,
                # What the business eats: fraud value that sailed through.
                "missed_fraud_count": int(fn),
            }
        )
    return metrics


def compare_to_baseline(current: dict, baseline: dict) -> dict:
    """Deltas against the champion's promotion-time metrics.

    Relative for PR-AUC (a 0.02 drop means something different at 0.85 than at 0.30),
    absolute for precision/recall (those are read as percentage points by humans).
    """
    def _rel_drop(now: float | None, before: float | None) -> float | None:
        if now is None or before in (None, 0) or not np.isfinite(before):
            return None
        return float((before - now) / before)

    return {
        "pr_auc_relative_drop": _rel_drop(current.get("pr_auc"), baseline.get("pr_auc")),
        "recall_absolute_drop": (
            float(baseline["recall"] - current["recall"])
            if baseline.get("recall") is not None
            else None
        ),
        "precision_absolute_drop": (
            float(baseline["precision"] - current["precision"])
            if baseline.get("precision") is not None
            else None
        ),
        "baseline": baseline,
    }


def performance_report(
    batch: pd.DataFrame,
    scores: Sequence[float],
    threshold: float,
    baseline: dict,
    *,
    label_column: str = "Class",
    min_labeled_rows: int = 5000,
    min_labeled_frauds: int = 30,
) -> dict:
    """Full supervised block for one batch, or a documented reason it was skipped."""
    if label_column not in batch.columns:
        return {"status": "UNLABELED", "reason": f"No {label_column!r} column in batch"}

    labeled = batch[batch[label_column].notna()]
    scores = np.asarray(scores, dtype=float)[batch[label_column].notna().to_numpy()]

    if len(labeled) < min_labeled_rows:
        return {
            "status": "INSUFFICIENT",
            "reason": f"{len(labeled)} labeled rows < min_labeled_rows={min_labeled_rows}",
            "n_rows": int(len(labeled)),
        }

    metrics = compute_performance(labeled[label_column], scores, threshold)
    if metrics["n_frauds"] < min_labeled_frauds:
        return {
            "status": "INSUFFICIENT",
            "reason": (
                f"{metrics['n_frauds']} frauds < min_labeled_frauds={min_labeled_frauds}; "
                "metrics would be dominated by sampling noise"
            ),
            "metrics": metrics,
        }

    return {"status": "OK", "metrics": metrics, "comparison": compare_to_baseline(metrics, baseline)}
