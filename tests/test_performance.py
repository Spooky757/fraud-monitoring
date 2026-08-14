from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_monitoring.performance import (
    compare_to_baseline,
    compute_performance,
    is_batch_matured,
    performance_report,
)


def test_metrics_are_correct_on_a_hand_checked_case():
    y_true = [0, 0, 1, 1, 0, 1]
    y_prob = [0.1, 0.4, 0.9, 0.6, 0.7, 0.2]
    metrics = compute_performance(y_true, y_prob, threshold=0.5)

    # Predicted positive: indices 2, 3, 4 -> tp=2 (2,3), fp=1 (4), fn=1 (5)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["false_discovery_rate"] == pytest.approx(1 / 3)
    assert metrics["alert_volume"] == 3


def test_ranking_metrics_are_none_on_single_class_batch():
    metrics = compute_performance([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert metrics["pr_auc"] is None and metrics["roc_auc"] is None
    assert metrics["recall"] == 0.0  # still defined, still reported


def test_threshold_moves_precision_and_recall_in_opposite_directions():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.02, size=5000)
    prob = np.clip(rng.beta(1, 20, size=5000) + y * 0.5, 0, 1)

    loose = compute_performance(y, prob, 0.2)
    tight = compute_performance(y, prob, 0.6)
    assert loose["recall"] >= tight["recall"]
    assert loose["alert_volume"] >= tight["alert_volume"]


def test_baseline_comparison_uses_relative_pr_auc_and_absolute_recall():
    comparison = compare_to_baseline(
        {"pr_auc": 0.72, "recall": 0.78, "precision": 0.85},
        {"pr_auc": 0.80, "recall": 0.85, "precision": 0.90},
    )
    assert comparison["pr_auc_relative_drop"] == pytest.approx(0.1)
    assert comparison["recall_absolute_drop"] == pytest.approx(0.07)
    assert comparison["precision_absolute_drop"] == pytest.approx(0.05)


def test_baseline_comparison_tolerates_a_missing_baseline():
    comparison = compare_to_baseline({"pr_auc": 0.7, "recall": 0.8, "precision": 0.9}, {})
    assert comparison["pr_auc_relative_drop"] is None
    assert comparison["recall_absolute_drop"] is None


def test_report_refuses_to_grade_a_batch_with_too_few_frauds():
    rng = np.random.default_rng(2)
    batch = pd.DataFrame({"Class": rng.binomial(1, 0.0005, size=6000)})
    scores = rng.random(6000)

    report = performance_report(batch, scores, 0.5, {}, min_labeled_rows=1000, min_labeled_frauds=30)
    assert report["status"] == "INSUFFICIENT"
    assert "sampling noise" in report["reason"]


def test_report_refuses_to_grade_a_tiny_batch():
    batch = pd.DataFrame({"Class": [0, 1, 0]})
    report = performance_report(batch, [0.1, 0.9, 0.2], 0.5, {}, min_labeled_rows=1000)
    assert report["status"] == "INSUFFICIENT"


def test_report_marks_unlabelled_batches_rather_than_guessing():
    batch = pd.DataFrame({"V1": [0.1, 0.2]})
    assert performance_report(batch, [0.1, 0.2], 0.5, {})["status"] == "UNLABELED"


def test_report_grades_a_sufficient_batch():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.02, size=10_000)
    scores = np.clip(rng.beta(1, 15, size=10_000) + y * 0.6, 0, 1)
    batch = pd.DataFrame({"Class": y})

    report = performance_report(
        batch, scores, 0.5, {"pr_auc": 0.9, "recall": 0.9, "precision": 0.9},
        min_labeled_rows=5000, min_labeled_frauds=30,
    )
    assert report["status"] == "OK"
    assert report["metrics"]["n_frauds"] >= 30
    assert report["comparison"]["pr_auc_relative_drop"] is not None


def test_label_maturity_gate_blocks_fresh_batches():
    now = pd.Timestamp("2026-08-14", tz="UTC")
    fresh = pd.DataFrame({"event_time": pd.date_range("2026-08-10", periods=3, tz="UTC")})
    matured = pd.DataFrame({"event_time": pd.date_range("2026-05-01", periods=3, tz="UTC")})

    assert not is_batch_matured(fresh, "event_time", 30, now=now)
    assert is_batch_matured(matured, "event_time", 30, now=now)
    # No timestamp column: the caller is asserting maturity, so we do not block.
    assert is_batch_matured(pd.DataFrame({"x": [1]}), "event_time", 30, now=now)
