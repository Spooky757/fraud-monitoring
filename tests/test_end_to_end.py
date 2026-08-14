"""End-to-end: train a stand-in champion, build a reference, and prove the monitor
stays quiet on healthy traffic and escalates on injected drift.

This is the test that would have caught every integration bug the unit tests can't
see — column ordering, scaler reuse, profile/champion version coupling.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from fraud_monitoring.artifacts import Champion
from fraud_monitoring.config import load_config
from fraud_monitoring.monitor import monitor_batch
from fraud_monitoring.performance import compute_performance
from fraud_monitoring.reference import build_reference_profile
from fraud_monitoring.rules import load_state
from fraud_monitoring.simulate import FEATURE_COLUMNS, inject_drift, make_synthetic_dataset

xgboost = pytest.importorskip("xgboost")


@pytest.fixture(scope="module")
def config():
    cfg = load_config("dev")
    cfg["drift"]["min_batch_rows"] = 1000
    cfg["performance"]["min_labeled_rows"] = 4000
    cfg["performance"]["min_labeled_frauds"] = 10
    return cfg


@pytest.fixture(scope="module")
def trained(config):
    """A champion trained exactly the way the training repo trains: time-ordered
    split, scaler fit on train Amount only, XGBoost with scale_pos_weight."""
    df = make_synthetic_dataset(40_000, fraud_rate=0.01, random_state=42)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    scaler = StandardScaler()
    X_train = train[FEATURE_COLUMNS].copy()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])

    positives = int(train["Class"].sum())
    model = xgboost.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.1, eval_metric="logloss",
        random_state=42, scale_pos_weight=(len(train) - positives) / positives,
    )
    model.fit(X_train, train["Class"])

    champion = Champion(model=model, scaler=scaler, threshold=0.5,
                        version="test-champion", source="test")
    eval_scores = champion.score(test, FEATURE_COLUMNS)
    baseline = compute_performance(test["Class"], eval_scores, champion.threshold)

    monitored = [f for f in FEATURE_COLUMNS if f != "Time"]
    profile = build_reference_profile(
        train, eval_scores, champion.threshold, monitored,
        model_version="test-champion",
        baseline_metrics={"pr_auc": baseline["pr_auc"], "recall": baseline["recall"],
                          "precision": baseline["precision"]},
    )
    return champion, profile, df, baseline


def test_the_stand_in_model_actually_learned_something(trained):
    _, _, _, baseline = trained
    assert baseline["pr_auc"] > 0.5, "a model that cannot rank makes the gate tests vacuous"
    assert baseline["recall"] > 0.5


def test_healthy_batch_produces_no_action(trained, config):
    champion, profile, _, _ = trained
    healthy = make_synthetic_dataset(15_000, fraud_rate=0.01, random_state=99)

    report, decision, _ = monitor_batch(
        healthy, champion, profile, config, load_state("/nonexistent"), batch_id="healthy"
    )
    assert report["drift"]["n_features_alert"] == 0
    assert decision.action in ("NO_ACTION", "WATCH")
    assert decision.severity in ("OK", "WARN")


def test_covariate_drift_is_detected(trained, config):
    champion, profile, _, _ = trained
    batch = inject_drift(
        make_synthetic_dataset(15_000, fraud_rate=0.01, random_state=100),
        kind="covariate", magnitude=1.5,
    )
    report, decision, _ = monitor_batch(
        batch, champion, profile, config, load_state("/nonexistent"), batch_id="covariate"
    )
    assert report["drift"]["n_features_alert"] >= 5
    assert report["drift"]["max_psi"] > 0.25
    assert decision.severity == "ALERT"


def test_amount_drift_is_detected(trained, config):
    champion, profile, _, _ = trained
    batch = inject_drift(
        make_synthetic_dataset(15_000, fraud_rate=0.01, random_state=101),
        kind="amount", magnitude=4.0,
    )
    report, _, _ = monitor_batch(
        batch, champion, profile, config, load_state("/nonexistent"), batch_id="amount"
    )
    amount = next(f for f in report["drift"]["features"] if f["feature"] == "Amount")
    assert amount["psi_level"] == "ALERT"


def test_concept_drift_shows_up_in_performance_not_in_feature_drift(trained, config):
    """The point of the whole two-layer design: concept drift leaves the inputs
    looking normal and only surfaces once labels mature."""
    champion, profile, _, baseline = trained
    batch = inject_drift(
        make_synthetic_dataset(15_000, fraud_rate=0.01, random_state=102),
        kind="concept", magnitude=0.8,
    )
    report, _, _ = monitor_batch(
        batch, champion, profile, config, load_state("/nonexistent"), batch_id="concept"
    )
    assert report["drift"]["n_features_alert"] == 0, "inputs look fine — that is the trap"
    assert report["performance"]["status"] == "OK"
    assert report["performance"]["metrics"]["recall"] < baseline["recall"]


def test_two_drifted_windows_escalate_to_action(trained, config):
    champion, profile, _, _ = trained
    state = load_state("/nonexistent")
    actions = []
    for seed in (200, 201):
        batch = inject_drift(
            make_synthetic_dataset(15_000, fraud_rate=0.01, random_state=seed),
            kind="covariate", magnitude=1.5,
        )
        _, decision, state = monitor_batch(
            batch, champion, profile, config, state, batch_id=f"drift-{seed}"
        )
        actions.append(decision.action)

    assert actions[0] == "WATCH", "first alerting window is held for confirmation"
    assert actions[1] in ("RETRAIN", "RECALIBRATE_THRESHOLD")


def test_tiny_batch_halts_instead_of_reporting_noise(trained, config):
    champion, profile, _, _ = trained
    tiny = make_synthetic_dataset(200, fraud_rate=0.01, random_state=103)
    _, decision, _ = monitor_batch(
        tiny, champion, profile, config, load_state("/nonexistent"), batch_id="tiny"
    )
    assert decision.action == "HALT"


def test_scoring_rejects_a_batch_missing_model_features(trained, config):
    champion, _, _, _ = trained
    batch = make_synthetic_dataset(2000, random_state=104).drop(columns=["V5"])
    with pytest.raises(ValueError, match="missing model features"):
        champion.score(batch, FEATURE_COLUMNS)


def test_report_renders_to_html_and_markdown(trained, config):
    from fraud_monitoring.report import to_html, to_markdown

    champion, profile, _, _ = trained
    batch = make_synthetic_dataset(15_000, fraud_rate=0.01, random_state=105)
    report, _, _ = monitor_batch(
        batch, champion, profile, config, load_state("/nonexistent"), batch_id="render"
    )
    html = to_html(report)
    assert "<!doctype html>" in html and "Feature drift" in html
    assert "Monitoring report" in to_markdown(report)
    assert not np.isnan(report["prediction"]["score_psi"])
