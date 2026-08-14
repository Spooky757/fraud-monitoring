from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_monitoring.config import load_config
from fraud_monitoring.retrain import build_training_window, check_training_data, evaluate_gate


@pytest.fixture
def config():
    return load_config("dev")


def _scores(y, separation, rng):
    """Probabilities whose separation from the labels we control directly."""
    y = np.asarray(y)
    return np.clip(rng.beta(1, 12, size=y.size) + y * separation, 0, 1)


def test_gate_promotes_a_genuinely_better_challenger(config):
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.02, size=20_000)
    result = evaluate_gate(
        champion_scores=_scores(y, 0.35, rng),
        challenger_scores=_scores(y, 0.75, rng),
        y_holdout=y,
        champion_threshold=0.5,
        challenger_threshold=0.5,
        config=config,
    )
    assert result.promote
    assert "PROMOTE" in result.summary
    assert result.challenger_metrics["pr_auc"] > result.champion_metrics["pr_auc"]


def test_gate_rejects_a_merely_equal_challenger(config):
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.02, size=20_000)
    scores = _scores(y, 0.5, rng)
    result = evaluate_gate(scores, scores.copy(), y, 0.5, 0.5, config)

    assert not result.promote
    assert "pr_auc_improvement" in result.summary


def test_gate_rejects_a_challenger_that_wins_on_auc_but_loses_recall(config):
    """The trap this gate exists for: better ranking, worse behaviour at the
    operating point the fraud team actually runs."""
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.03, size=20_000)
    champion = _scores(y, 0.55, rng)
    challenger = _scores(y, 0.75, rng) * 0.45  # better ordering, compressed downward

    result = evaluate_gate(champion, challenger, y, 0.5, 0.5, config)
    recall_check = next(c for c in result.checks if c["name"] == "recall_not_regressed")
    assert result.challenger_metrics["pr_auc"] > result.champion_metrics["pr_auc"]
    assert not recall_check["passed"]
    assert not result.promote


def test_gate_enforces_the_precision_floor(config):
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.02, size=20_000)
    # A challenger that flags almost everything: high recall, useless precision.
    challenger = np.clip(_scores(y, 0.6, rng) + 0.55, 0, 1)

    result = evaluate_gate(_scores(y, 0.5, rng), challenger, y, 0.5, 0.5, config)
    precision_check = next(c for c in result.checks if c["name"] == "precision_floor")
    assert not precision_check["passed"]
    assert not result.promote


def test_gate_records_every_check_even_when_it_passes(config):
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.02, size=20_000)
    result = evaluate_gate(_scores(y, 0.3, rng), _scores(y, 0.8, rng), y, 0.5, 0.5, config)
    names = {c["name"] for c in result.checks}
    assert names == {"pr_auc_improvement", "recall_not_regressed", "precision_floor"}


def test_gate_honours_extra_preflight_checks(config):
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.02, size=20_000)
    result = evaluate_gate(
        _scores(y, 0.3, rng), _scores(y, 0.8, rng), y, 0.5, 0.5, config,
        extra_checks=[{"name": "min_training_rows", "passed": False, "value": 10,
                       "threshold": 50_000, "detail": "too little data"}],
    )
    assert not result.promote, "a failed pre-flight check must veto promotion"


def test_training_window_is_time_ordered_and_holds_out_the_most_recent_slice():
    days = 300
    history = pd.DataFrame({
        "Time": np.arange(days * 24) * 3600.0,   # hourly rows over 300 days
        "Class": 0,
    })
    train, holdout = build_training_window(
        history, rolling_window_days=180, holdout_days=30
    )
    assert train["Time"].max() < holdout["Time"].min(), "no future leaks into training"
    assert holdout["Time"].max() == history["Time"].max()
    # Holdout ~30 days of hourly rows; training window capped at ~180 days.
    assert 700 <= len(holdout) <= 740
    assert 4300 <= len(train) <= 4340


def test_training_window_without_a_rolling_cap_keeps_all_history():
    history = pd.DataFrame({"Time": np.arange(1000) * 3600.0, "Class": 0})
    train, holdout = build_training_window(history, rolling_window_days=None, holdout_days=1)
    assert len(train) + len(holdout) == len(history)


def test_preflight_rejects_a_window_with_too_few_frauds(config):
    train = pd.DataFrame({"Class": [0] * 60_000 + [1] * 10})
    checks = {c["name"]: c for c in check_training_data(train, config)}
    assert checks["min_training_rows"]["passed"]
    assert not checks["min_training_frauds"]["passed"]
