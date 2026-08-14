"""The escalation policy is the highest-risk logic in the repo — it decides whether
a model gets retrained. Every branch of the ladder is pinned here."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fraud_monitoring.config import load_config
from fraud_monitoring.rules import Signal, collect_signals, decide, load_state

NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)


@pytest.fixture
def config():
    return load_config("dev")


@pytest.fixture
def fresh_state():
    return {"streak": 0, "last_action": "NO_ACTION", "last_retrain_at": None, "history": []}


def _signal(name, level, category):
    return Signal(name=name, level=level, value=1.0, threshold=0.5, message=name, category=category)


def test_clean_window_produces_no_action(config, fresh_state):
    decision, state = decide([_signal("psi", "OK", "drift")], config, fresh_state, now=NOW)
    assert decision.action == "NO_ACTION"
    assert state["streak"] == 0


def test_warn_only_window_watches_but_does_not_act(config, fresh_state):
    decision, state = decide([_signal("psi", "WARN", "drift")], config, fresh_state, now=NOW)
    assert decision.action == "WATCH"
    assert state["streak"] == 0, "a WARN must not accumulate toward a retrain"


def test_single_alert_window_is_held_for_confirmation(config, fresh_state):
    decision, state = decide(
        [_signal("pr_auc_degradation", "ALERT", "performance")], config, fresh_state, now=NOW
    )
    assert decision.action == "WATCH"
    assert state["streak"] == 1
    assert "1/2" in decision.reasons[0]


def test_two_consecutive_performance_alerts_trigger_retrain(config, fresh_state):
    signals = [_signal("pr_auc_degradation", "ALERT", "performance")]
    _, state = decide(signals, config, fresh_state, now=NOW)
    decision, state = decide(signals, config, state, now=NOW + timedelta(days=1))

    assert decision.action == "RETRAIN"
    assert decision.confirmed_streak == 2
    assert state["last_retrain_at"] is not None


def test_streak_resets_when_a_window_comes_back_clean(config, fresh_state):
    alert = [_signal("pr_auc_degradation", "ALERT", "performance")]
    _, state = decide(alert, config, fresh_state, now=NOW)
    assert state["streak"] == 1

    _, state = decide([_signal("psi", "OK", "drift")], config, state, now=NOW + timedelta(days=1))
    assert state["streak"] == 0

    decision, _ = decide(alert, config, state, now=NOW + timedelta(days=2))
    assert decision.action == "WATCH", "the clean window broke the streak"


def test_distribution_drift_with_intact_ranking_prefers_recalibration(config, fresh_state):
    # Score distribution alerting, PR-AUC fine -> the cheap fix is the right fix.
    signals = [
        _signal("score_distribution_psi", "ALERT", "prediction"),
        _signal("pr_auc_degradation", "OK", "performance"),
    ]
    _, state = decide(signals, config, fresh_state, now=NOW)
    decision, _ = decide(signals, config, state, now=NOW + timedelta(days=1))
    assert decision.action == "RECALIBRATE_THRESHOLD"


def test_drift_plus_ranking_collapse_escalates_to_retrain(config, fresh_state):
    signals = [
        _signal("score_distribution_psi", "ALERT", "prediction"),
        _signal("pr_auc_degradation", "ALERT", "performance"),
    ]
    _, state = decide(signals, config, fresh_state, now=NOW)
    decision, _ = decide(signals, config, state, now=NOW + timedelta(days=1))
    assert decision.action == "RETRAIN"


def test_cooldown_suppresses_a_second_retrain(config, fresh_state):
    config["decision"]["cooldown_days"] = 7   # the dev overlay disables cooldown
    state = dict(fresh_state, streak=1,
                 last_retrain_at=(NOW - timedelta(days=2)).isoformat())
    decision, _ = decide(
        [_signal("pr_auc_degradation", "ALERT", "performance")], config, state, now=NOW
    )
    assert decision.action == "WATCH"
    assert decision.cooldown_active
    assert "cooldown" in decision.reasons[0].lower()


def test_cooldown_expires(config, fresh_state):
    config["decision"]["cooldown_days"] = 7
    state = dict(fresh_state, streak=1,
                 last_retrain_at=(NOW - timedelta(days=30)).isoformat())
    decision, _ = decide(
        [_signal("pr_auc_degradation", "ALERT", "performance")], config, state, now=NOW
    )
    assert decision.action == "RETRAIN"
    assert not decision.cooldown_active


def test_data_quality_failure_halts_immediately_and_bypasses_confirmation(config, fresh_state):
    decision, state = decide(
        [_signal("batch_size", "ALERT", "data_quality"),
         _signal("pr_auc_degradation", "ALERT", "performance")],
        config, fresh_state, now=NOW,
    )
    assert decision.action == "HALT"
    assert state["streak"] == 0, "garbage input must not count toward a retrain"


def test_collect_signals_builds_the_full_set_from_a_report(config):
    report = {
        "n_rows": 20_000,
        "drift": {"features": [], "n_features_tested": 29, "n_features_alert": 0,
                  "drifted_fraction": 0.0, "max_psi": 0.02, "top_drifted": []},
        "prediction": {"score_psi": 0.01, "flag_rate": 0.002, "reference_flag_rate": 0.002,
                       "flag_rate_relative_change": 0.0, "mean_score": 0.01},
        "performance": {"status": "INSUFFICIENT", "reason": "labels pending"},
    }
    signals = collect_signals(report, config)
    categories = {s.category for s in signals}
    assert {"drift", "prediction", "performance"} <= categories
    assert all(s.level == "OK" for s in signals)


def test_load_state_returns_a_usable_default_when_no_file_exists(tmp_path):
    state = load_state(tmp_path / "nope.json")
    assert state["streak"] == 0 and state["last_retrain_at"] is None
