"""Drift maths must be right or every downstream decision is wrong. These tests pin
the properties that matter, not just the happy path."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_monitoring.drift import (
    bin_proportions,
    classify_psi,
    feature_drift,
    population_stability_index,
    prediction_drift,
    psi_from_proportions,
    quantile_bin_edges,
)
from fraud_monitoring.reference import build_reference_profile


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_psi_is_zero_for_identical_distributions(rng):
    sample = rng.normal(size=20_000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_near_zero_for_same_distribution_different_draws(rng):
    a, b = rng.normal(size=50_000), rng.normal(size=50_000)
    # Two honest samples from one population must sit well inside the "stable" band,
    # otherwise the monitor cries wolf on every batch.
    assert population_stability_index(a, b) < 0.01


def test_psi_grows_monotonically_with_shift(rng):
    reference = rng.normal(size=50_000)
    values = [
        population_stability_index(reference, rng.normal(loc=shift, size=50_000))
        for shift in (0.1, 0.5, 1.0, 2.0)
    ]
    assert values == sorted(values)
    assert values[0] < 0.1 < values[2], "a 1-sigma shift should clear the warn band"


def test_psi_is_symmetric_enough_to_be_directionless(rng):
    a, b = rng.normal(size=30_000), rng.normal(loc=0.6, size=30_000)
    forward = population_stability_index(a, b)
    backward = population_stability_index(b, a)
    assert forward == pytest.approx(backward, rel=0.35)


def test_psi_handles_empty_bins_without_blowing_up():
    expected = np.array([0.5, 0.5])
    actual = np.array([1.0, 0.0])  # a bin emptied completely
    value = psi_from_proportions(expected, actual)
    assert np.isfinite(value) and value > 0


def test_quantile_edges_are_open_ended_so_new_extremes_are_captured(rng):
    edges = quantile_bin_edges(rng.normal(size=1000), bins=10)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    # Values far outside the reference range still land in a bin rather than vanishing.
    assert bin_proportions([1e9, -1e9], edges).sum() == pytest.approx(1.0)


def test_constant_feature_does_not_crash():
    edges = quantile_bin_edges(np.zeros(500), bins=10)
    assert len(edges) >= 2
    assert np.isfinite(psi_from_proportions(bin_proportions(np.zeros(500), edges),
                                            bin_proportions(np.zeros(500), edges)))


@pytest.mark.parametrize(
    "value,expected", [(0.02, "OK"), (0.10, "WARN"), (0.24, "WARN"), (0.25, "ALERT"), (0.9, "ALERT")]
)
def test_psi_bands_match_the_documented_convention(value, expected):
    assert classify_psi(value, warn=0.10, alert=0.25) == expected


THRESHOLD = 0.05  # low enough that the reference has a non-zero flag rate to compare against


def _profile(rng, n=20_000):
    df = pd.DataFrame({"A": rng.normal(size=n), "B": rng.normal(size=n), "Class": 0})
    scores = rng.beta(1, 40, size=n)
    return df, build_reference_profile(df, scores, THRESHOLD, ["A", "B"], label_column="Class")


def test_feature_drift_flags_only_the_feature_that_moved(rng):
    df, profile = _profile(rng)
    batch = df.copy()
    batch["A"] = batch["A"] + 2.0

    result = feature_drift(profile, batch, ["A", "B"], 0.1, 0.25, 0.01)
    levels = {row["feature"]: row["psi_level"] for row in result["features"]}
    assert levels["A"] == "ALERT"
    assert levels["B"] == "OK"
    assert result["drifted_fraction"] == pytest.approx(0.5)
    assert result["top_drifted"][0] == "A"


def test_feature_drift_reports_missing_columns_rather_than_raising(rng):
    df, profile = _profile(rng)
    result = feature_drift(profile, df.drop(columns=["B"]), ["A", "B"], 0.1, 0.25, 0.01)
    missing = [r for r in result["features"] if r["missing_from_batch"]]
    assert [r["feature"] for r in missing] == ["B"]


def test_ks_alpha_is_bonferroni_corrected(rng):
    df, profile = _profile(rng, n=5_000)
    result = feature_drift(profile, df, ["A", "B"], 0.1, 0.25, ks_alpha=0.01)
    assert result["ks_alpha_corrected"] == pytest.approx(0.005)
    uncorrected = feature_drift(profile, df, ["A", "B"], 0.1, 0.25, 0.01, ks_correction="none")
    assert uncorrected["ks_alpha_corrected"] == pytest.approx(0.01)


def test_prediction_drift_is_quiet_on_an_honest_resample(rng):
    """The false-alarm guard: two draws from the same score distribution must not
    trip the alert band, or the daily monitor becomes noise everyone ignores."""
    _, profile = _profile(rng)
    same = rng.beta(1, 40, size=20_000)
    result = prediction_drift(profile, same, THRESHOLD)
    assert result["score_psi"] < 0.1


def test_prediction_drift_tracks_flag_rate_change(rng):
    _, profile = _profile(rng)
    ref_flag_rate = profile["predictions"]["flag_rate"]
    assert ref_flag_rate > 0

    # Push scores up: more transactions cross the threshold, the review queue grows.
    hotter = np.clip(rng.beta(1, 40, size=20_000) + 0.5, 0, 1)
    result = prediction_drift(profile, hotter, THRESHOLD)

    assert result["flag_rate"] > ref_flag_rate
    assert result["score_psi"] > 0.25
    assert np.isfinite(result["flag_rate_relative_change"])
    assert result["flag_rate_relative_change"] > 1.0


def test_score_bands_are_fixed_not_quantile(rng):
    """Score binning must not be re-cut per model, or PSI values are incomparable
    across model versions."""
    _, profile = _profile(rng)
    assert profile["predictions"]["binning"] == "fixed_decision_bands"
    assert profile["predictions"]["edges"][1] == pytest.approx(1e-4)
