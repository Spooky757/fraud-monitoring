"""Unsupervised drift detection — the half of monitoring that works without labels.

Two complementary tests per feature:

  PSI  — population stability index. Bins the reference distribution into quantiles
         and measures how much probability mass moved. Interpretable magnitude,
         insensitive to sample size, which is why risk teams use it as the headline
         number. Conventional reading: <0.10 stable, 0.10-0.25 moderate, >0.25 shifted.

  KS   — two-sample Kolmogorov-Smirnov. Catches shape changes PSI's coarse binning
         misses, and gives a p-value. It is *very* sensitive at large n, so it is
         used as a corroborating signal, never as the sole trigger, and its p-values
         are Bonferroni-corrected across the ~29 features tested simultaneously.

A feature is called drifted only when PSI clears its alert band. KS is recorded for
diagnosis and tie-breaking.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

EPSILON = 1e-6  # keeps log(0) out of PSI when a bin empties completely


def quantile_bin_edges(values: Sequence[float], bins: int = 10) -> list[float]:
    """Bin edges cut from the *reference* distribution, open at both ends.

    Quantile bins (not equal-width) so that heavy-tailed features like Amount get
    resolution where the mass actually is. Duplicate edges are collapsed, which is
    what makes this safe on near-constant features.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot compute bin edges from an empty reference sample")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(array, quantiles))
    if edges.size < 2:  # constant feature — one interior edge keeps PSI defined
        edges = np.array([edges[0], edges[0] + EPSILON])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(e) for e in edges]


def bin_proportions(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """Share of `values` falling in each bin defined by `edges`."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    counts, _ = np.histogram(array, bins=np.asarray(edges, dtype=float))
    total = counts.sum()
    if total == 0:
        return np.full(len(edges) - 1, np.nan)
    return counts / total


def psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum((actual - expected) * ln(actual / expected)) over bins.

    Symmetric-ish, always >= 0, and 0 only when the distributions match bin-for-bin.
    """
    expected = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], bins: int = 10
) -> float:
    """Convenience wrapper for ad-hoc use; the monitor path uses stored edges."""
    edges = quantile_bin_edges(reference, bins=bins)
    return psi_from_proportions(bin_proportions(reference, edges), bin_proportions(current, edges))


def ks_statistic(reference: Sequence[float], current: Sequence[float]) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan"), float("nan")
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def classify_psi(value: float, warn: float, alert: float) -> str:
    if not np.isfinite(value):
        return "UNKNOWN"
    if value >= alert:
        return "ALERT"
    if value >= warn:
        return "WARN"
    return "OK"


def feature_drift(
    reference_profile: dict,
    batch: pd.DataFrame,
    features: Iterable[str],
    psi_warn: float,
    psi_alert: float,
    ks_alpha: float,
    ks_correction: str = "bonferroni",
) -> dict:
    """Per-feature drift table plus a dataset-level roll-up.

    `reference_profile` is the JSON blob written by reference.py: it carries the bin
    edges and expected proportions frozen at training time, so drift is always
    measured against the distribution the champion actually learned — not against
    whatever batch happened to come before.
    """
    features = [f for f in features if f in reference_profile.get("features", {})]
    n_tests = max(len(features), 1)
    alpha = ks_alpha / n_tests if ks_correction == "bonferroni" else ks_alpha

    rows = []
    for feature in features:
        ref_entry = reference_profile["features"][feature]
        if feature not in batch.columns:
            rows.append(
                {
                    "feature": feature,
                    "psi": float("nan"),
                    "psi_level": "UNKNOWN",
                    "ks_statistic": float("nan"),
                    "ks_pvalue": float("nan"),
                    "ks_significant": False,
                    "reference_mean": ref_entry["mean"],
                    "current_mean": float("nan"),
                    "mean_shift_in_ref_sds": float("nan"),
                    "missing_from_batch": True,
                }
            )
            continue

        current = batch[feature].to_numpy(dtype=float)
        psi = psi_from_proportions(
            np.asarray(ref_entry["proportions"], dtype=float),
            bin_proportions(current, ref_entry["edges"]),
        )
        ks_stat, ks_p = ks_statistic(ref_entry["sample"], current)
        ref_std = ref_entry["std"] or EPSILON
        current_mean = float(np.nanmean(current)) if current.size else float("nan")

        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "psi_level": classify_psi(psi, psi_warn, psi_alert),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "ks_significant": bool(np.isfinite(ks_p) and ks_p < alpha),
                "reference_mean": ref_entry["mean"],
                "current_mean": current_mean,
                "mean_shift_in_ref_sds": (current_mean - ref_entry["mean"]) / ref_std,
                "missing_from_batch": False,
            }
        )

    table = pd.DataFrame(rows)
    n_alert = int((table["psi_level"] == "ALERT").sum()) if len(table) else 0
    n_warn = int((table["psi_level"] == "WARN").sum()) if len(table) else 0

    return {
        "features": rows,
        "n_features_tested": len(rows),
        "n_features_alert": n_alert,
        "n_features_warn": n_warn,
        "drifted_fraction": n_alert / n_tests,
        "max_psi": float(table["psi"].max()) if len(table) else float("nan"),
        "top_drifted": [
            r["feature"]
            for r in sorted(
                rows, key=lambda r: (-r["psi"] if np.isfinite(r["psi"]) else 0.0)
            )[:5]
        ],
        "ks_alpha_corrected": alpha,
    }


def prediction_drift(reference_profile: dict, scores: Sequence[float], threshold: float) -> dict:
    """Drift in the model's own output — the fastest-moving health signal available.

    Score PSI catches the model seeing a systematically different population. Flag
    rate (share above the operating threshold) catches the operational consequence:
    an analyst queue about to double, or a model that has gone quiet.
    """
    scores = np.asarray(scores, dtype=float)
    ref = reference_profile.get("predictions", {})
    score_psi = psi_from_proportions(
        np.asarray(ref["proportions"], dtype=float),
        bin_proportions(scores, ref["edges"]),
    )

    flag_rate = float((scores >= threshold).mean()) if scores.size else float("nan")
    ref_flag_rate = float(ref.get("flag_rate", float("nan")))
    relative_change = (
        abs(flag_rate - ref_flag_rate) / ref_flag_rate
        if ref_flag_rate and np.isfinite(ref_flag_rate) and ref_flag_rate > 0
        else float("nan")
    )

    return {
        "score_psi": score_psi,
        "mean_score": float(np.nanmean(scores)) if scores.size else float("nan"),
        "reference_mean_score": ref.get("mean_score"),
        "flag_rate": flag_rate,
        "reference_flag_rate": ref_flag_rate,
        "flag_rate_relative_change": relative_change,
        "threshold": float(threshold),
    }
