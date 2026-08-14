"""Builds the reference profile — the frozen snapshot every batch is compared to.

Why a profile and not the raw training set: the profile is a few hundred KB of JSON
that can live in git next to the model version it describes, so drift is always
measured against the exact distribution the *current champion* was trained on. Swap
the champion, rebuild the profile; the two move together and can never silently
disagree.

Per feature it stores the reference quantile bin edges, the expected mass in each
bin, mean/std, and a bounded random subsample used for the KS test (the full column
would bloat the file and KS saturates well before 20k points).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .drift import bin_proportions, quantile_bin_edges

logger = logging.getLogger(__name__)

KS_SAMPLE_SIZE = 20_000

# Score bins are FIXED bands, not quantiles — and this is a deliberate departure
# from how the feature bins are cut.
#
# A well-separated fraud model puts ~99% of its mass in the near-zero region, so
# equal-mass quantile bins spend nineteen of twenty bins resolving the difference
# between p=1.5e-5 and p=4e-5. Nobody makes a decision in that range; both are
# "obviously not fraud". Measured that way, score PSI hits 0.5+ on two honest
# samples of the *same* population — a monitor that alarms every single day, which
# is a monitor everyone learns to ignore.
#
# Fixed bands anchored on decision-relevant probabilities collapse that noise into
# one bucket and reserve resolution for the range a human would actually act on.
# The bands are also readable in an alert: "mass moved from the 1-5% band into the
# 25-50% band" means something; "bin 14 grew" does not.
SCORE_BANDS = [-np.inf, 1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, np.inf]


def _feature_entry(values: pd.Series, bins: int, rng: np.random.Generator) -> dict[str, Any]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    edges = quantile_bin_edges(array, bins=bins)
    sample = (
        rng.choice(array, size=KS_SAMPLE_SIZE, replace=False)
        if array.size > KS_SAMPLE_SIZE
        else array
    )
    return {
        "edges": edges,
        "proportions": [float(p) for p in bin_proportions(array, edges)],
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "p01": float(np.quantile(array, 0.01)),
        "p50": float(np.quantile(array, 0.50)),
        "p99": float(np.quantile(array, 0.99)),
        "n": int(array.size),
        "sample": [float(v) for v in sample],
    }


def build_reference_profile(
    training_df: pd.DataFrame,
    scores: Sequence[float],
    threshold: float,
    monitored_features: Sequence[str],
    *,
    bins: int = 10,
    label_column: str = "Class",
    model_version: str = "unknown",
    baseline_metrics: dict | None = None,
    random_state: int = 42,
) -> dict:
    """`training_df` should be the exact data the champion trained on (pre-scaling,
    so drift is reported in the units a human can reason about), and `scores` the
    champion's probabilities on the held-out evaluation window."""
    rng = np.random.default_rng(random_state)
    scores = np.asarray(scores, dtype=float)

    features = {
        feature: _feature_entry(training_df[feature], bins, rng)
        for feature in monitored_features
        if feature in training_df.columns
    }

    # Laplace floor on the reference proportions: an empty band in the reference
    # must not make PSI explode the first time a single transaction lands there.
    score_floor = 0.5 / max(scores.size, 1)
    score_proportions = np.clip(bin_proportions(scores, SCORE_BANDS), score_floor, None)

    fraud_rate = (
        float(training_df[label_column].mean()) if label_column in training_df.columns else None
    )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "n_reference_rows": int(len(training_df)),
        "reference_fraud_rate": fraud_rate,
        "operating_threshold": float(threshold),
        "monitored_features": list(features.keys()),
        "features": features,
        "predictions": {
            "edges": [float(e) for e in SCORE_BANDS],
            "binning": "fixed_decision_bands",
            "proportions": [float(p) for p in score_proportions],
            "mean_score": float(scores.mean()),
            "flag_rate": float((scores >= threshold).mean()),
            "n": int(scores.size),
        },
        "baseline_metrics": baseline_metrics or {},
    }


def save_reference_profile(profile: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    logger.info(
        "Wrote reference profile for model_version=%s (%s features, %s rows) to %s",
        profile.get("model_version"),
        len(profile.get("features", {})),
        profile.get("n_reference_rows"),
        path,
    )
    return path


def load_reference_profile(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No reference profile at {path}. Run `make reference` (or "
            "`python -m fraud_monitoring.cli build-reference`) after promoting a champion."
        )
    with open(path) as f:
        return json.load(f)
