"""Synthetic batch generation — so the monitoring loop is testable without the
284,807-row Kaggle file, and so drift detection can be proven to fire on demand.

A monitoring system nobody has ever seen alarm is not a monitoring system, it is a
hope. `inject_drift` produces batches with known, controllable corruption:

  covariate  — shift/scale the PCA features (upstream feature pipeline changed)
  amount     — inflate transaction values (inflation, new merchant mix, FX)
  prior      — change the fraud base rate (a new attack campaign)
  concept    — decouple the label from the learned signal (fraudsters adapted; the
               same features no longer mean the same thing — the one kind of drift
               no amount of recalibration fixes)
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def make_synthetic_dataset(
    n_rows: int = 40_000,
    fraud_rate: float = 0.0017,
    *,
    start_time: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """A stand-in for creditcard.csv with the same schema and a learnable signal.

    Fraud is encoded in a handful of components (mirroring how V14/V17/V12 carry most
    of the real dataset's signal) so a model trained on this is genuinely predictive
    rather than guessing — otherwise the gate tests would be vacuous.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(int(round(n_rows * fraud_rate)), 1)
    labels = np.zeros(n_rows, dtype=int)
    labels[rng.choice(n_rows, size=n_fraud, replace=False)] = 1

    data = {f"V{i}": rng.normal(0, 1, n_rows) for i in range(1, 29)}
    signal_features = {"V14": -3.2, "V17": -2.6, "V12": -2.1, "V4": 1.8, "V10": -1.5}
    for feature, shift in signal_features.items():
        data[feature] = data[feature] + labels * shift

    amount = rng.lognormal(mean=3.0, sigma=1.2, size=n_rows)
    amount[labels == 1] *= rng.uniform(1.2, 3.0, size=int(labels.sum()))

    df = pd.DataFrame(data)
    df["Time"] = np.sort(rng.uniform(0, 172_800, n_rows)) + start_time
    df["Amount"] = np.round(amount, 2)
    df["Class"] = labels
    return df[FEATURE_COLUMNS + ["Class"]]


def inject_drift(
    df: pd.DataFrame,
    *,
    kind: str = "covariate",
    magnitude: float = 1.0,
    features: Sequence[str] | None = None,
    random_state: int = 7,
) -> pd.DataFrame:
    """Return a copy of `df` with a known distortion applied.

    `magnitude` is in reference standard deviations for covariate drift, a multiplier
    for amount drift, and a multiplier on the base rate for prior drift.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()

    if kind == "covariate":
        targets = list(features or ["V1", "V2", "V3", "V5", "V6", "V7", "V9", "V11"])
        for feature in targets:
            if feature in out.columns:
                out[feature] = out[feature] * (1 + 0.35 * magnitude) + magnitude
    elif kind == "amount":
        out["Amount"] = np.round(out["Amount"] * magnitude, 2)
    elif kind == "prior":
        fraud_idx = out.index[out["Class"] == 1]
        legit_idx = out.index[out["Class"] == 0]
        target_n = int(len(fraud_idx) * magnitude)
        if target_n > len(fraud_idx):  # duplicate frauds to raise the base rate
            extra = rng.choice(fraud_idx, size=target_n - len(fraud_idx), replace=True)
            out = pd.concat([out, out.loc[extra]], ignore_index=True)
        elif target_n < len(fraud_idx):
            drop = rng.choice(fraud_idx, size=len(fraud_idx) - target_n, replace=False)
            out = out.drop(index=drop).reset_index(drop=True)
        _ = legit_idx
    elif kind == "concept":
        # Same feature values, relabelled: the learned mapping is now wrong. Detectable
        # only once labels mature — which is exactly the point of the demo.
        fraud_idx = out.index[out["Class"] == 1]
        flip_n = int(len(fraud_idx) * min(magnitude, 1.0))
        flip = rng.choice(fraud_idx, size=flip_n, replace=False)
        out.loc[flip, "Class"] = 0
        legit = out.index[out["Class"] == 0]
        new_frauds = rng.choice(legit, size=flip_n, replace=False)
        out.loc[new_frauds, "Class"] = 1
    else:
        raise ValueError(f"Unknown drift kind: {kind!r}")

    logger.info("Injected %s drift (magnitude=%s) -> %s rows", kind, magnitude, len(out))
    return out
