"""Loading the champion, and applying it exactly the way the training repo does.

This module is the seam between the two repos. The training repo
(Credit-Card-Fraud-testing) writes `model.pkl`, `scaler.pkl`, `threshold.json` via
`fraud_pipeline.inference.save_inference_artifacts()`. This repo consumes them and
must reproduce that transform bit-for-bit — the scaler is fit on training Amount
only, and applying a differently-fitted scaler is a silent accuracy leak that looks
exactly like drift. Hence: never re-fit anything here, only load and apply.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Champion:
    model: Any
    scaler: Any
    threshold: float
    version: str
    source: str

    def score(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
        return score_batch(df, self.model, self.scaler, feature_columns)


def _fingerprint(path: Path) -> str:
    """Short content hash — a model 'version' that is true even when nobody
    remembered to bump a tag."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def load_champion(artifacts_dir: Path) -> Champion:
    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / "model.pkl"
    scaler_path = artifacts_dir / "scaler.pkl"
    threshold_path = artifacts_dir / "threshold.json"

    missing = [p.name for p in (model_path, scaler_path, threshold_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Champion artifacts incomplete in {artifacts_dir}: missing {', '.join(missing)}. "
            "Fetch them from the training repo's pipeline output (models/) or its "
            "release assets — see README -> 'Wiring the two repos together'."
        )

    with open(threshold_path) as f:
        threshold = float(json.load(f)["threshold"])

    version_file = artifacts_dir / "VERSION"
    version = (
        version_file.read_text().strip() if version_file.exists() else _fingerprint(model_path)
    )

    logger.info("Loaded champion %s from %s (threshold=%.3f)", version, artifacts_dir, threshold)
    return Champion(
        model=joblib.load(model_path),
        scaler=joblib.load(scaler_path),
        threshold=threshold,
        version=version,
        source=str(artifacts_dir),
    )


def load_champion_from_registry(
    tracking_uri: str, model_name: str, alias: str, artifacts_dir: Path
) -> Champion:
    """Pull the aliased model from the training repo's MLflow registry.

    The scaler and threshold still come from `artifacts_dir` — they are pipeline
    artifacts, not the model object, and MLflow's registry versions only the latter.
    """
    import mlflow  # imported lazily: the file-based path must not require mlflow

    mlflow.set_tracking_uri(tracking_uri)
    uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(uri)

    with open(Path(artifacts_dir) / "threshold.json") as f:
        threshold = float(json.load(f)["threshold"])

    return Champion(
        model=model,
        scaler=joblib.load(Path(artifacts_dir) / "scaler.pkl"),
        threshold=threshold,
        version=f"{model_name}@{alias}",
        source=uri,
    )


def score_batch(
    df: pd.DataFrame, model: Any, scaler: Any, feature_columns: Sequence[str]
) -> np.ndarray:
    """Reproduces the training-time transform, then returns fraud probabilities.

    Column order is enforced explicitly. XGBoost keys on feature names when it has
    them, but reordering columns is the classic way a monitoring job quietly reports
    garbage, so this is not left to chance.
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Batch is missing model features: {', '.join(missing)}")

    features = df.loc[:, list(feature_columns)].copy()
    features["Amount"] = scaler.transform(features[["Amount"]])

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return np.asarray(model.predict(features), dtype=float).ravel()  # mlflow pyfunc
