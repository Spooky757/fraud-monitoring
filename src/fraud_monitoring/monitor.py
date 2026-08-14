"""Orchestrates one monitoring window: load champion -> score -> measure -> decide.

One run of `monitor_batch` == one row in the monitoring timeline. It is deliberately
side-effect-light: it returns a report dict, and the caller (cli.py) decides what to
persist. That makes it trivial to replay a month of historical batches through the
same code path when tuning thresholds.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .artifacts import Champion
from .drift import feature_drift, prediction_drift
from .performance import performance_report
from .rules import Decision, collect_signals, decide

logger = logging.getLogger(__name__)


def load_batch(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def monitor_batch(
    batch: pd.DataFrame,
    champion: Champion,
    reference_profile: dict,
    config: dict,
    state: dict,
    *,
    batch_id: str = "unnamed",
    now: datetime | None = None,
) -> tuple[dict, Decision, dict]:
    now = now or datetime.now(timezone.utc)
    data_cfg = config["data"]
    feature_columns = data_cfg["feature_columns"]
    monitored = [
        f for f in reference_profile.get("monitored_features", [])
        if f not in data_cfg.get("drift_exclude", [])
    ]

    scores = champion.score(batch, feature_columns)

    drift_block = feature_drift(
        reference_profile,
        batch,
        monitored,
        psi_warn=config["drift"]["psi"]["warn"],
        psi_alert=config["drift"]["psi"]["alert"],
        ks_alpha=config["drift"]["ks"]["alpha"],
        ks_correction=config["drift"]["ks"]["correction"],
    )
    prediction_block = prediction_drift(reference_profile, scores, champion.threshold)

    baseline = {
        **(reference_profile.get("baseline_metrics") or {}),
        **{k: v for k, v in (config["performance"].get("baseline") or {}).items() if v is not None},
    }
    performance_block = performance_report(
        batch,
        scores,
        champion.threshold,
        baseline,
        label_column=data_cfg["label_column"],
        min_labeled_rows=config["performance"]["min_labeled_rows"],
        min_labeled_frauds=config["performance"]["min_labeled_frauds"],
    )

    report = {
        "batch_id": batch_id,
        "evaluated_at": now.isoformat(),
        "environment": config.get("environment", "dev"),
        "model_version": champion.version,
        "reference_model_version": reference_profile.get("model_version"),
        "n_rows": int(len(batch)),
        "drift": drift_block,
        "prediction": prediction_block,
        "performance": performance_block,
    }

    if report["model_version"] != report["reference_model_version"]:
        logger.warning(
            "Champion (%s) does not match the reference profile (%s) — rebuild the "
            "reference profile after promoting a model.",
            report["model_version"],
            report["reference_model_version"],
        )
        report["reference_mismatch"] = True

    signals = collect_signals(report, config)
    decision, new_state = decide(signals, config, state, now=now)
    report["decision"] = decision.to_dict()
    return report, decision, new_state
