"""Retraining: what data, what gate, and who is allowed to promote.

The retraining *trigger* lives in rules.py. This module answers the harder question —
given that we are going to retrain, how do we avoid shipping something worse?

Three rules the design leans on:

1. **The challenger never auto-deploys.** It is trained, evaluated, and gated; a pass
   produces a promotion candidate and a PR. A fraud model is a money-and-customers
   decision surface, and the failure mode of a bad automatic promotion (silently
   letting fraud through, or freezing thousands of legitimate cards) is much worse
   than a day of staleness.

2. **Both models are judged on the same untouched, most-recent window.** The holdout
   is the last `holdout_days` of data, excluded from the challenger's training set.
   Comparing a fresh challenger against the champion's *historical* metrics would be
   comparing across different populations — which is precisely the drift we are
   supposed to be measuring.

3. **Better is not enough; it must not regress.** Fraud teams tune an operating point
   around a review capacity. A challenger with higher PR-AUC but lower recall at that
   operating point is not an upgrade, it is a different trade-off that a human has to
   consent to.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .performance import compute_performance

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    promote: bool
    checks: list[dict] = field(default_factory=list)
    champion_metrics: dict = field(default_factory=dict)
    challenger_metrics: dict = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_training_window(
    history: pd.DataFrame,
    *,
    time_column: str = "Time",
    rolling_window_days: int | None = 180,
    holdout_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split accumulated history into (challenger training set, shared holdout).

    Time-ordered, never random: a random split would let the model peek at the future,
    and fraud is overwhelmingly a temporal problem. The holdout is the *most recent*
    slice precisely because that is the population the next deployment will face.
    """
    history = history.sort_values(time_column).reset_index(drop=True)
    seconds = history[time_column].to_numpy(dtype=float)
    latest = seconds.max()

    holdout_cutoff = latest - holdout_days * 86_400
    holdout = history[history[time_column] > holdout_cutoff]
    train = history[history[time_column] <= holdout_cutoff]

    if rolling_window_days:
        window_start = holdout_cutoff - rolling_window_days * 86_400
        train = train[train[time_column] > window_start]

    logger.info(
        "Training window: %s rows (%s frauds); holdout: %s rows (%s frauds)",
        len(train), int(train["Class"].sum()) if "Class" in train else -1,
        len(holdout), int(holdout["Class"].sum()) if "Class" in holdout else -1,
    )
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def check_training_data(train: pd.DataFrame, config: dict, label_column: str = "Class") -> list[dict]:
    """Pre-flight checks — refuse to train on data that cannot support a good model."""
    cfg = config["retraining"]
    n_rows, n_frauds = len(train), int(train[label_column].sum())
    return [
        {
            "name": "min_training_rows",
            "passed": n_rows >= cfg["min_training_rows"],
            "value": n_rows,
            "threshold": cfg["min_training_rows"],
            "detail": f"{n_rows} rows in the training window",
        },
        {
            "name": "min_training_frauds",
            "passed": n_frauds >= cfg["min_training_frauds"],
            "value": n_frauds,
            "threshold": cfg["min_training_frauds"],
            "detail": f"{n_frauds} positive examples — too few and the model memorises them",
        },
    ]


def evaluate_gate(
    champion_scores: Sequence[float],
    challenger_scores: Sequence[float],
    y_holdout: Sequence[int],
    champion_threshold: float,
    challenger_threshold: float,
    config: dict,
    extra_checks: Sequence[dict] = (),
) -> GateResult:
    """The promotion decision. Pure function — no IO, no model objects, fully testable."""
    gate_cfg = config["retraining"]["gate"]
    y_holdout = np.asarray(y_holdout, dtype=int)

    champion = compute_performance(y_holdout, champion_scores, champion_threshold)
    challenger = compute_performance(y_holdout, challenger_scores, challenger_threshold)

    checks: list[dict] = list(extra_checks)

    champ_pr = champion.get("pr_auc")
    chall_pr = challenger.get("pr_auc")
    improvement = (
        float(chall_pr - champ_pr) if champ_pr is not None and chall_pr is not None else None
    )
    checks.append(
        {
            "name": "pr_auc_improvement",
            "passed": improvement is not None and improvement >= gate_cfg["min_pr_auc_improvement"],
            "value": improvement,
            "threshold": gate_cfg["min_pr_auc_improvement"],
            "detail": (
                f"challenger PR-AUC {chall_pr} vs champion {champ_pr}. A challenger that is "
                "merely equal is not worth the deployment risk."
            ),
        }
    )

    recall_regression = float(champion["recall"] - challenger["recall"])
    checks.append(
        {
            "name": "recall_not_regressed",
            "passed": recall_regression <= gate_cfg["max_recall_regression"],
            "value": recall_regression,
            "threshold": gate_cfg["max_recall_regression"],
            "detail": (
                f"recall {challenger['recall']:.4f} vs champion {champion['recall']:.4f} at each "
                "model's own operating threshold"
            ),
        }
    )

    checks.append(
        {
            "name": "precision_floor",
            "passed": challenger["precision"] >= gate_cfg["min_precision"],
            "value": challenger["precision"],
            "threshold": gate_cfg["min_precision"],
            "detail": (
                f"{challenger['alert_volume']} alerts on the holdout window; precision below the "
                "floor means the review queue drowns"
            ),
        }
    )

    promote = all(c["passed"] for c in checks)
    failed = [c["name"] for c in checks if not c["passed"]]
    summary = (
        "PROMOTE — challenger passed every gate check"
        if promote
        else f"REJECT — failed: {', '.join(failed)}. Champion stays live."
    )

    return GateResult(
        promote=promote,
        checks=checks,
        champion_metrics=champion,
        challenger_metrics=challenger,
        summary=summary,
    )


def run_training_repo(
    repo_url: str,
    ref: str,
    workdir: Path,
    data_csv: Path,
    *,
    timeout: int = 3600,
) -> Path:
    """Clone the training repo at a pinned ref and run its pipeline on our window.

    The training code is *not* vendored here on purpose: the challenger must be built
    by the same pipeline that built the champion, or the comparison is meaningless.
    Pinning `ref` makes a retrain reproducible after the fact.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    repo_dir = workdir / "training-repo"

    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, repo_url, str(repo_dir)],
            check=True, timeout=timeout,
        )

    raw_dir = repo_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "creditcard.csv"
    if Path(data_csv).resolve() != target.resolve():
        target.write_bytes(Path(data_csv).read_bytes())

    subprocess.run(
        ["python", "-m", "src.fraud_pipeline.pipeline"],
        cwd=repo_dir, check=True, timeout=timeout,
        env={**_clean_env(), "PYTHONPATH": str(repo_dir)},
    )

    models_dir = repo_dir / "models"
    if not (models_dir / "model.pkl").exists():
        raise RuntimeError(f"Training pipeline produced no model.pkl in {models_dir}")
    return models_dir


def _clean_env() -> dict:
    import os

    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}


def write_promotion_record(
    gate: GateResult, path: Path, *, champion_version: str, challenger_version: str
) -> Path:
    """The audit trail. Every promotion decision — including rejections — is a file in
    git, so 'why is this model live?' has an answer six months later."""
    import json

    record = {
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "champion_version": champion_version,
        "challenger_version": challenger_version,
        "promote": gate.promote,
        "summary": gate.summary,
        "checks": gate.checks,
        "champion_metrics": gate.champion_metrics,
        "challenger_metrics": gate.challenger_metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
