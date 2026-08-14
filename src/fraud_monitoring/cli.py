"""Command line entrypoints. Every GitHub Actions step calls one of these, so the
whole system is runnable by hand exactly the way CI runs it.

    python -m fraud_monitoring.cli build-reference --training-data data/raw/creditcard.csv
    python -m fraud_monitoring.cli monitor --batch data/incoming/2026-08-14.csv
    python -m fraud_monitoring.cli recalibrate --batch data/incoming/2026-08-14.csv
    python -m fraud_monitoring.cli gate --challenger-dir artifacts/challenger --holdout ...
    python -m fraud_monitoring.cli simulate --kind covariate --magnitude 1.5 --out batch.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_champion, load_champion_from_registry
from .config import REPO_ROOT, abspath, load_config, resolve
from .monitor import load_batch, monitor_batch
from .performance import compute_performance
from .reference import build_reference_profile, load_reference_profile, save_reference_profile
from .report import save_html_report, save_json_report, to_markdown
from .retrain import evaluate_gate, write_promotion_record
from .rules import load_state, save_state
from .simulate import inject_drift, make_synthetic_dataset

logger = logging.getLogger("fraud_monitoring")

EXIT_OK = 0
EXIT_ALERT = 1        # workflow may choose to treat this as failure
EXIT_ERROR = 2


def _champion(config: dict):
    registry = resolve(config, "model.registry", {}) or {}
    artifacts_dir = abspath(config, "model.artifacts_dir")
    if registry.get("enabled"):
        return load_champion_from_registry(
            registry["tracking_uri"], registry["registered_model_name"],
            registry["champion_alias"], artifacts_dir,
        )
    return load_champion(artifacts_dir)


def _emit_github_output(**values) -> None:
    """Expose decisions to downstream workflow steps without parsing logs."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def _emit_job_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(markdown + "\n")


# ------------------------------------------------------------------ build-reference


def cmd_build_reference(args, config: dict) -> int:
    champion = _champion(config)
    training_df = load_batch(Path(args.training_data))

    feature_columns = config["data"]["feature_columns"]
    label_column = config["data"]["label_column"]

    # The reference distribution is the training slice; the reference *score*
    # distribution comes from the held-out slice, because that is what live scoring
    # looks like. Using in-sample scores would set an unreachably confident baseline.
    split_index = int(len(training_df) * args.train_fraction)
    train_slice = training_df.sort_values(config["data"]["time_column"]).reset_index(drop=True)
    eval_slice = train_slice.iloc[split_index:]
    train_slice = train_slice.iloc[:split_index]

    eval_scores = champion.score(eval_slice, feature_columns)
    baseline = compute_performance(eval_slice[label_column], eval_scores, champion.threshold)

    monitored = [f for f in feature_columns if f not in config["data"].get("drift_exclude", [])]
    profile = build_reference_profile(
        train_slice,
        eval_scores,
        champion.threshold,
        monitored,
        bins=config["drift"]["psi"]["bins"],
        label_column=label_column,
        model_version=champion.version,
        baseline_metrics={
            "pr_auc": baseline["pr_auc"],
            "recall": baseline["recall"],
            "precision": baseline["precision"],
            "f1": baseline["f1"],
            "computed_on_rows": baseline["n_rows"],
        },
    )
    path = save_reference_profile(profile, abspath(config, "paths.reference_profile"))
    print(f"Reference profile written to {path}")
    print(f"  model_version   : {champion.version}")
    print(f"  monitored feats : {len(profile['features'])}")
    print(f"  baseline PR-AUC : {baseline['pr_auc']}")
    print(f"  baseline recall : {baseline['recall']:.4f}  precision: {baseline['precision']:.4f}")
    print(f"  reference flag rate: {profile['predictions']['flag_rate']:.5f}")
    return EXIT_OK


# -------------------------------------------------------------------------- monitor


def cmd_monitor(args, config: dict) -> int:
    champion = _champion(config)
    profile = load_reference_profile(abspath(config, "paths.reference_profile"))
    state_path = abspath(config, "paths.state_file")
    state = load_state(state_path)

    batch_path = Path(args.batch)
    batch = load_batch(batch_path)
    batch_id = args.batch_id or batch_path.stem

    report, decision, new_state = monitor_batch(
        batch, champion, profile, config, state, batch_id=batch_id
    )

    reports_dir = abspath(config, "paths.reports_dir")
    json_path = save_json_report(report, reports_dir, batch_id)
    html_path = save_html_report(report, reports_dir, batch_id)
    markdown = to_markdown(report)

    if not args.dry_run:
        save_state(new_state, state_path)

    print(markdown)
    print(f"\nJSON: {json_path}\nHTML: {html_path}")

    _emit_job_summary(markdown)
    _emit_github_output(
        action=decision.action,
        severity=decision.severity,
        batch_id=batch_id,
        needs_retrain=str(decision.action == "RETRAIN").lower(),
        report_path=str(json_path.relative_to(REPO_ROOT)),
    )

    if decision.action in ("RETRAIN", "HALT") and config["alerting"].get("fail_workflow_on_alert"):
        return EXIT_ALERT
    return EXIT_OK


# ---------------------------------------------------------------------- recalibrate


def cmd_recalibrate(args, config: dict) -> int:
    """The cheap remedy: re-cut the operating threshold on a recent labelled window.

    Only valid when ranking quality held up. Writes a proposed threshold.json rather
    than overwriting the live one — promotion stays a reviewed change.
    """
    champion = _champion(config)
    batch = load_batch(Path(args.batch))
    label_column = config["data"]["label_column"]
    if label_column not in batch.columns:
        print(f"Cannot recalibrate: batch has no {label_column} column", file=sys.stderr)
        return EXIT_ERROR

    scores = champion.score(batch, config["data"]["feature_columns"])
    y = batch[label_column].to_numpy(dtype=int)

    rows = []
    for threshold in np.arange(0.05, 1.0, 0.01):
        metrics = compute_performance(y, scores, float(threshold))
        rows.append({"threshold": float(threshold), **{k: metrics[k] for k in
                    ("precision", "recall", "f1", "alert_volume", "missed_fraud_count")}})
    sweep = pd.DataFrame(rows)

    if args.target_recall is not None:
        # Operating-point policy: hold recall, buy back precision. This is usually
        # what a fraud team wants — the recall floor is a risk commitment.
        eligible = sweep[sweep["recall"] >= args.target_recall]
        best = (eligible.sort_values("precision", ascending=False).iloc[0]
                if len(eligible) else sweep.sort_values("recall", ascending=False).iloc[0])
        rationale = f"highest precision subject to recall >= {args.target_recall}"
    else:
        best = sweep.sort_values("f1", ascending=False).iloc[0]
        rationale = "max F1 (same objective the training pipeline uses)"

    out_dir = abspath(config, "model.artifacts_dir").parent / "proposed"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)
    proposal = {
        "threshold": float(best["threshold"]),
        "previous_threshold": champion.threshold,
        "selection_rule": rationale,
        "metrics_at_proposed": {k: float(best[k]) for k in ("precision", "recall", "f1")},
        "model_version": champion.version,
        "proposed_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "threshold.json").write_text(json.dumps(proposal, indent=2))

    print(json.dumps(proposal, indent=2))
    print(f"\nProposal written to {out_dir/'threshold.json'} (not applied — review and promote).")
    _emit_github_output(proposed_threshold=proposal["threshold"])
    return EXIT_OK


# ------------------------------------------------------------------------------ gate


def cmd_gate(args, config: dict) -> int:
    """Champion vs challenger on a shared, most-recent holdout."""
    champion = _champion(config)
    challenger = load_champion(Path(args.challenger_dir))
    holdout = load_batch(Path(args.holdout))

    feature_columns = config["data"]["feature_columns"]
    label_column = config["data"]["label_column"]

    result = evaluate_gate(
        champion_scores=champion.score(holdout, feature_columns),
        challenger_scores=challenger.score(holdout, feature_columns),
        y_holdout=holdout[label_column],
        champion_threshold=champion.threshold,
        challenger_threshold=challenger.threshold,
        config=config,
    )

    record_path = write_promotion_record(
        result,
        abspath(config, "paths.reports_dir") / "promotions" /
        f"gate-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json",
        champion_version=champion.version,
        challenger_version=challenger.version,
    )

    lines = [f"## Promotion gate — {result.summary}", "",
             "| Check | Passed | Value | Threshold |", "|---|---|---|---|"]
    for check in result.checks:
        lines.append(
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | "
            f"{check['value']} | {check['threshold']} |"
        )
    lines += ["", f"Champion PR-AUC: {result.champion_metrics.get('pr_auc')} · "
                  f"Challenger PR-AUC: {result.challenger_metrics.get('pr_auc')}"]
    markdown = "\n".join(lines)

    print(markdown)
    print(f"\nPromotion record: {record_path}")
    _emit_job_summary(markdown)
    _emit_github_output(promote=str(result.promote).lower())
    return EXIT_OK if result.promote else EXIT_ALERT


# -------------------------------------------------------------------------- simulate


def cmd_simulate(args, config: dict) -> int:
    df = (
        make_synthetic_dataset(args.rows, args.fraud_rate, random_state=args.seed)
        if args.source is None
        else load_batch(Path(args.source)).sample(
            n=min(args.rows, len(load_batch(Path(args.source)))), random_state=args.seed
        )
    )
    if args.kind != "none":
        df = inject_drift(df, kind=args.kind, magnitude=args.magnitude, random_state=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({int(df['Class'].sum())} frauds) to {out} "
          f"[drift={args.kind}, magnitude={args.magnitude}]")
    return EXIT_OK


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud_monitoring", description=__doc__)
    parser.add_argument("--env", default=None, help="config overlay: dev | staging | prod")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-reference", help="freeze the reference profile for the champion")
    p.add_argument("--training-data", required=True)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.set_defaults(func=cmd_build_reference)

    p = sub.add_parser("monitor", help="run one monitoring window over a batch")
    p.add_argument("--batch", required=True)
    p.add_argument("--batch-id", default=None)
    p.add_argument("--dry-run", action="store_true", help="do not persist state")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("recalibrate", help="propose a new operating threshold")
    p.add_argument("--batch", required=True)
    p.add_argument("--target-recall", type=float, default=None)
    p.set_defaults(func=cmd_recalibrate)

    p = sub.add_parser("gate", help="champion vs challenger promotion gate")
    p.add_argument("--challenger-dir", required=True)
    p.add_argument("--holdout", required=True)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("simulate", help="generate a (optionally drifted) batch")
    p.add_argument("--out", required=True)
    p.add_argument("--source", default=None, help="sample from this CSV instead of synthesising")
    p.add_argument("--rows", type=int, default=20000)
    p.add_argument("--fraud-rate", type=float, default=0.0017)
    p.add_argument("--kind", default="none",
                   choices=["none", "covariate", "amount", "prior", "concept"])
    p.add_argument("--magnitude", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    config = load_config(args.env)
    try:
        return args.func(args, config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
