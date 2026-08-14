#!/usr/bin/env python
"""End-to-end walkthrough on synthetic data — no Kaggle download required.

Trains a stand-in champion the same way the training repo does, freezes a reference
profile, then feeds the monitor a healthy batch followed by increasingly drifted
ones so you can watch the escalation ladder actually climb:

    NO_ACTION -> WATCH -> RETRAIN/RECALIBRATE

Run: make demo
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import json  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from fraud_monitoring.artifacts import Champion  # noqa: E402
from fraud_monitoring.config import REPO_ROOT, load_config  # noqa: E402
from fraud_monitoring.monitor import monitor_batch  # noqa: E402
from fraud_monitoring.performance import compute_performance  # noqa: E402
from fraud_monitoring.reference import build_reference_profile, save_reference_profile  # noqa: E402
from fraud_monitoring.report import save_html_report, save_json_report  # noqa: E402
from fraud_monitoring.rules import load_state, save_state  # noqa: E402
from fraud_monitoring.simulate import FEATURE_COLUMNS, inject_drift, make_synthetic_dataset  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def train_stand_in_champion(config: dict) -> tuple[Champion, dict, dict]:
    print("→ Training a stand-in champion on synthetic data (mirrors the training repo)")
    df = make_synthetic_dataset(60_000, fraud_rate=0.005, random_state=42)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    scaler = StandardScaler()
    X_train = train[FEATURE_COLUMNS].copy()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    positives = int(train["Class"].sum())

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", random_state=42,
        scale_pos_weight=(len(train) - positives) / positives,
    )
    model.fit(X_train, train["Class"])

    artifacts_dir = REPO_ROOT / config["model"]["artifacts_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "model.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    (artifacts_dir / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
    (artifacts_dir / "VERSION").write_text("demo-champion")

    champion = Champion(model, scaler, 0.5, "demo-champion", str(artifacts_dir))
    eval_scores = champion.score(test, FEATURE_COLUMNS)
    baseline = compute_performance(test["Class"], eval_scores, 0.5)
    print(f"   baseline PR-AUC {baseline['pr_auc']:.4f} · recall {baseline['recall']:.4f} "
          f"· precision {baseline['precision']:.4f}")

    monitored = [f for f in FEATURE_COLUMNS if f not in config["data"]["drift_exclude"]]
    profile = build_reference_profile(
        train, eval_scores, 0.5, monitored, model_version="demo-champion",
        baseline_metrics={"pr_auc": baseline["pr_auc"], "recall": baseline["recall"],
                          "precision": baseline["precision"]},
    )
    save_reference_profile(profile, REPO_ROOT / config["paths"]["reference_profile"])
    print(f"   reference profile: {len(profile['features'])} features, "
          f"flag rate {profile['predictions']['flag_rate']:.5f}\n")
    return champion, profile, baseline


def main() -> int:
    config = load_config("dev")
    champion, profile, _ = train_stand_in_champion(config)

    scenarios = [
        ("healthy-1", dict(kind=None), "same population — nothing should fire"),
        ("healthy-2", dict(kind=None), "still healthy"),
        ("covariate-1", dict(kind="covariate", magnitude=1.5), "8 features shifted"),
        ("covariate-2", dict(kind="covariate", magnitude=1.5), "still shifted — confirmation"),
        ("concept-1", dict(kind="concept", magnitude=0.7), "inputs normal, labels rewired"),
        ("concept-2", dict(kind="concept", magnitude=0.7), "confirmed concept drift"),
    ]

    state = load_state(REPO_ROOT / config["paths"]["state_file"])
    reports_dir = REPO_ROOT / config["paths"]["reports_dir"]

    print(f"{'batch':<14}{'PSI feats':<11}{'max PSI':<10}{'recall':<10}{'decision':<24}note")
    print("-" * 100)

    for i, (name, drift_kwargs, note) in enumerate(scenarios):
        batch = make_synthetic_dataset(20_000, fraud_rate=0.005, random_state=1000 + i)
        if drift_kwargs.get("kind"):
            batch = inject_drift(batch, random_state=1000 + i, **drift_kwargs)

        report, decision, state = monitor_batch(
            batch, champion, profile, config, state, batch_id=name
        )
        save_json_report(report, reports_dir, name)
        save_html_report(report, reports_dir, name)

        drift = report["drift"]
        perf = report["performance"]
        recall = (
            f"{perf['metrics']['recall']:.3f}" if perf.get("status") == "OK" else "n/a"
        )
        print(
            f"{name:<14}{drift['n_features_alert']}/{drift['n_features_tested']:<9}"
            f"{drift['max_psi']:<10.3f}{recall:<10}{decision.action:<24}{note}"
        )

    save_state(state, REPO_ROOT / config["paths"]["state_file"])
    print(f"\nReports written to {reports_dir.relative_to(REPO_ROOT)}/ — open any .html file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
