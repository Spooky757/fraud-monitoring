#!/usr/bin/env bash
# ============================================================================
# run_full_pipeline.sh — End-to-end: train → copy artifacts → monitor
#
# This script chains the training repo's pipeline with the monitoring repo's
# CLI so everything runs in one go. It is a LOCAL DEMO — for CI automation,
# see the GitHub Actions workflows (release.yml, monitor.yml, retrain.yml).
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh               # uses paths below
#   TRAINING_REPO=../Credit-Card-Fraud-testing ./run_full_pipeline.sh
#
# Prerequisites:
#   - Python 3.10+
#   - creditcard.csv in the training repo's data/raw/ directory
#     (download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
#   - pip install -r requirements.txt  (in BOTH repos)
# ============================================================================
set -euo pipefail

# ---------- configurable paths ----------
TRAINING_REPO="${TRAINING_REPO:-../Credit-Card-Fraud-testing}"
MONITORING_REPO="${MONITORING_REPO:-.}"                   # run this from the monitoring repo root
ENV="${ENV:-dev}"                                         # dev is less strict for demos

# ---------- colours (optional, degrades gracefully) ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
step() { echo -e "\n${CYAN}━━━ Step $1: $2 ━━━${NC}"; }
ok()   { echo -e "${GREEN}✔ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✘ $1${NC}"; exit 1; }

# ---------- sanity checks ----------
[[ -d "$TRAINING_REPO/src/fraud_pipeline" ]] || fail "Training repo not found at $TRAINING_REPO"
[[ -d "$MONITORING_REPO/src/fraud_monitoring" ]] || fail "Run this script from the monitoring repo root (or set MONITORING_REPO)"

TRAINING_DATA="$TRAINING_REPO/data/raw/creditcard.csv"
if [[ ! -f "$TRAINING_DATA" ]]; then
    warn "creditcard.csv not found at $TRAINING_DATA"
    echo "  Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo "  Place it at:      $TRAINING_DATA"
    echo ""
    echo "Generating SYNTHETIC training data instead (results will differ from real data)..."
    TRAINING_DATA="$TRAINING_REPO/data/raw/creditcard.csv"
    PYTHONPATH="$MONITORING_REPO/src" python -m fraud_monitoring.cli simulate \
        --out "$TRAINING_DATA" --rows 50000 --fraud-rate 0.0017 --kind none
    ok "Synthetic dataset written to $TRAINING_DATA"
fi

# ==============================
# STEP 1 — Train the model
# ==============================
step 1 "Training the fraud model"
cd "$TRAINING_REPO"
PYTHONPATH=. python -m src.fraud_pipeline.pipeline
ok "Model trained — artifacts in $TRAINING_REPO/models/"
ls -lh models/model.pkl models/scaler.pkl models/threshold.json
cd - > /dev/null

# ==============================
# STEP 2 — Copy artifacts to monitoring repo
# ==============================
step 2 "Copying champion artifacts to monitoring repo"
mkdir -p "$MONITORING_REPO/artifacts/champion"
cp "$TRAINING_REPO/models/model.pkl"       "$MONITORING_REPO/artifacts/champion/model.pkl"
cp "$TRAINING_REPO/models/scaler.pkl"      "$MONITORING_REPO/artifacts/champion/scaler.pkl"
cp "$TRAINING_REPO/models/threshold.json"  "$MONITORING_REPO/artifacts/champion/threshold.json"

# Tag the version so the monitoring system knows which model it is watching
COMMIT_HASH=$(git -C "$TRAINING_REPO" rev-parse --short HEAD 2>/dev/null || echo "local")
echo "$COMMIT_HASH" > "$MONITORING_REPO/artifacts/champion/VERSION"
ok "Champion artifacts copied (version: $COMMIT_HASH)"

# ==============================
# STEP 3 — Build the reference profile
# ==============================
step 3 "Building reference profile"
cd "$MONITORING_REPO"
PYTHONPATH=src python -m fraud_monitoring.cli --env "$ENV" build-reference \
    --training-data "$TRAINING_DATA"
ok "Reference profile frozen"

# ==============================
# STEP 4 — Generate test batches and run monitoring
# ==============================
step 4 "Generating test batches and running monitoring"

echo -e "\n${YELLOW}--- Batch 1: Healthy (no drift) ---${NC}"
PYTHONPATH=src python -m fraud_monitoring.cli simulate \
    --out data/incoming/healthy.csv --rows 20000 --kind none
PYTHONPATH=src python -m fraud_monitoring.cli --env "$ENV" monitor \
    --batch data/incoming/healthy.csv --batch-id healthy-1

echo -e "\n${YELLOW}--- Batch 2: Covariate drift (feature distributions shift) ---${NC}"
PYTHONPATH=src python -m fraud_monitoring.cli simulate \
    --out data/incoming/covariate.csv --rows 20000 --kind covariate --magnitude 1.5
PYTHONPATH=src python -m fraud_monitoring.cli --env "$ENV" monitor \
    --batch data/incoming/covariate.csv --batch-id covariate-1

echo -e "\n${YELLOW}--- Batch 3: Concept drift (labels rewired, features unchanged) ---${NC}"
PYTHONPATH=src python -m fraud_monitoring.cli simulate \
    --out data/incoming/concept.csv --rows 20000 --kind concept --magnitude 0.8
PYTHONPATH=src python -m fraud_monitoring.cli --env "$ENV" monitor \
    --batch data/incoming/concept.csv --batch-id concept-1

ok "All monitoring windows complete"

# ==============================
# STEP 5 — Summary
# ==============================
step 5 "Summary"
echo ""
echo "Reports generated:"
ls -1 monitoring/reports/*.html 2>/dev/null || echo "  (none)"
echo ""
echo "State file:"
cat monitoring/state/monitor_state.json 2>/dev/null || echo "  (not yet created)"
echo ""
echo -e "${GREEN}━━━ Pipeline complete ━━━${NC}"
echo "Open any .html report in your browser to see the full drift analysis."
echo ""
echo "Next steps:"
echo "  • To recalibrate:  PYTHONPATH=src python -m fraud_monitoring.cli --env $ENV recalibrate --batch <labelled.csv>"
echo "  • To run more batches: PYTHONPATH=src python -m fraud_monitoring.cli --env $ENV monitor --batch <new-batch.csv>"
echo "  • For GitHub Actions automation: see release.yml + monitor.yml + retrain.yml"
