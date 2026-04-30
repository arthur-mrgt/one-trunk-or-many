#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper:
# 1) run benchmark to create a run_id
# 2) precompute null distribution for that run_id
# 3) re-run metrics with null enrichment
#
# Usage:
#   bash scripts/run_null_pipeline.sh [n_scenes] [n_draws] [tracking] [run_id] [reuse_if_exists] [name]
# Examples:
#   bash scripts/run_null_pipeline.sh 10 500 wandb_on "" true "B"
#   bash scripts/run_null_pipeline.sh 10 500 wandb_off rq1_cka_pipeline-20260430-103348 false "A_rerun"

N_SCENES="${1:-2}"
N_DRAWS="${2:-200}"
TRACKING="${3:-wandb_off}"
RUN_ID="${4:-}"
REUSE_IF_EXISTS="${5:-true}"
RUN_NAME="${6:-}"

if [[ -z "${RUN_ID}" ]]; then
  echo "[INFO] Step 1/3: running benchmark (n_scenes=${N_SCENES}, tracking=${TRACKING})"
  python -m src.run_benchmark "data.n_scenes=${N_SCENES}" "tracking=${TRACKING}"

  RUN_ID="$(ls -1t results/runs | grep -v null_distributions | grep -v hydra | head -n 1)"
  if [[ -z "${RUN_ID}" ]]; then
    echo "[ERROR] Could not determine RUN_ID from results/runs"
    exit 1
  fi
else
  echo "[INFO] Step 1/3: skipped benchmark; using provided RUN_ID=${RUN_ID}"
fi

echo "[INFO] RUN_ID=${RUN_ID}"
echo "[INFO] Step 2/3: precomputing null distribution (n_draws=${N_DRAWS}, reuse_if_exists=${REUSE_IF_EXISTS})"
python -m src.run_null_distribution \
  analysis.null_distribution.enabled=true \
  "runtime.metrics_input_run_id=${RUN_ID}" \
  "analysis.null_distribution.n_draws=${N_DRAWS}" \
  "analysis.null_distribution.reuse_if_exists=${REUSE_IF_EXISTS}"

echo "[INFO] Step 3/3: running metrics with null enrichment"
WANDB_RUN_NAME="${RUN_ID}-${N_SCENES}-${N_DRAWS}"
NULL_FILENAME="null_distribution_${N_SCENES}scenes_${N_DRAWS}draws.csv"
echo "[INFO] W&B run name: ${WANDB_RUN_NAME}"
echo "[INFO] Null file: ${NULL_FILENAME}"
python -m src.run_metrics \
  "runtime.metrics_input_run_id=${RUN_ID}" \
  "runtime.null_input_run_id=${RUN_ID}" \
  "runtime.null_input_filename=${NULL_FILENAME}" \
  "tracking=${TRACKING}" \
  "tracking.wandb.run_name=${WANDB_RUN_NAME}" \
  runtime.null_missing_behavior=error

# Register run in registry
REGISTRY="results/runs_registry.json"
if [[ -n "${RUN_NAME}" ]]; then
  python - <<PY
import json, pathlib
reg = pathlib.Path("${REGISTRY}")
data = json.loads(reg.read_text()) if reg.exists() else {}
entry = data.get("${RUN_ID}", {
    "name": "${RUN_NAME}",
    "n_scenes": ${N_SCENES},
    "n_draws": ${N_DRAWS},
    "tracking": "${TRACKING}",
    "notes": ""
})
# Accumulate multiple experiments under the same run_id
experiments = entry.setdefault("experiments", {})
experiments["${RUN_NAME}"] = {
    "n_draws": ${N_DRAWS},
    "tracking": "${TRACKING}"
}
# Keep top-level name pointing to the first registered experiment
if "name" not in entry or entry["name"] == "${RUN_NAME}":
    entry["name"] = list(experiments.keys())[0]
data["${RUN_ID}"] = entry
reg.write_text(json.dumps(data, indent=2))
print("[INFO] Registered experiment '${RUN_NAME}' under run_id ${RUN_ID} in ${REGISTRY}")
PY
fi

echo "[DONE] Null pipeline completed."
echo "[DONE] Metrics file: results/runs/${RUN_ID}/metrics/metrics.csv"
