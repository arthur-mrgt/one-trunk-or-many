#!/usr/bin/env bash
set -euo pipefail

# Unified pipeline runner with explicit modes.
#
# Modes:
#   full                   extraction + metrics + null + metrics_with_null
#   metrics_null_enriched  metrics + null + metrics_with_null
#   null_enriched          null + metrics_with_null
#   metrics                metrics only
#   null                   null only
#
# Usage examples:
#   bash scripts/run_pipeline.sh full --n-scenes 20 --n-draws 1000 --tracking wandb_on \
#     --base-run-name rq1_20scenes_base --final-run-name rq1_<RUN_ID>_with_null_1000draws
#
#   bash scripts/run_pipeline.sh metrics_null_enriched --run-id <RUN_ID> --n-draws 1000 \
#     --tracking wandb_on --base-run-name rq1_<RUN_ID>_base_metrics --final-run-name rq1_<RUN_ID>_with_null_1000draws
#
#   bash scripts/run_pipeline.sh null_enriched --run-id <RUN_ID> --n-draws 1000 \
#     --tracking wandb_on --final-run-name rq1_<RUN_ID>_with_null_1000draws
#
#   bash scripts/run_pipeline.sh metrics --run-id <RUN_ID> --tracking wandb_on \
#     --base-run-name rq1_<RUN_ID>_base_metrics
#
#   bash scripts/run_pipeline.sh null --run-id <RUN_ID> --n-draws 1000

if [[ $# -lt 1 ]]; then
  echo "[ERROR] Missing mode."
  exit 1
fi

MODE="$1"
shift

N_SCENES=20
N_DRAWS=1000
RUN_ID=""
TRACKING="wandb_on"
BASE_RUN_NAME=""
FINAL_RUN_NAME=""
REUSE_IF_EXISTS="true"
EXTRA_OVERRIDES=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pipeline.sh <mode> [options]

Modes:
  full
  metrics_null_enriched
  null_enriched
  metrics
  null

Options:
  --n-scenes <int>            Number of scenes for full mode benchmark (default: 20)
  --n-draws <int>             Number of null draws (default: 1000)
  --run-id <id>               Existing run id (required for modes without extraction)
  --tracking <preset>         tracking preset (wandb_on|wandb_off), default: wandb_on
  --base-run-name <name>      W&B run name for base metrics run
  --final-run-name <name>     W&B run name for metrics-with-null run
  --reuse-if-exists <bool>    Reuse existing null artifact (default: true)
  --extra-overrides "<args>"  Extra Hydra overrides for benchmark/null/metrics commands

Examples:
  bash scripts/run_pipeline.sh full --n-scenes 20 --n-draws 1000 --tracking wandb_on \
    --base-run-name rq1_20scenes_base --final-run-name rq1_<RUN_ID>_with_null_1000draws

  bash scripts/run_pipeline.sh metrics_null_enriched --run-id <RUN_ID> --n-draws 1000 \
    --base-run-name rq1_<RUN_ID>_base_metrics --final-run-name rq1_<RUN_ID>_with_null_1000draws

  bash scripts/run_pipeline.sh null_enriched --run-id <RUN_ID> --n-draws 1000 \
    --final-run-name rq1_<RUN_ID>_with_null_1000draws
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-scenes)
      N_SCENES="$2"; shift 2 ;;
    --n-draws)
      N_DRAWS="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --tracking)
      TRACKING="$2"; shift 2 ;;
    --base-run-name)
      BASE_RUN_NAME="$2"; shift 2 ;;
    --final-run-name)
      FINAL_RUN_NAME="$2"; shift 2 ;;
    --reuse-if-exists)
      REUSE_IF_EXISTS="$2"; shift 2 ;;
    --extra-overrides)
      EXTRA_OVERRIDES="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 1 ;;
  esac
done

latest_run_id() {
  python3 - <<'PY'
from pathlib import Path
runs_root = Path("results/runs")
candidates = sorted(
    [p for p in runs_root.glob("*") if p.is_dir() and p.name not in {"hydra", "null_distributions"}]
)
print(candidates[-1].name if candidates else "")
PY
}

require_run_id() {
  if [[ -z "${RUN_ID}" ]]; then
    RUN_ID="$(latest_run_id)"
    if [[ -n "${RUN_ID}" ]]; then
      echo "[INFO] --run-id not provided. Using latest run: ${RUN_ID}"
    fi
  fi
  if [[ -z "${RUN_ID}" ]]; then
    echo "[ERROR] run_id is required for mode '${MODE}' and no existing run was found."
    exit 1
  fi
}

run_benchmark_step() {
  local run_name="${BASE_RUN_NAME}"
  if [[ -z "${run_name}" ]]; then
    run_name="rq1_${N_SCENES}scenes_base"
  fi
  echo "[INFO] Running benchmark: n_scenes=${N_SCENES}, tracking=${TRACKING}, run_name=${run_name}"
  python -m src.run_benchmark \
    "data.n_scenes=${N_SCENES}" \
    "tracking=${TRACKING}" \
    "tracking.wandb.run_name=${run_name}" \
    ${EXTRA_OVERRIDES}

  RUN_ID="$(latest_run_id)"
  if [[ -z "${RUN_ID}" ]]; then
    echo "[ERROR] Could not determine RUN_ID after benchmark."
    exit 1
  fi
  echo "[INFO] RUN_ID=${RUN_ID}"
}

run_metrics_step() {
  local run_name="${BASE_RUN_NAME}"
  if [[ -z "${run_name}" ]]; then
    run_name="rq1_${RUN_ID}_base_metrics"
  fi
  echo "[INFO] Running metrics-only stage for RUN_ID=${RUN_ID}"
  python -m src.run_metrics \
    "runtime.metrics_input_run_id=${RUN_ID}" \
    "tracking=${TRACKING}" \
    "tracking.wandb.run_name=${run_name}" \
    ${EXTRA_OVERRIDES}
}

run_null_step() {
  echo "[INFO] Running null precompute: RUN_ID=${RUN_ID}, n_draws=${N_DRAWS}, reuse_if_exists=${REUSE_IF_EXISTS}"
  python -m src.run_null_distribution \
    analysis.null_distribution.enabled=true \
    "runtime.metrics_input_run_id=${RUN_ID}" \
    "analysis.null_distribution.n_draws=${N_DRAWS}" \
    "analysis.null_distribution.reuse_if_exists=${REUSE_IF_EXISTS}" \
    ${EXTRA_OVERRIDES}
}

run_metrics_enriched_step() {
  local run_name="${FINAL_RUN_NAME}"
  if [[ -z "${run_name}" ]]; then
    run_name="rq1_${RUN_ID}_with_null_${N_DRAWS}draws"
  fi
  echo "[INFO] Running metrics with null enrichment for RUN_ID=${RUN_ID}"
  python -m src.run_metrics \
    "runtime.metrics_input_run_id=${RUN_ID}" \
    "runtime.null_input_run_id=${RUN_ID}" \
    runtime.null_missing_behavior=error \
    "tracking=${TRACKING}" \
    "tracking.wandb.run_name=${run_name}" \
    ${EXTRA_OVERRIDES}
}

case "${MODE}" in
  full)
    run_benchmark_step
    run_null_step
    run_metrics_enriched_step
    ;;
  metrics_null_enriched)
    require_run_id
    run_metrics_step
    run_null_step
    run_metrics_enriched_step
    ;;
  null_enriched)
    require_run_id
    run_null_step
    run_metrics_enriched_step
    ;;
  metrics)
    require_run_id
    run_metrics_step
    ;;
  null)
    require_run_id
    run_null_step
    ;;
  *)
    echo "[ERROR] Unknown mode: ${MODE}"
    usage
    exit 1
    ;;
esac

echo "[DONE] Mode '${MODE}' completed."
echo "[DONE] RUN_ID=${RUN_ID}"
echo "[DONE] Metrics output: results/runs/${RUN_ID}/metrics/metrics.csv"
