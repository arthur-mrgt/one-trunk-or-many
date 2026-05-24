#!/usr/bin/env bash
# Interactive launcher for the benchmark pipeline (foreground execution).
#
# Use this on a local workstation OR inside an interactive SLURM allocation
# (e.g. after `srun --pty bash`). For non-interactive SLURM batch submission,
# use `scripts/run/submit_slurm.sh` instead.
#
# Usage:
#   bash scripts/run/run_interactive.sh                                          # final RQ1 preset
#   PRESET=benchmark_rq1_smoke_hypersim bash scripts/run/run_interactive.sh      # smoke preset
#   PRESET=<preset> EXTRA="data.n_scenes=10" bash scripts/run/run_interactive.sh # custom overrides
#
# Env vars:
#   PRESET     Hydra preset name (default: benchmark_rq1_final_hypersim).
#   CONDA_ENV  Conda env to activate (default: trunk).
#   PYTHON_BIN Override python entrypoint (default: python).
#   EXTRA      Extra Hydra overrides appended to the command.

set -euo pipefail

PRESET="${PRESET:-benchmark_rq1_final_hypersim}"
CONDA_ENV="${CONDA_ENV:-trunk}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA="${EXTRA:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
  else
    echo "[WARN] Conda env '${CONDA_ENV}' not found; using current environment."
  fi
fi

echo "[INFO] Repo root  : ${REPO_ROOT}"
echo "[INFO] Preset     : ${PRESET}"
echo "[INFO] Python bin : ${PYTHON_BIN}"
nvidia-smi || echo "[WARN] nvidia-smi not available; GPU presets will fail."

latest_run_id() {
  ${PYTHON_BIN} - <<'PY'
from pathlib import Path
runs_root = Path("results/runs")
candidates = sorted(
    [p for p in runs_root.glob("*") if p.is_dir() and p.name not in {"hydra", "null_distributions"}]
)
print(candidates[-1].name if candidates else "")
PY
}

echo "[INFO] Launching: ${PYTHON_BIN} -m src.run_benchmark --config-name ${PRESET} ${EXTRA}"
${PYTHON_BIN} -m src.run_benchmark --config-name "${PRESET}" ${EXTRA}

RUN_ID="$(latest_run_id)"
echo
echo "[DONE] Preset      : ${PRESET}"
echo "[DONE] RUN_ID      : ${RUN_ID}"
echo "[DONE] Metrics CSV : results/runs/${RUN_ID}/metrics/metrics.csv"
echo "[DONE] Null CSV    : results/runs/null_distributions/${RUN_ID}/null_distribution.csv"
echo "[NEXT] Open notebooks/pvalue_diagnostics.ipynb and set RUN_ID='${RUN_ID}' to render plots."
