#!/bin/bash
#SBATCH --job-name=trunk-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err
#SBATCH --chdir=.

# SLURM submission script for the benchmark pipeline (SCITAS et al.).
#
# Usage:
#   sbatch scripts/run/submit_slurm.sh                                          # final RQ1 preset
#   PRESET=benchmark_rq1_smoke_hypersim sbatch scripts/run/submit_slurm.sh      # smoke preset
#   PRESET=<preset> EXTRA="data.n_scenes=10" sbatch scripts/run/submit_slurm.sh # custom overrides
#
# Adapt the #SBATCH directives above to your cluster (partition, gres, time).
#
# Env vars (forwarded by sbatch):
#   PRESET     Hydra preset name (default: benchmark_rq1_final_hypersim).
#   CONDA_ENV  Conda env to activate (default: trunk).
#   EXTRA      Extra Hydra overrides appended to the command.

set -euo pipefail

PRESET="${PRESET:-benchmark_rq1_final_hypersim}"
CONDA_ENV="${CONDA_ENV:-trunk}"
EXTRA="${EXTRA:-}"

mkdir -p results/slurm
echo "[INFO] Host    : $(hostname)"
echo "[INFO] Started : $(date)"
echo "[INFO] Preset  : ${PRESET}"
nvidia-smi || true

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
set -u

python -u -m src.run_benchmark --config-name "${PRESET}" ${EXTRA}

echo "[DONE] Finished : $(date)"
