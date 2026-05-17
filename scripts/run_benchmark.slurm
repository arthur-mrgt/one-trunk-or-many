#!/bin/bash
#SBATCH --job-name=cka-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail

mkdir -p results/slurm
echo "[INFO] Host: $(hostname)"
echo "[INFO] Started at: $(date)"

set +u
source ~/.bashrc
conda activate trunk
set -u

python -u -m src.run_benchmark