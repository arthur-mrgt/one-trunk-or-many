#!/bin/bash
#SBATCH --job-name=cka-extraction
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail

mkdir -p results/slurm
echo "[INFO] Host: $(hostname)"
echo "[INFO] Started at: $(date)"

# source ~/.bashrc
# conda activate trunk

python -m src.run_extraction data.name=hypersim metrics.pairs='[[rgb,depth]]'
