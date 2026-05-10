#!/bin/bash
#SBATCH --job-name=cka-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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

python -m src.run_benchmark \
    data.n_scenes=100 \
    data.frames_per_scene=null \
    data.max_total_samples=4000 \
    "data.modalities=[rgb,depth,normals]" \
    tracking=wandb_on