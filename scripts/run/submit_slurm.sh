#!/bin/bash
#SBATCH --job-name=trunk-benchmark
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err
#SBATCH --chdir=.

# Generic SLURM submission script for the benchmark pipeline.
#
# The defaults above target a single-GPU run on SCITAS Izar. Override any
# #SBATCH directive (gres, time, mem, nodes, …) from the sbatch CLI.
#
# Single-GPU run (smoke):
#   PRESET=benchmark_rq1_smoke_hypersim sbatch \
#     --gres=gpu:1 --time=02:00:00 --mem=64G --cpus-per-task=10 \
#     --job-name=trunk-smoke \
#     scripts/run/submit_slurm.sh
#
# Single-node 2-GPU run (recommended for the final RQ1 benchmark):
#   sbatch --gres=gpu:2 --time=12:00:00 --job-name=trunk-final \
#     scripts/run/submit_slurm.sh
#
# Multi-node multi-GPU run (only when the single-node 2-GPU run is the
# bottleneck — inter-node NCCL barriers eat a large part of the speedup):
#   sbatch --nodes=2 --gres=gpu:2 --ntasks-per-node=1 --time=08:00:00 \
#     --job-name=trunk-multinode \
#     EXTRA="paths.resources_root=$SCRATCH/otm_resources" \
#     scripts/run/submit_slurm.sh
#
# Env vars (forwarded by sbatch):
#   PRESET     Hydra preset name (default: benchmark_rq1_final_hypersim).
#   CONDA_ENV  Conda env to activate (default: trunk).
#   EXTRA      Extra Hydra overrides appended to the command.
#
# The script auto-detects the number of GPUs allocated by SLURM
# (SLURM_GPUS_ON_NODE / SLURM_GPUS / --gres=gpu:N) and uses torchrun for
# multi-GPU and multi-node runs, plain python otherwise.

set -euo pipefail

PRESET="${PRESET:-benchmark_rq1_final_hypersim}"
CONDA_ENV="${CONDA_ENV:-trunk}"
EXTRA="${EXTRA:-}"

mkdir -p results/slurm
echo "[INFO] Host         : $(hostname)"
echo "[INFO] Started      : $(date)"
echo "[INFO] Preset       : ${PRESET}"
echo "[INFO] Nodes        : ${SLURM_JOB_NUM_NODES:-1}"
echo "[INFO] Nodelist     : ${SLURM_JOB_NODELIST:-$(hostname)}"
nvidia-smi || true

set +u
source ~/.bashrc
conda activate "${CONDA_ENV}"
set -u

# Detect GPUs allocated per node.
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"
TOTAL_PROCS=$(( NPROC_PER_NODE * NNODES ))

# Share CPU threads fairly between local ranks; avoid oversubscription.
THREADS_PER_RANK=$(( ${SLURM_CPUS_PER_TASK:-20} / NPROC_PER_NODE ))
if [[ "${THREADS_PER_RANK}" -lt 1 ]]; then THREADS_PER_RANK=1; fi
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export MKL_NUM_THREADS="${THREADS_PER_RANK}"

echo "[INFO] GPUs/node    : ${NPROC_PER_NODE}"
echo "[INFO] Total procs  : ${TOTAL_PROCS}"
echo "[INFO] OMP threads  : ${OMP_NUM_THREADS}"

if [[ "${TOTAL_PROCS}" -le 1 ]]; then
  # Plain single-GPU (or CPU) run.
  python -u -m src.run_benchmark --config-name "${PRESET}" ${EXTRA}
elif [[ "${NNODES}" -eq 1 ]]; then
  # Single-node multi-GPU run via torchrun standalone rendezvous.
  export MASTER_ADDR="127.0.0.1"
  export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
  torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m src.run_benchmark \
    --config-name "${PRESET}" \
    runtime.distributed.enabled=true \
    ${EXTRA}
else
  # Multi-node run: srun launches one torchrun per node with a shared rendezvous.
  export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
  export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
  srun --kill-on-bad-exit=1 bash -c "
    torchrun \
      --nnodes=${NNODES} \
      --node_rank=\${SLURM_NODEID} \
      --nproc_per_node=${NPROC_PER_NODE} \
      --rdzv_id=${SLURM_JOB_ID} \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      -m src.run_benchmark \
      --config-name ${PRESET} \
      runtime.distributed.enabled=true \
      ${EXTRA}
  "
fi

echo "[DONE] Finished : $(date)"
