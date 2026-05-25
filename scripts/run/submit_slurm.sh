#!/bin/bash
#SBATCH --job-name=trunk-benchmark
#SBATCH --partition=gpu
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
#   PRESET                   Hydra preset name (default: benchmark_rq1_final_hypersim).
#   CONDA_ENV                Conda env to activate (default: trunk).
#   EXTRA                    Extra Hydra overrides appended to the command.
#   STAGE_DATASET            One of: auto (default), hypersim, diode, none.
#                            "auto" picks `hypersim` if `_hypersim` is in the
#                            preset name, `diode` if `_diode` is in it, else
#                            `none`. The selected dataset is rsynced to
#                            /tmp/$USER/<dataset> at job start. Once staged,
#                            Hydra's paths.datasets_root is overridden to /tmp.
#   STAGE_HYPERSIM           Legacy override: set to 0 to force-skip the
#                            Hypersim staging when STAGE_DATASET=auto would
#                            have picked it. Defaults to 1.
#   STAGE_DIODE              Legacy override: set to 1 to force DIODE staging
#                            when STAGE_DATASET=auto would not pick it.
#   SNAPSHOT_INTERVAL        Seconds between background tar snapshots of
#                            /tmp/$USER/trunk_acts → /home (default: 900 = 15 min,
#                            0 disables). Snapshots land in
#                            results/snapshots/job_<JOBID>/snapshot_*.tar.
#   RESTORE_FROM_SNAPSHOT    Absolute path to a previous snapshot .tar; if set,
#                            the script extracts it to /tmp/$USER before launch
#                            so a crashed run can resume with
#                            `runtime.reuse.activations=true`.
#
# The script auto-detects the number of GPUs allocated by SLURM
# (SLURM_GPUS_ON_NODE / SLURM_GPUS / --gres=gpu:N) and uses torchrun for
# multi-GPU and multi-node runs, plain python otherwise.
#
# Dataset staging (R1 optimization): reads of HDF5 (Hypersim) and PNG/NPY
# (DIODE) files are massively faster from /tmp (local NVMe SSD, 2.9 TB)
# than from /home (Lustre). When STAGE_DATASET picks a dataset, it is
# copied to /tmp/$USER/<dataset> once at job start (~10-15 min for 70 GB
# Hypersim, a few minutes for the smaller DIODE subsets), then Hydra is
# told to read from there. Activations are NOT staged — they remain on
# the persistent shared filesystem so the pipeline can resume from a
# partial run with:
#   EXTRA="runtime.activation_input_run_id=<run_id_from_crashed_job> \
#          runtime.reuse.activations=true \
#          analysis.null_distribution.resume_if_partial=true"

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

# ─── Stage dataset to fast local NVMe (R1 optimization) ──────────────────
# HDF5/PNG/NPY reads dominate extraction time on shared filesystems
# (Lustre serves ~100-500 MB/s with high per-file metadata latency). The
# /tmp partition on Izar compute nodes is a 2.9 TB local NVMe SSD that
# delivers ~3-5 GB/s with low latency, so staging the dataset there gives
# a ~50-100x read speedup. Activations are NOT staged here — they live in
# `paths.results_root` (default `./results/runs/<run_id>/activations/` on
# /home), so they persist across jobs and the pipeline can resume via:
#   EXTRA="runtime.activation_input_run_id=<run_id> \
#          runtime.reuse.activations=true \
#          analysis.null_distribution.resume_if_partial=true"
#
# Resolve which dataset to stage. STAGE_DATASET wins when set explicitly;
# otherwise we infer from the preset name. Legacy STAGE_HYPERSIM /
# STAGE_DIODE flags still work as last-mile overrides.
# ─────────────────────────────────────────────────────────────────────────
STAGE_DATASET="${STAGE_DATASET:-auto}"
if [[ "${STAGE_DATASET}" == "auto" ]]; then
  if [[ "${PRESET}" == *"_diode"* ]]; then
    STAGE_DATASET="diode"
  elif [[ "${PRESET}" == *"_hypersim"* ]]; then
    STAGE_DATASET="hypersim"
  else
    STAGE_DATASET="none"
  fi
fi
# Legacy single-dataset flags: keep them as overrides for backward compat.
if [[ "${STAGE_HYPERSIM:-}" == "0" && "${STAGE_DATASET}" == "hypersim" ]]; then
  STAGE_DATASET="none"
fi
if [[ "${STAGE_DIODE:-}" == "1" ]]; then
  STAGE_DATASET="diode"
fi

stage_dataset() {
  local name="$1"
  local remote="resources/datasets/${name}"
  local local_dir="/tmp/${USER}/${name}"
  local sentinel="${local_dir}/.stage_complete"
  if [[ ! -d "${remote}" ]]; then
    echo "[WARN] Cannot stage ${name}: ${remote} does not exist on this node. Falling back to /home."
    return 1
  fi
  if [[ ! -f "${sentinel}" ]]; then
    echo "[INFO] Staging ${name} from ${remote} to ${local_dir} ..."
    mkdir -p "${local_dir}"
    time rsync -a "${remote}/" "${local_dir}/"
    touch "${sentinel}"
    echo "[INFO] Stage complete. Size: $(du -sh "${local_dir}" | cut -f1)"
  else
    echo "[INFO] ${name} already cached at ${local_dir} (skipping stage)"
  fi
  return 0
}

if [[ "${STAGE_DATASET}" == "hypersim" ]] || [[ "${STAGE_DATASET}" == "diode" ]]; then
  if stage_dataset "${STAGE_DATASET}"; then
    EXTRA="paths.datasets_root=/tmp/${USER} ${EXTRA}"
  else
    echo "[INFO] Continuing without staging."
  fi
else
  echo "[INFO] STAGE_DATASET=${STAGE_DATASET}; using ${PWD}/resources/datasets for I/O."
fi

# ─── Restore activations from a previous snapshot (resume after crash) ───
# Set RESTORE_FROM_SNAPSHOT=<abs path to .tar> to extract a prior snapshot
# back to /tmp/$USER before launching python. The activation_index_*.csv
# files (always on /home) reference absolute /tmp paths, so the restore
# only works on the same user account; they will resolve correctly once
# the .npy files are back on /tmp.
# ─────────────────────────────────────────────────────────────────────────
if [[ -n "${RESTORE_FROM_SNAPSHOT:-}" ]]; then
  if [[ -f "${RESTORE_FROM_SNAPSHOT}" ]]; then
    echo "[INFO] Restoring activations from ${RESTORE_FROM_SNAPSHOT} to /tmp/${USER}/ ..."
    mkdir -p "/tmp/${USER}"
    time tar xf "${RESTORE_FROM_SNAPSHOT}" -C "/tmp/${USER}/"
    echo "[INFO] Restore complete. trunk_acts size: $(du -sh /tmp/${USER}/trunk_acts 2>/dev/null | cut -f1)"
  else
    echo "[WARN] RESTORE_FROM_SNAPSHOT=${RESTORE_FROM_SNAPSHOT} does not exist; continuing without restore."
  fi
fi

# ─── Background snapshot watchdog ────────────────────────────────────────
# Periodically tar /tmp/$USER/trunk_acts → results/snapshots/job_<JOBID>/
# so a crashed run can be resumed via RESTORE_FROM_SNAPSHOT. Snapshots are
# monolithic .tar files (1 Lustre create + 1 sequential write ≈ 1-2 sec)
# rather than 96k small file copies (would take ~2 hours on Lustre).
# Disable with SNAPSHOT_INTERVAL=0.
# ─────────────────────────────────────────────────────────────────────────
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-900}"
SNAPSHOT_DIR="${PWD}/results/snapshots/job_${SLURM_JOB_ID:-local}"
SNAPSHOT_PID=""
if [[ "${SNAPSHOT_INTERVAL}" -gt 0 ]]; then
  mkdir -p "${SNAPSHOT_DIR}"
  echo "[INFO] Snapshot watchdog enabled: every ${SNAPSHOT_INTERVAL}s → ${SNAPSHOT_DIR}"
  (
    while sleep "${SNAPSHOT_INTERVAL}"; do
      if [[ -d "/tmp/${USER}/trunk_acts" ]]; then
        ts=$(date +%H%M%S)
        out="${SNAPSHOT_DIR}/snapshot_${ts}.tar"
        if tar cf "${out}" -C "/tmp/${USER}" trunk_acts/ 2>/dev/null; then
          echo "[BACKUP $(date +%H:%M:%S)] $(basename "${out}") ($(du -sh "${out}" | cut -f1))"
        fi
      fi
    done
  ) &
  SNAPSHOT_PID=$!
  # On script exit (normal or crash), kill watchdog and take a final snapshot.
  cleanup_snapshot() {
    [[ -n "${SNAPSHOT_PID}" ]] && kill "${SNAPSHOT_PID}" 2>/dev/null || true
    if [[ -d "/tmp/${USER}/trunk_acts" ]]; then
      ts=$(date +%H%M%S)
      out="${SNAPSHOT_DIR}/snapshot_FINAL_${ts}.tar"
      if tar cf "${out}" -C "/tmp/${USER}" trunk_acts/ 2>/dev/null; then
        echo "[BACKUP final] $(basename "${out}") ($(du -sh "${out}" | cut -f1))"
      fi
    fi
  }
  trap cleanup_snapshot EXIT
fi

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
