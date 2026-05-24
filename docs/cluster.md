# Cluster and Local Launchers

Two launchers wrap the Hydra entry point. Both forward a `PRESET=<config-name>`
environment variable (default `benchmark_rq1_final_hypersim`) and an optional
`EXTRA="<hydra overrides>"` string.

## Interactive (`scripts/run/run_interactive.sh`)

Use on a local workstation or inside an interactive SLURM allocation
(`srun --pty bash`). Runs in the foreground and prints the resolved `RUN_ID`
plus paths to `metrics.csv` and `null_distribution.csv` at the end.

```bash
# Final RQ1 preset
bash scripts/run/run_interactive.sh

# Smoke preset
PRESET=benchmark_rq1_smoke_hypersim bash scripts/run/run_interactive.sh

# Custom Hydra overrides
PRESET=benchmark_rq1_smoke_hypersim \
  EXTRA="data.n_scenes=10 tracking=wandb_off" \
  bash scripts/run/run_interactive.sh
```

The script activates the conda env named `trunk` if present, falls back to
the current shell otherwise. Override with `CONDA_ENV=<name>`.

## SLURM batch (`scripts/run/submit_slurm.sh`)

Plain `sbatch` submission. The `#SBATCH` directives at the top of the script
(`--partition=gpu`, `--gres=gpu:1`, `--time=24:00:00`, …) should be adapted
to your cluster (e.g. SCITAS Izar, Kuma).

```bash
sbatch scripts/run/submit_slurm.sh                                       # final
PRESET=benchmark_rq1_smoke_hypersim sbatch scripts/run/submit_slurm.sh   # smoke
PRESET=<preset> EXTRA="<overrides>" sbatch scripts/run/submit_slurm.sh   # custom
```

Job stdout/stderr go to `results/slurm/%x_%j.out` and `.err`.

## SCITAS Izar — typical sbatch invocations

`submit_slurm.sh` detects the number of GPUs allocated by SLURM and switches
between plain `python`, `torchrun --standalone` (single-node multi-GPU), and
`srun + torchrun` (multi-node) automatically. Override the `#SBATCH`
directives from the sbatch CLI to size each job.

```bash
# 1) Smoke — 1 GPU, ~20 min, verifies env/dataset/model
PRESET=benchmark_rq1_smoke_hypersim sbatch \
  --gres=gpu:1 --time=02:00:00 --mem=64G --cpus-per-task=10 \
  --job-name=trunk-smoke \
  scripts/run/submit_slurm.sh

# 2) Final on 2 GPUs single-node — recommended, ~3–5 h
sbatch --gres=gpu:2 --time=12:00:00 --job-name=trunk-final \
  scripts/run/submit_slurm.sh

# 3) Final on 2 nodes × 2 GPUs — only if (2) is the bottleneck
EXTRA="paths.resources_root=$SCRATCH/otm_resources" \
  sbatch --nodes=2 --gres=gpu:2 --ntasks-per-node=1 --time=08:00:00 \
  --job-name=trunk-multinode \
  scripts/run/submit_slurm.sh
```

### Why 2 GPUs single-node is the sweet spot

The pipeline alternates between GPU-heavy compute (extraction, metrics, null
draws) and rank-0 I/O (writing CSVs, W&B logging). With 2 V100s on the same
node, NCCL gather operations cost ~µs and the GPUs stay saturated. Going to
4 GPUs across 2 nodes adds Infiniband round-trips at every barrier and
typically only delivers 1.3–1.8× speedup over single-node 2-GPU, not the
nominal 2×.

When you do go multi-node, make sure `paths.resources_root` resolves to a
path visible from every node (typically `$SCRATCH/otm_resources`).

## Multi-GPU on a single node (generic)

The pipeline supports DDP-style sharding via `torch.distributed`. Launch with
`torchrun` and flip the distributed flag:

```bash
torchrun --nproc_per_node=4 -m src.run_benchmark \
  --config-name benchmark_rq1_final_hypersim \
  runtime.distributed.enabled=true
```

Activations are extracted in parallel, then gathered to rank 0 for the
metrics and null stages. W&B logging stays on rank 0 only.

## Resources on the cluster

If the dataset and model live outside the repo (e.g. on scratch), point Hydra
at the right roots via CLI overrides:

```bash
sbatch scripts/run/submit_slurm.sh EXTRA="paths.resources_root=/scratch/$USER/otm_resources"
```

See [`docs/data.md`](data.md) for the expected resource layout.

## Stage Hypersim to fast local storage (default on Izar)

When `submit_slurm.sh` runs on Izar with `STAGE_HYPERSIM=1` (the default),
the script copies `resources/datasets/hypersim/` to `/tmp/$USER/hypersim/`
on the compute node before launching Python. This bypasses Lustre for the
read-heavy HDF5 access during extraction:

| Source | Read speed | Notes |
|---|---|---|
| `/home` (Lustre) | ~100-500 MB/s | High per-file metadata latency |
| `/tmp` (NVMe SSD, 2.9 TB) | ~3-5 GB/s | Local to compute node, ephemeral |

For 100 Hypersim scenes (~70 GB), the initial `rsync` takes ~10-15 min, and
the cache survives for the lifetime of the node — re-using the same compute
node skips the stage entirely.

The override is added automatically:

```
EXTRA="paths.datasets_root=/tmp/$USER ${EXTRA}"
```

Disable when running on a different dataset:

```bash
STAGE_HYPERSIM=0 sbatch scripts/run/submit_slurm.sh
```

## Resuming a partial run

Activations are written to `paths.results_root` (default
`results/runs/<run_id>/activations/` on `/home`), so they persist across
jobs. If a run crashes mid-way (timeout, node failure, etc.), relaunch
with these overrides to skip what is already done:

```bash
sbatch scripts/run/submit_slurm.sh \
  EXTRA="runtime.activation_input_run_id=<RUN_ID_OF_CRASHED_JOB> \
         runtime.reuse.activations=true \
         analysis.null_distribution.resume_if_partial=true"
```

What gets re-used:

- **Per-pair activations** if `activation_index_<pair>.csv` is on disk in
  the resolved run dir → that pair's extraction is skipped entirely.
- **Partial null draws** stored in
  `results/runs/null_distributions/<run_id>/null_distribution.csv` →
  the adaptive sampler appends new draws on top of existing ones.

The `RUN_ID_OF_CRASHED_JOB` is printed at the top of every run (e.g.
`rq1_final_hypersim-20260524-205615`) and is also visible in W&B.

## Activations on fast ephemeral storage (recommended for long runs)

Lustre is catastrophically slow for the small `.npy` writes the extraction
stage emits (we measured **~107 ms per file create** on Izar Lustre vs
**~0.04 ms** on local NVMe — a 3000× ratio). For a 4000-sample × 3-pair
run, that translates to ~8h30 of pure write-wait on Lustre vs ~10s on
`/tmp`. Recommended override:

```bash
sbatch scripts/run/submit_slurm.sh \
  EXTRA="paths.activations_root=/tmp/$USER/trunk_acts"
```

Only the `.npy` activation vectors land on `/tmp`. The
`activation_index_*.csv` files (which the metrics and null stages need)
**always live on `/home`** in `results/runs/<run_id>/artifacts/` because
they are produced by `_resolve_activations_dir` in `src/utils/config.py`.
That keeps the small but critical metadata persistent.

### Snapshot watchdog (safety net for the ephemeral activations)

`submit_slurm.sh` automatically tars `/tmp/$USER/trunk_acts` to
`results/snapshots/job_<JOBID>/snapshot_*.tar` every 15 minutes (and once
more on script exit). Each snapshot is a single ~70-100 MB monolithic
file → 1 Lustre create + 1 sequential write ≈ 1-2 seconds, instead of
the ~2 hours that copying 96k small `.npy` files would cost.

Tune via env var:

```bash
SNAPSHOT_INTERVAL=600 sbatch scripts/run/submit_slurm.sh ...   # every 10 min
SNAPSHOT_INTERVAL=0   sbatch scripts/run/submit_slurm.sh ...   # disable
```

### Resuming after a crash with snapshots

When a job dies (OOM, timeout, node failure) the `activation_index_*.csv`
on `/home` references absolute `/tmp` paths that no longer exist. Pass
`RESTORE_FROM_SNAPSHOT=<path>.tar` so the new job's launcher extracts the
last snapshot back to `/tmp/$USER/` before launching Python:

```bash
LATEST=$(ls -t results/snapshots/job_*/snapshot_FINAL_*.tar 2>/dev/null | head -1)
sbatch \
  --export=ALL,WANDB_MODE=offline,RESTORE_FROM_SNAPSHOT="$LATEST",\
EXTRA="paths.activations_root=/tmp/$USER/trunk_acts \
       runtime.activation_input_run_id=<RUN_ID_OF_CRASHED_JOB> \
       runtime.reuse.activations=true \
       analysis.null_distribution.resume_if_partial=true" \
  -A cs-503 --qos=cs-503 --gres=gpu:2 --cpus-per-task=40 --mem=180G \
  --time=08:00:00 --job-name=trunk-resume \
  scripts/run/submit_slurm.sh
```

The launcher prints `[INFO] Restoring activations from <tar>` then the
pipeline replays only the missing pairs and resumes the partial null
distribution.
