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
