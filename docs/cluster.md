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

## Multi-GPU on a single node

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
