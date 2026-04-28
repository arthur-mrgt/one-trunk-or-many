# CKA Pipeline Guide

## Overview

The benchmark is Hydra-driven and fully YAML-configurable.

- Dataset backend is selected via `data` config group.
- Model backend is selected via `model` config group.
- Metrics are selected via `metrics.enabled` and `metrics.pairs`.
- Tracking is controlled via `tracking` config group.

Main entrypoint:

```bash
python -m src.run_benchmark
```

## Config groups

- `configs/default.yaml` - main composition and shared sections.
- `configs/data/*.yaml` - dataset root/name/modality defaults.
- `configs/model/*.yaml` - encoder backend and layer policy.
- `configs/metrics/*.yaml` - enabled metrics and options.
- `configs/tracking/*.yaml` - W&B on/off modes.
- `configs/runtime/*.yaml` - batch/worker/device settings.
- `configs/slurm/*.yaml` - cluster defaults.

## Output schema

Each run creates `results/runs/<run_id>/` with:

- `activations/` - saved vectors as `.npy`.
- `artifacts/activation_index_<pair>.csv` - index of saved activations.
- `metrics/metrics.csv` - metric table (`metric`, `pair`, `layer`, `value`, ...).
- `run_summary.json` - run-level summary.
- `resolved_config.json` - fully resolved config snapshot.

## Typical commands

Local POC:

```bash
python -m src.run_benchmark \
  data.name=hypersim \
  data.n_scenes=10 \
  metrics.pairs='[[rgb,depth]]'
```

Enable W&B:

```bash
python -m src.run_benchmark tracking=wandb_on
```

Switch model backend config:

```bash
python -m src.run_benchmark model=fourm_real
```

## SLURM

Use the provided scripts:

- `scripts/run_benchmark.slurm`
- `scripts/run_extraction.slurm`
- `scripts/run_metrics.slurm`

Submit with:

```bash
sbatch scripts/run_benchmark.slurm
```

## Extension points

- Add metrics by implementing a function and registering it in `src/metrics/registry.py`.
- Add datasets by implementing a loader and wiring it in `src/data/registry.py`.
- Null distribution is scaffolded in config under `analysis.null_distribution` and in `src/analysis/null_distribution.py`.
