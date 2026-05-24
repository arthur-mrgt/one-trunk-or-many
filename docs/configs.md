# Hydra Configuration Guide

All runs are configured via [Hydra](https://hydra.cc/). The entry point
composes a single resolved config from one of two sources:

- a **top-level preset** (`--config-name <preset>`), or
- the **default composition** (`configs/default.yaml`) plus CLI overrides.

## Top-level presets

| Preset | Purpose |
|---|---|
| `benchmark_rq1_final_hypersim` | Production RQ1 run: Hypersim, 100 scenes, ~4000 paired samples, 3 pairs × 3 metrics, adaptive null with `min_total_draws=1000`, `alpha=0.01`. |
| `benchmark_rq1_smoke_hypersim` | Same wiring, scaled down: 5 scenes, 200 samples, 100 null draws. Used to validate the full pipeline end-to-end in minutes. |

Run them with:

```bash
python -m src.run_benchmark --config-name benchmark_rq1_smoke_hypersim
python -m src.run_benchmark --config-name benchmark_rq1_final_hypersim
```

## Config groups

`configs/default.yaml` composes one file from each of these groups:

| Group | Files | Purpose |
|---|---|---|
| `data` | `hypersim.yaml`, `diode.yaml` | Dataset root, modalities, sampling caps, null-sampling metadata. |
| `model` | `fourm.yaml`, `fourm_mock.yaml` | Encoder backend, layer policy, embedding shape. |
| `metrics` | `rq1.yaml` | Enabled metrics, modality pairs, PCA/FAISS backends, joint-pass toggle. |
| `runtime` | `local.yaml`, `full_gpu_faiss.yaml` | Device, workers, distributed flags, reuse policy. |
| `tracking` | `wandb_on.yaml`, `wandb_off.yaml` | W&B project, entity, tags. |
| `slurm` | `izar.yaml` | Cluster defaults (partition, gres, time). |

The top-level `analysis.null_distribution.*` block in
`configs/default.yaml` controls the adaptive null stage.

## Common CLI overrides

```bash
# Subset Hypersim
python -m src.run_benchmark --config-name benchmark_rq1_smoke_hypersim \
  data.n_scenes=20 data.max_total_samples=1000

# Restrict modality pairs
python -m src.run_benchmark "metrics.pairs=[[rgb,depth]]"

# Force CPU / single GPU / multi-GPU
python -m src.run_benchmark runtime.device=cpu
python -m src.run_benchmark runtime.device=cuda
python -m src.run_benchmark runtime.device=cuda runtime.multi_gpu_strategy=data_parallel

# Enable / disable W&B
python -m src.run_benchmark tracking=wandb_on
python -m src.run_benchmark tracking=wandb_off

# Reuse activations from a previous run
python -m src.run_benchmark \
  runtime.activation_input_run_id=<run_id> runtime.reuse.activations=true

# Force null recomputation from scratch
python -m src.run_benchmark runtime.reuse.null_distribution=false

# Enable the joint forward pass for one pair (off by default)
python -m src.run_benchmark metrics.joint_pass.enabled=true
```

## Adding a preset

A preset is a top-level YAML in `configs/` that composes the defaults and
overrides specific sections. Example skeleton:

```yaml
# configs/benchmark_my_experiment.yaml
defaults:
  - default
  - override runtime: full_gpu_faiss
  - override tracking: wandb_on
  - _self_

project:
  stage: my_experiment

data:
  n_scenes: 50
  max_total_samples: 2000

analysis:
  null_distribution:
    min_total_draws: 500
```

Launch with `python -m src.run_benchmark --config-name benchmark_my_experiment`.
