# Benchmark Pipeline Guide

## Overview

The benchmark is Hydra-driven and fully YAML-configurable.

- Dataset backend is selected via `data` config group.
- Model backend is selected via `model` config group.
- Metrics are selected via `metrics.enabled` and `metrics.pairs`.
- Tracking is controlled via `tracking` config group.

Main end-to-end entrypoint (single orchestrator):

```bash
python -m src.run_benchmark
```

Optional stage entrypoints:

```bash
python -m src.run_extraction
python -m src.run_null_distribution runtime.metrics_input_run_id=<run_id>
python -m src.run_metrics runtime.metrics_input_run_id=<run_id>
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
- `metrics/null_stop_evaluation.csv` - adaptive null stop diagnostics (when null is enabled).
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
python -m src.run_benchmark model=fourm
```

Device controls:

```bash
# auto (default): cuda if available else cpu
python -m src.run_benchmark runtime.device=auto

# force cpu
python -m src.run_benchmark runtime.device=cpu

# force single-gpu
python -m src.run_benchmark runtime.device=cuda

# multi-gpu via DataParallel
python -m src.run_benchmark runtime.device=cuda runtime.multi_gpu_strategy=data_parallel
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

## Null distribution (adaptive)

Null computation is now integrated into `python -m src.run_benchmark`:

- Extraction runs first (or is reused if `runtime.reuse.activations=true` and artifacts exist).
- Null draws run next when `analysis.null_distribution.enabled=true`.
- Metrics stage runs last and adds significance columns (`p_value`, `p_value_adjusted`, `n_null_draws`, etc.) when null artifacts are available.

Adaptive stop is controlled in YAML:

- `analysis.null_distribution.adaptive_stop.alpha`
- `analysis.null_distribution.adaptive_stop.correction`
- `analysis.null_distribution.adaptive_stop.require_all_hypotheses`
- `analysis.null_distribution.min_total_draws`
- `analysis.null_distribution.max_total_draws`

Manual interrupt:

- `Ctrl+C` during null stage persists partial null artifacts and the benchmark still continues to metrics (default `on_keyboard_interrupt: save_and_continue`).

Reuse behavior (YAML):

- `runtime.reuse.activations=true`: reuse extraction artifacts instead of recomputing.
- `runtime.activation_input_run_id=<run_id>`: if set, reuse activations from this run.
- if `runtime.activation_input_run_id=null`, the latest run is used as activation source.
- `runtime.reuse.null_distribution=true|false`: reuse/continue existing null artifact or force recomputation.

## Extension points

- Add metrics by implementing a function and registering it in `src/metrics/registry.py`.
- Add datasets by implementing a loader and wiring it in `src/data/registry.py`.
- Null distribution internals are split across:
  - `src/analysis/null_runner.py`
  - `src/analysis/null_sampling.py`
  - `src/analysis/null_artifacts.py`
  - `src/analysis/null_stop.py`
  - public API: `src/analysis/null_distribution.py`
- Benchmark internals are split across:
  - `src/pipeline/benchmark_config.py`
  - `src/pipeline/benchmark_tables.py`
  - `src/pipeline/benchmark_significance.py`
  - `src/pipeline/benchmark_plots.py`
  - stage orchestrator: `src/pipeline/benchmark.py`
