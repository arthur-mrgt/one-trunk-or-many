# Pipeline Guide

The benchmark is a Hydra-driven, single-orchestrator pipeline. Configuration
options are documented in [`configs.md`](configs.md), output schemas in
[`results.md`](results.md), and cluster launchers in
[`cluster.md`](cluster.md). This document focuses on the **stages** and the
**extension points**.

## Stage flow

```text
run_benchmark
    │
    ├── extraction stage
    │     for each modality pair:
    │       load_dataset_pairs(...)              ← honors data.n_scenes / max_total_samples / seed
    │       split_by_rank(...)                   ← DDP sharding when runtime.distributed.enabled
    │       run_extraction(model, samples)       ← writes .npy + activation_index_<pair>.csv
    │
    ├── observed-metrics stage
    │     compute CKA / PWCCA / kNN-overlap per (pair, layer) from activations
    │
    ├── null-distribution stage              ← analysis.null_distribution.enabled
    │     adaptive sampling of mismatched pairs until adaptive_stop or max_total_draws
    │     checkpoints partial null_distribution.csv every batch
    │
    └── metrics-finalization stage
          enrich observed metrics with p_value, p_value_adjusted, z_score, ...
          write metrics.csv + log to W&B if enabled
```

All stages share a single `RUN_ID` and a single resolved config snapshot
(`resolved_config.json`).

## Stage entry points

The full pipeline:

```bash
python -m src.run_benchmark
```

Individual stages, useful when iterating without re-running extraction:

```bash
python -m src.run_extraction
python -m src.run_null_distribution runtime.metrics_input_run_id=<run_id>
python -m src.run_metrics            runtime.metrics_input_run_id=<run_id>
```

## Adaptive null distribution

Controlled via the `analysis.null_distribution.*` block:

| Key | Role |
|---|---|
| `enabled` | Master switch. |
| `metrics_enabled` | Subset of metrics for which a null is computed. |
| `sampling_mode` | `cross_scene_type_random` (default), `within_scene_type_random`, or `within_scene_random`. See `configs/data/hypersim.yaml` for the trade-offs. |
| `draws_per_batch` | Draws per checkpoint batch. |
| `min_total_draws` / `max_total_draws` | Hard floor / cap on draws per hypothesis. |
| `adaptive_stop.alpha` | Significance level for the early-stop test. |
| `adaptive_stop.correction` | `bh_fdr` or `bonferroni`. |
| `adaptive_stop.require_all_hypotheses` | If true, every hypothesis must satisfy the criterion. |

`Ctrl-C` during the null stage persists the partial artifact and the pipeline
continues to the metrics stage (`on_keyboard_interrupt: save_and_continue`).

## Reuse policies

`runtime.reuse.*` flags let you skip stages when artifacts already exist:

```bash
# Reuse activations from the latest run, recompute null and metrics
python -m src.run_benchmark runtime.reuse.activations=true

# Reuse activations from a specific run
python -m src.run_benchmark \
  runtime.activation_input_run_id=<run_id> runtime.reuse.activations=true

# Reuse a precomputed null artifact
python -m src.run_benchmark runtime.reuse.null_distribution=true
```

## Extension points

- **New metric** — implement a function and register it in
  `src/metrics/registry.py`. Activate via `metrics.enabled`.
- **New dataset** — implement a loader and wire it in
  `src/data/registry.py`. Add a `configs/data/<name>.yaml`.
- **New model backend** — add a class in `src/models/`, wire it in
  `src/models/registry.py`, add a `configs/model/<name>.yaml`.

## Internal module map

Null distribution internals:

- `src/analysis/null_runner.py` — adaptive loop and DDP coordination.
- `src/analysis/null_sampling.py` — hypothesis caches and mismatched-sample drawing.
- `src/analysis/null_artifacts.py` — CSV/Parquet checkpointing and state files.
- `src/analysis/null_stop.py` — adaptive-stop decision logic.
- `src/analysis/null_distribution.py` — public façade.

Benchmark orchestrator (`src/pipeline/`):

- `benchmark.py` — top-level stage runners.
- `benchmark_config.py` — Hydra-derived path and pair resolution.
- `benchmark_tables.py` — activation index loading and metric computation.
- `benchmark_significance.py` — null-vs-observed enrichment.
- `benchmark_plots.py` — W&B plot helpers.

Distributed and tracking utilities live in `src/utils/`.
