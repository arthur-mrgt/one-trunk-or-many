## Null Hypothesis / Noise Integration Handoff

This note describes the implemented null-hypothesis workflow and where to extend it.

Current behavior:

1. Compute null draws from image-level mismatched samples.
2. Reuse existing extraction/null artifacts when configured.
3. Adaptively continue null draws in batches until the configured significance rule is met or a cap/interrupt is reached.
4. Report p-values, adjusted p-values, and effect-size stats in benchmark metrics outputs.

---

## 1) Conceptual Goal (implemented)

For each `(pair, layer, metric)` hypothesis, observed scores are computed from matched samples.

Null draws are built from image-level mismatched samples:

- observed example: `rgb(scene_i, frame_t)` vs `depth(scene_i, frame_t)`
- null example: `rgb(scene_i, frame_t)` vs `depth(scene_j, frame_u)` with `scene_j != scene_i`

Then compute the one-sided p-value:

- `p = (1 + count(null_cka >= observed_cka)) / (N + 1)`

Where `N` is the number of null draws.

---

## 2) Practical Workflow

Main path:

1. `python -m src.run_benchmark` orchestrates extraction -> null -> metrics.
2. Null stage can reuse and continue from previous null artifacts.
3. Metrics stage enriches rows with significance fields using the latest null table.

Optional path:

- Run null only with `python -m src.run_null_distribution runtime.metrics_input_run_id=<run_id>`.

---

## 3) Main Code Locations

- `src/analysis/null_distribution.py`
  - adaptive null loop, checkpoint writing, image-level mismatched sampling
  - stop evaluation helper (`evaluate_adaptive_stop`)
- `src/pipeline/benchmark.py`
  - unified orchestration and artifact reuse logic
  - significance enrichment and adjusted p-values in metrics stage
- `src/run_null_distribution.py`
  - null-only entrypoint routed through benchmark stage helpers

---

## 4) YAML / Hydra Controls

Core null config (`analysis.null_distribution`):

- artifact files: `artifact_dir`, `artifact_filename`, `state_filename`
- sampling: `sampling_mode`, `replace`, `min_scenes`
- batching: `draws_per_batch`, `min_total_draws`, `max_total_draws`, `max_batches`
- interrupt: `on_keyboard_interrupt`
- adaptive stop: `adaptive_stop.enabled`, `adaptive_stop.alpha`, `adaptive_stop.correction`, `adaptive_stop.require_all_hypotheses`

Runtime reuse controls (`runtime.reuse`):

- `extraction`
- `null_distribution`
- `metrics_inputs`

---

## 5) Data Contract for Null Artifacts

Define one stable artifact schema so all stages can interoperate.

Main columns:

- `metric`
- `pair`
- `layer`
- `draw_id`
- `draw_batch_id`
- `null_value`
- metadata fields (`n_samples`, `sampling`, `seed`, `run_id`, scene IDs/types)

Storage options:

- CSV for readability
- Parquet for speed and size

Recommended: write both when configured.

---

## 6) Metrics Output Contract

Metrics rows include:

- `p_value`
- `p_value_adjusted`
- `is_significant`
- `n_null_draws`
- `null_mean`
- `null_std`
- `delta_vs_null_mean` (observed - null_mean)
- `z_score`

This should be added in:

- `results/runs/<run_id>/metrics/metrics.csv`
- W&B table logging

---

## 7) Sampling Semantics

Null draws enforce cross-scene mismatching:

- left and right samples are selected at image level
- each null draw uses a target sample size comparable to observed sample size
- replacement policy is configurable

---

## 8) Remaining Extensions (optional)

1. Add stronger resume metadata (per-hypothesis draw counters and timestamps).
2. Add optional confidence-interval-based stopping criteria.
3. Add metric-specific stop policies (e.g., stop on CKA only, continue for others).
4. Add dashboard panels for null progression over batches.

