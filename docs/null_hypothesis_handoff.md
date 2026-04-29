## Null Hypothesis / Noise Integration Handoff

This note explains what must be implemented to add null-hypothesis significance testing to the current benchmark pipeline.

The target behavior is:

1. Compute a null distribution (noise floor) from mismatched samples.
2. Reuse that null distribution during regular benchmark runs.
3. Report p-values (and optional effect-size stats) together with CKA.
4. Keep everything configurable via YAML and Hydra overrides.

---

## 1) Conceptual Goal

For each modality pair and each layer, we currently compute an observed CKA using matched samples.

To test if this CKA is meaningful, we need a null distribution built from mismatched pairs:

- matched example: `rgb(scene_i, frame_t)` vs `depth(scene_i, frame_t)`
- null example: `rgb(scene_i, frame_t)` vs `depth(scene_j, frame_u)` with `scene_j != scene_i`

Then compute the one-sided p-value:

- `p = (1 + count(null_cka >= observed_cka)) / (N + 1)`

Where `N` is the number of null draws.

---

## 2) Recommended Practical Workflow

Use a two-step workflow:

### Step A: Precompute null/noise

Run a dedicated command/script that:

- loads activation indexes
- builds mismatched sample sets for each pair/layer
- computes many null CKA values (`N` draws)
- saves them as reusable artifacts

### Step B: Run benchmark with significance

During normal metrics stage:

- compute observed CKA as today
- load precomputed null distribution
- compute p-value and optional summary stats
- append those fields to metrics outputs

This avoids recomputing null every run and keeps benchmark runs fast.

---

## 3) Where To Implement

### Core files to extend

- `src/analysis/null_distribution.py`
  - implement actual null distribution computation
  - this is currently scaffolded and should become the main null engine

- `src/pipeline/benchmark.py`
  - in metrics stage, add optional logic to load null artifacts
  - compute p-values for each `(pair, layer, metric)` row
  - write enriched `metrics.csv`

- `src/run_metrics.py` and/or a new entrypoint (recommended):
  - add a dedicated command to compute null only, for example:
    - `python -m src.run_null_distribution`

- `src/utils/io.py`
  - add helper read/write functions for null artifacts if needed

### Optional new module

- `src/analysis/significance.py`
  - put pure statistical helpers here, for example:
    - `compute_p_value(observed, null_values, side="greater")`
    - optional BH-FDR correction helper

---

## 4) YAML / Hydra Configuration Changes

Keep the feature fully config-driven.

### Extend existing scaffold in `configs/default.yaml`

Under `analysis.null_distribution`, add practical fields such as:

- `enabled: false`
- `n_draws: 1000`
- `sampling: cross_scene_random`
- `seed: 42`
- `replace: true`
- `artifact_dir: ${paths.runs_root}/null_distributions`
- `reuse_if_exists: true`
- `min_scenes: 2`

### Runtime flags

Add runtime fields (or a dedicated config group) to control:

- whether to load null artifacts during metrics stage
- null artifact run id or path
- behavior if null is missing (`error` vs `warn_and_skip`)

### Example usage pattern

- Null precompute:
  - `python -m src.run_null_distribution analysis.null_distribution.enabled=true ...`
- Benchmark with p-values:
  - `python -m src.run_benchmark runtime.null_input_run_id=<id> ...`

---

## 5) Data Contract for Null Artifacts

Define one stable artifact schema so all stages can interoperate.

Suggested columns:

- `metric` (for now `cka`)
- `pair` (for example `rgb-depth`)
- `layer` (for example `layer_03`)
- `draw_id`
- `null_value`
- metadata fields (`n_samples`, `sampling`, `seed`, `run_id`)

Storage options:

- CSV for readability
- Parquet for speed and size

Recommended: write both when configured.

---

## 6) Metrics Output Contract (after integration)

Extend current metrics rows with significance fields:

- `p_value`
- `null_mean`
- `null_std`
- `delta_vs_null_mean` (observed - null_mean)
- optional `z_score`

This should be added in:

- `results/runs/<run_id>/metrics/metrics.csv`
- W&B table logging

---

## 7) Sampling Guidance

For the null, prefer cross-scene mismatching:

- enforce different scenes when possible
- sample random mismatches instead of exhaustive combinations

Why:

- exhaustive combinations are expensive
- Monte Carlo draws are usually enough and scalable

---

## 8) Minimal Implementation Plan for Next Developer

1. Implement null computation in `src/analysis/null_distribution.py`.
2. Add a null-only entrypoint (`src/run_null_distribution.py`).
3. Add YAML fields and config docs for null settings.
4. Integrate null loading + p-value computation in `src/pipeline/benchmark.py`.
5. Extend metrics output schema and W&B logging.
6. Add a short smoke test:
   - run null precompute on small data
   - run benchmark with null artifact
   - verify p-values exist in `metrics.csv`.

---

## 9) Acceptance Criteria

The feature is complete when:

- null distribution can be computed independently and cached
- benchmark can consume cached null artifacts
- p-values are reported per pair and layer
- all controls are exposed via Hydra YAML
- behavior is documented and reproducible from command line

