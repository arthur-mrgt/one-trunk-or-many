## RQ1 Full Metrics Handoff (CKA + PWCCA + k-NN)

This document explains what to implement to complete the full RQ1 pipeline from the proposal:

- CKA
- PWCCA
- k-NN overlap
- null-hypothesis testing with mismatched scenes
- all modality pairs of interest
- large-scale Hypersim runs

The goal is to help a new developer continue implementation quickly.

---

## 1) Conceptual target from the proposal

RQ1 asks whether representations converge across modalities of the same scene, and whether this is consistent across geometric, directional, and topological metrics.

From the proposal:

- CKA is the primary metric (global relational structure, robust with `N x N` Gram matrices).
- PWCCA is a directional alignment metric and needs care when `N < d`.
- k-NN overlap captures local topology and should be stabilized with dimensionality reduction.
- Significance must be tested against an empirical null based on mismatched scenes.

Practical implication:

1. Compute observed similarity per `(pair, layer, metric)` on matched samples.
2. Compute null distributions per `(pair, layer, metric)` with mismatched scenes.
3. Report p-values and summary stats in benchmark outputs.

---

## 2) What must be added in code

### A) Implement PWCCA and k-NN metric logic

Current status:

- `src/metrics/pwcca.py` is a placeholder.
- `src/metrics/knn_overlap.py` is a placeholder.
- `src/metrics/registry.py` only supports `cka`.

Required:

- implement `pwcca(x, y, cfg)` in `src/metrics/pwcca.py`
- implement `knn_overlap(x, y, cfg)` in `src/metrics/knn_overlap.py`
- register both in `src/metrics/registry.py`

### B) Add metric-specific config in YAML

Current status:

- `configs/metrics/cka.yaml` has only `cka` config.

Required:

- extend this file (or split into metric group files) with:
  - `pwcca` section
  - `knn_overlap` section
  - shared reduction settings where needed

Suggested fields:

- `pwcca.enabled`, `pwcca.pca.enabled`, `pwcca.pca.n_components`, `pwcca.eps`
- `knn_overlap.enabled`, `knn_overlap.k`, `knn_overlap.pca.enabled`, `knn_overlap.pca.n_components`

### C) Add PCA preprocessing utility

Proposal-specific requirement:

- PWCCA and k-NN should be computed after PCA projection in high-dimensional settings.

Implement one reusable helper (for both metrics), for example:

- new file: `src/metrics/reduction.py`
- function: `fit_transform_pair(x, y, cfg)` with:
  - fit on stacked `[x; y]` or training-only strategy
  - deterministic seed support
  - clear handling for `n_components > min(N-1, d)`

### D) Extend benchmark metrics stage output schema

Current status:

- `src/pipeline/metrics.py` computes one metric value per layer.

Required:

- support all enabled metrics in config
- include metric-specific metadata columns when useful:
  - `metric`
  - `value`
  - `n_samples`
  - optional `reduction_used`, `k`

### E) Integrate null significance for all metrics

Use the null workflow described in `docs/null_hypothesis_handoff.md` and apply it to:

- CKA
- PWCCA
- k-NN overlap

At reporting time, each metric row should include:

- `null_mean`
- `null_std`
- `p_value`
- `delta_vs_null_mean`

---

## 3) Modality pairs: how to scale

RQ1 should run on all target modality pairs, not only `rgb-depth`.

Recommended pair generation:

- define modality list in config (for example `rgb, depth, normals, semseg`)
- generate all unordered pairs automatically, or define explicitly in YAML

If automated, keep one explicit switch:

- `metrics.pairs_mode: explicit | all_combinations`

For reproducibility and auditing, always write resolved pairs into:

- `run_summary.json`
- `resolved_config.json`

---

## 4) Execution order in practice

To run the complete RQ1 pipeline in production:

1. Download as much Hypersim as possible (scene count constrained by storage/time).
2. Run extraction and metric computation artifacts generation.
3. Run null/noise precompute script on mismatched scenes.
4. Run benchmark metrics with null artifacts loaded.
5. Export final tables and plots (including W&B).

Important:

- Null should be computed first when p-values are required.
- Benchmark should fail or warn clearly if significance is requested but null artifacts are missing.

---

## 5) Where each part belongs

### Metrics implementation

- `src/metrics/pwcca.py`
- `src/metrics/knn_overlap.py`
- `src/metrics/registry.py`
- optional: `src/metrics/reduction.py`

### Pipeline integration

- `src/pipeline/metrics.py`
- `src/pipeline/benchmark.py`

### Null hypothesis integration

- `src/analysis/null_distribution.py`
- optional: `src/analysis/significance.py`
- new entrypoint recommended: `src/run_null_distribution.py`

### Config

- `configs/metrics/cka.yaml` (or split to dedicated files)
- `configs/default.yaml` under `analysis.null_distribution`
- runtime knobs for null artifact loading

### Docs

- update `docs/pipeline.md`
- keep `docs/null_hypothesis_handoff.md` as the reference for significance workflow

---

## 6) Resource and complexity guidance

Do not compute all mismatched combinations exhaustively.

Use Monte Carlo sampling:

- fixed number of null draws per `(pair, layer, metric)` (for example 1000)
- cross-scene mismatches only
- deterministic seed for reproducibility

This is much cheaper and usually sufficient for stable p-values.

For large Hypersim:

- keep extraction artifacts on fast storage
- use chunked or batched metric computations to avoid memory spikes
- run on SLURM with pair-wise or metric-wise job partitioning if needed

---

## 7) Minimal acceptance criteria for handoff completion

The next developer can consider this done when:

1. `metrics.enabled: [cka, pwcca, knn_overlap]` runs end-to-end.
2. PWCCA and k-NN use configurable PCA preprocessing.
3. Full modality pair set is configurable and reproducible.
4. Null distributions are computed and reused in benchmark.
5. `metrics.csv` includes p-values for each metric and layer.
6. W&B logs include per-metric, per-layer curves/tables.

