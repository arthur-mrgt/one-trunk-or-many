# One Trunk or Many?

**Probing Representational Convergence in the 4M Multimodal Encoder.**

Project page (source of truth for results and figures):
<https://arthur-mrgt.github.io/one-trunk-or-many/>

EPFL CS-503 *Visual Intelligence*, Spring 2026 —
Arthur Margeat, Adrien Clement, Martina Gatti, Albert Fares.

---

## Abstract

4M-21 ([`EPFL-VILAB/4M-7_B_CC12M`](https://huggingface.co/EPFL-VILAB/4M-7_B_CC12M))
processes every modality through the same 12-layer transformer encoder with
shared weights, but shared weights do not guarantee a shared representation.
This repository measures, layer by layer, how strongly that encoder unifies
RGB, depth, and surface normals on Hypersim (synthetic indoor) and DIODE
(real indoor + outdoor). We score each (layer, modality pair) with **CKA**,
**PWCCA**, and **k-NN overlap**, tested against an **adaptive permutation
null** built from mismatched scenes with Benjamini–Hochberg FDR control, then
quantify how *complete* the convergence is via a **transmodal triangulation**
analysis that bounds the residual cross-modal structure inaccessible by
linear projection through a third modality. The encoder is *one trunk*, but
a *measurably imperfect* one under linear probing — see the project page for
the full results.

---

## Research question

**Does the 4M encoder build a single unified representation across modalities,
and if so, how complete is that unification?** We decompose it into four
sub-questions, each owned by a section of the project page:

- **(a)** Layer-wise activation geometry of each modality —
  [§4.1](https://arthur-mrgt.github.io/one-trunk-or-many/#results-rq-a).
- **(b)** Paired-scene vs mismatched scalar similarity —
  [§4.2](https://arthur-mrgt.github.io/one-trunk-or-many/#results-rq-b).
- **(c)** Effect of joint forward passes and the imprint on solo
  representations —
  [§4.3](https://arthur-mrgt.github.io/one-trunk-or-many/#results-rq-c).
- **(d)** Completeness of the convergence via transmodal triangulation —
  [§4.4](https://arthur-mrgt.github.io/one-trunk-or-many/#results-rq-d).

---

## Reproducing the results (Linux)

The full pipeline is Hydra-driven, GPU + FAISS end-to-end, with single-node
multi-GPU via `torchrun`. PowerShell equivalents for every setup step live
in [`docs/data.md`](docs/data.md).

### 1. Environment

```bash
conda create -n trunk python=3.11 -y && conda activate trunk

bash scripts/setup/install_python_deps.sh        # GPU FAISS (default)
# bash scripts/setup/install_python_deps.sh --cpu  # CPU FAISS fallback (~3-5x slower k-NN)
```

**Why this script and not plain `pip install -r requirements.txt`?**
Two packages must be installed *after* the pinned requirements and in this
specific order — the script handles it for you:

1. `pip install git+https://github.com/apple/ml-4m.git` — `fourm` is not on
   PyPI.
2. `pip install --no-deps faiss-gpu-cu12==1.9.0.post1` — `--no-deps` is
   required because `faiss-gpu-cu12` wheels `>=1.10` pull `numpy>=2`, which
   breaks `fourm` via a torch ABI mismatch; we keep `numpy==1.26.4`.

CPU fallback uses `pip install --no-deps faiss-cpu==1.8.0` for the same
reason. The script ends with an environment sanity check (numpy, torch,
fourm, faiss versions).

### 2. Model and datasets

```bash
# 4M-7B weights → resources/models/4m/EPFL-VILAB/4M-7_B_CC12M
bash scripts/setup/setup_base.sh EPFL-VILAB/4M-7_B_CC12M

# Hypersim subset (100 scenes, stratified by Scene type, reproducible via --seed)
bash scripts/data/download_hypersim_subset.sh \
  --n-random 100 --seed 42 --jobs 8 --cleanup-zip \
  --include-rgb --include-depth --include-normals --include-metadata

# DIODE subset (2 random train scans per scene + full val, ~13 GB after extraction)
bash scripts/data/download_diode.sh --train-sample 2 --with-val
```

Datasets and weights land under `resources/` (gitignored). Override the root
with `export OTM_RESOURCES_ROOT=/path/to/otm_resources` — see
[`docs/data.md`](docs/data.md) for the full layout and SCITAS notes.

### 3. End-to-end runs

```bash
# Smoke presets (~minutes, validate the full wiring)
python -m src.run_benchmark --config-name benchmark_rq1_smoke_hypersim
python -m src.run_benchmark --config-name benchmark_rq1_smoke_diode

# Production presets (100 scenes, ~4000 paired samples, 1000+ null draws / hypothesis)
python -m src.run_benchmark --config-name benchmark_rq1_final_hypersim
python -m src.run_benchmark --config-name benchmark_rq1_final_diode
```

Each run prints a `RUN_ID` of the form `<stage>-<YYYYMMDD>-<HHMMSS>` and
writes everything under `results/runs/<RUN_ID>/` — see
[Run outputs](#run-outputs) below.

### 4. Re-run a single stage (iteration shortcut)

The benchmark is split into three composable Hydra entry points. Use them
when you want to recompute metrics or the null without re-extracting
activations:

```bash
python -m src.run_extraction                                                # extract only
python -m src.run_metrics            runtime.metrics_input_run_id=<RUN_ID>  # metrics only
python -m src.run_null_distribution  runtime.metrics_input_run_id=<RUN_ID>  # null only
```

See [`docs/pipeline.md`](docs/pipeline.md) for the reuse policies
(`runtime.reuse.activations`, `runtime.reuse.null_distribution`, …).

### 5. Completeness analysis (transmodal triangulation)

Pure post-processing on an existing run directory — no forward passes:

```bash
python -m src.run_rq2_triangulation \
  --run_id <RUN_ID> \
  --runs_root results/runs \
  --n_null_draws 1000 --n_boots 200 --seed 42
```

Writes `results/runs/<RUN_ID>/rq2/fragmentation_results.csv`. The CSV is
consumed by `notebooks/rq2_analysis.ipynb` (figures) and by
`scripts/build_rq2_json.py` (publishes the JSON payload used by the project
page's interactive chart).

### 6. Notebooks

```bash
jupyter lab notebooks/
```

| Notebook | Purpose |
|---|---|
| [`analysis_notebook.ipynb`](notebooks/analysis_notebook.ipynb) | t-SNE / UMAP paired-line plots, inter-cloud distances, CKA layer×layer matrix, activation entropy, intrinsic dim (from pre-computed parquet frames). |
| [`pca_tsne_umap_visualizations_all_pairs.ipynb`](notebooks/pca_tsne_umap_visualizations_all_pairs.ipynb) | Activation geometry visualisations for sub-question **(a)**. |
| [`rq1_results_analysis.ipynb`](notebooks/rq1_results_analysis.ipynb) | Metrics + null + significance across all modality pairs; produces the report-ready significance tables. |
| [`rq1_multipair_hypersim_diode.ipynb`](notebooks/rq1_multipair_hypersim_diode.ipynb) | Hypersim vs DIODE multi-pair comparison plots. |
| [`rq1_visual_report.ipynb`](notebooks/rq1_visual_report.ipynb) | Renders the figures used in the report and on the project page for sub-questions **(b)–(c)**. |
| [`rq2_analysis.ipynb`](notebooks/rq2_analysis.ipynb) | Transmodal triangulation: fragmentation curves for sub-question **(d)**; emits the fragmentation summary CSV. |

Each notebook starts by setting a `RUN_ID` constant in the first cell —
point it at the run directory you want to visualise.

### 7. Tests

```bash
pytest tests/
```

Covers the metrics metadata contract, pair resolution, PWCCA sanity, and a
small null-distribution smoke test.

### 8. Cluster (SCITAS / SLURM)

[`docs/cluster.md`](docs/cluster.md) covers the interactive and `sbatch`
launchers ([`scripts/run/run_interactive.sh`](scripts/run/run_interactive.sh),
[`scripts/run/submit_slurm.sh`](scripts/run/submit_slurm.sh)), single-node
multi-GPU via `torchrun`, dataset staging to `/tmp`, activation snapshotting
to survive node failures, and pulling a finished run back to a workstation
with [`scripts/data/recover_run_from_cluster.py`](scripts/data/recover_run_from_cluster.py).

---

## Repository structure

```text
.
├── configs/                Hydra config groups + top-level presets
│   ├── data/               hypersim.yaml, diode.yaml
│   ├── model/              fourm.yaml, fourm_mock.yaml
│   ├── metrics/            rq1.yaml
│   ├── runtime/            local.yaml, full_gpu_faiss.yaml
│   ├── tracking/           wandb_on.yaml, wandb_off.yaml
│   ├── slurm/              izar.yaml
│   ├── default.yaml
│   └── benchmark_rq1_{smoke,final}_{hypersim,diode}.yaml
│
├── src/                    Python package
│   ├── run_benchmark.py            full pipeline (extraction → metrics → null)
│   ├── run_extraction.py           extraction stage only
│   ├── run_metrics.py              metrics stage only
│   ├── run_null_distribution.py    adaptive null stage only
│   ├── run_rq2_triangulation.py    completeness / triangulation analysis
│   ├── pipeline/                   benchmark orchestrator, rq2_triangulation
│   ├── metrics/                    CKA, PWCCA, kNN-overlap + registry
│   ├── models/                     4M backend + registry
│   ├── data/                       dataset loaders + registry
│   ├── analysis/                   adaptive null sampling, BH-FDR stop
│   ├── plotting/                   headless figure renderer
│   └── utils/                      distributed, config, tracking helpers
│
├── scripts/
│   ├── setup/              install_python_deps.sh, setup_base.sh, bootstrap_resources.sh (+ .ps1)
│   ├── data/               download_hypersim_subset.{sh,py,ps1}, download_diode.{sh,ps1},
│   │                       recover_run_from_cluster.py, rewrite_activation_index_paths.py
│   ├── run/                run_interactive.sh, submit_slurm.sh
│   ├── smoke/              smoke_test_joint_pass.py, smoke_test_null_modes.py
│   └── build_rq2_json.py   serialises the fragmentation CSV for the project page
│
├── notebooks/              see table above
├── tests/                  pytest suite
├── docs/                   specialised documentation (see index below)
├── website/                static project page (deployed via GitHub Pages)
├── figures/                exported PDF figures used in the report
├── resources/              datasets + model weights (gitignored)
└── results/                runs/, wandb/, slurm/, snapshots/ (gitignored)
```

---

## Configuration

All runs compose a single Hydra config from
[`configs/default.yaml`](configs/default.yaml) plus six config groups:

| Group | Files | Purpose |
|---|---|---|
| `data` | `hypersim.yaml`, `diode.yaml` | Dataset root, modalities, sampling caps, null-sampling metadata. |
| `model` | `fourm.yaml`, `fourm_mock.yaml` | Encoder backend, layer policy, embedding shape. |
| `metrics` | `rq1.yaml` | Enabled metrics, modality pairs, PCA/FAISS backends, joint-pass toggle. |
| `runtime` | `local.yaml`, `full_gpu_faiss.yaml` | Device, workers, distributed flags, reuse policy. |
| `tracking` | `wandb_on.yaml`, `wandb_off.yaml` | W&B project, entity, tags. |
| `slurm` | `izar.yaml` | Cluster defaults. |

The top-level `analysis.null_distribution.*` block controls the adaptive
null stage (sampling mode, `min/max_total_draws`, `adaptive_stop.alpha`,
BH-FDR vs Bonferroni, …).

Most common CLI overrides:

```bash
# Subset on the fly
python -m src.run_benchmark --config-name benchmark_rq1_smoke_hypersim \
  data.n_scenes=20 data.max_total_samples=1000

# Restrict modality pairs
python -m src.run_benchmark "metrics.pairs=[[rgb,depth]]"

# Force CPU / single GPU / multi-GPU
python -m src.run_benchmark runtime.device=cpu
torchrun --nproc_per_node=4 -m src.run_benchmark \
  --config-name benchmark_rq1_final_hypersim runtime.distributed.enabled=true

# Toggle W&B
python -m src.run_benchmark tracking=wandb_off

# Reuse activations from a previous run
python -m src.run_benchmark \
  runtime.activation_input_run_id=<RUN_ID> runtime.reuse.activations=true
```

Full reference: [`docs/configs.md`](docs/configs.md).

---

## Run outputs

```text
results/
├── runs/
│   ├── <RUN_ID>/
│   │   ├── activations/                # one .npy per (sample, modality, layer)
│   │   ├── artifacts/                  # activation_index_<pair>.csv
│   │   ├── metrics/                    # metrics.csv + null_stop_evaluation.csv
│   │   ├── rq2/                        # fragmentation_results.csv (after run_rq2_triangulation)
│   │   ├── figures/                    # PNGs written by src.plotting.metrics_null
│   │   ├── resolved_config.json
│   │   └── run_summary.json
│   └── null_distributions/<RUN_ID>/    # null_distribution.csv + state.json
├── wandb/                              # local W&B cache (when tracking=wandb_on)
├── slurm/                              # job stdout/stderr
└── snapshots/                          # tar snapshots of /tmp activations
```

Headless plotting for any finished run:

```bash
python -m src.plotting.metrics_null --run-id <RUN_ID>
# or auto-detect the latest run:
python -m src.plotting.metrics_null
```

Column schema for `metrics.csv` and `null_distribution.csv`:
[`docs/results.md`](docs/results.md).

---

## Documentation index

| Topic | Document |
|---|---|
| Project page (source of truth) | <https://arthur-mrgt.github.io/one-trunk-or-many/> |
| Datasets and 4M model setup | [`docs/data.md`](docs/data.md) |
| Pipeline stages and extension points | [`docs/pipeline.md`](docs/pipeline.md) |
| Hydra config layout and overrides | [`docs/configs.md`](docs/configs.md) |
| Cluster launchers (interactive, SLURM) | [`docs/cluster.md`](docs/cluster.md) |
| Run outputs, W&B, diagnostics | [`docs/results.md`](docs/results.md) |
| Project proposal | [`docs/proposal.pdf`](docs/proposal.pdf) |
| Progress report | [`docs/progress.pdf`](docs/progress.pdf) |
| Final presentation | [`docs/final_presentation.pdf`](docs/final_presentation.pdf) |

---

## License

MIT License. See [`LICENSE`](LICENSE).
