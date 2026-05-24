# One Trunk or Many?

Measuring representational convergence in any-to-any vision models.

This repository currently provides a Hydra-driven RQ1 pipeline around 4M-7B (`EPFL-VILAB/4M-7_B_CC12M`) with:

- layer-wise activation extraction
- CKA, PWCCA, and k-NN overlap computation per layer and modality pair
- adaptive null-distribution estimation with p-value enrichment
- run-scoped outputs in `results/runs/<run_id>/`
- optional Weights and Biases tracking, including metric-vs-layer and significance plots

Project proposal: [`docs/proposal.pdf`](docs/proposal.pdf)

## Quick Start

This is the full startup path:

1. Create the conda environment and install dependencies.
2. Run the base setup script to create resource folders and download 4M.
3. Download datasets.
4. Run the benchmark, optionally with W&B enabled.

### 1) Create environment

```bash
conda create -n trunk python=3.11 -y
conda activate trunk
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Optional CUDA setup

The default runtime setting is `runtime.device=auto`, which selects CUDA if available.

Check CUDA availability:

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If CUDA is not detected while you have an NVIDIA GPU, install a CUDA-enabled PyTorch build in your active environment.

Example for CUDA 12.8:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

### 3) Run initial setup and download 4M

This creates local resource folders and downloads `EPFL-VILAB/4M-7_B_CC12M`.

Linux/macOS:

```bash
bash scripts/setup/setup_base.sh EPFL-VILAB/4M-7_B_CC12M
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/setup_base.ps1 -ModelRepo EPFL-VILAB/4M-7_B_CC12M
```

By default, resources are stored in `<repo>/resources`.  
Override path with `OTM_RESOURCES_ROOT` when needed.

### 4) Download datasets

#### Hypersim subset (recommended for POC)

Linux/macOS:

```bash
bash scripts/data/download_hypersim_subset.sh \
  --scenes ai_001_001 ai_001_002 \
  --include-rgb --include-depth --include-metadata
```

```bash
bash scripts/data/download_hypersim_subset.sh \
  --scenes ai_024_010 ai_001_001 ai_001_006 ai_009_001 ai_005_010 ai_023_004 ai_013_002 ai_027_005 ai_001_005 ai_053_001 ai_016_009 ai_001_004 \
  --include-rgb --include-depth --include-metadata
```

```bash
bash scripts/data/download_hypersim_subset.sh \
  --scenes ai_024_010 ai_001_001 ai_001_006 ai_009_001 ai_005_010 ai_023_004 ai_013_002 ai_027_005 ai_001_005 ai_053_001 ai_016_009 ai_001_004 ai_001_002 ai_005_003 ai_047_002 ai_008_001 ai_006_006 ai_002_007 ai_006_002 ai_005_005 \
  --include-rgb --include-depth --include-metadata
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/data/download_hypersim_subset.ps1 `
  -Scenes ai_001_001,ai_001_002 `
  -IncludeRgb -IncludeDepth -IncludeMetadata
```

#### DIODE

Linux/macOS:

```bash
bash scripts/data/download_diode.sh
# or smaller depth-only download
bash scripts/data/download_diode.sh --depth-only
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/data/download_diode.ps1
# or smaller depth-only download
powershell -ExecutionPolicy Bypass -File scripts/data/download_diode.ps1 -DepthOnly
```

### 5) Run benchmark

Default:

```bash
python -m src.run_benchmark
```

POC example with W&B:

```bash
python -m src.run_benchmark data.n_scenes=1 tracking=wandb_on
```

Final run (20 scenes, rgb-depth, null from scratch, 3 metrics) with W&B:

```bash
python -m src.run_benchmark tracking=wandb_on
```

Useful overrides:

```bash
# Force CPU
python -m src.run_benchmark runtime.device=cpu

# Force single GPU
python -m src.run_benchmark runtime.device=cuda

# Multi GPU DataParallel
python -m src.run_benchmark runtime.device=cuda runtime.multi_gpu_strategy=data_parallel

# Use tokenized RGB input instead of pixel RGB
python -m src.run_benchmark model.rgb_input_mode=tokenized
```

## W&B setup

W&B presets:

- `configs/tracking/wandb_off.yaml`
- `configs/tracking/wandb_on.yaml`

With `tracking=wandb_on`, the pipeline logs:

- metrics table
- summary values
- metric-vs-layer (line and scatter) per `(metric, pair)`
- significance plots when null is available:
  - `pvalue_vs_layer`
  - `pvalue_adjusted_vs_layer`
  - `delta_vs_null_mean`

## Configuration model (Hydra)

Main composition lives in `configs/default.yaml`:

- `data: hypersim`
- `model: fourm`
- `metrics: rq1`
- `tracking: wandb_off`
- `runtime: local`
- `slurm: izar`

Common overrides:

```bash
# Run on a subset of scenes
python -m src.run_benchmark data.n_scenes=10

# Change modality pair list
python -m src.run_benchmark "metrics.pairs=[[rgb,depth]]"

# Enable W&B
python -m src.run_benchmark tracking=wandb_on

# Force null recomputation from scratch
python -m src.run_benchmark runtime.reuse.null_distribution=false

# Reuse activations from a specific run
python -m src.run_benchmark runtime.activation_input_run_id=<run_id> runtime.reuse.activations=true
```

## Repository architecture

```text
.
├── configs/                  # Hydra config groups and defaults
│   ├── data/                 # Dataset configs
│   ├── model/                # Model backend and layer settings
│   ├── metrics/              # Metrics and modality pairs
│   ├── runtime/              # Device, workers, multi-GPU strategy
│   ├── tracking/             # W&B presets
│   └── slurm/                # Cluster defaults
├── scripts/                  # Setup, download, and SLURM scripts
├── src/
│   ├── data/                 # Dataset loaders and registry
│   ├── models/               # 4M wrappers and registry
│   ├── metrics/              # CKA, PWCCA, k-NN overlap, reductions
│   ├── pipeline/             # Benchmark orchestration + stage helpers
│   ├── analysis/             # Null runner/sampling/stop/artifacts modules
│   └── utils/                # I/O, tracking, config helpers
├── docs/                     # Setup and pipeline docs
├── resources/                # Local heavy assets (gitignored)
└── results/                  # Run outputs (mostly gitignored)
```

## Pipeline flow

1. `src.run_benchmark` resolves Hydra config.
2. Extraction stage loads aligned sample pairs from the selected dataset.
3. Model wrapper encodes each modality and saves vectors in `activations/`.
4. Activation index is written in `artifacts/activation_index_<pair>.csv`.
5. Observed metrics are computed per layer (`CKA`, `PWCCA`, `kNN overlap`).
6. If enabled, adaptive null draws run from mismatched samples and stop by threshold/caps.
7. Metrics stage enriches rows with significance statistics from null artifacts.
8. Metrics are written to `metrics/metrics.csv` and optionally logged to W&B.

## RQ1 final and smoke presets

Two Hydra presets wrap the end-to-end pipeline for the RQ1 deliverables:

- `benchmark_rq1_final_hypersim.yaml` — Hypersim, 100 scenes, ~4000 randomly
  sampled paired examples, 3 modality pairs (`rgb-depth`, `rgb-normals`,
  `depth-normals`), 3 metrics (CKA, PWCCA, kNN overlap), adaptive null with
  `min_total_draws=1000` and `alpha=0.01`.
- `benchmark_rq1_smoke_hypersim.yaml` — same wiring with 5 scenes, 200 samples
  and 100 null draws for fast end-to-end validation.

Run locally (after the dataset is downloaded):

```bash
python -m src.run_benchmark --config-name benchmark_rq1_smoke_hypersim
python -m src.run_benchmark --config-name benchmark_rq1_final_hypersim
```

Run interactively on a GPU node (local or `srun --pty bash`):

```bash
# Smoke run
PRESET=benchmark_rq1_smoke_hypersim bash scripts/run/run_interactive.sh
# Final run
bash scripts/run/run_interactive.sh
```

Submit as a SLURM batch job (e.g. SCITAS):

```bash
# Smoke run
PRESET=benchmark_rq1_smoke_hypersim sbatch scripts/run/submit_slurm.sh
# Final run
sbatch scripts/run/submit_slurm.sh
```

The launcher prints the resolved `RUN_ID` along with the paths to
`metrics.csv` and `null_distribution.csv`. To render the layer-wise plots
and p-value diagnostics, open `notebooks/pvalue_diagnostics.ipynb`, set
`RUN_ID` to the printed value, and run all cells.

## Run outputs

Each run writes to:

`results/runs/<run_id>/`

Typical files:

- `activations/` vector `.npy` files
- `artifacts/activation_index_<pair>.csv`
- `metrics/metrics.csv`
- `metrics/metrics.parquet` (if enabled)
- `metrics/null_stop_evaluation.csv` (when null stage runs)
- `run_summary.json`
- `resolved_config.json`

Null artifacts are stored separately under:

`results/runs/null_distributions/<run_id>/`

- `null_distribution.csv`
- `null_distribution.parquet` (if enabled)
- `null_distribution_state.json`

## Stage entrypoints

```bash
python -m src.run_benchmark
python -m src.run_extraction
python -m src.run_null_distribution runtime.metrics_input_run_id=<run_id>
python -m src.run_metrics runtime.metrics_input_run_id=<run_id>
```

## SCITAS / SLURM

Submit any Hydra preset as a batch job:

```bash
sbatch scripts/run/submit_slurm.sh                                       # final RQ1 preset
PRESET=benchmark_rq1_smoke_hypersim sbatch scripts/run/submit_slurm.sh   # smoke preset
```

Adapt the `#SBATCH` directives in `scripts/run/submit_slurm.sh` (partition,
`--gres`, `--time`, `--mem`) to your cluster.

## More docs

- Data and model setup: [`docs/data.md`](docs/data.md)
- Pipeline details: [`docs/pipeline.md`](docs/pipeline.md)

## License

MIT License. See [`LICENSE`](LICENSE).
