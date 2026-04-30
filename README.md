# One Trunk or Many?

Measuring representational convergence in any-to-any vision models.

This repository currently provides a Hydra-driven RQ1 pipeline around 4M-7B (`EPFL-VILAB/4M-7_B_CC12M`) with:

- layer-wise activation extraction
- CKA computation per layer and modality pair
- run-scoped outputs in `results/runs/<run_id>/`
- optional Weights and Biases tracking, including CKA-vs-layer plots

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
bash scripts/setup_base.sh EPFL-VILAB/4M-7_B_CC12M
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_base.ps1 -ModelRepo EPFL-VILAB/4M-7_B_CC12M
```

By default, resources are stored in `<repo>/resources`.  
Override path with `OTM_RESOURCES_ROOT` when needed.

### 4) Download datasets

#### Hypersim subset (recommended for POC)

Linux/macOS:

```bash
bash scripts/download_hypersim_subset.sh \
  --scenes ai_001_001 ai_001_002 \
  --include-rgb --include-depth --include-metadata
```

```bash
bash scripts/download_hypersim_subset.sh \
  --scenes ai_024_010 ai_001_001 ai_001_006 ai_009_001 ai_005_010 ai_023_004 ai_013_002 ai_027_005 ai_001_005 ai_053_001 ai_016_009 ai_001_004 \
  --include-rgb --include-depth --include-metadata
```

```bash
bash scripts/download_hypersim_subset.sh \
  --scenes ai_024_010 ai_001_001 ai_001_006 ai_009_001 ai_005_010 ai_023_004 ai_013_002 ai_027_005 ai_001_005 ai_053_001 ai_016_009 ai_001_004 ai_001_002 ai_005_003 ai_047_002 ai_008_001 ai_006_006 ai_002_007 ai_006_002 ai_005_005 \
  --include-rgb --include-depth --include-metadata
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_hypersim_subset.ps1 `
  -Scenes ai_001_001,ai_001_002 `
  -IncludeRgb -IncludeDepth -IncludeMetadata
```

#### DIODE

Linux/macOS:

```bash
bash scripts/download_diode.sh
# or smaller depth-only download
bash scripts/download_diode.sh --depth-only
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_diode.ps1
# or smaller depth-only download
powershell -ExecutionPolicy Bypass -File scripts/download_diode.ps1 -DepthOnly
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
- CKA-vs-layer visualizations for each modality pair

## Configuration model (Hydra)

Main composition lives in `configs/default.yaml`:

- `data: hypersim`
- `model: fourm`
- `metrics: cka`
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
│   ├── metrics/              # CKA and metric registry
│   ├── pipeline/             # Extraction/metrics orchestration
│   ├── analysis/             # Null-distribution scaffold
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
5. Metrics stage loads indices and computes CKA per layer.
6. Metrics are written to `metrics/metrics.csv` and optionally logged to W&B.

## Run outputs

Each run writes to:

`results/runs/<run_id>/`

Typical files:

- `activations/` vector `.npy` files
- `artifacts/activation_index_<pair>.csv`
- `metrics/metrics.csv`
- `metrics/metrics.parquet` (if enabled)
- `run_summary.json`
- `resolved_config.json`

## Stage entrypoints

```bash
python -m src.run_benchmark
python -m src.run_extraction
python -m src.run_metrics runtime.metrics_input_run_id=<run_id>
```

## SCITAS / SLURM

Templates:

- `scripts/run_benchmark.slurm`
- `scripts/run_extraction.slurm`
- `scripts/run_metrics.slurm`

Submit:

```bash
sbatch scripts/run_benchmark.slurm
```

## More docs

- Data and model setup: [`docs/data.md`](docs/data.md)
- Pipeline details: [`docs/pipeline.md`](docs/pipeline.md)

## License

MIT License. See [`LICENSE`](LICENSE).
