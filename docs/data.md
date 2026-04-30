# Data and Model Setup

This document defines where to store heavy resources (datasets and model
weights) and how to download them for reproducible onboarding.

## 1) Storage layout (local and SCITAS)

Use a single root directory for large resources.

Default (recommended for this repo):

- Local machine: `<repo>/resources`
- SCITAS: `<repo>/resources` (or switch to `$SCRATCH/otm_resources` if needed)

Recommended structure:

```text
resources/
  models/
    4m/
      EPFL-VILAB/
        4M-7_B_CC12M/
  datasets/
    hypersim/
      metadata_camera_trajectories.csv   ← scene-type metadata (auto-downloaded)
      ai_001_001/                        ← downloaded scene data
      ai_001_002/
      ...
    diode/
```

Optional env var override (only if you do not want to use `<repo>/resources`):

- Linux/macOS: `export OTM_RESOURCES_ROOT=/path/to/otm_resources`
- PowerShell: `$env:OTM_RESOURCES_ROOT = "D:\otm_resources"`

If not set, bootstrap scripts and configs default to `<repo>/resources`.

## 2) 4M-7B model download (Hugging Face)

For first POC, use:

- `EPFL-VILAB/4M-7_B_CC12M`

### Option A (recommended): Hugging Face CLI

1. Install and login:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

2. Download snapshot:

```bash
huggingface-cli download EPFL-VILAB/4M-7_B_CC12M \
  --local-dir "$OTM_RESOURCES_ROOT/models/4m/EPFL-VILAB/4M-7_B_CC12M"
```

### Option B: Python API

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="EPFL-VILAB/4M-7_B_CC12M",
    local_dir="/path/to/otm_resources/models/4m/EPFL-VILAB/4M-7_B_CC12M",
    local_dir_use_symlinks=False,
)
```

## 3) Datasets to download

For this project:

- **Hypersim** (primary, for initial RQ1 POC)
- **DIODE** (secondary validation on real images)

The official sources and scripts may evolve. Keep your raw downloads in:

- `.../datasets/hypersim/`
- `.../datasets/diode/`

and keep project-specific preprocessing outputs in:

- `<repo>/results/processed/` (small metadata)
- or `$OTM_RESOURCES_ROOT/datasets/<name>/processed/` (large artifacts)

## 4) Reproducible onboarding with separate scripts

We keep setup modular: one script per concern.

### 4.1 Base setup (folders + 4M model)

Linux/SCITAS:

```bash
bash scripts/setup_base.sh EPFL-VILAB/4M-7_B_CC12M
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_base.ps1 -ModelRepo EPFL-VILAB/4M-7_B_CC12M
```

### 4.2 DIODE dataset (optional)

Linux/SCITAS:

```bash
# full depth + normals
bash scripts/download_diode.sh
# depth only (smaller)
bash scripts/download_diode.sh --depth-only
```

Windows:

```powershell
# full depth + normals
powershell -ExecutionPolicy Bypass -File scripts/download_diode.ps1
# depth only (smaller)
powershell -ExecutionPolicy Bypass -File scripts/download_diode.ps1 -DepthOnly
```

### 4.3 Hypersim scene-type metadata

`resources/datasets/hypersim/metadata_camera_trajectories.csv` is a lightweight
CSV file sourced from Apple's official Hypersim release. It is **automatically
copied** from the cloned `ml-hypersim` repo the first time you run any
`download_hypersim_subset.sh` command — no manual step needed. It maps every camera trajectory to its scene
and scene type (e.g. Bathroom, Office, Living room).

Columns used by the pipeline:

| Column | Example | Purpose |
|---|---|---|
| `Animation` | `ai_001_001_cam_00` | Identifies the trajectory; first 3 `_`-parts give the scene ID (`ai_001_001`) |
| `Scene type` | `Bathroom` | Used by the null-distribution engine to enforce cross-scene-type sampling |

The null distribution step reads this file automatically when
`analysis.null_distribution.sampling=cross_scene_type_random` (the default).
If the file is missing, the pipeline falls back to plain cross-scene sampling
with a warning.

### 4.4 Hypersim download (full by default, subset via args)

This uses Apple’s contrib downloader that supports partial file selection:
[`contrib/99991`](https://github.com/apple/ml-hypersim/tree/main/contrib/99991).

Default (full dataset):

Linux/SCITAS:

```bash
bash scripts/download_hypersim_subset.sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_hypersim_subset.ps1
```

Subset examples:

Linux/SCITAS:

```bash
bash scripts/download_hypersim_subset.sh \
  --scenes ai_001_001 ai_001_002 ai_001_003 \
  --include-rgb --include-depth --include-metadata
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_hypersim_subset.ps1 `
  -Scenes ai_001_001,ai_001_002,ai_001_003 `
  -IncludeRgb -IncludeDepth -IncludeMetadata
```

Other modality flags are available:

- `--include-normals` / `-IncludeNormals`
- `--include-semantic` / `-IncludeSemantic`

Notes:

- Hypersim full download is very large (~1.9TB).
- For POC, use scene subsets and only needed files (RGB + depth + minimal metadata).

## 5) Notes for SCITAS (Izar)

- Prefer storing heavy files in `$SCRATCH` rather than `$HOME`.
- In SLURM jobs, export:

```bash
export OTM_RESOURCES_ROOT="$SCRATCH/otm_resources"
```

- Keep run outputs in project `results/` or in scratch, then copy final
  summaries/figures to the repository.
