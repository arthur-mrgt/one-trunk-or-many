# One Trunk or Many?

**Measuring representational convergence in any-to-any vision models.**

This repository benchmarks how strongly the encoder of 4M-7B
([`EPFL-VILAB/4M-7_B_CC12M`](https://huggingface.co/EPFL-VILAB/4M-7_B_CC12M))
aligns the representations of different visual modalities (RGB, depth, surface
normals) across its layers. It is the experimental backbone of
**Research Question 1** of the project proposal
([`docs/proposal.pdf`](docs/proposal.pdf)): *Does the 4M encoder build a
single shared trunk, or many parallel ones?*

The pipeline is Hydra-driven and produces, for each modality pair and each
encoder layer:

- three similarity metrics: **CKA**, **PWCCA**, **k-NN overlap**;
- an **adaptive null distribution** of mismatched-pair scores with
  Benjamini–Hochberg-adjusted p-values;
- ready-to-plot CSVs, W&B logs, and a diagnostics notebook.

GPU + FAISS are used end-to-end; single-node multi-GPU is supported via
`torchrun`.

---

## Quick start

```bash
# 1. Environment
conda create -n trunk python=3.11 -y && conda activate trunk
python -m pip install -U pip && python -m pip install -r requirements.txt

# 2. Download the 4M-7B weights and the Hypersim subset
bash scripts/setup/setup_base.sh EPFL-VILAB/4M-7_B_CC12M
bash scripts/data/download_hypersim_subset.sh --scenes ai_001_001 ai_001_002 \
  --include-rgb --include-depth --include-normals --include-metadata

# 3. Run the smoke preset end-to-end (5 scenes, 200 samples, ~minutes)
python -m src.run_benchmark --config-name benchmark_rq1_smoke_hypersim

# 4. Render layer-wise plots and p-value diagnostics
#    Open notebooks/pvalue_diagnostics.ipynb, set RUN_ID, run all cells.
```

For the full RQ1 production run (100 scenes, ~4000 paired samples,
1 000+ null draws per hypothesis), use:

```bash
python -m src.run_benchmark --config-name benchmark_rq1_final_hypersim
```

See [`docs/cluster.md`](docs/cluster.md) for launching the same preset
interactively on a GPU node or as a SLURM batch job.

---

## What lives where

```text
.
├── configs/        Hydra config groups + top-level RQ1 presets
├── src/            Python package (pipeline, metrics, models, analysis, utils)
├── scripts/        Setup, data download, run launchers, smoke tests
├── notebooks/      Diagnostics notebook (pvalue_diagnostics.ipynb)
├── docs/           Specialized documentation (see index below)
├── tests/          pytest suite
├── resources/      Local heavy assets — datasets, model weights (gitignored)
└── results/        Run outputs — runs/, wandb/, slurm/ (gitignored)
```

---

## Documentation index

| Topic | Document |
|---|---|
| Datasets and 4M model setup | [`docs/data.md`](docs/data.md) |
| Pipeline stages, configs, extension points | [`docs/pipeline.md`](docs/pipeline.md) |
| Hydra config layout and common overrides | [`docs/configs.md`](docs/configs.md) |
| Cluster launchers (interactive, SLURM) | [`docs/cluster.md`](docs/cluster.md) |
| Run outputs, W&B, diagnostics notebook | [`docs/results.md`](docs/results.md) |
| Project proposal (RQ1–RQ3) | [`docs/proposal.pdf`](docs/proposal.pdf) |

---

## License

MIT License. See [`LICENSE`](LICENSE).
