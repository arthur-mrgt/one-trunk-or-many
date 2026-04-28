# One Trunk or Many?

**Measuring representational convergence in any-to-any vision models.**

Any-to-any models like [4M](https://github.com/apple/ml-4m) share a single Transformer encoder across modalities, but each modality enters through its own learned embedding. This raises a question: does the encoder actually unify modalities into a shared representation, or does it route them through effectively disjoint sub-networks?

This project develops a multi-metric protocol to probe representational convergence inside 4M-21's shared encoder, with statistical significance testing against a mismatched-scene null distribution.

[Project Proposal (PDF)](docs/proposal.pdf)

## Research Questions

**RQ1.** Does the encoder exhibit measurable representational convergence when processing different modalities of the same scene, and is this signal consistent across geometric, directional, and topological similarity metrics?

**RQ2.** Can complementary protocols (cross-modal transfer, residual analysis) address the functional and ceiling blindspots left open by RQ1?

## Approach

We pass each modality (RGB, depth, normals, segmentation) of a given scene separately through 4M-21's encoder and extract layer-wise activations. For each layer and each pair of modalities, we compute three complementary similarity metrics:

- **CKA**: global relational structure (scale-invariant, primary metric)
- **PWCCA**: directional alignment of principal axes
- **k-NN overlap**: local topological similarity

To distinguish encoder-driven convergence from architectural biases or dataset regularities, we test against an empirical null distribution constructed from mismatched-scene pairs. Only convergence significantly above this floor is attributed to genuine unification.

## Repository Structure

```
.
├── src/
│   ├── data/           # Hypersim & DIODE dataloaders
│   ├── models/         # 4M-21 wrapper + activation hooks
│   ├── metrics/        # CKA, PWCCA, k-NN overlap
│   └── analysis/       # Null distribution & significance testing
├── notebooks/          # Exploratory analysis & figures
├── configs/            # Experiment configs
├── scripts/            # SLURM job scripts (SCITAS)
├── results/            # Saved activations, metric values, figures
└── docs/               # Proposal, progress report, slides
```

## Setup

```bash
# Clone
git clone https://github.com/<user>/one-trunk-or-many.git
cd one-trunk-or-many

# Environment
conda create -n trunk python=3.11
conda activate trunk
pip install -r requirements.txt

# Setup base resources (folders + 4M model)
# Linux/SCITAS:
bash scripts/setup_base.sh EPFL-VILAB/4M-7_B_CC12M
# Windows PowerShell:
powershell -ExecutionPolicy Bypass -File scripts/setup_base.ps1 -ModelRepo EPFL-VILAB/4M-7_B_CC12M
```

Datasets (Hypersim, DIODE) must be downloaded separately into the configured
resource root. Full instructions are in [`docs/data.md`](docs/data.md).

## Reproducing Results

```bash
# 1) Run full CKA benchmark pipeline (default config)
python -m src.run_benchmark

# 2) Example override: Hypersim rgb-depth POC, 20 scenes, W&B on
python -m src.run_benchmark \
  data.name=hypersim \
  data.n_scenes=20 \
  metrics.pairs='[[rgb,depth]]' \
  tracking=wandb_on

# 3) Stage entrypoints (separated)
python -m src.run_extraction
python -m src.run_metrics runtime.metrics_input_run_id=<run_id>
```

Full SCITAS job scripts are in [`scripts/`](scripts/).

Detailed pipeline and config reference: [`docs/pipeline.md`](docs/pipeline.md).

## Status

Work in progress. Milestones tracked in [Issues](../../issues).

- [x] Project proposal submitted
- [ ] RQ1 pipeline (extraction + CKA + null testing)
- [ ] Multi-metric analysis (PWCCA, k-NN)
- [ ] Midterm presentation
- [ ] RQ2 exploratory protocols
- [ ] Final webpage

## Authors

- **Arthur Margeat** (330258)
- **Adrien Clement** (345535)
- **Albert Fares** (341018)
- **Martina Gatti** (341013)

EPFL, CS-503 Visual Intelligence, Spring 2026.

## References

Key references. Full list in the [proposal](docs/proposal.pdf).

- Mizrahi et al. *4M: Massively Multimodal Masked Modeling.* NeurIPS 2023.
- Bachmann et al. *4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities.* NeurIPS 2024.
- Huh et al. *The Platonic Representation Hypothesis.* ICML 2024.
- Kornblith et al. *Similarity of Neural Network Representations Revisited.* ICML 2019.
- Morcos et al. *Insights on Representational Similarity in Neural Networks with Canonical Correlation.* NeurIPS 2018.

## License

MIT License. See [`LICENSE`](LICENSE).
