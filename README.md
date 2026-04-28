# One Trunk or Many?

**Measuring representational convergence in any-to-any vision models.**

Any-to-any models like [4M](https://github.com/apple/ml-4m) share a single Transformer encoder across modalities, but each modality enters through its own learned embedding. This raises a question: does the encoder actually unify modalities into a shared representation, or does it route them through effectively disjoint sub-networks?

This project develops a multi-metric protocol to probe representational convergence inside 4M-21's shared encoder, with statistical significance testing against a mismatched-scene null distribution.

> 📄 [Project Proposal (PDF)](docs/proposal.pdf) · 🎓 CS-503 Visual Intelligence · EPFL Spring 2026

---

## Research Questions

**RQ1.** Does the encoder exhibit measurable representational convergence when processing different modalities of the same scene, and is this signal consistent across geometric, directional, and topological similarity metrics?

**RQ2.** Can complementary protocols (cross-modal transfer, residual analysis) address the functional and ceiling blindspots left open by RQ1?

## Approach

We pass each modality (RGB, depth, normals, segmentation) of a given scene separately through 4M-21's encoder and extract layer-wise activations. For each layer and each pair of modalities, we compute three complementary similarity metrics:

- **CKA** — global relational structure (scale-invariant, primary metric)
- **PWCCA** — directional alignment of principal axes
- **k-NN overlap** — local topological similarity

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

# Download 4M-21 checkpoint (from HuggingFace)
python scripts/download_checkpoint.py --model 4m-21-b
```

Datasets (Hypersim, DIODE) must be downloaded separately — see [`docs/data.md`](docs/data.md).

## Reproducing Results

```bash
# 1. Extract activations
python -m src.run_extraction --dataset hypersim --n_scenes 500

# 2. Compute metrics + null distribution
python -m src.run_metrics --pairs rgb-depth rgb-normals depth-normals

# 3. Generate figures
python -m src.make_figures
```

Full SCITAS job scripts are in [`scripts/`](scripts/).

## Status

🚧 **Work in progress** — milestones tracked in [Issues](../../issues).

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

EPFL · CS-503 Visual Intelligence · Spring 2026

## References

Key references — full list in the [proposal](docs/proposal.pdf).

- Mizrahi et al. *4M: Massively Multimodal Masked Modeling.* NeurIPS 2023.
- Bachmann et al. *4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities.* NeurIPS 2024.
- Huh et al. *The Platonic Representation Hypothesis.* ICML 2024.
- Kornblith et al. *Similarity of Neural Network Representations Revisited.* ICML 2019.
- Morcos et al. *Insights on Representational Similarity in Neural Networks with Canonical Correlation.* NeurIPS 2018.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
