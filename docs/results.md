# Run Outputs and Diagnostics

Each benchmark run is identified by a `RUN_ID` of the form
`<stage>-<YYYYMMDD>-<HHMMSS>` (e.g. `rq1_final_hypersim-20260524-110045`).
All outputs of a run live under `results/runs/<RUN_ID>/`.

## On-disk layout

```text
results/
├── runs/
│   ├── <RUN_ID>/
│   │   ├── activations/                        # one .npy per (sample, modality, layer)
│   │   ├── artifacts/
│   │   │   └── activation_index_<pair>.csv     # row-level index of saved activations
│   │   ├── metrics/
│   │   │   ├── metrics.csv                     # observed values + significance columns
│   │   │   ├── metrics.parquet                 # same, when enabled
│   │   │   └── null_stop_evaluation.csv        # adaptive-stop decision per hypothesis
│   │   ├── diagnostics/                        # CSVs written by the diagnostics notebook
│   │   ├── figures/                            # PNGs written by src.plotting.metrics_null
│   │   ├── resolved_config.json                # fully resolved Hydra config
│   │   └── run_summary.json                    # high-level run summary
│   ├── null_distributions/<RUN_ID>/
│   │   ├── null_distribution.csv               # one row per draw, layer, pair, metric
│   │   ├── null_distribution.parquet           # same, when enabled
│   │   └── null_distribution_state.json        # checkpoint + adaptive-stop state
│   └── hydra/                                  # Hydra runtime logs
├── wandb/                                      # W&B local cache (gitignored)
└── slurm/                                      # SLURM stdout/stderr (gitignored)
```

`null_distributions/<RUN_ID>/` is kept as a sibling of `<RUN_ID>/` for
historical reasons; null artifacts are reusable across reruns of the same
extraction.

## `metrics.csv` columns

| Column | Description |
|---|---|
| `metric` | `cka`, `pwcca`, or `knn_overlap`. |
| `pair` | Modality pair, e.g. `rgb-depth`. |
| `layer` | Layer name (`layer_00` … `layer_11` for 4M-7B). |
| `value` | Observed metric value on matched pairs. |
| `n_samples` | Number of aligned samples used. |
| `p_value` | Two-sided right-tailed empirical p-value vs the null. |
| `p_value_adjusted` | BH-FDR adjusted p-value. |
| `null_mean`, `null_std` | Summary stats of the null distribution at that hypothesis. |
| `z_score` | `(value − null_mean) / null_std`. |
| `n_null_draws` | Number of draws used to compute the null. |

`null_distribution.csv` has the same primary keys plus a `null_value` column
(one row per draw).

## Weights & Biases

When `tracking=wandb_on`, the pipeline logs:

- the resolved config as run config;
- the full `metrics_table`;
- per-metric per-pair line and scatter plots vs layer index;
- significance plots when null is available (`pvalue_vs_layer`,
  `pvalue_adjusted_vs_layer`, `delta_vs_null_mean`);
- stage durations and a final summary scalar block.

The local W&B cache lives under `results/wandb/`. Disable with
`tracking=wandb_off` or set `WANDB_MODE=offline`.

## Diagnostics notebook

[`notebooks/pvalue_diagnostics.ipynb`](../notebooks/pvalue_diagnostics.ipynb)
renders the figures used in the progress report:

1. Set `RUN_ID` in the first cell to the value printed by the launcher.
2. Run all cells. The notebook reads:
   - `results/runs/<RUN_ID>/metrics/metrics.csv`
   - `results/runs/null_distributions/<RUN_ID>/null_distribution.csv`
3. It writes report-ready CSVs to
   `results/runs/<RUN_ID>/diagnostics/`:
   - per-metric tail enrichment by scene type;
   - `report_ready_significance_table.csv`.

The notebook covers:

- observed value vs null cloud per layer (boxplot + strip);
- log-scale null mean vs observed;
- z-score and fold-change summaries;
- volcano-style significance plot per metric;
- null-leakage sanity checks (cross-scene, cross-type, draw consistency).

## Static plot script

For headless rendering without the notebook:

```bash
python -m src.plotting.metrics_null --run-id <RUN_ID>
# or auto-detect the latest run:
python -m src.plotting.metrics_null
```

Writes `figures/layers_<metric>.png` and `figures/zscores_summary.png` under
the run directory.
