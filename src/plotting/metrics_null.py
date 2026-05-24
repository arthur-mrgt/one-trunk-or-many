"""Generate per-metric layer-trend plots overlaying observed values on null clouds.

For each enabled metric (cka, pwcca, knn_overlap, ...) this writes a figure with
one subplot per modality pair, showing on the same axes:

  * the null distribution at each layer (boxplot + jittered strip of all draws),
  * the observed (matched-pair) value as a red line with markers,
  * a stars annotation indicating significance vs the BH-FDR threshold,
  * the null mean as a dashed reference line.

A companion z-score summary figure plots ``z_score`` per (pair, layer) so the
relative ranking of layers is still visible when raw p-values are clamped.

Usage
-----
Auto-detect the latest run::

    python -m src.plotting.metrics_null

Explicit run id::

    python -m src.plotting.metrics_null --run-id rq1_full_pipeline-20260510-160357
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _layer_order(layer: str) -> int:
    """Return the trailing integer of a layer name (``layer_03 -> 3``)."""
    try:
        return int(str(layer).split("_")[-1])
    except Exception:
        return 10**9


def _resolve_run_id(repo_root: Path, run_id: str | None) -> str:
    """Return the requested run id or fall back to the most recent run on disk."""
    if run_id is not None:
        return run_id
    runs_root = repo_root / "results" / "runs"
    candidates = sorted(
        p for p in runs_root.glob("*") if p.is_dir() and (p / "metrics" / "metrics.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No run with metrics.csv found under {runs_root}")
    return candidates[-1].name


def _load(repo_root: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = repo_root / "results" / "runs" / run_id / "metrics" / "metrics.csv"
    null_path = (
        repo_root / "results" / "runs" / "null_distributions" / run_id / "null_distribution.csv"
    )
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not null_path.exists():
        raise FileNotFoundError(null_path)
    metrics = pd.read_csv(metrics_path)
    null = pd.read_csv(null_path)
    metrics["layer_idx"] = metrics["layer"].map(_layer_order)
    null["layer_idx"] = null["layer"].map(_layer_order)
    return metrics, null


def _significance_label(p_adj: float | None, alpha: float = 0.05) -> str:
    """Return a star annotation for an adjusted p-value."""
    if p_adj is None or pd.isna(p_adj):
        return ""
    if p_adj <= 0.001:
        return "***"
    if p_adj <= 0.01:
        return "**"
    if p_adj <= alpha:
        return "*"
    return ""


def plot_metric_with_null(
    metric: str,
    metrics: pd.DataFrame,
    null: pd.DataFrame,
    out_path: Path,
    alpha: float = 0.05,
) -> None:
    """Write a single PNG with one subplot per pair for the given metric."""
    sub_metric = metrics[metrics["metric"] == metric].copy()
    sub_null = null[null["metric"] == metric].copy()
    pairs = sorted(sub_metric["pair"].unique())
    if not pairs:
        return

    fig, axes = plt.subplots(
        1, len(pairs), figsize=(5.5 * len(pairs), 5.0), sharey=True, squeeze=False
    )
    axes = axes[0]
    rng = np.random.default_rng(0)

    for ax, pair in zip(axes, pairs):
        mp = sub_metric[sub_metric["pair"] == pair].sort_values("layer_idx")
        layers = mp["layer_idx"].tolist()
        observed = mp["value"].to_numpy()
        null_means = mp["null_mean"].to_numpy()
        p_adj = mp["p_value_adjusted"].to_numpy() if "p_value_adjusted" in mp.columns else None

        null_by_layer = (
            sub_null[sub_null["pair"] == pair]
            .groupby("layer_idx")["null_value"]
            .apply(list)
            .to_dict()
        )

        for li in layers:
            vals = np.asarray(null_by_layer.get(li, []), dtype=float)
            if vals.size == 0:
                continue
            jitter = (rng.random(vals.size) - 0.5) * 0.35
            ax.scatter(np.full(vals.size, li) + jitter, vals, s=6, alpha=0.18, color="#7f7f7f", zorder=1)

        box_data = [np.asarray(null_by_layer.get(li, []), dtype=float) for li in layers]
        if any(arr.size > 0 for arr in box_data):
            ax.boxplot(
                box_data,
                positions=layers,
                widths=0.55,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor="#cccccc", edgecolor="#444444", alpha=0.55),
                medianprops=dict(color="#222222"),
                whiskerprops=dict(color="#444444"),
                capprops=dict(color="#444444"),
                zorder=2,
            )

        ax.plot(layers, null_means, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.8, zorder=3, label="null mean")
        ax.plot(layers, observed, color="#d62728", marker="o", linewidth=2.0, zorder=4, label="observed (matched)")

        if p_adj is not None:
            y_offset = 0.02 * (np.nanmax(observed) - np.nanmin(observed) + 1e-12)
            for li, obs_val, padj in zip(layers, observed, p_adj):
                star = _significance_label(padj, alpha=alpha)
                if star:
                    ax.text(li, obs_val + y_offset, star, ha="center", va="bottom", color="#d62728", fontsize=10)

        ax.set_title(pair)
        ax.set_xlabel("layer index")
        ax.set_xticks(layers)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.97))
    fig.suptitle(f"{metric}: observed vs null distribution across layers", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_zscores(metrics: pd.DataFrame, out_path: Path) -> None:
    """Companion figure: z-score per (pair, layer) for each metric."""
    metric_names = sorted(metrics["metric"].unique())
    pairs = sorted(metrics["pair"].unique())
    if not metric_names or not pairs:
        return

    fig, axes = plt.subplots(
        len(metric_names), 1, figsize=(8, 3.2 * len(metric_names)), sharex=True, squeeze=False
    )
    axes = [ax[0] for ax in axes]

    for ax, m in zip(axes, metric_names):
        sub = metrics[metrics["metric"] == m]
        for pair in pairs:
            mp = sub[sub["pair"] == pair].sort_values("layer_idx")
            ax.plot(mp["layer_idx"], mp["z_score"], marker="o", linewidth=1.8, label=pair)
        ax.set_title(m)
        ax.set_ylabel("z-score (observed vs null)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="#444444", linewidth=0.8)

    axes[-1].set_xlabel("layer index")
    axes[-1].set_xticks(sorted(metrics["layer_idx"].unique()))
    axes[0].legend(loc="best", title="pair")
    fig.suptitle("Per-layer z-score (higher = more aligned vs null)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None, help="Run id under results/runs/. Defaults to latest.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Project root (defaults to CWD).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for figures.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for stars.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_id = _resolve_run_id(repo_root, args.run_id)
    out_dir = args.out_dir or (repo_root / "results" / "runs" / run_id / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, null = _load(repo_root, run_id)
    print(f"[plot] run_id={run_id}")
    print(f"[plot] metrics rows={len(metrics)}, null rows={len(null)}")
    print(f"[plot] writing to {out_dir}")

    for metric in sorted(metrics["metric"].unique()):
        out_path = out_dir / f"layers_{metric}.png"
        plot_metric_with_null(metric, metrics, null, out_path, alpha=args.alpha)
        print(f"[plot]  - {out_path.name}")

    zscore_path = out_dir / "zscores_summary.png"
    plot_zscores(metrics, zscore_path)
    print(f"[plot]  - {zscore_path.name}")


if __name__ == "__main__":
    main()
