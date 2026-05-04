"""Tracker plotting helpers for benchmark metric and significance outputs."""

from __future__ import annotations

import pandas as pd

from src.pipeline.benchmark_tables import sorted_layer_df


def log_metric_plots(tracker, metric_table: pd.DataFrame) -> None:
    """Log value-vs-layer plots for each `(metric, pair)`."""
    if metric_table.empty:
        return
    for metric_name in sorted(metric_table["metric"].unique().tolist()):
        m_df = metric_table[metric_table["metric"] == metric_name].copy()
        for pair in sorted(m_df["pair"].unique().tolist()):
            pair_df = sorted_layer_df(m_df[m_df["pair"] == pair])
            if pair_df.empty:
                continue
            plot_df = pair_df[["layer_idx", "layer", "value"]].copy()
            tracker.log_line_plot(
                name=f"{metric_name}_vs_layer/{pair}",
                table=plot_df,
                x="layer_idx",
                y="value",
                title=f"{metric_name.upper()} vs Layer ({pair})",
            )
            tracker.log_scatter_plot(
                name=f"{metric_name}_vs_layer_points/{pair}",
                table=plot_df,
                x="layer_idx",
                y="value",
                title=f"{metric_name.upper()} vs Layer Points ({pair})",
            )


def log_significance_plots(tracker, metric_table: pd.DataFrame) -> None:
    """Log p-value and delta-vs-null-mean curves when significance exists."""
    needed = {"p_value", "delta_vs_null_mean", "pair", "layer", "metric"}
    if not needed.issubset(metric_table.columns):
        return
    sig_df = metric_table.dropna(subset=["p_value"]).copy()
    if sig_df.empty:
        return

    for metric_name in sorted(sig_df["metric"].unique().tolist()):
        m_df = sig_df[sig_df["metric"] == metric_name]
        for pair in sorted(m_df["pair"].unique().tolist()):
            pair_df = sorted_layer_df(m_df[m_df["pair"] == pair])
            if pair_df.empty:
                continue
            tracker.log_line_plot(
                name=f"pvalue_vs_layer/{metric_name}/{pair}",
                table=pair_df[["layer_idx", "layer", "p_value"]].copy(),
                x="layer_idx",
                y="p_value",
                title=f"p-value vs Layer - {metric_name.upper()} ({pair})",
            )
            if "p_value_adjusted" in pair_df.columns:
                tracker.log_line_plot(
                    name=f"pvalue_adjusted_vs_layer/{metric_name}/{pair}",
                    table=pair_df[["layer_idx", "layer", "p_value_adjusted"]].copy(),
                    x="layer_idx",
                    y="p_value_adjusted",
                    title=f"Adjusted p-value vs Layer - {metric_name.upper()} ({pair})",
                )
            tracker.log_line_plot(
                name=f"delta_vs_null_mean/{metric_name}/{pair}",
                table=pair_df[["layer_idx", "layer", "delta_vs_null_mean"]].copy(),
                x="layer_idx",
                y="delta_vs_null_mean",
                title=f"{metric_name.upper()} - null_mean vs Layer ({pair})",
            )
