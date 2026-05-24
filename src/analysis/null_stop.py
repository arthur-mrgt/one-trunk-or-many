"""Stop-rule and observed-table helpers for adaptive null runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.significance import bh_fdr_correction, compute_p_value


def normalize_observed_metrics(
    observed_metrics: pd.DataFrame,
    metrics_cfg: dict[str, Any],
    null_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalize observed metric rows used by adaptive stop evaluation."""
    if observed_metrics.empty:
        return pd.DataFrame()
    null_cfg = null_cfg or {}
    null_metrics = null_cfg.get("metrics_enabled")
    enabled_source = null_metrics if null_metrics else metrics_cfg.get("enabled", ["cka"])
    enabled = set(str(m) for m in enabled_source)
    required_cols = {"pair", "layer", "metric", "value", "n_samples"}
    if not required_cols.issubset(observed_metrics.columns):
        return pd.DataFrame()

    df = observed_metrics.copy()
    df = df[df["metric"].isin(enabled)].dropna(subset=["value", "n_samples"])
    if df.empty:
        return pd.DataFrame()
    df["layer"] = df["layer"].astype(str)
    df["pair"] = df["pair"].astype(str)
    df["metric"] = df["metric"].astype(str)
    df["n_samples"] = df["n_samples"].astype(int)
    return df[["pair", "layer", "metric", "value", "n_samples"]].reset_index(drop=True)


def build_observed_stub(
    activation_index: pd.DataFrame,
    metrics_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Build minimal observed rows from activation index for legacy API."""
    if activation_index.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    metric_names = list(metrics_cfg.get("enabled", ["cka"]))
    for pair in sorted(activation_index["pair"].unique().tolist()):
        left_mod, right_mod = tuple(pair.split("-", 1))
        pair_df = activation_index[activation_index["pair"] == pair]
        for layer in sorted(pair_df["layer"].astype(str).unique().tolist()):
            left_df = pair_df[
                (pair_df["layer"].astype(str) == layer)
                & (pair_df["modality"] == left_mod)
            ]
            right_df = pair_df[
                (pair_df["layer"].astype(str) == layer)
                & (pair_df["modality"] == right_mod)
            ]
            merged = left_df.merge(
                right_df,
                on=["run_id", "pair", "scene_id", "sample_key", "layer"],
                suffixes=("_left", "_right"),
            )
            if merged.empty:
                continue
            for metric_name in metric_names:
                rows.append(
                    {
                        "pair": pair,
                        "layer": layer,
                        "metric": metric_name,
                        "value": 0.0,
                        "n_samples": int(len(merged)),
                    }
                )
    return pd.DataFrame.from_records(rows)


def evaluate_adaptive_stop(
    observed_metrics: pd.DataFrame,
    null_df: pd.DataFrame,
    alpha: float,
    correction: str,
    require_all_hypotheses: bool,
    min_total_draws: int,
) -> tuple[bool, pd.DataFrame]:
    """Return stop decision and per-hypothesis p-value table."""
    if observed_metrics.empty:
        return False, pd.DataFrame()

    eval_rows: list[dict[str, Any]] = []
    for _, row in observed_metrics.iterrows():
        key_mask = (
            (null_df["pair"] == row["pair"])
            & (null_df["layer"].astype(str) == str(row["layer"]))
            & (null_df["metric"] == row["metric"])
        ) if not null_df.empty else np.array([], dtype=bool)
        null_values = (
            null_df.loc[key_mask, "null_value"].to_numpy(dtype=float)
            if not null_df.empty
            else np.array([], dtype=float)
        )
        eval_rows.append(
            {
                "pair": row["pair"],
                "layer": str(row["layer"]),
                "metric": row["metric"],
                "p_value_raw": compute_p_value(float(row["value"]), null_values, side="greater"),
                "n_null_draws": int(len(null_values)),
            }
        )

    eval_df = pd.DataFrame.from_records(eval_rows)
    if eval_df.empty:
        return False, eval_df

    valid_mask = eval_df["p_value_raw"].notna().to_numpy()
    p_adj = np.full(len(eval_df), np.nan, dtype=float)
    if np.any(valid_mask):
        pvals = eval_df.loc[valid_mask, "p_value_raw"].to_numpy(dtype=float)
        if correction == "none":
            p_adj_vals = pvals
        elif correction == "bh_fdr":
            _, p_adj_vals = bh_fdr_correction(pvals, alpha=alpha)
        elif correction == "bonferroni":
            p_adj_vals = np.minimum(1.0, pvals * len(pvals))
        else:
            raise ValueError(
                f"Unknown adaptive correction '{correction}'. "
                "Supported: none, bh_fdr, bonferroni."
            )
        p_adj[valid_mask] = p_adj_vals
    eval_df["p_value_adjusted"] = p_adj

    enough_draws = eval_df["n_null_draws"] >= int(min_total_draws)
    eval_df["is_significant"] = enough_draws & (eval_df["p_value_adjusted"] <= float(alpha))

    should_stop = bool(eval_df["is_significant"].all()) if require_all_hypotheses else bool(eval_df["is_significant"].any())
    return should_stop, eval_df
