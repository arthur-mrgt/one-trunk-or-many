"""Significance enrichment helpers for benchmark metric tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.significance import bh_fdr_correction, compute_p_value


def apply_pvalue_correction(
    metric_table: pd.DataFrame,
    correction: str,
    alpha: float,
) -> pd.DataFrame:
    """Compute adjusted p-values and significance flags."""
    out = metric_table.copy()
    out["p_value_adjusted"] = float("nan")
    out["is_significant"] = False
    valid = out["p_value"].notna()
    if not valid.any():
        return out

    pvals = out.loc[valid, "p_value"].to_numpy(dtype=float)
    if correction == "none":
        p_adj = pvals
    elif correction == "bh_fdr":
        _, p_adj = bh_fdr_correction(pvals, alpha=alpha)
    elif correction == "bonferroni":
        p_adj = np.minimum(1.0, pvals * len(pvals))
    else:
        raise ValueError(f"Unknown correction '{correction}'. Supported: none, bh_fdr, bonferroni.")

    out.loc[valid, "p_value_adjusted"] = p_adj
    out.loc[valid, "is_significant"] = out.loc[valid, "p_value_adjusted"] <= float(alpha)
    return out


def enrich_with_significance(
    metric_table: pd.DataFrame,
    null_df: pd.DataFrame,
    correction: str,
    alpha: float,
) -> pd.DataFrame:
    """Join observed metric rows with null draws and compute significance stats."""
    out = metric_table.copy()
    sig_cols = [
        "p_value",
        "p_value_adjusted",
        "is_significant",
        "n_null_draws",
        "null_mean",
        "null_std",
        "delta_vs_null_mean",
        "z_score",
    ]
    for col in sig_cols:
        out[col] = float("nan") if col != "is_significant" else False
    if null_df.empty:
        return out

    for idx, row in out.iterrows():
        mask = (
            (null_df["pair"] == row["pair"])
            & (null_df["layer"].astype(str) == str(row["layer"]))
            & (null_df["metric"] == row["metric"])
        )
        null_values = null_df.loc[mask, "null_value"].to_numpy(dtype=float)
        if len(null_values) == 0:
            continue
        observed = float(row["value"])
        null_mean = float(np.mean(null_values))
        null_std = float(np.std(null_values))

        out.at[idx, "p_value"] = compute_p_value(observed, null_values, side="greater")
        out.at[idx, "n_null_draws"] = int(len(null_values))
        out.at[idx, "null_mean"] = null_mean
        out.at[idx, "null_std"] = null_std
        out.at[idx, "delta_vs_null_mean"] = observed - null_mean
        out.at[idx, "z_score"] = (observed - null_mean) / null_std if null_std > 0 else float("nan")

    return apply_pvalue_correction(out, correction=correction, alpha=alpha)
