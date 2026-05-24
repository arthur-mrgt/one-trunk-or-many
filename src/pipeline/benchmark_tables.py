"""Table assembly helpers for benchmark extraction and metrics stages."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from src.pipeline.benchmark_config import list_activation_index_files
from src.pipeline.metrics import run_metrics
from src.utils.config import RunContext


def layer_sort_key(layer_name: str) -> tuple[int, str]:
    """Build stable sort key from layer string identifier."""
    try:
        return (int(str(layer_name).split("_")[-1]), str(layer_name))
    except Exception:
        return (10**9, str(layer_name))


def sorted_layer_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted by layer order with helper index columns."""
    out = df.copy()
    out["layer"] = out["layer"].astype(str)
    out["layer_order"] = out["layer"].map(lambda x: layer_sort_key(x)[0])
    out = out.sort_values(["layer_order", "layer"]).reset_index(drop=True)
    out["layer_idx"] = out["layer_order"].astype(int)
    return out


def load_activation_indices(run_ctx: RunContext) -> pd.DataFrame:
    """Load and concatenate all activation index CSV files for one run."""
    files = list_activation_index_files(run_ctx)
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def compute_metric_table_from_indices(
    cfg_dict: dict,
    run_ctx: RunContext,
    show_progress: bool,
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Compute metric table by reading saved activation index artifacts."""
    frames: list[pd.DataFrame] = []
    for idx_file in list_activation_index_files(run_ctx):
        activation_index = pd.read_csv(idx_file)
        pair_name = idx_file.stem.replace("activation_index_", "")
        if log_fn is not None:
            log_fn(f"Processing metrics for pair={pair_name}")
        try:
            left_mod, right_mod = tuple(pair_name.split("-", 1))
        except ValueError:
            continue
        metric_df = run_metrics(
            activation_index=activation_index,
            metric_names=list(cfg_dict["metrics"]["enabled"]),
            metrics_cfg=cfg_dict["metrics"],
            pair_modalities=(left_mod, right_mod),
            show_progress=show_progress,
        )
        frames.append(metric_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
