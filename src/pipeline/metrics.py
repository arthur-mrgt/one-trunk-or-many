"""Metrics stage implementation over saved activations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.metrics.registry import build_metric
from src.utils.distributed import get_dist_context, split_by_rank


def _stack_vectors(paths: list[str]) -> np.ndarray:
    """Load and stack vectors from NPY files."""
    return np.stack([np.load(Path(p)) for p in paths], axis=0)


def _metric_metadata(metric_name: str, metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return metric-specific metadata fields for output rows."""
    reduction_used = False
    k = float("nan")

    metric_cfg = metrics_cfg.get(metric_name, {})
    pca_cfg = metric_cfg.get("pca", {})
    if isinstance(pca_cfg, dict):
        reduction_used = bool(pca_cfg.get("enabled", False))

    if metric_name == "knn_overlap":
        k = int(metric_cfg.get("k", 10))

    return {
        "reduction_used": reduction_used,
        "k": k,
    }


def run_metrics(
    activation_index: pd.DataFrame,
    metric_names: list[str],
    metrics_cfg: dict[str, Any],
    pair_modalities: tuple[str, str],
    show_progress: bool = True,
    log_fn: Any | None = None,
) -> pd.DataFrame:
    """Compute all enabled metrics per layer for a modality pair.

    Parameters
    ----------
    activation_index:
        Index DataFrame produced by the extraction stage.
    metric_names:
        List of metric names to compute, e.g. ``["cka", "pwcca", "knn_overlap"]``.
    metrics_cfg:
        Full metrics config dict (contains sub-configs for each metric).
    pair_modalities:
        ``(left_modality, right_modality)`` tuple, e.g. ``("rgb", "depth")``.
    show_progress:
        Whether to show a tqdm progress bar.
    """
    if activation_index.empty:
        return pd.DataFrame(
            columns=[
                "run_id", "pair", "layer", "metric",
                "value", "n_samples", "left_modality", "right_modality",
                "reduction_used", "k",
            ]
        )

    cka_cfg: dict = metrics_cfg.get("cka", {})
    left_mod, right_mod = pair_modalities
    layers = sorted(activation_index["layer"].unique().tolist())
    dist_ctx = get_dist_context()
    local_layers = split_by_rank(layers, ctx=dist_ctx)
    pair_name = f"{left_mod}-{right_mod}"

    if log_fn is not None:
        log_fn(
            f"Metrics start for pair={pair_name}: "
            f"metrics={metric_names} layers={len(layers)} local_layers={len(local_layers)}"
        )

    rows: list[dict[str, Any]] = []

    for metric_name in metric_names:
        metric_start = time.perf_counter()
        metric_meta = _metric_metadata(metric_name, metrics_cfg)
        metric_fn = build_metric(
            metric_name=metric_name,
            cka_cfg=cka_cfg,
            metrics_cfg=metrics_cfg,
        )
        completed_layers = 0

        layer_iter = tqdm(
            local_layers,
            desc=f"Metric[{metric_name}][{left_mod}-{right_mod}]",
            unit="layer",
            disable=(not show_progress) or (dist_ctx.enabled and not dist_ctx.is_main),
            dynamic_ncols=True,
        )
        for layer in layer_iter:
            left_df = activation_index[
                (activation_index["layer"] == layer)
                & (activation_index["modality"] == left_mod)
            ]
            right_df = activation_index[
                (activation_index["layer"] == layer)
                & (activation_index["modality"] == right_mod)
            ]
            merged = left_df.merge(
                right_df,
                on=["run_id", "pair", "scene_id", "sample_key", "layer"],
                suffixes=("_left", "_right"),
            )
            if merged.empty:
                continue

            x = _stack_vectors(merged["activation_path_left"].tolist())
            y = _stack_vectors(merged["activation_path_right"].tolist())

            try:
                value = metric_fn(x, y)
            except Exception as exc:
                value = float("nan")
                if show_progress:
                    tqdm.write(
                        f"[WARN] {metric_name} failed for layer {layer}: {exc}"
                    )

            rows.append(
                {
                    "run_id": merged.iloc[0]["run_id"],
                    "pair": merged.iloc[0]["pair"],
                    "layer": layer,
                    "metric": metric_name,
                    "value": value,
                    "n_samples": int(len(merged)),
                    "left_modality": left_mod,
                    "right_modality": right_mod,
                    "reduction_used": metric_meta["reduction_used"],
                    "k": metric_meta["k"],
                }
            )
            completed_layers += 1
            if show_progress:
                layer_iter.set_postfix_str(f"n={len(merged)}")

        if log_fn is not None:
            elapsed = time.perf_counter() - metric_start
            log_fn(
                f"Metrics completed for pair={pair_name} metric={metric_name}: "
                f"rows={completed_layers} elapsed_s={elapsed:.1f}"
            )

    return pd.DataFrame.from_records(rows)
