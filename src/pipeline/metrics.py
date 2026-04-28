from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.registry import build_metric


def _stack_vectors(paths: list[str]) -> np.ndarray:
    return np.stack([np.load(Path(p)) for p in paths], axis=0)


def run_metrics(
    activation_index: pd.DataFrame,
    metric_name: str,
    cka_cfg: dict[str, Any],
    pair_modalities: tuple[str, str],
) -> pd.DataFrame:
    if activation_index.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "pair",
                "layer",
                "metric",
                "value",
                "n_samples",
                "left_modality",
                "right_modality",
            ]
        )

    metric_fn = build_metric(metric_name=metric_name, cka_cfg=cka_cfg)
    left_mod, right_mod = pair_modalities

    rows: list[dict[str, Any]] = []
    layers = sorted(activation_index["layer"].unique().tolist())

    for layer in layers:
        left_df = activation_index[
            (activation_index["layer"] == layer) & (activation_index["modality"] == left_mod)
        ]
        right_df = activation_index[
            (activation_index["layer"] == layer) & (activation_index["modality"] == right_mod)
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
        value = metric_fn(x, y)
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
            }
        )

    return pd.DataFrame.from_records(rows)
