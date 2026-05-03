"""Tests for metric-specific metadata columns in metrics output."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.metrics import run_metrics


def _write_vec(path, vec) -> str:
    np.save(path, vec.astype(np.float32))
    return str(path)


def test_run_metrics_writes_reduction_used_and_k(tmp_path) -> None:
    rows = []
    run_id = "run_meta"
    pair = "rgb-depth"
    layer = "layer_00"
    for i in range(3):
        sample_key = f"s_{i:03d}"
        rgb_path = tmp_path / f"{sample_key}_rgb.npy"
        depth_path = tmp_path / f"{sample_key}_depth.npy"
        rows.append(
            {
                "run_id": run_id,
                "pair": pair,
                "scene_id": "ai_001_001",
                "sample_key": sample_key,
                "layer": layer,
                "modality": "rgb",
                "activation_path": _write_vec(rgb_path, np.random.randn(8)),
            }
        )
        rows.append(
            {
                "run_id": run_id,
                "pair": pair,
                "scene_id": "ai_001_001",
                "sample_key": sample_key,
                "layer": layer,
                "modality": "depth",
                "activation_path": _write_vec(depth_path, np.random.randn(8)),
            }
        )

    activation_index = pd.DataFrame(rows)
    metrics_cfg = {
        "cka": {"center_gram": True},
        "pwcca": {"pca": {"enabled": True, "n_components": 4, "seed": 0}},
        "knn_overlap": {"k": 2, "pca": {"enabled": True, "n_components": 4, "seed": 0}},
    }

    out = run_metrics(
        activation_index=activation_index,
        metric_names=["cka", "pwcca", "knn_overlap"],
        metrics_cfg=metrics_cfg,
        pair_modalities=("rgb", "depth"),
        show_progress=False,
    )

    assert "reduction_used" in out.columns
    assert "k" in out.columns

    cka_row = out[out["metric"] == "cka"].iloc[0]
    pwcca_row = out[out["metric"] == "pwcca"].iloc[0]
    knn_row = out[out["metric"] == "knn_overlap"].iloc[0]

    assert bool(cka_row["reduction_used"]) is False
    assert np.isnan(cka_row["k"])
    assert bool(pwcca_row["reduction_used"]) is True
    assert np.isnan(pwcca_row["k"])
    assert bool(knn_row["reduction_used"]) is True
    assert int(knn_row["k"]) == 2

