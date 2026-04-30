"""Smoke test for the null-distribution significance pipeline.

What this test covers
---------------------
1. ``compute_null_distribution`` runs on synthetic activation data and
   produces the expected CSV schema.
2. ``_enrich_with_significance`` correctly joins null values onto a
   metrics table and writes valid p-values in [0, 1].
3. The significance helpers (``compute_p_value``, ``bh_fdr_correction``)
   behave correctly on simple synthetic data.

Run with:
    pytest tests/test_null_smoke.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.significance import bh_fdr_correction, compute_p_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_activation_index(
    tmp_path,
    n_scenes: int = 3,
    n_frames_per_scene: int = 4,
    n_layers: int = 3,
    dim: int = 16,
    pair: str = "rgb-depth",
) -> pd.DataFrame:
    """Write fake .npy activation files and return a matching activation index."""
    left_mod, right_mod = pair.split("-", 1)
    rows: list[dict] = []
    rng = np.random.default_rng(0)

    for scene_idx in range(n_scenes):
        scene_id = f"ai_{scene_idx:03d}_001"
        for frame_idx in range(n_frames_per_scene):
            for layer_idx in range(n_layers):
                layer_name = f"layer_{layer_idx:02d}"
                for modality in (left_mod, right_mod):
                    vec = rng.standard_normal(dim).astype(np.float32)
                    act_path = (
                        tmp_path
                        / f"{scene_id}_{frame_idx:04d}_{layer_name}_{modality}.npy"
                    )
                    np.save(act_path, vec)
                    rows.append(
                        {
                            "run_id": "smoke_run",
                            "pair": pair,
                            "scene_id": scene_id,
                            "frame_id": frame_idx,
                            "layer": layer_name,
                            "modality": modality,
                            "activation_path": str(act_path),
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputePValue:
    def test_observed_above_all_null(self):
        null = np.zeros(99)
        p = compute_p_value(observed=1.0, null_values=null, side="greater")
        # (1 + 0) / (99 + 1) == 0.01
        assert p == pytest.approx(0.01)

    def test_observed_below_all_null(self):
        null = np.ones(99)
        p = compute_p_value(observed=0.0, null_values=null, side="greater")
        # (1 + 99) / (99 + 1) == 1.0
        assert p == pytest.approx(1.0)

    def test_two_sided(self):
        null = np.linspace(-1, 1, 99)
        p = compute_p_value(observed=0.0, null_values=null, side="two")
        assert 0.0 < p <= 1.0

    def test_empty_null(self):
        p = compute_p_value(observed=0.5, null_values=np.array([]))
        assert np.isnan(p)

    def test_invalid_side(self):
        with pytest.raises(ValueError, match="side"):
            compute_p_value(1.0, np.ones(10), side="both")


class TestBhFdrCorrection:
    def test_all_significant(self):
        p_values = np.array([0.001, 0.002, 0.003])
        rejected, p_adj = bh_fdr_correction(p_values, alpha=0.05)
        assert all(rejected)
        assert len(p_adj) == len(p_values)

    def test_none_significant(self):
        p_values = np.array([0.9, 0.95, 1.0])
        rejected, _ = bh_fdr_correction(p_values, alpha=0.05)
        assert not any(rejected)

    def test_empty(self):
        rejected, p_adj = bh_fdr_correction(np.array([]))
        assert len(rejected) == 0 and len(p_adj) == 0

    def test_adjusted_geq_original(self):
        p_values = np.array([0.01, 0.04, 0.03, 0.02])
        _, p_adj = bh_fdr_correction(p_values)
        assert np.all(p_adj >= p_values - 1e-9)


class TestNullDistributionSmoke:
    def test_compute_null_distribution(self, tmp_path):
        """End-to-end: build synthetic data, run null engine, check output."""
        from src.analysis.null_distribution import compute_null_distribution

        activation_index = _make_activation_index(tmp_path)
        out_dir = tmp_path / "null_out"
        out_dir.mkdir()

        null_cfg = {
            "n_draws": 10,
            "seed": 0,
            "replace": True,
            "min_scenes": 2,
            "sampling": "cross_scene_random",
        }
        metrics_cfg = {
            "enabled": ["cka"],
            "cka": {"kernel": "linear"},
            "output": {"save_parquet": False},
        }

        null_df = compute_null_distribution(
            activation_index=activation_index,
            null_cfg=null_cfg,
            metrics_cfg=metrics_cfg,
            run_id="smoke_run",
            out_dir=out_dir,
        )

        # Schema check
        required_cols = {"run_id", "pair", "layer", "metric", "draw_id",
                         "null_value", "n_samples", "sampling", "seed"}
        assert required_cols.issubset(null_df.columns), (
            f"Missing columns: {required_cols - set(null_df.columns)}"
        )

        # Should have at least one row per layer
        assert len(null_df) > 0, "null_df is empty"

        # null_value must be finite floats in [0, 1] for CKA
        assert null_df["null_value"].between(0.0, 1.0).all(), (
            "CKA null values outside [0, 1]"
        )

        # CSV written to disk
        assert (out_dir / "null_distribution.csv").exists()

    def test_enrich_with_significance(self, tmp_path):
        """_enrich_with_significance adds p-value columns with valid values."""
        from src.pipeline.benchmark import _enrich_with_significance

        # Build a minimal metrics table (3 layers, 1 pair)
        layers = ["layer_00", "layer_01", "layer_02"]
        metric_rows = [
            {"pair": "rgb-depth", "layer": l, "metric": "cka", "value": 0.6}
            for l in layers
        ]
        metric_table = pd.DataFrame(metric_rows)

        # Build matching null distribution (10 draws per layer)
        rng = np.random.default_rng(1)
        null_rows = []
        for l in layers:
            for i in range(10):
                null_rows.append({
                    "pair": "rgb-depth",
                    "layer": l,
                    "metric": "cka",
                    "draw_id": i,
                    "null_value": float(rng.uniform(0.0, 0.5)),
                })
        null_df = pd.DataFrame(null_rows)

        enriched = _enrich_with_significance(metric_table, null_df)

        sig_cols = ["p_value", "null_mean", "null_std",
                    "delta_vs_null_mean", "z_score"]
        for col in sig_cols:
            assert col in enriched.columns, f"Missing column: {col}"

        assert enriched["p_value"].between(0.0, 1.0).all(), (
            "p_values outside [0, 1]"
        )
        assert (enriched["delta_vs_null_mean"] > 0).all(), (
            "Observed CKA (0.6) should be above null mean (<0.5)"
        )
