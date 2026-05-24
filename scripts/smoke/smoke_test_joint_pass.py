"""Smoke test for the joint forward pass plumbing.

Verifies, end-to-end, on the deterministic mock backend:
  1. FourMMockEncoder.encode_pair_joint emits both modalities per layer in
     alphabetical order.
  2. run_joint_extraction writes the expected .npy files + index rows.
  3. build_joint_comparison_indices builds three comparison index frames
     with the right pair/modality combinations, and joint-side rows
     reference newly written .npy files while single-side rows reuse the
     original single-mod .npy paths verbatim.
  4. The metric engine's merge predicate (run_id, pair, scene_id,
     sample_key, layer) yields non-empty merges on all three comparisons.

Run with:  python scripts/smoke/smoke_test_joint_pass.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.hypersim import PairSample
from src.models.model_fourm import FourMMockEncoder
from src.pipeline.extraction import (
    joint_slice_modality_name,
    run_extraction,
    run_joint_extraction,
)
from src.pipeline.joint_indices import build_joint_comparison_indices


def make_fake_samples(tmpdir: Path, modalities: tuple[str, ...]) -> list[PairSample]:
    """Two scenes × two frames, one fake file per (sample, modality)."""
    samples: list[PairSample] = []
    for s_idx in range(2):
        scene = f"scene_{s_idx:02d}"
        for f_idx in range(2):
            sample_key = f"frame_{f_idx:02d}"
            paths = {}
            for mod in modalities:
                p = tmpdir / scene / sample_key / f"{mod}.npy"
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, np.zeros(4, dtype=np.float32))  # content irrelevant for mock encoder
                paths[mod] = p
            samples.append(
                PairSample(scene_id=scene, sample_key=sample_key, modality_paths=paths),
            )
    return samples


def main() -> None:
    layers = [f"layer_{i:02d}" for i in range(3)]
    embedding_dim = 8
    encoder = FourMMockEncoder(layers=layers, embedding_dim=embedding_dim)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        modalities = ("rgb", "depth", "normals")
        samples = make_fake_samples(tmp_path / "input", modalities)

        # ── Step 1 — encoder.encode_pair_joint contract
        first = samples[0]
        joint_layers = encoder.encode_pair_joint(
            file_paths={m: first.modality_paths[m] for m in ("rgb", "depth")},
            modalities=("rgb", "depth"),
        )
        assert sorted(joint_layers.keys()) == sorted(layers), (
            f"Expected one entry per layer, got {sorted(joint_layers.keys())}"
        )
        for layer, slice_dict in joint_layers.items():
            assert sorted(slice_dict.keys()) == ["depth", "rgb"], (
                f"Joint dict at {layer} should hold both modalities, got {list(slice_dict)}"
            )
            for mod, vec in slice_dict.items():
                assert vec.shape == (embedding_dim,), (
                    f"Layer {layer} modality {mod}: expected shape ({embedding_dim},), got {vec.shape}"
                )

        # ── Step 2 — full extraction + joint extraction for one pair
        out_dir = tmp_path / "activations"

        single_index = run_extraction(
            model=encoder,
            samples=samples,
            pair_name="rgb-depth",
            run_id="smoke",
            out_dir=out_dir,
            show_progress=False,
        )
        joint_index = run_joint_extraction(
            model=encoder,
            samples=samples,
            pair_name="rgb-depth",
            modalities=("rgb", "depth"),
            run_id="smoke",
            out_dir=out_dir,
            show_progress=False,
        )

        # Joint index modalities use the synthetic names from joint_slice_modality_name
        rgb_joint = joint_slice_modality_name("rgb", "depth")
        depth_joint = joint_slice_modality_name("depth", "rgb")
        assert set(joint_index["modality"].unique()) == {rgb_joint, depth_joint}, (
            f"Joint extraction should emit only synthetic modalities, got {set(joint_index['modality'].unique())}"
        )

        # ── Step 3 — comparison indices
        comparisons = build_joint_comparison_indices(
            single_mod_index=single_index,
            joint_index=joint_index,
            left_modality="rgb",
            right_modality="depth",
        )
        assert len(comparisons) == 3, f"Expected 3 comparison frames, got {len(comparisons)}"

        expected_pairs = {
            f"rgb-{rgb_joint}",
            f"depth-{depth_joint}",
            f"{rgb_joint}-{depth_joint}",
        }
        got_pairs = {c.pair_name for c in comparisons}
        assert got_pairs == expected_pairs, f"Pair names: expected {expected_pairs}, got {got_pairs}"

        # Each comparison frame must merge non-empty on the 5-key predicate
        merge_keys = ["run_id", "pair", "scene_id", "sample_key", "layer"]
        for comp in comparisons:
            left = comp.table[comp.table["modality"] == comp.left_modality]
            right = comp.table[comp.table["modality"] == comp.right_modality]
            assert not left.empty, f"{comp.pair_name}: left modality '{comp.left_modality}' rows are empty"
            assert not right.empty, f"{comp.pair_name}: right modality '{comp.right_modality}' rows are empty"
            merged = left.merge(right, on=merge_keys, suffixes=("_l", "_r"))
            expected_rows = len(samples) * len(layers)
            assert len(merged) == expected_rows, (
                f"{comp.pair_name}: merge produced {len(merged)} rows, expected {expected_rows}"
            )

        # ── Step 4 — single-side rows in comparisons point to ORIGINAL single-mod files
        comp_a = next(c for c in comparisons if c.pair_name == f"rgb-{rgb_joint}")
        single_rgb_paths_in_comp = set(
            comp_a.table.loc[comp_a.table["modality"] == "rgb", "activation_path"]
        )
        original_rgb_paths = set(
            single_index.loc[single_index["modality"] == "rgb", "activation_path"]
        )
        assert single_rgb_paths_in_comp == original_rgb_paths, (
            "Single-mod rows in the comparison frame must reuse the original .npy paths verbatim"
        )

        # And joint-side rows point to *different* files under joint/
        joint_rgb_paths_in_comp = set(
            comp_a.table.loc[comp_a.table["modality"] == rgb_joint, "activation_path"]
        )
        assert joint_rgb_paths_in_comp.isdisjoint(original_rgb_paths), (
            "Joint-side rows must reference newly written .npy files, not the single-mod ones"
        )
        for path in joint_rgb_paths_in_comp:
            assert "/joint/" in path, f"Joint .npy path missing 'joint' segment: {path}"

    print("ALL PASSED")


if __name__ == "__main__":
    main()
