"""Deterministic mock encoder for smoke tests and CI."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np



class FourMMockEncoder:
    """Deterministic mock encoder used for smoke tests."""

    def __init__(self, layers: list[str], embedding_dim: int) -> None:
        """Store layer names and embedding size."""
        self.layers = layers
        self.embedding_dim = embedding_dim

    def encode_path(self, file_path: Path, modality: str) -> dict[str, np.ndarray]:
        """Generate deterministic mock vectors for one file."""
        seed_material = f"{file_path.as_posix()}::{modality}"
        seed = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return {
            layer: rng.standard_normal(self.embedding_dim).astype(np.float32)
            for layer in self.layers
        }

    def encode_pair_joint(
        self,
        file_paths: dict[str, Path],
        modalities: tuple[str, str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Mock joint forward pass: deterministic vectors per (layer, modality).

        Mirrors :meth:`FourMEncoder.encode_pair_joint`. The output structure is
        ``{layer: {modality_a: vec, modality_b: vec}}`` where the modalities are
        always emitted in **alphabetical order** to match the production path.
        """
        if len(modalities) != 2 or modalities[0] == modalities[1]:
            raise ValueError(
                f"encode_pair_joint requires two distinct modalities, got {modalities!r}"
            )
        sorted_mods = tuple(sorted(modalities))
        joint_tag = (
            f"{sorted_mods[0]}={file_paths[sorted_mods[0]].as_posix()}::"
            f"{sorted_mods[1]}={file_paths[sorted_mods[1]].as_posix()}"
        )
        out: dict[str, dict[str, np.ndarray]] = {}
        for layer in self.layers:
            slices: dict[str, np.ndarray] = {}
            for mod in sorted_mods:
                seed_material = f"joint::{joint_tag}::{layer}::{mod}"
                seed = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                slices[mod] = rng.standard_normal(self.embedding_dim).astype(np.float32)
            out[layer] = slices
        return out
