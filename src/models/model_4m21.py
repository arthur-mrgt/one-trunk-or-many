from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class FourMMockEncoder:
    """
    Deterministic mock encoder for pipeline validation.
    Replace with a real 4M wrapper in production.
    """

    def __init__(self, layers: list[str], embedding_dim: int) -> None:
        self.layers = layers
        self.embedding_dim = embedding_dim

    def encode_path(self, file_path: Path, modality: str) -> dict[str, np.ndarray]:
        seed_material = f"{file_path.as_posix()}::{modality}"
        seed = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return {
            layer: rng.standard_normal(self.embedding_dim).astype(np.float32)
            for layer in self.layers
        }


class FourMRealEncoder:
    def __init__(self, model_dir: Path, layers: list[str] | str) -> None:
        self.model_dir = model_dir
        self.layers = layers
        raise NotImplementedError(
            "Real 4M runtime integration is intentionally left as an extension point. "
            "Use model.backend=fourm_mock for current CKA pipeline runs."
        )
