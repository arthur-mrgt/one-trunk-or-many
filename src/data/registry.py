"""Dataset loader registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data import diode, hypersim


def load_dataset_pairs(
    dataset_name: str,
    root: Path,
    modalities: tuple[str, str],
    n_scenes: int,
    scene_stride: int,
) -> list[Any]:
    """Dispatch pair loading to the selected dataset adapter."""
    if dataset_name == "hypersim":
        return hypersim.load_pairs(
            root=root,
            modalities=modalities,
            n_scenes=n_scenes,
            scene_stride=scene_stride,
        )
    if dataset_name == "diode":
        return diode.load_pairs(
            root=root,
            modalities=modalities,
            n_scenes=n_scenes,
            scene_stride=scene_stride,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")
