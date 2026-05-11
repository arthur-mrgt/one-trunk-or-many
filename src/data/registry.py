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
    exclude_scenes: list[str] | None = None,
    frames_per_scene: int | None = None,
    max_total_samples: int | None = None,
    seed: int = 42,
    split: str = "train",
    environment: str = "indoors",
) -> list[Any]:
    """Dispatch pair loading to the selected dataset adapter."""
    if dataset_name == "hypersim":
        return hypersim.load_pairs(
            root=root,
            modalities=modalities,
            n_scenes=n_scenes,
            scene_stride=scene_stride,
            exclude_scenes=exclude_scenes,
            frames_per_scene=frames_per_scene,
            max_total_samples=max_total_samples,
            seed=seed,
        )
    if dataset_name == "diode":
        return diode.load_pairs(
            root=root,
            modalities=modalities,
            n_scenes=n_scenes,
            scene_stride=scene_stride,
            split=split,
            environment=environment,
            exclude_scenes=exclude_scenes,
            frames_per_scene=frames_per_scene,
            seed=seed,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")
