"""DIODE dataset adapter for aligned modality pair sampling."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from src.data.hypersim import PairSample

log = logging.getLogger(__name__)

# Maps modality name → suffix relative to the RGB file stem
_MODALITY_SUFFIXES: dict[str, str] = {
    "rgb":     "",           # <stem>.png  (the RGB file itself)
    "depth":   "_depth.npy",
    "normals": "_normal.npy",
}


def _modality_path(rgb_path: Path, modality: str) -> Path:
    """Return the file path for the requested modality given the RGB path."""
    if modality not in _MODALITY_SUFFIXES:
        raise ValueError(
            f"Unsupported DIODE modality: '{modality}'. "
            f"Available: {sorted(_MODALITY_SUFFIXES)}"
        )
    suffix = _MODALITY_SUFFIXES[modality]
    if not suffix:
        return rgb_path
    return rgb_path.parent / (rgb_path.stem + suffix)


def _list_scene_ids(env_root: Path) -> list[str]:
    """List all scene directories under the given environment root."""
    return sorted(
        d.name for d in env_root.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    )


def _list_scene_rgb_files(scene_root: Path) -> list[Path]:
    """Return sorted RGB .png paths across all scans in a scene."""
    return sorted(
        p
        for scan_dir in sorted(scene_root.iterdir())
        if scan_dir.is_dir() and scan_dir.name.startswith("scan_")
        for p in scan_dir.glob("*.png")
    )


def load_pairs(
    root: Path,
    modalities: tuple[str, str],
    n_scenes: int,
    scene_stride: int,
    split: str = "train",
    environment: str = "indoors",
    exclude_scenes: list[str] | None = None,
    frames_per_scene: int | None = None,
    seed: int = 42,
) -> list[PairSample]:
    """Load aligned pair samples for the requested modalities from DIODE.

    Parameters
    ----------
    root:
        Path to the top-level DIODE directory (contains train/val/test).
    modalities:
        Two-element tuple of modality names, e.g. ``("rgb", "depth")``.
        Supported: ``"rgb"``, ``"depth"``, ``"normals"``.
    n_scenes:
        Number of scenes to include (after stride and exclusion).
    scene_stride:
        Step size when walking the sorted scene list (1 = every scene).
    split:
        Dataset split: one of ``"train"``, ``"val"``, ``"test"``.
    environment:
        Scene environment: one of ``"indoors"``, ``"outdoor"``.
    exclude_scenes:
        Optional scene IDs to skip.
    frames_per_scene:
        If set, randomly sample this many frames per scene.
        Scenes with fewer frames contribute all their frames.
        Pass ``None`` to keep every frame.
    seed:
        Random seed for frame sampling.
    """
    env_root = root / split / environment
    if not env_root.exists():
        raise FileNotFoundError(
            f"DIODE environment directory not found: {env_root}\n"
            f"Expected layout: <root>/{split}/{environment}/scene_XXXXX/scan_XXXXX/*.png"
        )

    excluded = set(exclude_scenes or [])
    scene_ids = [s for s in _list_scene_ids(env_root) if s not in excluded]
    sampled_scene_ids = scene_ids[:: max(scene_stride, 1)][:n_scenes]
    output: list[PairSample] = []
    rng = random.Random(seed)

    for scene_id in sampled_scene_ids:
        scene_root = env_root / scene_id
        if not scene_root.exists():
            log.warning("DIODE scene %s not found, skipping.", scene_id)
            continue

        rgb_files = _list_scene_rgb_files(scene_root)

        if frames_per_scene is not None and frames_per_scene < len(rgb_files):
            rgb_files = sorted(rng.sample(rgb_files, frames_per_scene))

        for rgb_path in rgb_files:
            mod_paths: dict[str, Path] = {}
            skip = False
            for mod in modalities:
                path = _modality_path(rgb_path, mod)
                if not path.exists():
                    log.debug(
                        "Missing %s file for %s (expected %s), skipping sample.",
                        mod, rgb_path.name, path.name,
                    )
                    skip = True
                    break
                mod_paths[mod] = path

            if skip:
                continue

            scan_name = rgb_path.parent.name       # e.g. "scan_00001"
            sample_key = f"{scan_name}:{rgb_path.stem}"

            output.append(
                PairSample(
                    scene_id=scene_id,
                    sample_key=sample_key,
                    modality_paths=mod_paths,
                )
            )

    log.info(
        "DIODE loaded %d samples | split=%s | env=%s | scenes=%d",
        len(output), split, environment, len(sampled_scene_ids),
    )
    return output
