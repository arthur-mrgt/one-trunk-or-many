"""DIODE dataset adapter for aligned modality pair sampling."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
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


def _normalize_environments(environment: str | Sequence[str]) -> list[str]:
    """Return the list of environments to load, preserving order and uniqueness.

    Accepts a single string (``"indoors"``), a list/tuple of strings
    (``["indoors", "outdoor"]``), or the sentinel ``"both"`` / ``"all"``.
    """
    if isinstance(environment, str):
        if environment.lower() in {"both", "all"}:
            return ["indoors", "outdoor"]
        return [environment]
    seen: set[str] = set()
    result: list[str] = []
    for env in environment:
        env_str = str(env)
        if env_str not in seen:
            seen.add(env_str)
            result.append(env_str)
    if not result:
        raise ValueError("environment must be a non-empty string or sequence of strings")
    return result


def load_pairs(
    root: Path,
    modalities: tuple[str, str],
    n_scenes: int,
    scene_stride: int,
    split: str = "train",
    environment: str | Sequence[str] = "indoors",
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
        Number of scenes to include per environment (after stride and
        exclusion). When multiple environments are requested, the cap is
        applied independently to each one, so the total scene count can be
        up to ``n_scenes * len(environments)``.
    scene_stride:
        Step size when walking the sorted scene list (1 = every scene).
    split:
        Dataset split: one of ``"train"``, ``"val"``, ``"test"``.
    environment:
        Scene environment. Either a single value (``"indoors"`` or
        ``"outdoor"``), the sentinel ``"both"`` / ``"all"``, or a list such
        as ``["indoors", "outdoor"]``. When multiple environments are
        selected, ``scene_id`` is prefixed with the environment name to keep
        IDs unique across environments.

    Notes
    -----
    Each sample's ``scene_id`` resolves to the **scan-level** path so that
    every DIODE scan is treated as a distinct "scene" by downstream null
    sampling. With multiple environments selected, IDs look like
    ``"indoors/scene_00021/scan_00190"``; with a single environment, they
    look like ``"scene_00021/scan_00190"``. The parent of each ``scene_id``
    (e.g. ``"indoors/scene_00021"``) groups scans into a DIODE-scene, which
    the null-distribution stage uses as the "scene type" for the
    ``cross_scene_type=scene`` constraint.
    exclude_scenes:
        Optional scene IDs to skip. Each entry is matched against the bare
        scene directory name (e.g. ``"scene_00019"``).
    frames_per_scene:
        If set, randomly sample this many frames per scene.
        Scenes with fewer frames contribute all their frames.
        Pass ``None`` to keep every frame.
    seed:
        Random seed for frame sampling.
    """
    environments = _normalize_environments(environment)
    namespace_scenes = len(environments) > 1
    excluded = set(exclude_scenes or [])
    rng = random.Random(seed)

    output: list[PairSample] = []
    total_sampled_scenes = 0

    for env in environments:
        env_root = root / split / env
        if not env_root.exists():
            raise FileNotFoundError(
                f"DIODE environment directory not found: {env_root}\n"
                f"Expected layout: <root>/{split}/{env}/scene_XXXXX/scan_XXXXX/*.png"
            )

        scene_ids = [s for s in _list_scene_ids(env_root) if s not in excluded]
        sampled_scene_ids = scene_ids[:: max(scene_stride, 1)][:n_scenes]
        total_sampled_scenes += len(sampled_scene_ids)

        env_samples_before = len(output)
        for scene_dir in sampled_scene_ids:
            scene_root = env_root / scene_dir
            if not scene_root.exists():
                log.warning("DIODE scene %s/%s not found, skipping.", env, scene_dir)
                continue

            rgb_files = _list_scene_rgb_files(scene_root)

            if frames_per_scene is not None and frames_per_scene < len(rgb_files):
                rgb_files = sorted(rng.sample(rgb_files, frames_per_scene))

            scene_id = f"{env}/{scene_dir}" if namespace_scenes else scene_dir

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
                scan_scene_id = f"{scene_id}/{scan_name}"
                sample_key = rgb_path.stem

                output.append(
                    PairSample(
                        scene_id=scan_scene_id,
                        sample_key=sample_key,
                        modality_paths=mod_paths,
                    )
                )

        log.info(
            "DIODE %s/%s: %d samples from %d scenes",
            split, env, len(output) - env_samples_before, len(sampled_scene_ids),
        )

    log.info(
        "DIODE loaded %d samples | split=%s | env=%s | scenes=%d",
        len(output), split, ",".join(environments), total_sampled_scenes,
    )
    return output
