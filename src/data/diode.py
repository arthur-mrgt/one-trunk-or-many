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


def _normalize_splits(split: str | Sequence[str]) -> list[str]:
    """Return the list of DIODE splits to load.

    Accepts a single string (``"train"``, ``"val"``, ``"test"``), a list/tuple
    of strings, or the sentinels ``"train+val"`` / ``"all"`` for the common
    multi-split combinations. ``"+"``-separated single strings (e.g.
    ``"train+val"`` or ``"train+val+test"``) are also accepted so the
    multi-split case stays expressible on a Hydra CLI override.
    """
    if isinstance(split, str):
        token = split.strip().lower()
        if token == "all":
            return ["train", "val", "test"]
        if "+" in token:
            parts = [p.strip() for p in token.split("+") if p.strip()]
            if not parts:
                raise ValueError(f"Empty split specification: {split!r}")
            return list(dict.fromkeys(parts))
        return [split]
    seen: set[str] = set()
    result: list[str] = []
    for s in split:
        s_str = str(s)
        if s_str not in seen:
            seen.add(s_str)
            result.append(s_str)
    if not result:
        raise ValueError("split must be a non-empty string or sequence of strings")
    return result


def load_pairs(
    root: Path,
    modalities: tuple[str, str],
    n_scenes: int,
    scene_stride: int,
    split: str | Sequence[str] = "train",
    environment: str | Sequence[str] = "indoors",
    exclude_scenes: list[str] | None = None,
    frames_per_scene: int | None = None,
    max_total_samples: int | None = None,
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
        Number of scenes to include per ``(split, environment)`` tuple
        (after stride and exclusion). When multiple splits or environments
        are requested, the cap is applied independently to each combination,
        so the total scene count can be up to
        ``n_scenes * len(splits) * len(environments)``.
    scene_stride:
        Step size when walking the sorted scene list (1 = every scene).
    split:
        Dataset split. Either a single string (``"train"`` / ``"val"`` /
        ``"test"``), a list (``["train", "val"]``), the sentinel ``"all"``
        (= ``["train", "val", "test"]``), or a ``+``-joined string such as
        ``"train+val"`` (handy for Hydra CLI overrides where lists are
        awkward). DIODE scene IDs are globally unique across splits, so no
        extra namespacing is added when combining splits.
    environment:
        Scene environment. Either a single value (``"indoors"`` /
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
    max_total_samples:
        If set, after per-scene sampling, randomly subsample the global
        pool of ``(scene, frame)`` tuples down to this many samples.
        Sampling is done without replacement and is deterministic w.r.t.
        ``seed``. Pass ``None`` (default) to skip the global cap. Mirrors
        the same knob in the Hypersim adapter for parity across datasets.
    seed:
        Random seed for frame sampling.
    """
    splits = _normalize_splits(split)
    environments = _normalize_environments(environment)
    namespace_scenes = len(environments) > 1
    excluded = set(exclude_scenes or [])
    rng = random.Random(seed)

    output: list[PairSample] = []
    total_sampled_scenes = 0

    for split_name in splits:
        for env in environments:
            env_root = root / split_name / env
            if not env_root.exists():
                raise FileNotFoundError(
                    f"DIODE environment directory not found: {env_root}\n"
                    f"Expected layout: <root>/{split_name}/{env}/scene_XXXXX/scan_XXXXX/*.png"
                )

            scene_ids = [s for s in _list_scene_ids(env_root) if s not in excluded]
            sampled_scene_ids = scene_ids[:: max(scene_stride, 1)][:n_scenes]
            total_sampled_scenes += len(sampled_scene_ids)

            section_samples_before = len(output)
            for scene_dir in sampled_scene_ids:
                scene_root = env_root / scene_dir
                if not scene_root.exists():
                    log.warning(
                        "DIODE scene %s/%s/%s not found, skipping.",
                        split_name, env, scene_dir,
                    )
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
                split_name, env, len(output) - section_samples_before, len(sampled_scene_ids),
            )

    if max_total_samples is not None and max_total_samples < len(output):
        output = rng.sample(output, max_total_samples)
        output.sort(key=lambda s: (s.scene_id, s.sample_key))

    log.info(
        "DIODE loaded %d samples | splits=%s | env=%s | scenes=%d",
        len(output), ",".join(splits), ",".join(environments), total_sampled_scenes,
    )
    return output
