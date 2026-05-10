"""Hypersim adapter for aligned modality pair sampling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PairSample:
    """Container for one aligned multimodal sample."""

    scene_id: str
    sample_key: str
    modality_paths: dict[str, Path]


def _frame_key(path: Path) -> str:
    """Build a stable key from camera and frame id."""
    cam_match = re.search(r"scene_(cam_\d+)_", path.as_posix())
    frame_match = re.search(r"frame\.(\d+)", path.name)
    cam = cam_match.group(1) if cam_match else "cam_unknown"
    frame = frame_match.group(1) if frame_match else "frame_unknown"
    return f"{cam}:{frame}"


def _list_scene_ids(root: Path) -> list[str]:
    """List available Hypersim scene ids under supported layouts."""
    # Support both layouts:
    # 1) <root>/scenes/ai_XXX_YYY/...
    # 2) <root>/ai_XXX_YYY/...
    scenes_root = root / "scenes"
    if scenes_root.exists():
        base = scenes_root
    else:
        base = root

    return sorted(
        p.name for p in base.glob("ai_*_*") if p.is_dir() and p.name.startswith("ai_")
    )


_MODALITY_PATTERNS: dict[str, str] = {
    "rgb":     "images/scene_cam_*_final_hdf5/frame.*.color.hdf5",
    "depth":   "images/scene_cam_*_geometry_hdf5/frame.*.depth_meters.hdf5",
    "normals": "images/scene_cam_*_geometry_hdf5/frame.*.normal_cam.hdf5"
}


def _index_scene_files(scene_root: Path, modality: str) -> dict[str, Path]:
    """Index files for one modality by frame key."""
    if modality not in _MODALITY_PATTERNS:
        raise ValueError(
            f"Unsupported Hypersim modality: '{modality}'. "
            f"Available: {sorted(_MODALITY_PATTERNS)}"
        )
    pattern = _MODALITY_PATTERNS[modality]

    mapping: dict[str, Path] = {}
    for file_path in scene_root.glob(pattern):
        mapping[_frame_key(file_path)] = file_path
    return mapping


def load_pairs(
    root: Path,
    modalities: tuple[str, str],
    n_scenes: int,
    scene_stride: int,
    exclude_scenes: list[str] | None = None,
    frames_per_scene: int | None = None,
    max_total_samples: int | None = None,
    seed: int = 42,
) -> list[PairSample]:
    """Load aligned pair samples for the requested modalities.

    Parameters
    ----------
    n_scenes:
        Number of scenes to load (after stride and exclusion).
    scene_stride:
        Step size when walking the sorted scene list (1 = every scene).
    exclude_scenes:
        Optional list of scene IDs to skip.
    frames_per_scene:
        If set, randomly sample this many frames per scene instead of
        using all available frames. Scenes with fewer frames than this
        value contribute all their frames. Pass ``None`` (default) to
        keep every frame.
    max_total_samples:
        If set, after per-scene sampling, randomly subsample the global
        pool of ``(scene, frame)`` tuples down to this many samples.
        Sampling is done without replacement and is deterministic w.r.t.
        ``seed``. Pass ``None`` (default) to keep every sample.
    seed:
        Random seed used for frame and global sampling.
    """
    import random

    excluded = set(exclude_scenes or [])
    scene_ids = [s for s in _list_scene_ids(root) if s not in excluded]
    sampled_scene_ids = scene_ids[:: max(scene_stride, 1)][:n_scenes]
    output: list[PairSample] = []

    rng = random.Random(seed)
    scenes_root = root / "scenes" if (root / "scenes").exists() else root

    for scene_id in sampled_scene_ids:
        scene_root = scenes_root / scene_id
        if not scene_root.exists():
            continue

        left = _index_scene_files(scene_root, modalities[0])
        right = _index_scene_files(scene_root, modalities[1])
        common_keys = sorted(set(left).intersection(right))

        if frames_per_scene is not None and frames_per_scene < len(common_keys):
            common_keys = rng.sample(common_keys, frames_per_scene)
            common_keys = sorted(common_keys)

        for key in common_keys:
            output.append(
                PairSample(
                    scene_id=scene_id,
                    sample_key=key,
                    modality_paths={
                        modalities[0]: left[key],
                        modalities[1]: right[key],
                    },
                )
            )

    if max_total_samples is not None and max_total_samples < len(output):
        output = rng.sample(output, max_total_samples)
        output.sort(key=lambda s: (s.scene_id, s.sample_key))

    return output
