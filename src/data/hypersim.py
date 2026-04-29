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


def _index_scene_files(scene_root: Path, modality: str) -> dict[str, Path]:
    """Index files for one modality by frame key."""
    if modality == "rgb":
        pattern = "images/scene_cam_*_final_hdf5/frame.*.color.hdf5"
    elif modality == "depth":
        pattern = "images/scene_cam_*_geometry_hdf5/frame.*.depth_meters.hdf5"
    else:
        raise ValueError(f"Unsupported Hypersim modality: {modality}")

    mapping: dict[str, Path] = {}
    for file_path in scene_root.glob(pattern):
        mapping[_frame_key(file_path)] = file_path
    return mapping


def load_pairs(
    root: Path,
    modalities: tuple[str, str],
    n_scenes: int,
    scene_stride: int,
) -> list[PairSample]:
    """Load aligned pair samples for the requested modalities."""
    scene_ids = _list_scene_ids(root)
    sampled_scene_ids = scene_ids[:: max(scene_stride, 1)][:n_scenes]
    output: list[PairSample] = []

    scenes_root = root / "scenes" if (root / "scenes").exists() else root

    for scene_id in sampled_scene_ids:
        scene_root = scenes_root / scene_id
        if not scene_root.exists():
            continue

        left = _index_scene_files(scene_root, modalities[0])
        right = _index_scene_files(scene_root, modalities[1])
        common_keys = sorted(set(left).intersection(right))

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
    return output
