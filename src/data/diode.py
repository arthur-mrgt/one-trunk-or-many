from __future__ import annotations

from pathlib import Path


def load_pairs(
    root: Path,
    modalities: tuple[str, str],
    n_scenes: int,
    scene_stride: int,
) -> list[dict]:
    raise NotImplementedError(
        "DIODE adapter scaffold is in place but not implemented yet. "
        "Use data.name=hypersim for now."
    )
