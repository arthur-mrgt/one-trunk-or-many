"""Activation extraction stage implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.utils.io import save_vector

EXTRACTION_COLUMNS: list[str] = [
    "run_id",
    "pair",
    "scene_id",
    "sample_key",
    "modality",
    "layer",
    "activation_path",
]


def _safe_sample_key(sample_key: str) -> str:
    """Filesystem-safe representation of ``sample_key``."""
    return str(sample_key).replace(":", "_")


def joint_slice_modality_name(modality: str, partner: str) -> str:
    """Return the synthetic modality name for a joint-pass token slice.

    Naming is symmetric (``rgb_joint_with_depth`` for the rgb slice in a joint
    rgb+depth pass). Documented once here so producers (extraction) and
    consumers (comparison index construction, metrics config) agree.
    """
    if modality == partner:
        raise ValueError(f"Joint slice modality and partner must differ: {modality!r}")
    return f"{modality}_joint_with_{partner}"


def run_extraction(
    model: Any,
    samples: list[Any],
    pair_name: str,
    run_id: str,
    out_dir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract vectors for each sample, modality, and layer."""
    records: list[dict[str, Any]] = []

    sample_iter = tqdm(
        samples,
        desc=f"Extract[{pair_name}]",
        unit="sample",
        disable=not show_progress,
    )

    for sample in sample_iter:
        for modality, file_path in sample.modality_paths.items():
            acts = model.encode_path(file_path=file_path, modality=modality)
            for layer, vec in acts.items():
                rel_path = Path(
                    pair_name,
                    layer,
                    modality,
                    f"{sample.scene_id}__{_safe_sample_key(sample.sample_key)}.npy",
                )
                abs_path = out_dir / rel_path
                save_vector(abs_path, vec)
                records.append(
                    {
                        "run_id": run_id,
                        "pair": pair_name,
                        "scene_id": sample.scene_id,
                        "sample_key": sample.sample_key,
                        "modality": modality,
                        "layer": layer,
                        "activation_path": str(abs_path),
                    }
                )
        if show_progress:
            sample_iter.set_postfix_str(f"scene={sample.scene_id}")

    return pd.DataFrame.from_records(records, columns=EXTRACTION_COLUMNS)


def run_joint_extraction(
    model: Any,
    samples: list[Any],
    pair_name: str,
    modalities: tuple[str, str],
    run_id: str,
    out_dir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run one joint forward pass per sample and save per-modality token slices.

    Mirrors :func:`run_extraction` but calls :meth:`model.encode_pair_joint`,
    saving two ``.npy`` files per sample/layer (one per token slice). Output
    ``modality`` values use :func:`joint_slice_modality_name` so the rows can
    be merged directly into comparison activation indices without renaming.

    Files are stored under ``out_dir / 'joint' / pair_name / layer / modality``
    to keep them physically separate from single-modality extractions.
    """
    if len(modalities) != 2 or modalities[0] == modalities[1]:
        raise ValueError(
            f"run_joint_extraction requires two distinct modalities, got {modalities!r}"
        )
    left_mod, right_mod = modalities
    records: list[dict[str, Any]] = []

    sample_iter = tqdm(
        samples,
        desc=f"ExtractJoint[{pair_name}]",
        unit="sample",
        disable=not show_progress,
    )

    for sample in sample_iter:
        if left_mod not in sample.modality_paths or right_mod not in sample.modality_paths:
            continue
        file_paths = {
            left_mod: sample.modality_paths[left_mod],
            right_mod: sample.modality_paths[right_mod],
        }
        per_layer = model.encode_pair_joint(file_paths=file_paths, modalities=modalities)
        for layer, slice_dict in per_layer.items():
            for modality, vec in slice_dict.items():
                partner = right_mod if modality == left_mod else left_mod
                pseudo_modality = joint_slice_modality_name(modality, partner)
                rel_path = Path(
                    "joint",
                    pair_name,
                    layer,
                    pseudo_modality,
                    f"{sample.scene_id}__{_safe_sample_key(sample.sample_key)}.npy",
                )
                abs_path = out_dir / rel_path
                save_vector(abs_path, vec)
                records.append(
                    {
                        "run_id": run_id,
                        "pair": pair_name,
                        "scene_id": sample.scene_id,
                        "sample_key": sample.sample_key,
                        "modality": pseudo_modality,
                        "layer": layer,
                        "activation_path": str(abs_path),
                    }
                )
        if show_progress:
            sample_iter.set_postfix_str(f"scene={sample.scene_id}")

    return pd.DataFrame.from_records(records, columns=EXTRACTION_COLUMNS)
