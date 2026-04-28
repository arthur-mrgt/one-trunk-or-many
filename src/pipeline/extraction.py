from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.utils.io import save_vector


def run_extraction(
    model: Any,
    samples: list[Any],
    pair_name: str,
    run_id: str,
    out_dir: Path,
    show_progress: bool = True,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    columns = [
        "run_id",
        "pair",
        "scene_id",
        "sample_key",
        "modality",
        "layer",
        "activation_path",
    ]

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
                    f"{sample.scene_id}__{sample.sample_key.replace(':', '_')}.npy",
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

    return pd.DataFrame.from_records(records, columns=columns)
