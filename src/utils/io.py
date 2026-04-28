from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def write_optional_parquet(path: Path, table: pd.DataFrame, enabled: bool) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def save_vector(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector)
