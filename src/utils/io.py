"""I/O helpers for tables, JSON, and vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload to disk with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Write a dataframe to CSV, always including the header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def write_optional_parquet(path: Path, table: pd.DataFrame, enabled: bool) -> None:
    """Write a dataframe to Parquet when enabled."""
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def save_vector(path: Path, vector: np.ndarray) -> None:
    """Save an activation vector as NPY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector)


def read_null_distribution(path: Path) -> pd.DataFrame:
    """Load a precomputed null distribution from CSV or Parquet.

    Tries ``path`` first, then swaps the suffix between ``.csv`` and
    ``.parquet`` so callers can pass either extension.
    Returns an empty DataFrame when no file is found or the file is empty.
    """
    for candidate in _null_distribution_candidates(path):
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".parquet":
                return pd.read_parquet(candidate)
            return pd.read_csv(candidate)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _null_distribution_candidates(path: Path) -> list[Path]:
    """Return path variants (CSV / Parquet) to try in order."""
    if path.suffix == ".csv":
        return [path, path.with_suffix(".parquet")]
    if path.suffix == ".parquet":
        return [path, path.with_suffix(".csv")]
    return [path]
