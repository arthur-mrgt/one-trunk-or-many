"""Tests for modality-pair resolution in benchmark config."""

from __future__ import annotations

import pytest

from src.pipeline.benchmark import _resolve_metric_pairs


def test_resolve_pairs_explicit() -> None:
    cfg = {
        "pairs_mode": "explicit",
        "pairs": [["rgb", "depth"], ["rgb", "normals"]],
    }
    assert _resolve_metric_pairs(cfg) == [["rgb", "depth"], ["rgb", "normals"]]


def test_resolve_pairs_all_combinations() -> None:
    cfg = {
        "pairs_mode": "all_combinations",
        "modalities": ["rgb", "depth", "normals"],
    }
    assert _resolve_metric_pairs(cfg) == [
        ["rgb", "depth"],
        ["rgb", "normals"],
        ["depth", "normals"],
    ]


def test_resolve_pairs_all_combinations_requires_two_modalities() -> None:
    cfg = {
        "pairs_mode": "all_combinations",
        "modalities": ["rgb"],
    }
    with pytest.raises(ValueError, match="at least two"):
        _resolve_metric_pairs(cfg)

