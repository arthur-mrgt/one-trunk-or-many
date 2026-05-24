"""Configuration and run-context helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class RunContext:
    """Paths and identifiers associated with one run."""

    run_id: str
    run_dir: Path
    activations_dir: Path
    metrics_dir: Path
    artifacts_dir: Path


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_context(cfg: DictConfig) -> RunContext:
    """Create run folder structure and metadata."""
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id = f"{cfg.project.stage}-{stamp}"
    run_dir = ensure_dir(Path(cfg.paths.runs_root) / run_id)
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        activations_dir=ensure_dir(run_dir / "activations"),
        metrics_dir=ensure_dir(run_dir / "metrics"),
        artifacts_dir=ensure_dir(run_dir / "artifacts"),
    )


def make_run_context_from_id(runs_root: Path, run_id: str) -> RunContext:
    """Build a run context from an already agreed run identifier."""
    run_dir = ensure_dir(Path(runs_root) / str(run_id))
    return RunContext(
        run_id=str(run_id),
        run_dir=run_dir,
        activations_dir=ensure_dir(run_dir / "activations"),
        metrics_dir=ensure_dir(run_dir / "metrics"),
        artifacts_dir=ensure_dir(run_dir / "artifacts"),
    )


def cfg_to_container(cfg: DictConfig) -> dict[str, Any]:
    """Convert Hydra config object to a plain dictionary."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
