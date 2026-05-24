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


def _resolve_activations_dir(cfg_paths: Any, run_id: str, default: Path) -> Path:
    """Resolve where activation .npy files should physically live.

    If ``paths.activations_root`` is set in the config, place activation files
    under ``<activations_root>/<run_id>/activations`` (typically a fast local
    disk like ``/tmp`` to bypass slow shared filesystems such as Lustre).
    Otherwise fall back to the in-run-dir default.
    """
    activations_root = None
    try:
        activations_root = cfg_paths.get("activations_root", None)  # type: ignore[union-attr]
    except AttributeError:
        activations_root = getattr(cfg_paths, "activations_root", None)
    if activations_root:
        return ensure_dir(Path(str(activations_root)) / run_id / "activations")
    return ensure_dir(default)


def make_run_context(cfg: DictConfig) -> RunContext:
    """Create run folder structure and metadata."""
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id = f"{cfg.project.stage}-{stamp}"
    run_dir = ensure_dir(Path(cfg.paths.runs_root) / run_id)
    activations_dir = _resolve_activations_dir(
        cfg.paths, run_id=run_id, default=run_dir / "activations"
    )
    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        activations_dir=activations_dir,
        metrics_dir=ensure_dir(run_dir / "metrics"),
        artifacts_dir=ensure_dir(run_dir / "artifacts"),
    )


def make_run_context_from_id(
    runs_root: Path,
    run_id: str,
    cfg_paths: Any | None = None,
) -> RunContext:
    """Build a run context from an already agreed run identifier.

    When ``cfg_paths`` is provided and ``paths.activations_root`` is set,
    activations are resolved to the fast local path so all ranks agree on
    where to read/write activation files.
    """
    run_dir = ensure_dir(Path(runs_root) / str(run_id))
    if cfg_paths is not None:
        activations_dir = _resolve_activations_dir(
            cfg_paths, run_id=str(run_id), default=run_dir / "activations"
        )
    else:
        activations_dir = ensure_dir(run_dir / "activations")
    return RunContext(
        run_id=str(run_id),
        run_dir=run_dir,
        activations_dir=activations_dir,
        metrics_dir=ensure_dir(run_dir / "metrics"),
        artifacts_dir=ensure_dir(run_dir / "artifacts"),
    )


def cfg_to_container(cfg: DictConfig) -> dict[str, Any]:
    """Convert Hydra config object to a plain dictionary."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
