from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    activations_dir: Path
    metrics_dir: Path
    artifacts_dir: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_context(cfg: DictConfig) -> RunContext:
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


def cfg_to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
