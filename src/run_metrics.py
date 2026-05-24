"""Hydra entrypoint for metrics-only execution."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pipeline.benchmark import run_metrics_stage
from src.utils.config import cfg_to_container
from src.utils.distributed import init_distributed, shutdown_distributed
from src.utils.hydra_setup import register_hydra_resolvers

register_hydra_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run only the metrics stage."""
    cfg_dict = cfg_to_container(cfg)
    init_distributed(cfg_dict.get("runtime", {}))
    try:
        run_ctx = run_metrics_stage(cfg)
        print(f"[DONE] Metrics completed for run: {run_ctx.run_id}")
        print(f"[DONE] Metrics dir: {run_ctx.metrics_dir}")
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
