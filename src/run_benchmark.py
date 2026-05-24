"""Hydra entrypoint for full benchmark execution."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pipeline.benchmark import run_benchmark
from src.utils.config import cfg_to_container
from src.utils.distributed import init_distributed, shutdown_distributed
from src.utils.hydra_setup import register_hydra_resolvers

register_hydra_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run extraction and metrics stages in sequence."""
    cfg_dict = cfg_to_container(cfg)
    init_distributed(cfg_dict.get("runtime", {}))
    try:
        run_ctx = run_benchmark(cfg)
        print(f"[DONE] Run completed: {run_ctx.run_id}")
        print(f"[DONE] Results directory: {run_ctx.run_dir}")
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
