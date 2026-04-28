from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pipeline.benchmark import run_benchmark


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_ctx = run_benchmark(cfg)
    print(f"[DONE] Run completed: {run_ctx.run_id}")
    print(f"[DONE] Results directory: {run_ctx.run_dir}")


if __name__ == "__main__":
    main()
