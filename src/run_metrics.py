from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pipeline.benchmark import run_benchmark


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
