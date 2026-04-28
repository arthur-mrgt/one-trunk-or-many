from __future__ import annotations

from datetime import datetime, timezone

from omegaconf import OmegaConf


def _now_utc(fmt: str) -> str:
    return datetime.now(timezone.utc).strftime(fmt)


def register_hydra_resolvers() -> None:
    if not OmegaConf.has_resolver("now_utc"):
        OmegaConf.register_new_resolver("now_utc", _now_utc)
