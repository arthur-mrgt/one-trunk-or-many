"""Helpers for single-node multi-process distributed execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TypeVar

import pandas as pd

T = TypeVar("T")


@dataclass(frozen=True)
class DistContext:
    """Runtime metadata for the distributed process group."""

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    backend: str

    @property
    def is_main(self) -> bool:
        """Return True when this process is global rank zero."""
        return self.rank == 0


_CTX = DistContext(enabled=False, rank=0, world_size=1, local_rank=0, backend="none")


def init_distributed(runtime_cfg: dict[str, Any]) -> DistContext:
    """Initialize torch.distributed when requested or launched with torchrun."""
    global _CTX

    enabled_cfg = bool((runtime_cfg.get("distributed") or {}).get("enabled", False))
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    should_enable = enabled_cfg or world_size_env > 1
    if not should_enable:
        _CTX = DistContext(enabled=False, rank=0, world_size=1, local_rank=0, backend="none")
        return _CTX

    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        backend = str(dist.get_backend())
        _CTX = DistContext(True, rank, world_size, local_rank, backend)
        return _CTX

    distributed_cfg = runtime_cfg.get("distributed") or {}
    backend = str(distributed_cfg.get("backend", "auto")).lower()
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    _CTX = DistContext(True, rank, world_size, local_rank, backend)
    return _CTX


def shutdown_distributed() -> None:
    """Destroy distributed process group when initialized."""
    global _CTX
    if not _CTX.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    _CTX = DistContext(enabled=False, rank=0, world_size=1, local_rank=0, backend="none")


def get_dist_context() -> DistContext:
    """Return current distributed context."""
    return _CTX


def barrier() -> None:
    """Synchronize all ranks when distributed mode is enabled."""
    ctx = get_dist_context()
    if not ctx.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


def split_by_rank(items: Sequence[T], ctx: DistContext | None = None) -> list[T]:
    """Deterministically shard a sequence across ranks."""
    resolved = ctx or get_dist_context()
    if not resolved.enabled or resolved.world_size <= 1:
        return list(items)
    return [item for idx, item in enumerate(items) if idx % resolved.world_size == resolved.rank]


def all_gather_objects(obj: Any, ctx: DistContext | None = None) -> list[Any]:
    """All-gather Python objects across ranks."""
    resolved = ctx or get_dist_context()
    if not resolved.enabled or resolved.world_size <= 1:
        return [obj]
    import torch.distributed as dist

    out: list[Any] = [None for _ in range(resolved.world_size)]
    dist.all_gather_object(out, obj)
    return out


def broadcast_object(obj: Any, src: int = 0, ctx: DistContext | None = None) -> Any:
    """Broadcast one Python object from source rank to all ranks."""
    resolved = ctx or get_dist_context()
    if not resolved.enabled or resolved.world_size <= 1:
        return obj
    import torch.distributed as dist

    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def any_rank_true(flag: bool, ctx: DistContext | None = None) -> bool:
    """Return True if any rank reports True."""
    resolved = ctx or get_dist_context()
    if not resolved.enabled or resolved.world_size <= 1:
        return bool(flag)
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([1 if flag else 0], dtype=torch.int32)
    if torch.cuda.is_available() and resolved.backend == "nccl":
        tensor = tensor.cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(int(tensor.item()) > 0)


def concat_gathered_tables(local_df: pd.DataFrame, ctx: DistContext | None = None) -> pd.DataFrame:
    """Gather dataframe rows from all ranks and concatenate them."""
    parts = all_gather_objects(local_df, ctx=ctx)
    frames = [p for p in parts if isinstance(p, pd.DataFrame) and not p.empty]
    if not frames:
        return pd.DataFrame(columns=local_df.columns)
    return pd.concat(frames, ignore_index=True)

