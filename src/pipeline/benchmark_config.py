"""Configuration and run-context helpers for benchmark orchestration."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.utils.config import RunContext, ensure_dir, make_run_context


def normalize_pair(pair: Any) -> list[str]:
    """Validate one modality pair and return `[left, right]`."""
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(
            f"Invalid modality pair: {pair!r}. Expected two entries, e.g. [rgb, depth]."
        )
    left, right = str(pair[0]), str(pair[1])
    if left == right:
        raise ValueError(f"Invalid modality pair: {pair!r}. Modalities must differ.")
    return [left, right]


def resolve_metric_pairs(metrics_cfg: dict[str, Any]) -> list[list[str]]:
    """Resolve modality pairs from metrics config."""
    pairs_mode = str(metrics_cfg.get("pairs_mode", "explicit"))
    explicit_pairs = metrics_cfg.get("pairs", []) or []

    if pairs_mode == "explicit":
        if not explicit_pairs:
            raise ValueError("metrics.pairs is empty while metrics.pairs_mode=explicit.")
        normalized = [normalize_pair(p) for p in explicit_pairs]
    elif pairs_mode == "all_combinations":
        modalities_raw = metrics_cfg.get("modalities", []) or []
        modalities = list(dict.fromkeys(str(m) for m in modalities_raw))
        if len(modalities) < 2:
            raise ValueError(
                "metrics.modalities must contain at least two unique entries when "
                "metrics.pairs_mode=all_combinations."
            )
        normalized = [[left, right] for left, right in combinations(modalities, 2)]
    else:
        raise ValueError(
            f"Unknown metrics.pairs_mode='{pairs_mode}'. "
            "Supported values: explicit, all_combinations."
        )

    seen: set[tuple[str, str]] = set()
    resolved: list[list[str]] = []
    for left, right in normalized:
        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        resolved.append([left, right])
    return resolved


def list_activation_index_files(run_ctx: RunContext) -> list[Path]:
    """List activation index CSV artifacts for one run context."""
    return sorted(run_ctx.artifacts_dir.glob("activation_index_*.csv"))


def resolve_run_ctx_for_metrics(cfg_dict: dict[str, Any]) -> RunContext:
    """Resolve which run directory to use for metrics-like stages.

    Resolution order:
      1. Explicit ``runtime.activation_input_run_id`` / ``metrics_input_run_id``.
      2. Latest run dir whose name starts with ``{project.stage}-``. Filtering
         by stage prevents silent reuse of artifacts from an unrelated stage
         (which was previously possible because the candidate list was the
         lexicographic max of *all* run dirs).
      3. If ``project.stage`` is empty, fall back to the legacy behaviour
         (latest run dir of any name) so old configs keep working.
    """
    runtime_cfg = cfg_dict.get("runtime", {})
    run_id = runtime_cfg.get("activation_input_run_id") or runtime_cfg.get("metrics_input_run_id")
    runs_root = Path(cfg_dict["paths"]["runs_root"])
    if run_id:
        run_dir = runs_root / run_id
    else:
        stage = str(cfg_dict.get("project", {}).get("stage", "")).strip()
        glob_pat = f"{stage}-*" if stage else "*"
        candidates = sorted(p for p in runs_root.glob(glob_pat) if p.is_dir())
        if not candidates:
            hint = f"stage={stage!r}" if stage else "any stage"
            raise FileNotFoundError(
                f"No run directory found for {hint}. "
                "Run extraction first or set runtime.activation_input_run_id."
            )
        run_dir = candidates[-1]
    return RunContext(
        run_id=run_dir.name,
        run_dir=run_dir,
        activations_dir=run_dir / "activations",
        metrics_dir=ensure_dir(run_dir / "metrics"),
        artifacts_dir=ensure_dir(run_dir / "artifacts"),
    )


def resolve_run_ctx_for_extraction(cfg: DictConfig, cfg_dict: dict[str, Any]) -> RunContext:
    """Resolve run context for extraction with optional reuse."""
    reuse_cfg = dict(cfg_dict.get("runtime", {}).get("reuse", {}))
    reuse_activations = bool(reuse_cfg.get("activations", reuse_cfg.get("extraction", False)))
    if reuse_activations:
        try:
            run_ctx = resolve_run_ctx_for_metrics(cfg_dict)
            if list_activation_index_files(run_ctx):
                return run_ctx
        except Exception:
            pass
    return make_run_context(cfg)


def resolve_null_artifact_path(cfg_dict: dict[str, Any], run_ctx: RunContext | None = None) -> Path | None:
    """Resolve null artifact file path from runtime and config values."""
    artifact_dir = Path(
        cfg_dict.get("analysis", {}).get("null_distribution", {}).get("artifact_dir", "")
    )
    if not artifact_dir:
        return None

    null_run_id = cfg_dict.get("runtime", {}).get("null_input_run_id")
    if null_run_id:
        run_dir = artifact_dir / str(null_run_id)
    elif run_ctx is not None:
        run_dir = artifact_dir / run_ctx.run_id
    else:
        return None

    explicit = cfg_dict.get("runtime", {}).get("null_input_filename")
    if explicit:
        return run_dir / str(explicit)

    candidate = run_dir / str(
        cfg_dict.get("analysis", {}).get("null_distribution", {}).get("artifact_filename", "null_distribution.csv")
    )
    if candidate.exists():
        return candidate

    versioned = sorted(run_dir.glob("null_distribution_*scenes_*draws.csv"))
    if versioned:
        return versioned[-1]
    return candidate


def read_null_state(path: Path) -> dict[str, Any]:
    """Read null-stage checkpoint JSON and return empty dict on failure."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
