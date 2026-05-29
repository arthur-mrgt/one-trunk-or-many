"""RQ2 triangulation pipeline — fragmentation analysis over existing RQ1 activations.

Loads the three activation index CSVs produced by the RQ1 run, merges them
into a single trimodal index, and computes fragmentation scores layer by layer
for all six directed triplets (A, B, C).

No new forward passes are required; all computation operates on the .npy
activation files already on disk.

Outputs
-------
``<out_dir>/rq2/fragmentation_results.csv`` — one row per (layer, A, B, C)
containing observed fragmentation scores, null statistics, bootstrap CIs, and
BH-FDR corrected p-values.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.analysis.significance import bh_fdr_correction, compute_p_value
from src.analysis.transmodal import (
    compute_residual,
    compute_residual_fast,
    frag_cka,
    frag_global,
    precompute_gram_inverse,
    sim_raw,
)
from src.utils.io import write_table

log = logging.getLogger(__name__)

# All CKA, OLS residuals, PWCCA and kNN are computed in the top-PCA_DIM
# principal subspace of each modality.  PCA is fit once per modality per
# layer and reused across all triplets, null draws, and bootstrap iterations.
# This reduces the dominant O(N × D²) matrix-multiply cost by (512/64)² = 64×,
# bringing total runtime from ~7 h to ~15 min with n_null_draws=50, n_boots=25.
PCA_DIM: int = 64

# Canonical order of the three modalities for stable pair naming.
_MODALITIES = ("depth", "normals", "rgb")

# All six directed triplets (A, B, C) — C is the witness.
TRIPLETS: list[tuple[str, str, str]] = [
    ("rgb",     "depth",   "normals"),
    ("depth",   "rgb",     "normals"),
    ("rgb",     "normals", "depth"),
    ("normals", "rgb",     "depth"),
    ("depth",   "normals", "rgb"),
    ("normals", "depth",   "rgb"),
]

_RESULT_COLUMNS = [
    "run_id", "layer", "A", "B", "C",
    # Global fragmentation
    "frag_global", "frag_global_ci_lo", "frag_global_ci_hi",
    # CKA-based fragmentation (primary, normalised ratio)
    "frag_cka", "denom_cka", "denom_cka_valid",
    "num_cka",
    "frag_cka_null_mean", "frag_cka_null_std",
    "frag_cka_z", "frag_cka_p", "frag_cka_p_adj",
    "frag_cka_ci_lo", "frag_cka_ci_hi",
    # PWCCA raw triangulation
    "sim_r_pwcca", "sim_b_pwcca",
    "sim_r_pwcca_ci_lo", "sim_r_pwcca_ci_hi",
    # kNN raw triangulation
    "sim_r_knn", "sim_b_knn",
    "sim_r_knn_ci_lo", "sim_r_knn_ci_hi",
    "n_samples",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _canonical_pair(m1: str, m2: str) -> str:
    """Return the canonical pair name (alphabetical order)."""
    return "-".join(sorted([m1, m2]))


def load_trimodal_index(run_dir: Path) -> pd.DataFrame:
    """Merge the three RQ1 activation index CSVs into a single trimodal table.

    Each row of the output corresponds to one (scene_id, sample_key, layer)
    triple with columns ``path_rgb``, ``path_depth``, and ``path_normals``
    pointing to the three .npy files.

    Parameters
    ----------
    run_dir:
        Root directory of the RQ1 run (contains ``artifacts/``).

    Returns
    -------
    pd.DataFrame with columns:
        run_id, layer, scene_id, sample_key, path_rgb, path_depth, path_normals
    """
    artifacts = run_dir / "artifacts"
    expected_pairs = {
        "rgb": ("rgb-depth", "rgb"),
        "depth": ("rgb-depth", "depth"),
        "normals": ("rgb-normals", "normals"),
    }

    frames: dict[str, pd.DataFrame] = {}
    for mod, (pair_name, modality) in expected_pairs.items():
        csv_path = artifacts / f"activation_index_{pair_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Activation index not found: {csv_path}. "
                "Run the RQ1 benchmark first."
            )
        df = pd.read_csv(csv_path)
        df = df[df["modality"] == modality][
            ["run_id", "scene_id", "sample_key", "layer", "activation_path"]
        ].rename(columns={"activation_path": f"path_{mod}"})
        frames[mod] = df

    merged = frames["rgb"].merge(
        frames["depth"],
        on=["run_id", "scene_id", "sample_key", "layer"],
        how="inner",
    ).merge(
        frames["normals"],
        on=["run_id", "scene_id", "sample_key", "layer"],
        how="inner",
    )

    # Sanity check: all three modalities must have the same N per layer.
    counts = merged.groupby("layer").size()
    n_unique = counts.nunique()
    if n_unique != 1:
        log.warning(
            "Unequal sample counts across layers after trimodal merge: %s",
            counts.to_dict(),
        )

    return merged.reset_index(drop=True)


def stack_layer(
    trimodal_df: pd.DataFrame,
    layer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and stack activation vectors for one layer.

    Parameters
    ----------
    trimodal_df:
        Trimodal index as returned by :func:`load_trimodal_index`.
    layer:
        Layer name, e.g. ``"layer_05"``.

    Returns
    -------
    X_rgb, X_depth, X_normals:
        Activation matrices of shape (N, D).
    scene_ids:
        Array of scene ID strings, shape (N,), used for bootstrap resampling.
    """
    sub = trimodal_df[trimodal_df["layer"] == layer].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows found for layer={layer!r}.")

    def _load(col: str) -> np.ndarray:
        return np.stack([np.load(p) for p in sub[col].tolist()], axis=0).astype(
            np.float64
        )

    X_rgb     = _load("path_rgb")
    X_depth   = _load("path_depth")
    X_normals = _load("path_normals")
    scene_ids = sub["scene_id"].to_numpy(dtype=str)
    return X_rgb, X_depth, X_normals, scene_ids


# ---------------------------------------------------------------------------
# Null threshold from RQ1
# ---------------------------------------------------------------------------

def _rq1_null_thresholds(
    null_csv: Path,
    n_sigma: float = 2.0,
) -> dict[tuple[str, str], float]:
    """Compute CKA denom thresholds from the RQ1 null distribution.

    Returns a dict keyed by ``(canonical_pair, layer)`` whose value is
    ``null_mean_cka + n_sigma * null_std_cka``.  If the null CSV is missing,
    returns an empty dict (no thresholding applied).

    Parameters
    ----------
    null_csv:
        Path to ``null_distribution.csv`` from the RQ1 run.
    n_sigma:
        Number of standard deviations above the null mean (default 2).
    """
    if not null_csv.exists():
        log.warning("RQ1 null CSV not found at %s — no denom thresholding.", null_csv)
        return {}

    df = pd.read_csv(null_csv)
    cka_df = df[df["metric"] == "cka"]
    thresholds: dict[tuple[str, str], float] = {}
    for (pair, layer), group in cka_df.groupby(["pair", "layer"]):
        vals = group["null_value"].to_numpy(dtype=float)
        thresholds[(str(pair), str(layer))] = float(
            np.mean(vals) + n_sigma * np.std(vals)
        )
    return thresholds


# ---------------------------------------------------------------------------
# PCA pre-reduction helpers
# ---------------------------------------------------------------------------

def _pca_basis(X: np.ndarray, n_components: int = PCA_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA basis for X.

    Parameters
    ----------
    X:
        Activation matrix, shape (N, D).
    n_components:
        Number of principal components to retain.

    Returns
    -------
    mean:
        Column mean of X, shape (D,).
    vt:
        Top-k right singular vectors, shape (k, D).  Apply via
        ``(X - mean) @ vt.T`` to get the k-dim projection.
    """
    X64 = X.astype(np.float64, copy=False)
    mean = X64.mean(axis=0)
    centered = X64 - mean
    k = min(n_components, X64.shape[0] - 1, X64.shape[1])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:k]


def _apply_pca(X: np.ndarray, mean: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """Project X onto a precomputed PCA basis.

    Parameters
    ----------
    X:
        Activation matrix, shape (N, D).
    mean:
        Column mean used when the basis was computed, shape (D,).
    vt:
        Right singular vectors (principal axes), shape (k, D).

    Returns
    -------
    X_r:
        Projected matrix, shape (N, k).  Mean-zero by construction.
    """
    return (X.astype(np.float64, copy=False) - mean) @ vt.T


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def _bootstrap_triplet_fast(
    mods_r: dict[str, np.ndarray],
    scene_ids: np.ndarray,
    A_name: str,
    B_name: str,
    C_name: str,
    M: np.ndarray,
    n_boots: int,
    rng: np.random.Generator,
    denom_threshold: float | None,
) -> dict[str, np.ndarray]:
    """Fast scene-level bootstrap using precomputed PCA reduction and inverse Gram.

    All matrices in ``mods_r`` are expected to be PCA-reduced (mean-zero, shape
    ``(N, PCA_DIM)``).  Because the column mean is already zero, no extra
    centring step is needed before calling ``compute_residual_fast``.

    The OLS is approximated by reusing the inverse Gram ``M`` from the full
    dataset.  Bootstrap-resample Grams converge to the population Gram, so the
    approximation error is O(1/√N) ≈ 1.6% for N=4000.  CKA is computed on the
    PCA-reduced representations (PCA_DIM=64 dimensions), consistent with all
    other metrics in the pipeline.

    PWCCA and kNN are excluded from the bootstrap loop.  Their SVD cost scales
    as O(N·k²) even for k=PCA_DIM and would dominate runtime.  Their
    ``ci_lo/ci_hi`` columns are reported as NaN; only observed values matter for
    the directional triangulation claims.

    Parameters
    ----------
    mods_r:
        Dict ``{"rgb": X_rgb_r, ...}`` of PCA-reduced, mean-zero matrices,
        each shape ``(N, PCA_DIM)``.
    scene_ids:
        Scene ID per sample, shape (N,).
    A_name, B_name, C_name:
        Modality names for the directed triplet.
    M:
        Precomputed ``(A_r.T @ A_r + λI)^{-1}``, shape (PCA_DIM, PCA_DIM).
    n_boots:
        Number of bootstrap iterations.
    rng:
        Numpy random generator (shared, advances in place).
    denom_threshold:
        Minimum CKA(B, C) required for a valid frag_cka ratio.

    Returns
    -------
    dict of arrays, each shape (n_boots,):
        ``frag_global``, ``frag_cka`` — bootstrapped values.
        ``sim_r_pwcca``, ``sim_b_pwcca``, ``sim_r_knn``, ``sim_b_knn``
        — all NaN (not bootstrapped; observed values reported separately).
    """
    unique_scenes = np.unique(scene_ids)
    frag_global_boot = np.full(n_boots, np.nan)
    frag_cka_boot    = np.full(n_boots, np.nan)

    for i in range(n_boots):
        sampled_scenes = rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
        idx = np.concatenate([np.where(scene_ids == s)[0] for s in sampled_scenes])
        if len(idx) < 4:
            continue

        # mods_r entries are already mean-zero (PCA-centred on the full dataset).
        XA_c = mods_r[A_name][idx]
        XB_c = mods_r[B_name][idx]
        XC_b = mods_r[C_name][idx]

        # Fast OLS residual: reuse precomputed M (approximation for resampled data).
        R = compute_residual_fast(XA_c, XB_c, M)

        norm_b = float(np.sum(XB_c ** 2))
        if norm_b > 1e-12:
            frag_global_boot[i] = float(np.sum(R ** 2) / norm_b)

        denom = float(linear_cka_numpy(XB_c, XC_b))
        if (denom_threshold is None or denom >= denom_threshold) and denom > 1e-12:
            frag_cka_boot[i] = float(linear_cka_numpy(R, XC_b)) / denom

    nan_arr = np.full(n_boots, np.nan)
    return {
        "frag_global":  frag_global_boot,
        "frag_cka":     frag_cka_boot,
        "sim_r_pwcca":  nan_arr,
        "sim_b_pwcca":  nan_arr,
        "sim_r_knn":    nan_arr,
        "sim_b_knn":    nan_arr,
    }


def _ci(arr: np.ndarray) -> tuple[float, float]:
    """Return 2.5 and 97.5 percentiles, ignoring NaN."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_rq2_triangulation(
    run_dir: Path,
    out_dir: Path,
    n_null_draws: int = 1000,
    n_boots: int = 200,
    ridge: float = 1e-4,
    seed: int = 42,
    null_csv: Path | None = None,
) -> pd.DataFrame:
    """Run the full RQ2 fragmentation analysis for all triplets and layers.

    Parameters
    ----------
    run_dir:
        Root directory of the RQ1 run.
    out_dir:
        Output directory; results are written to ``out_dir/rq2/``.
    n_null_draws:
        Number of row-permutation draws for the null distribution of frag_cka.
    n_boots:
        Number of bootstrap iterations for confidence intervals.  Can be
        reduced to 200 if runtime is a bottleneck.
    ridge:
        Relative ridge coefficient for OLS regularisation.
    seed:
        Random seed for null draws and bootstrap.
    null_csv:
        Optional explicit path to the RQ1 null distribution CSV.  If None,
        the pipeline searches for it at the canonical location relative to
        ``run_dir``.

    Returns
    -------
    pd.DataFrame with schema matching ``_RESULT_COLUMNS``.
    """
    rng = np.random.default_rng(seed)
    run_id = run_dir.name

    # Locate RQ1 null distribution for denom thresholds.
    if null_csv is None:
        _candidate = (
            run_dir.parent.parent
            / "null_distributions"
            / run_id
            / "null_distribution.csv"
        )
        null_csv = _candidate if _candidate.exists() else Path("__nonexistent__")

    denom_thresholds = _rq1_null_thresholds(null_csv)

    log.info("Loading trimodal index from %s", run_dir)
    trimodal_df = load_trimodal_index(run_dir)
    layers = sorted(trimodal_df["layer"].unique().tolist())

    rows: list[dict[str, Any]] = []
    timed_first = False

    for layer in tqdm(layers, desc="RQ2 layers", unit="layer"):
        X_rgb, X_depth, X_normals, scene_ids = stack_layer(trimodal_df, layer)
        mods: dict[str, np.ndarray] = {
            "rgb": X_rgb, "depth": X_depth, "normals": X_normals,
        }

        # Pre-compute PCA reduction to PCA_DIM dimensions once per modality.
        # All subsequent computation (CKA, OLS, PWCCA, kNN) operates in this
        # subspace.  The 3 SVDs here cost ~1 s total (N=4000, D=512) and
        # reduce the dominant matrix-multiply cost by (512/64)² = 64×.
        _no_pca_cfg: dict = {"pca": {"enabled": False}}
        mods_r: dict[str, np.ndarray] = {}
        for mod_name, X in mods.items():
            mean_m, vt_m = _pca_basis(X, n_components=PCA_DIM)
            mods_r[mod_name] = _apply_pca(X, mean_m, vt_m)

        for triplet_idx, (A_name, B_name, C_name) in enumerate(TRIPLETS):
            XA_r = mods_r[A_name]
            XB_r = mods_r[B_name]
            XC_r = mods_r[C_name]

            # Denom threshold for this B–C pair from the RQ1 null.
            bc_pair = _canonical_pair(B_name, C_name)
            denom_threshold = denom_thresholds.get((bc_pair, layer))

            # ---- Timer on first iteration ----
            if not timed_first:
                t0 = time.perf_counter()

            # ---- Observed scores (all in PCA_DIM-dimensional subspace) ----
            obs_fg = frag_global(XA_r, XB_r, ridge=ridge)
            obs_fc = frag_cka(XA_r, XB_r, XC_r, ridge=ridge, denom_threshold=denom_threshold)

            # Compute residual once; reuse for PWCCA/kNN.  Pass _no_pca_cfg so
            # PWCCA and kNN skip internal PCA (matrices are already reduced).
            R_obs, _, B_c = compute_residual(XA_r, XB_r, ridge=ridge)
            obs_sim_b_pwcca = sim_raw(XB_r, XC_r, "pwcca", cfg=_no_pca_cfg)
            obs_sim_b_knn   = sim_raw(XB_r, XC_r, "knn",   cfg=_no_pca_cfg)
            obs_sim_r_pwcca = sim_raw(R_obs, XC_r, "pwcca", cfg=_no_pca_cfg)
            obs_sim_r_knn   = sim_raw(R_obs, XC_r, "knn",   cfg=_no_pca_cfg)

            # ---- Null distribution for frag_cka (optimised Gram path) ----
            # denom_n = CKA(B_r, C_r) is constant across null draws; compute once.
            A_c, M = precompute_gram_inverse(XA_r, ridge=ridge)
            denom_n = float(linear_cka_numpy(XB_r, XC_r))
            null_frags: list[float] = []
            perm_rng = np.random.default_rng(int(rng.integers(1 << 31)))
            for _ in range(n_null_draws):
                perm = perm_rng.permutation(len(XA_r))
                A_perm = A_c[perm]  # already centred; permutation preserves mean
                R_null = compute_residual_fast(A_perm, B_c, M)
                num_n  = float(linear_cka_numpy(R_null, XC_r))
                null_frags.append(
                    (num_n / denom_n) if (
                        denom_n > 1e-12
                        and obs_fc["valid"]
                    ) else float("nan")
                )

            null_arr = np.array(null_frags, dtype=float)
            valid_null = null_arr[np.isfinite(null_arr)]
            null_mean = float(np.mean(valid_null)) if len(valid_null) > 0 else float("nan")
            null_std  = float(np.std(valid_null))  if len(valid_null) > 0 else float("nan")

            obs_frag = obs_fc["frag_cka"]
            if np.isfinite(obs_frag) and len(valid_null) > 0:
                p_val = compute_p_value(obs_frag, valid_null, side="less")
                z_score = (
                    (obs_frag - null_mean) / null_std
                    if null_std > 0
                    else float("nan")
                )
            else:
                p_val = float("nan")
                z_score = float("nan")

            # ---- Bootstrap CIs (fast approximation via precomputed Gram) ----
            # mods_r matrices are already PCA-reduced and mean-zero.
            # M was computed above for the null loop and is reused here.
            boot_results = _bootstrap_triplet_fast(
                mods_r=mods_r,
                scene_ids=scene_ids,
                A_name=A_name,
                B_name=B_name,
                C_name=C_name,
                M=M,
                n_boots=n_boots,
                rng=rng,
                denom_threshold=denom_threshold,
            )

            if not timed_first:
                elapsed = time.perf_counter() - t0
                timed_first = True
                log.info(
                    "RQ2 first-iteration timing (layer=%s, triplet=%s→%s|%s): %.1fs. "
                    "Estimated total: %.0fs.",
                    layer, A_name, B_name, C_name, elapsed,
                    elapsed * len(layers) * len(TRIPLETS),
                )
                if elapsed > 30:
                    log.warning(
                        "First iteration took %.1fs > 30s.  Expected ~2-5s after "
                        "PCA pre-reduction.  Check that mods_r has shape (N, %d).",
                        elapsed, PCA_DIM,
                    )

            rows.append({
                "run_id": run_id,
                "layer": layer,
                "A": A_name,
                "B": B_name,
                "C": C_name,
                "frag_global": obs_fg,
                "frag_global_ci_lo": _ci(boot_results["frag_global"])[0],
                "frag_global_ci_hi": _ci(boot_results["frag_global"])[1],
                "frag_cka": obs_frag,
                "denom_cka": obs_fc["denom_cka"],
                "denom_cka_valid": obs_fc["valid"],
                "num_cka": obs_fc["num_cka"],
                "frag_cka_null_mean": null_mean,
                "frag_cka_null_std": null_std,
                "frag_cka_z": z_score,
                "frag_cka_p": p_val,
                "frag_cka_p_adj": float("nan"),  # filled after BH-FDR below
                "frag_cka_ci_lo": _ci(boot_results["frag_cka"])[0],
                "frag_cka_ci_hi": _ci(boot_results["frag_cka"])[1],
                "sim_r_pwcca": obs_sim_r_pwcca,
                "sim_b_pwcca": obs_sim_b_pwcca,
                "sim_r_pwcca_ci_lo": _ci(boot_results["sim_r_pwcca"])[0],
                "sim_r_pwcca_ci_hi": _ci(boot_results["sim_r_pwcca"])[1],
                "sim_r_knn": obs_sim_r_knn,
                "sim_b_knn": obs_sim_b_knn,
                "sim_r_knn_ci_lo": _ci(boot_results["sim_r_knn"])[0],
                "sim_r_knn_ci_hi": _ci(boot_results["sim_r_knn"])[1],
                "n_samples": len(XA_r),
            })

    results_df = pd.DataFrame.from_records(rows, columns=_RESULT_COLUMNS)

    # ---- BH-FDR correction across all frag_cka p-values ----
    valid_mask = results_df["frag_cka_p"].notna()
    if valid_mask.any():
        pvals = results_df.loc[valid_mask, "frag_cka_p"].to_numpy(dtype=float)
        _, p_adj = bh_fdr_correction(pvals, alpha=0.05)
        results_df.loc[valid_mask, "frag_cka_p_adj"] = p_adj

    # ---- Write output ----
    out_rq2 = out_dir / "rq2"
    out_rq2.mkdir(parents=True, exist_ok=True)
    out_path = out_rq2 / "fragmentation_results.csv"
    write_table(out_path, results_df)
    log.info("RQ2 results written to %s (%d rows)", out_path, len(results_df))

    return results_df


# ---------------------------------------------------------------------------
# Local CKA numpy helper (avoids overhead of auto-dispatch in null loop)
# ---------------------------------------------------------------------------

def linear_cka_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Lightweight CKA using the numpy path directly (for null loop speed)."""
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    x64 = x64 - x64.mean(axis=0, keepdims=True)
    y64 = y64 - y64.mean(axis=0, keepdims=True)
    xty = x64.T @ y64
    xtx = x64.T @ x64
    yty = y64.T @ y64
    hsic  = float(np.sum(xty * xty))
    norm_x = float(np.sqrt(np.sum(xtx * xtx)))
    norm_y = float(np.sqrt(np.sum(yty * yty)))
    denom = max(norm_x * norm_y, 1e-12)
    return hsic / denom
