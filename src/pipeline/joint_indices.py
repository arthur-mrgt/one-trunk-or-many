"""Build derived activation index files for joint-pass comparisons.

Given the activation indices produced by

  * single-modality extraction for one pair (A, B)  — ``run_extraction``
  * the joint forward pass for the same pair (A, B) — ``run_joint_extraction``

this module emits the three derived comparison indices the metrics engine
needs to compute CKA / PWCCA / kNN on the joint-pass slices:

  1. (A, A_joint_with_B)            — does joint processing shift A's
                                       representation vs the single-mod pass?
  2. (B, B_joint_with_A)            — same question, for B.
  3. (A_joint_with_B, B_joint_with_A)— are the two slices more aligned after
                                       joint processing than they were before?

The comparison indices reference the existing ``.npy`` files on disk
verbatim — no copying. Only the ``pair`` and (for joint-only rows) the
``modality`` columns are rewritten so the metric engine's merge succeeds.

Filenames follow the convention ``activation_index_<left>-<right>.csv`` so
that the regular ``compute_metric_table_from_indices`` glob picks them up
without any code changes.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.pipeline.extraction import joint_slice_modality_name


class ComparisonIndex(NamedTuple):
    """One derived activation index file to write."""

    pair_name: str       # e.g. "rgb-rgb_joint_with_depth"
    left_modality: str   # e.g. "rgb"
    right_modality: str  # e.g. "rgb_joint_with_depth"
    table: pd.DataFrame


def _retag(
    df: pd.DataFrame,
    new_pair: str,
    *,
    new_modality: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``pair`` (and optionally ``modality``) rewritten."""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["pair"] = new_pair
    if new_modality is not None:
        out["modality"] = new_modality
    return out


def build_joint_comparison_indices(
    *,
    single_mod_index: pd.DataFrame,
    joint_index: pd.DataFrame,
    left_modality: str,
    right_modality: str,
) -> list[ComparisonIndex]:
    """Build the three comparison indices for one joint pair.

    Parameters
    ----------
    single_mod_index:
        ``activation_index_<A>-<B>.csv`` content for the same ``(A, B)`` pair.
    joint_index:
        Activation index returned by :func:`run_joint_extraction` for the
        same ``(A, B)`` pair. ``modality`` values are expected to be the
        synthetic ``<m>_joint_with_<p>`` names produced by that function.
    left_modality, right_modality:
        Same ``(A, B)`` as the single-mod extraction call. Order is preserved
        in the comparison file names so output stays deterministic.
    """
    if left_modality == right_modality:
        raise ValueError("left and right modalities must differ")

    a, b = left_modality, right_modality
    a_joint = joint_slice_modality_name(a, b)
    b_joint = joint_slice_modality_name(b, a)

    # 1) single-mod A rows  vs  joint A-slice rows
    single_a = single_mod_index[single_mod_index["modality"] == a]
    joint_a = joint_index[joint_index["modality"] == a_joint]
    pair_a = f"{a}-{a_joint}"
    left_a = _retag(single_a, new_pair=pair_a)
    right_a = _retag(joint_a, new_pair=pair_a)
    table_a = pd.concat([left_a, right_a], ignore_index=True) if not (left_a.empty and right_a.empty) else pd.DataFrame()

    # 2) single-mod B rows  vs  joint B-slice rows
    single_b = single_mod_index[single_mod_index["modality"] == b]
    joint_b = joint_index[joint_index["modality"] == b_joint]
    pair_b = f"{b}-{b_joint}"
    left_b = _retag(single_b, new_pair=pair_b)
    right_b = _retag(joint_b, new_pair=pair_b)
    table_b = pd.concat([left_b, right_b], ignore_index=True) if not (left_b.empty and right_b.empty) else pd.DataFrame()

    # 3) joint A-slice  vs  joint B-slice  (cross-modality alignment in the joint pass)
    pair_cross = f"{a_joint}-{b_joint}"
    table_cross = (
        pd.concat([_retag(joint_a, new_pair=pair_cross), _retag(joint_b, new_pair=pair_cross)], ignore_index=True)
        if not (joint_a.empty and joint_b.empty)
        else pd.DataFrame()
    )

    return [
        ComparisonIndex(pair_name=pair_a, left_modality=a, right_modality=a_joint, table=table_a),
        ComparisonIndex(pair_name=pair_b, left_modality=b, right_modality=b_joint, table=table_b),
        ComparisonIndex(pair_name=pair_cross, left_modality=a_joint, right_modality=b_joint, table=table_cross),
    ]
