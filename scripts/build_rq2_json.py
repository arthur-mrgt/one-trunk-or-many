"""Build website/assets/metrics/rq2.json from the RQ2 fragmentation summary CSV.

Reads the (pair, layer)-level fragmentation summary produced by `notebooks/rq2_analysis.ipynb`
and serialises it into the JSON payload consumed by the interactive Plotly chart in
`website/index.html` (section 4.7 -- Cross-modal unification, transmodal triangulation).

Only the fragmentation block is emitted; the directional-asymmetry CSV is no longer
surfaced on the site (Table 3 already exposes the layer-11 directional values).

Run from the repo root:

    python scripts/build_rq2_json.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAG_CSV = REPO_ROOT / "notebooks" / "rq2_rq1_final_hypersim-20260525-000411_fragmentation_summary.csv"
OUT_JSON = REPO_ROOT / "website" / "assets" / "metrics" / "rq2.json"

PAIR_ORDER = ["depth-rgb", "normals-rgb", "depth-normals"]

L11_SUMMARY = {
    "depth-rgb":     {"frag": 0.086, "unified_pct": 91},
    "normals-rgb":   {"frag": 0.125, "unified_pct": 87},
    "depth-normals": {"frag": 0.194, "unified_pct": 81},
}


def _round(value: float, ndigits: int = 4) -> float:
    return round(float(value), ndigits)


def build_payload() -> dict:
    df = pd.read_csv(FRAG_CSV)

    layers = sorted(int(x) for x in df["layer_idx"].unique())
    assert layers == list(range(12)), f"Expected layers 0..11, got {layers}"

    fragmentation: dict[str, list[dict[str, float]]] = {}
    for pair in PAIR_ORDER:
        rows = df[df["pair"] == pair].sort_values("layer_idx")
        assert len(rows) == 12, f"Pair {pair!r}: expected 12 rows, got {len(rows)}"
        fragmentation[pair] = [
            {
                "mean":  _round(r.frag_cka_mean),
                "ci_lo": _round(r.frag_cka_ci_lo),
                "ci_hi": _round(r.frag_cka_ci_hi),
                "p_adj": _round(r.frag_cka_p_adj_min, 6),
            }
            for r in rows.itertuples(index=False)
        ]

    return {
        "dataset": "hypersim",
        "pairs": PAIR_ORDER,
        "layers": layers,
        "fragmentation": fragmentation,
        "l11_summary": L11_SUMMARY,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    n_rows = sum(len(v) for v in payload["fragmentation"].values())
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)} ({n_rows} pair*layer rows).")


if __name__ == "__main__":
    main()
