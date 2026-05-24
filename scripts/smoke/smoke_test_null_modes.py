"""Smoke test for all 4 null sampling modes.

Run with:  python scripts/smoke/smoke_test_null_modes.py
Verifies that build_right_candidates_by_left_index returns the expected
candidates for each mode on a tiny synthetic DIODE-shaped index.
"""

from __future__ import annotations

import numpy as np

from src.analysis.null_sampling import (
    MODES_NEEDING_TYPE_MAP,
    SUPPORTED_SAMPLING_MODES,
    build_right_candidates_by_left_index,
)


def main() -> None:
    # Layout: 2 DIODE-scenes, 2 scans each, 2 frames per scan → 8 rows.
    scene_ids = np.array([
        "indoors/scene_00021/scan_00189",  # 0
        "indoors/scene_00021/scan_00189",  # 1
        "indoors/scene_00021/scan_00190",  # 2
        "indoors/scene_00021/scan_00190",  # 3
        "indoors/scene_00022/scan_00191",  # 4
        "indoors/scene_00022/scan_00191",  # 5
        "indoors/scene_00022/scan_00192",  # 6
        "indoors/scene_00022/scan_00192",  # 7
    ])
    sample_keys = np.array(["frame_a", "frame_b"] * 4)
    type_map = {sid: sid.rsplit("/", 1)[0] for sid in set(scene_ids)}

    left_idx = 0  # scan_00189 / frame_a / DIODE-scene indoors/scene_00021
    print(f"LEFT[{left_idx}]: scene={scene_ids[left_idx]}, key={sample_keys[left_idx]}")
    print(f"          parent: {type_map[scene_ids[left_idx]]}\n")

    expected = {
        "cross_scene_type_random":  {4, 5, 6, 7},
        "within_scene_type_random": {2, 3},
        "within_scene_random":      {1},
    }

    failures = 0
    for mode in sorted(SUPPORTED_SAMPLING_MODES):
        cands = build_right_candidates_by_left_index(
            left_scene_ids=scene_ids,
            right_scene_ids=scene_ids,
            sampling_mode=mode,
            scene_type_map=type_map if mode in MODES_NEEDING_TYPE_MAP else {},
            left_sample_keys=sample_keys,
            right_sample_keys=sample_keys,
        )
        got = set(cands.get(left_idx, np.array([], dtype=int)).tolist())
        ok = got == expected[mode]
        print(f"  {mode:30s}  got={sorted(got)}  expected={sorted(expected[mode])}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    print()
    print(f"{'ALL PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")


if __name__ == "__main__":
    main()
