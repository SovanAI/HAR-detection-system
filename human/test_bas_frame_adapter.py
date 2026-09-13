"""
Test BASFrameAdapter using a real Person + Hand + HMR fusion JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from human.bas_frame_adapter import BASFrameAdapter


INPUT_FILE = (
    ROOT
    / "test_results"
    / "person_hand_hmr_fusion"
    / "fused_human_frame.json"
)

OUTPUT_FILE = (
    ROOT
    / "test_results"
    / "person_hand_hmr_fusion"
    / "bas_canonical_frame.json"
)


def main():

    print("=" * 60)
    print("BAS FRAME ADAPTER TEST")
    print("=" * 60)

    print(f"\nInput:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        print("\nERROR: Fusion JSON does not exist.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        fused_data = json.load(f)

    print("\nInput persons:")
    print(len(fused_data.get("persons", [])))

    adapter = BASFrameAdapter()

    canonical = adapter.adapt(fused_data)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            canonical,
            f,
            indent=2
        )

    print("\nCanonical representation:")
    print(
        f"  Frame ID: {canonical['frame_id']}"
    )

    print(
        f"  Persons: "
        f"{len(canonical['persons'])}"
    )

    print(
        f"  YOLO persons: "
        f"{len(canonical['yolo']['persons'])}"
    )

    print(
        f"  HMR persons: "
        f"{len(canonical['hmr']['persons'])}"
    )

    print(
        f"  Hands: "
        f"{len(canonical['hands']['hands'])}"
    )

    print(
        f"  Unassociated hands: "
        f"{len(canonical['unassociated_hands'])}"
    )

    print("\nSchema checks:")

    assert "yolo" in canonical
    assert "hmr" in canonical
    assert "hands" in canonical
    assert "persons" in canonical

    assert len(
        canonical["persons"]
    ) == len(
        canonical["yolo"]["persons"]
    )

    assert len(
        canonical["persons"]
    ) == len(
        canonical["hmr"]["persons"]
    )

    print("  YOLO schema: PASS")
    print("  HMR schema: PASS")
    print("  Hand schema: PASS")
    print("  Unified schema: PASS")

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("BAS FRAME ADAPTER TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
