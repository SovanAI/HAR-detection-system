"""
Test:

Fusion JSON
    ↓
BASFrameAdapter
    ↓
HumanStateExtractor
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from human.bas_frame_adapter import BASFrameAdapter
from human.state_extractor import HumanStateExtractor


INPUT_FILE = (
    ROOT
    / "test_results"
    / "person_hand_hmr_fusion"
    / "fused_human_frame.json"
)


def main():

    print("=" * 60)
    print("BAS HUMAN STATE PIPELINE TEST")
    print("=" * 60)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        fused_data = json.load(f)

    # ----------------------------------------------------------
    # Step 1
    # ----------------------------------------------------------

    adapter = BASFrameAdapter()

    canonical = adapter.adapt(
        fused_data
    )

    print(
        f"\nCanonical persons: "
        f"{len(canonical['persons'])}"
    )

    # ----------------------------------------------------------
    # Step 2
    # ----------------------------------------------------------

    extractor = HumanStateExtractor()

    state = extractor.extract_from_fused(
        canonical
    )

    print(
        f"Extracted persons: "
        f"{len(state['persons'])}"
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    assert len(state["persons"]) == len(
        canonical["persons"]
    )

    for person in state["persons"]:

        print(
            f"\nPerson {person['person_id']}"
        )

        print(
            "  Camera translation:",
            person["camera_translation"]
        )

        print(
            "  Joint count:",
            len(person["joints_3d"])
        )

        assert len(
            person["joints_3d"]
        ) == 44

    print("\n" + "=" * 60)
    print("HUMAN STATE PIPELINE: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
