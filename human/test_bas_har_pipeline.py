"""
BAS End-to-End Human Activity Recognition Test

Pipeline:

Fusion JSON
    ↓
BASFrameAdapter
    ↓
HumanFeatureAnalyzer
    ↓
TemporalFeatureBuffer
    ↓
HAREngine
    ↓
Final Activity Result

This test uses an existing real HMR fusion frame.

Because only one real fusion frame is currently available,
the same frame is fed repeatedly to validate the complete
software pipeline. This is a SOFTWARE INTEGRATION TEST,
not an activity-accuracy test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from human.bas_frame_adapter import (
    BASFrameAdapter,
)

from human.feature_analyzer import (
    HumanFeatureAnalyzer,
)

from human.temporal_buffer import (
    TemporalFeatureBuffer,
)

from human.har_engine import (
    HAREngine,
)


INPUT_FILE = (
    ROOT
    / "test_results"
    / "person_hand_hmr_fusion"
    / "fused_human_frame.json"
)


OUTPUT_FILE = (
    ROOT
    / "test_results"
    / "bas_har_pipeline"
    / "final_har_result.json"
)


def main():

    print("=" * 70)
    print("BAS END-TO-END HUMAN ACTIVITY RECOGNITION TEST")
    print("=" * 70)

    # ==========================================================
    # Check input
    # ==========================================================

    if not INPUT_FILE.exists():

        print()
        print("ERROR:")
        print("Fusion JSON not found:")
        print(INPUT_FILE)

        sys.exit(1)

    # ==========================================================
    # Load real fusion result
    # ==========================================================

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        fused_data = json.load(f)

    print()
    print("Input fusion frame:")
    print(
        f"  Frame ID: {fused_data.get('frame_id')}"
    )

    print(
        f"  Persons: "
        f"{len(fused_data.get('persons', []))}"
    )

    # ==========================================================
    # Adapter
    # ==========================================================

    adapter = BASFrameAdapter()

    canonical = adapter.adapt(
        fused_data
    )

    print()
    print("BAS Frame Adapter:")
    print(
        f"  Persons: "
        f"{len(canonical['persons'])}"
    )

    # ==========================================================
    # Feature analyzer
    # ==========================================================

    analyzer = HumanFeatureAnalyzer()

    # ==========================================================
    # Temporal buffer
    # ==========================================================

    buffer = TemporalFeatureBuffer(
        max_length=30
    )

    # ==========================================================
    # IMPORTANT
    #
    # We only have one real fusion frame right now.
    #
    # To test the complete software chain, we process it
    # repeatedly.
    #
    # This does NOT simulate real motion.
    # ==========================================================

    print()
    print("Building temporal feature buffer...")

    for index in range(30):

        # Give each integration frame a unique ID/time
        # so that the temporal pipeline can be exercised.

        test_frame = json.loads(
            json.dumps(canonical)
        )

        test_frame["frame_id"] = index

        test_frame["timestamp"] = (
            float(
                canonical["timestamp"]
            )
            + index * 0.1
        )

        features = analyzer.process_frame(
            test_frame
        )

        buffer.add(
            features
        )

        if index == 0:

            print(
                "  Frame 1/30 processed"
            )

        elif index == 9:

            print(
                "  Frame 10/30 processed"
            )

        elif index == 19:

            print(
                "  Frame 20/30 processed"
            )

        elif index == 29:

            print(
                "  Frame 30/30 processed"
            )

    # ==========================================================
    # Buffer validation
    # ==========================================================

    print()
    print("Temporal Buffer:")

    print(
        f"  Size: {buffer.size()}"
    )

    print(
        f"  Ready: {buffer.is_ready()}"
    )

    assert buffer.size() == 30

    assert buffer.is_ready()

    sequence = buffer.get_all()

    # ==========================================================
    # HAR
    # ==========================================================

    har_engine = HAREngine()

    har_result = har_engine.classify(
        sequence
    )

    # ==========================================================
    # Display results
    # ==========================================================

    print()
    print("HAR RESULT:")
    print()

    print(
        f"  Start frame: "
        f"{har_result.get('start_frame')}"
    )

    print(
        f"  End frame: "
        f"{har_result.get('end_frame')}"
    )

    print(
        f"  Sequence length: "
        f"{har_result.get('sequence_length')}"
    )

    print()

    for result in har_result.get(
        "persons",
        [],
    ):

        print(
            f"  Person {result['person_id']}:"
        )

        print(
            f"    Activity: "
            f"{result['activity']}"
        )

        print(
            f"    Confidence: "
            f"{result['confidence']}"
        )

        print(
            f"    Reason: "
            f"{result['reason']}"
        )

    # ==========================================================
    # Output
    # ==========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            har_result,
            f,
            indent=2,
        )

    # ==========================================================
    # Final assertions
    # ==========================================================

    assert "persons" in har_result

    assert len(
        har_result["persons"]
    ) == len(
        canonical["persons"]
    )

    # ==========================================================
    # PASS
    # ==========================================================

    print()
    print("=" * 70)
    print("BAS END-TO-END HAR PIPELINE: PASS")
    print("=" * 70)

    print()
    print("Output:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
