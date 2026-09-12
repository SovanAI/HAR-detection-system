import glob
import json

from human.feature_analyzer import HumanFeatureAnalyzer
from human.temporal_buffer import TemporalFeatureBuffer
from human.har_engine import HAREngine


def main():

    files = sorted(
        glob.glob(
            "output/*_fused.json"
        )
    )

    if len(files) < 8:
        raise RuntimeError(
            "At least 8 fused frames are required."
        )

    analyzer = HumanFeatureAnalyzer()

    buffer = TemporalFeatureBuffer(
        max_length=8
    )

    har = HAREngine()

    print("=" * 70)
    print("HAR ENGINE TEST")
    print("=" * 70)

    for path in files:

        with open(
            path,
            "r"
        ) as f:
            fused = json.load(f)

        features = analyzer.process_frame(
            fused
        )

        buffer.add(
            features
        )

        if not buffer.is_ready():
            continue

        sequence = buffer.get_all()

        result = har.classify(
            sequence
        )

        print()
        print(
            f"Window: "
            f"{result['start_frame']} "
            f"-> "
            f"{result['end_frame']}"
        )

        for person in result["persons"]:

            print(
                f"  Person "
                f"{person['person_id']}"
            )

            print(
                f"    Activity: "
                f"{person['activity']}"
            )

            print(
                f"    Confidence: "
                f"{person['confidence']:.2f}"
            )

            print(
                f"    Reason: "
                f"{person['reason']}"
            )

    print()
    print("=" * 70)
    print("HAR ENGINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
