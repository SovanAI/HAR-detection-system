import glob
import json
import os

from human.feature_analyzer import HumanFeatureAnalyzer
from human.temporal_buffer import TemporalFeatureBuffer


def main():

    files = sorted(
        glob.glob(
            "output/*_fused.json"
        )
    )

    if len(files) == 0:
        raise RuntimeError(
            "No fused JSON files found."
        )

    analyzer = HumanFeatureAnalyzer()

    buffer = TemporalFeatureBuffer(
        max_length=8
    )

    print("=" * 70)
    print("TEMPORAL FEATURE BUFFER TEST")
    print("=" * 70)

    for path in files:

        with open(path, "r") as f:
            fused = json.load(f)

        features = analyzer.process_frame(
            fused
        )

        buffer.add(
            features
        )

        print(
            f"Added frame "
            f"{features['frame_id']} | "
            f"buffer={buffer.size()}/8 | "
            f"ready={buffer.is_ready()}"
        )

        if buffer.is_ready():

            sequence = buffer.get_all()

            first_frame = sequence[0]["frame_id"]
            last_frame = sequence[-1]["frame_id"]

            print(
                f"  HAR window: "
                f"{first_frame} -> {last_frame}"
            )

    print()
    print("=" * 70)

    print(
        f"Final buffer size: "
        f"{buffer.size()}"
    )

    print(
        f"Final buffer ready: "
        f"{buffer.is_ready()}"
    )

    print(
        "Frames currently in buffer:"
    )

    for frame in buffer.get_all():

        print(
            f"  {frame['frame_id']}"
        )

    print("=" * 70)
    print("TEMPORAL BUFFER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
