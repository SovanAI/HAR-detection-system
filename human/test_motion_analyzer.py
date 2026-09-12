import glob
import json
import os

from human.state_extractor import HumanStateExtractor
from human.motion_analyzer import MotionAnalyzer


def main():

    files = sorted(
        glob.glob(
            "output/*_fused.json"
        )
    )

    if len(files) < 2:
        raise RuntimeError(
            "At least two fused JSON files are required."
        )

    extractor = HumanStateExtractor()
    analyzer = MotionAnalyzer()

    print("=" * 70)
    print("HUMAN MOTION ANALYSIS TEST")
    print("=" * 70)

    for path in files:

        with open(path, "r") as f:
            fused = json.load(f)

        state = extractor.extract_from_fused(
            fused
        )

        motion = analyzer.analyze(
            state
        )

        print()
        print(
            f"Frame {motion['frame_id']} "
            f"| timestamp={motion['timestamp']:.3f}"
        )

        for person in motion["persons"]:

            person_id = person["person_id"]
            data = person["motion"]

            print(
                f"  Person {person_id}"
            )

            print(
                f"    movement: "
                f"{data.get('movement_state')}"
            )

            if data.get("dt") is not None:
                print(
                    f"    dt: "
                    f"{data['dt']:.3f}s"
                )

            if "left_hand_speed" in data:
                print(
                    f"    left hand speed: "
                    f"{data['left_hand_speed']:.4f}"
                )

            if "right_hand_speed" in data:
                print(
                    f"    right hand speed: "
                    f"{data['right_hand_speed']:.4f}"
                )

            if "pelvis_speed" in data:
                print(
                    f"    pelvis speed: "
                    f"{data['pelvis_speed']:.4f}"
                )

            if "average_joint_speed" in data:
                print(
                    f"    average joint speed: "
                    f"{data['average_joint_speed']:.4f}"
                )

    print()
    print("=" * 70)
    print("MOTION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
