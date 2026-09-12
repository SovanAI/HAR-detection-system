import glob
import os

from human.feature_analyzer import (
    HumanFeatureAnalyzer,
    load_fused_file,
    save_features,
)


def main():

    files = sorted(
        glob.glob(
            "output/*_fused.json"
        )
    )

    if len(files) < 2:
        raise RuntimeError(
            "At least two fused frames are required."
        )

    analyzer = HumanFeatureAnalyzer()

    os.makedirs(
        "output/features",
        exist_ok=True,
    )

    print("=" * 70)
    print("COMBINED HUMAN FEATURE ANALYSIS")
    print("=" * 70)

    for path in files:

        fused = load_fused_file(
            path
        )

        features = analyzer.process_frame(
            fused
        )

        output_name = os.path.basename(
            path
        ).replace(
            "_fused.json",
            "_features.json",
        )

        output_path = os.path.join(
            "output",
            "features",
            output_name,
        )

        save_features(
            features,
            output_path,
        )

        print()
        print(
            f"Frame {features['frame_id']}"
        )

        for person in features["persons"]:

            person_id = person["person_id"]

            motion = person["motion"]
            posture = person["posture"]

            print(
                f"  Person {person_id}"
            )

            print(
                f"    movement: "
                f"{motion.get('movement_state')}"
            )

            print(
                f"    average speed: "
                f"{motion.get('average_joint_speed')}"
            )

            print(
                f"    left hand speed: "
                f"{motion.get('left_hand_speed')}"
            )

            print(
                f"    right hand speed: "
                f"{motion.get('right_hand_speed')}"
            )

            print(
                f"    posture: "
                f"{posture.get('posture_state')}"
            )

            print(
                f"    body orientation: "
                f"{posture.get('body_orientation')}"
            )

    print()
    print("=" * 70)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "Generated files:"
    )

    print(
        "output/features/"
    )


if __name__ == "__main__":
    main()
