import json

from human.state_extractor import HumanStateExtractor
from human.posture_analyzer import PostureAnalyzer


def main():

    input_file = (
        "output/frame_000570_fused.json"
    )

    with open(
        input_file,
        "r"
    ) as f:
        fused = json.load(f)

    extractor = HumanStateExtractor()
    posture_analyzer = PostureAnalyzer()

    human_state = extractor.extract_from_fused(
        fused
    )

    posture = posture_analyzer.analyze(
        human_state
    )

    print("=" * 70)
    print("POSTURE ANALYSIS TEST")
    print("=" * 70)

    print(
        f"Frame: {posture['frame_id']}"
    )

    for person in posture["persons"]:

        print()
        print(
            f"Person: {person['person_id']}"
        )

        data = person["posture"]

        print(
            f"Status: "
            f"{data['status']}"
        )

        if data["status"] != "ok":
            print(
                "Missing joints:",
                data["missing_joints"]
            )
            continue

        print(
            f"Body orientation: "
            f"{data['body_orientation']}"
        )

        print(
            f"Posture state: "
            f"{data['posture_state']}"
        )

        angles = data["angles"]

        print(
            f"Left knee: "
            f"{angles['left_knee']:.2f} deg"
        )

        print(
            f"Right knee: "
            f"{angles['right_knee']:.2f} deg"
        )

        print(
            f"Left elbow: "
            f"{angles['left_elbow']:.2f} deg"
        )

        print(
            f"Right elbow: "
            f"{angles['right_elbow']:.2f} deg"
        )

        print(
            f"Torso vertical angle: "
            f"{angles['torso_vertical']:.2f} deg"
        )

        print(
            f"Left arm angle: "
            f"{angles['left_arm_vertical']:.2f} deg"
        )

        print(
            f"Right arm angle: "
            f"{angles['right_arm_vertical']:.2f} deg"
        )

    print()
    print("=" * 70)
    print("POSTURE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
