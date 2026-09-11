import os
import sys
import json


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT
# ============================================================

from fusion.synchronizer import FrameSynchronizer


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLO + HMR FUSION TEST")
    print("=" * 60)

    # Create synchronizer
    synchronizer = FrameSynchronizer(
        max_buffer_size=30,
        max_time_difference=0.050
    )

    print("[SYSTEM] Synchronizer created")


    # ========================================================
    # YOLO DATA
    # ========================================================

    yolo_data = {
        "frame_id": 100,
        "timestamp": 1000.000,
        "model": "YOLO11n",
        "device": "cpu",

        "image": {
            "width": 640,
            "height": 480
        },

        "persons": [
            {
                "person_id": 0,
                "confidence": 0.95,

                "bbox": {
                    "x1": 100,
                    "y1": 50,
                    "x2": 400,
                    "y2": 450
                }
            }
        ]
    }


    # ========================================================
    # HMR DATA
    # ========================================================

    hmr_data = {
        "frame_id": 100,
        "timestamp": 1000.020,
        "model": "HMR2",
        "device": "cpu",

        "persons": [
            {
                "person_id": 0,

                "camera_translation": [
                    0.12,
                    -0.31,
                    3.20
                ],

                "pose": {
                    "joint_count": 44,

                    "joints_3d": [
                        {
                            "joint_id": 0,
                            "x": 0.10,
                            "y": -0.20,
                            "z": 3.10
                        }
                    ]
                }
            }
        ]
    }


    # ========================================================
    # ADD YOLO
    # ========================================================

    print()
    print("[1] Adding YOLO frame...")

    result = synchronizer.add_yolo(
        yolo_data
    )

    print("[SYNC] YOLO added")


    # ========================================================
    # ADD HMR
    # ========================================================

    print()
    print("[2] Adding HMR frame...")

    result = synchronizer.add_hmr(
        hmr_data
    )


    # ========================================================
    # CHECK RESULT
    # ========================================================

    if result is None:

        print()
        print("[ERROR] Frames were not synchronized.")

        return


    print()
    print("[3] SUCCESS!")
    print("[SYNC] YOLO + HMR synchronized")


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print()
    print("-" * 60)

    print(
        "Frame ID:",
        result["frame_id"]
    )

    print(
        "Synchronization status:",
        result["synchronization"]["status"]
    )

    print(
        "Timestamp difference:",
        result["synchronization"]
        ["timestamp_difference"],
        "seconds"
    )

    print(
        "YOLO persons:",
        len(result["yolo"]["persons"])
    )

    print(
        "HMR persons:",
        len(result["hmr"]["persons"])
    )


    # ========================================================
    # DISPLAY 3D INFORMATION
    # ========================================================

    if result["hmr"]["persons"]:

        person = result["hmr"]["persons"][0]

        print()
        print("3D HUMAN INFORMATION")
        print("-" * 60)

        print(
            "Camera translation:",
            person["camera_translation"]
        )

        print(
            "Joint count:",
            person["pose"]["joint_count"]
        )

        print(
            "First joint:",
            person["pose"]["joints_3d"][0]
        )


    # ========================================================
    # SAVE FUSED JSON
    # ========================================================

    output_dir = os.path.join(
        PROJECT_ROOT,
        "output"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        "fusion_test.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )


    print()
    print("[SAVED]", output_file)

    print()
    print("=" * 60)
    print("FUSION TEST COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
