from ultralytics import YOLO


MODEL = "yolo11n-pose.pt"

DATA = (
    "/home/hawkeye/bas-hmr/"
    "datasets/hand_pose/data.yaml"
)

PROJECT = "/home/hawkeye/bas-hmr/runs"


def main():

    print()
    print("=" * 60)
    print("BAS 21-KEYPOINT HAND POSE TRAINING")
    print("=" * 60)
    print()

    print("[MODEL] Loading YOLO11n-Pose...")

    model = YOLO(
        MODEL
    )

    print(
        "[MODEL] Loaded successfully."
    )

    print()
    print("[TRAIN] Starting full training...")
    print()

    model.train(

        data=DATA,

        epochs=50,

        imgsz=416,

        batch=2,

        workers=0,

        device="cpu",

        cache=False,

        project=PROJECT,

        name="hand_pose_yolo11n",

        exist_ok=True,

        patience=10,

        plots=True,

        pretrained=True,

        verbose=True,
    )

    print()
    print("=" * 60)
    print("HAND POSE TRAINING COMPLETE")
    print("=" * 60)
    print()

    print(
        "Search for best.pt with:"
    )

    print(
        "find ~/bas-hmr/runs "
        "-path '*hand_pose_yolo11n*/weights/best.pt' "
        "-print"
    )


if __name__ == "__main__":
    main()
