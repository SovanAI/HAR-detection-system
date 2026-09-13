from ultralytics import YOLO

MODEL = "/home/hawkeye/bas-hmr/runs/hand_pose_yolo11n/weights/best.pt"
DATA = "/home/hawkeye/bas-hmr/datasets/hand_pose/data.yaml"


def main():
    print("Loading trained hand-pose model...")
    model = YOLO(MODEL)

    print("Starting validation...")

    results = model.val(
        data=DATA,
        imgsz=416,
        batch=2,
        workers=0,
        device="cpu",
        plots=True,
    )

    print("\n===== VALIDATION COMPLETE =====")
    print(results)


if __name__ == "__main__":
    main()
