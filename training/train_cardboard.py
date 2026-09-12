from ultralytics import YOLO


MODEL = "/home/hawkeye/bas-hmr/yolo11n.pt"

DATA = (
    "/home/hawkeye/bas-hmr/"
    "datasets/cardboard_yolo/data.yaml"
)

PROJECT = "/home/hawkeye/bas-hmr/runs"


def main():

    print()
    print("=" * 60)
    print("BAS CARDBOARD BOX YOLO TRAINING")
    print("=" * 60)
    print()

    model = YOLO(
        MODEL
    )

    model.train(

        data=DATA,

        epochs=50,

        imgsz=416,

        batch=2,

        workers=0,

        device="cpu",

        cache=False,

        project=PROJECT,

        name="cardboard_yolo11n",

        exist_ok=True,

        patience=10,

        plots=True,

        pretrained=True,
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print()

    print(
        "Best model:"
    )

    print(
        "/home/hawkeye/bas-hmr/"
        "runs/cardboard_yolo11n/weights/best.pt"
    )


if __name__ == "__main__":
    main()
