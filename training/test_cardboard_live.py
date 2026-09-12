from __future__ import annotations

import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "/home/hawkeye/bas-hmr/"
    "runs/cardboard_yolo11n/weights/best.pt"
)

CAMERA_DEVICE = "/dev/video0"

CONFIDENCE = 0.25

OUTPUT_DIR = Path(
    "/home/hawkeye/bas-hmr/"
    "output/cardboard_live_test"
)

SAVE_EVERY_N_FRAMES = 10


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " BAS CUSTOM CARDBOARD DETECTOR"
    )
    print("=" * 60)
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "[MODEL] Loading trained model..."
    )

    model = YOLO(
        MODEL_PATH
    )

    print(
        f"[MODEL] Loaded: {MODEL_PATH}"
    )

    print(
        f"[MODEL] Classes: {model.names}"
    )

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    print(
        "[CAMERA] Opening..."
    )

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open /dev/video0"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    print(
        "[CAMERA] Ready."
    )

    print()
    print(
        "Live detection running."
    )
    print(
        "Press Ctrl+C to stop."
    )
    print()

    frame_id = 0

    total_detections = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print(
                    "[CAMERA] Frame read failed."
                )

                time.sleep(
                    0.1
                )

                continue

            frame_id += 1

            # ------------------------------------------------
            # YOLO inference
            # ------------------------------------------------

            results = model.predict(
                source=frame,
                device="cpu",
                conf=CONFIDENCE,
                verbose=False,
            )

            result = results[0]

            detection_count = 0

            if result.boxes is not None:

                boxes = result.boxes

                for i in range(
                    len(boxes)
                ):

                    class_id = int(
                        boxes.cls[i].item()
                    )

                    confidence = float(
                        boxes.conf[i].item()
                    )

                    x1, y1, x2, y2 = (
                        boxes.xyxy[i]
                        .cpu()
                        .numpy()
                    )

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    class_name = (
                        model.names[class_id]
                    )

                    center_x = (
                        x1 + x2
                    ) // 2

                    center_y = (
                        y1 + y2
                    ) // 2

                    detection_count += 1

                    total_detections += 1

                    print(
                        f"[FRAME {frame_id}] "
                        f"{class_name} | "
                        f"conf={confidence:.3f} | "
                        f"center=({center_x}, {center_y}) | "
                        f"bbox=("
                        f"{x1},{y1},"
                        f"{x2},{y2}"
                        f")"
                    )

                    # ----------------------------------------
                    # Draw box
                    # ----------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    # ----------------------------------------
                    # Draw center
                    # ----------------------------------------

                    cv2.circle(
                        frame,
                        (
                            center_x,
                            center_y,
                        ),
                        5,
                        (0, 0, 255),
                        -1,
                    )

                    # ----------------------------------------
                    # Draw label
                    # ----------------------------------------

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1,
                            max(
                                y1 - 10,
                                20,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            print(
                f"[FRAME {frame_id}] "
                f"detections={detection_count}"
            )

            # ------------------------------------------------
            # Save annotated frame periodically
            # ------------------------------------------------

            if (
                frame_id
                % SAVE_EVERY_N_FRAMES
                == 0
            ):

                output_path = (
                    OUTPUT_DIR
                    / f"frame_{frame_id:06d}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    frame,
                )

                print(
                    f"[IMAGE] Saved: "
                    f"{output_path}"
                )

    except KeyboardInterrupt:

        print()
        print(
            "[TEST] Ctrl+C received."
        )

    finally:

        cap.release()

        print()
        print(
            "=" * 60
        )
        print(
            " TEST COMPLETE"
        )
        print(
            "=" * 60
        )

        print(
            f"Frames processed: "
            f"{frame_id}"
        )

        print(
            f"Total detections: "
            f"{total_detections}"
        )

        print(
            f"Saved frames: "
            f"{OUTPUT_DIR}"
        )


if __name__ == "__main__":
    main()