import cv2
import time
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "/home/hawkeye/bas-hmr/runs/hand_pose_yolo11n/weights/best.pt"

CAMERA = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10

CONF = 0.20

OUTPUT_DIR = Path(
    "/home/hawkeye/bas-hmr/test_results/hand_webcam"
)

SAVE_EVERY = 10


# ============================================================
# MAIN
# ============================================================

def main():

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BAS-HMR HAND POSE WEBCAM TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading hand pose model...")
    print(f"Model: {MODEL}")

    model = YOLO(MODEL)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    print(f"\nOpening camera: {CAMERA}")

    cap = cv2.VideoCapture(
        CAMERA,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA}"
        )

    # --------------------------------------------------------
    # Configure stable camera mode
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS
    )

    print("\nRequested camera settings:")
    print(f"  Resolution : {WIDTH} x {HEIGHT}")
    print(f"  FPS        : {FPS}")
    print("  Format     : MJPEG")

    # Give the camera a moment to stabilize
    time.sleep(1)

    # --------------------------------------------------------
    # Read actual camera settings
    # --------------------------------------------------------

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print("\nActual camera settings:")
    print(f"  Resolution : {actual_width} x {actual_height}")
    print(f"  FPS        : {actual_fps:.2f}")

    print("\nCamera opened successfully.")
    print("Press Ctrl+C to stop.\n")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    frame_id = 0
    detected_frames = 0
    total_hands = 0

    start_time = time.time()

    try:

        while True:

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok or frame is None:

                print(
                    f"Frame {frame_id}: "
                    "FAILED TO READ"
                )

                continue

            timestamp = time.time()

            # ------------------------------------------------
            # Run hand pose model
            # ------------------------------------------------

            results = model.predict(
                source=frame,
                imgsz=416,
                conf=CONF,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            # ------------------------------------------------
            # Count hands
            # ------------------------------------------------

            hands = 0

            if result.boxes is not None:
                hands = len(result.boxes)

            if hands > 0:

                detected_frames += 1
                total_hands += hands

                print(
                    f"\nFrame {frame_id} | "
                    f"hands={hands}"
                )

                # --------------------------------------------
                # Print hand information
                # --------------------------------------------

                if result.boxes is not None:

                    for hand_id, box in enumerate(
                        result.boxes
                    ):

                        confidence = float(
                            box.conf[0]
                        )

                        x1, y1, x2, y2 = map(
                            float,
                            box.xyxy[0]
                        )

                        print(
                            f"  Hand {hand_id}: "
                            f"conf={confidence:.3f} "
                            f"bbox=("
                            f"{x1:.1f}, "
                            f"{y1:.1f}, "
                            f"{x2:.1f}, "
                            f"{y2:.1f})"
                        )

                # --------------------------------------------
                # Print 21 keypoints
                # --------------------------------------------

                if result.keypoints is not None:

                    for hand_id, hand in enumerate(
                        result.keypoints.data
                    ):

                        print(
                            f"  Hand {hand_id} "
                            f"keypoints:"
                        )

                        for kp_id, kp in enumerate(hand):

                            x, y, kp_conf = (
                                kp.tolist()
                            )

                            print(
                                f"    "
                                f"{kp_id:02d}: "
                                f"x={x:.1f}, "
                                f"y={y:.1f}, "
                                f"conf={kp_conf:.3f}"
                            )

            # ------------------------------------------------
            # Create annotated frame
            # ------------------------------------------------

            annotated = result.plot()

            # Add basic information to image
            cv2.putText(
                annotated,
                f"Frame: {frame_id}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                f"Hands: {hands}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Save periodic frames
            # ------------------------------------------------

            if frame_id % SAVE_EVERY == 0:

                output_file = (
                    OUTPUT_DIR /
                    f"frame_{frame_id:06d}.jpg"
                )

                saved = cv2.imwrite(
                    str(output_file),
                    annotated
                )

                if saved:

                    print(
                        f"  Saved: {output_file}"
                    )

                else:

                    print(
                        f"  WARNING: "
                        f"Could not save "
                        f"{output_file}"
                    )

            frame_id += 1

    except KeyboardInterrupt:

        print(
            "\n\nStopping webcam test..."
        )

    finally:

        cap.release()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print(
        f"Total frames      : {frame_id}"
    )

    print(
        f"Frames with hands : {detected_frames}"
    )

    print(
        f"Total hands found : {total_hands}"
    )

    if frame_id > 0:

        detection_rate = (
            detected_frames /
            frame_id *
            100
        )

        print(
            f"Detection rate    : "
            f"{detection_rate:.2f}%"
        )

    if elapsed > 0:

        processing_fps = (
            frame_id /
            elapsed
        )

        print(
            f"Processing FPS    : "
            f"{processing_fps:.2f}"
        )

    print(
        f"\nSaved frames      : "
        f"{OUTPUT_DIR}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
