from __future__ import annotations

import time
import cv2

from human.hand_processor import HandProcessor


MODEL = "/home/hawkeye/bas-hmr/runs/hand_pose_yolo11n/weights/best.pt"
CAMERA = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10


def main():

    print("=" * 60)
    print("BAS-HMR HAND PROCESSOR TEST")
    print("=" * 60)

    # ------------------------------------------------------------
    # Load processor
    # ------------------------------------------------------------

    processor = HandProcessor(
        model_path=MODEL,
        device="cpu",
        confidence=0.20,
        image_size=416,
    )

    # ------------------------------------------------------------
    # Open camera
    # ------------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {CAMERA}"
        )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Use exactly the same stable camera configuration
    # as the validated hand webcam tester.
    # ------------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FPS,
    )

    print("\nRequested camera settings:")
    print(f"  Resolution : {WIDTH} x {HEIGHT}")
    print(f"  FPS        : {FPS}")
    print("  Format     : MJPEG")

    # Give camera time to stabilize
    time.sleep(1)

    # ------------------------------------------------------------
    # Read actual settings
    # ------------------------------------------------------------

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
    print(
        f"  Resolution : "
        f"{actual_width} x {actual_height}"
    )

    print(
        f"  FPS        : "
        f"{actual_fps:.2f}"
    )

    print("\nCamera opened successfully.")
    print("Press Ctrl+C to stop.\n")

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    frame_id = 0
    detected_frames = 0
    total_hands = 0

    start_time = time.time()

    try:

        while True:

            ok, frame = cap.read()

            if not ok or frame is None:

                print(
                    f"Frame {frame_id}: "
                    "FAILED TO READ"
                )

                continue

            timestamp = time.time()

            # ----------------------------------------------------
            # Process frame
            # ----------------------------------------------------

            result = processor.process_frame(
                frame=frame,
                frame_id=frame_id,
                timestamp=timestamp,
            )

            hands = result["hands"]

            # ----------------------------------------------------
            # Statistics
            # ----------------------------------------------------

            if len(hands) > 0:

                detected_frames += 1
                total_hands += len(hands)

                print(
                    f"\nFrame {frame_id} | "
                    f"hands={len(hands)}"
                )

                for hand in hands:

                    bbox = hand["bbox"]

                    print(
                        f"  Hand {hand['hand_id']} | "
                        f"conf={hand['confidence']:.3f} | "
                        f"kp={hand['mean_keypoint_confidence']:.3f} | "
                        f"bbox=("
                        f"{bbox['x1']:.1f}, "
                        f"{bbox['y1']:.1f}, "
                        f"{bbox['x2']:.1f}, "
                        f"{bbox['y2']:.1f})"
                    )

            else:

                print(
                    f"Frame {frame_id:06d} | "
                    "hands=0"
                )

            frame_id += 1

    except KeyboardInterrupt:

        print("\n[TEST] Stopped")

    finally:

        cap.release()

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    elapsed = time.time() - start_time

    processing_fps = (
        frame_id / elapsed
        if elapsed > 0
        else 0
    )

    detection_rate = (
        detected_frames / frame_id * 100
        if frame_id > 0
        else 0
    )

    print("\n")
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    print(
        f"Frames              : {frame_id}"
    )

    print(
        f"Frames with hands   : "
        f"{detected_frames}"
    )

    print(
        f"Total hands found   : "
        f"{total_hands}"
    )

    print(
        f"Detection rate      : "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Processing FPS      : "
        f"{processing_fps:.2f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
