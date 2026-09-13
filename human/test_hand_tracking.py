from __future__ import annotations

import cv2
import time
import json
import os

from hand_processor import HandProcessor
from hand_tracker import HandTracker


CAMERA = "/dev/video0"

MODEL = "/home/hawkeye/bas-hmr/runs/hand_pose_yolo11n/weights/best.pt"

WIDTH = 640
HEIGHT = 480
FPS = 10

CONFIDENCE = 0.20

OUTPUT_DIR = "/home/hawkeye/bas-hmr/test_results/hand_tracking"

TEST_DURATION = 30
SAVE_EVERY = 10


os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("BAS HAND TRACKING VERIFICATION")
print("=" * 60)


# ------------------------------------------------------------
# Hand detector
# ------------------------------------------------------------

processor = HandProcessor(
    model_path=MODEL,
    device="cpu",
    confidence=CONFIDENCE,
    image_size=416,
)


# ------------------------------------------------------------
# Hand tracker
# ------------------------------------------------------------

tracker = HandTracker(
    max_distance=150,
    max_missed_frames=10,
)


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2,
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA}"
    )


# MJPEG is important for your webcam in WSL
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

time.sleep(1)


actual_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

actual_fps = cap.get(
    cv2.CAP_PROP_FPS
)

print(
    f"Camera: {actual_width}x{actual_height} "
    f"@ {actual_fps:.1f} FPS"
)

print()
print("Move your hand around during the test.")
print("Test duration:", TEST_DURATION, "seconds")
print()


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

frame_count = 0
frames_with_hands = 0
total_detections = 0

unique_ids = set()

start_time = time.time()


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

try:

    while True:

        elapsed = time.time() - start_time

        if elapsed >= TEST_DURATION:
            break


        ret, frame = cap.read()

        if not ret:
            print("[CAMERA] Failed to read frame")
            continue


        frame_id = frame_count

        timestamp = time.time()


        # ----------------------------------------------------
        # Hand detection
        # ----------------------------------------------------

        result = processor.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp,
        )


        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        result = tracker.update(result)


        hands = result.get(
            "hands",
            []
        )


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if hands:

            frames_with_hands += 1

            total_detections += len(hands)


        for hand in hands:

            unique_ids.add(
                hand["hand_id"]
            )


        # ----------------------------------------------------
        # Print every 10 frames
        # ----------------------------------------------------

        if frame_id % 10 == 0:

            print(
                f"Frame {frame_id:04d} | "
                f"Hands={len(hands)} | "
                f"ActiveTracks="
                f"{result['tracking']['active_tracks']}"
            )

            for hand in hands:

                hand_id = hand["hand_id"]

                confidence = hand["confidence"]

                center = hand["center"]

                tracking = hand["tracking"]

                speed = tracking["speed"]

                print(
                    f"    "
                    f"ID={hand_id} | "
                    f"conf={confidence:.3f} | "
                    f"center=("
                    f"{center['x']:.1f},"
                    f"{center['y']:.1f}"
                    f") | "
                    f"speed={speed:.2f}"
                )


        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

        if frame_id % SAVE_EVERY == 0:

            json_path = os.path.join(
                OUTPUT_DIR,
                f"frame_{frame_id:06d}.json",
            )

            with open(
                json_path,
                "w",
            ) as f:

                json.dump(
                    result,
                    f,
                    indent=2,
                )


        # ----------------------------------------------------
        # Save annotated image
        # ----------------------------------------------------

        if frame_id % SAVE_EVERY == 0:

            annotated = frame.copy()


            for hand in hands:

                bbox = hand["bbox"]

                x1 = int(bbox["x1"])
                y1 = int(bbox["y1"])
                x2 = int(bbox["x2"])
                y2 = int(bbox["y2"])


                # Bounding box
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2,
                )


                # Tracking label
                label = (
                    f"ID {hand['hand_id']} "
                    f"conf={hand['confidence']:.2f} "
                    f"speed="
                    f"{hand['tracking']['speed']:.1f}"
                )


                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )


                # Keypoints
                for point in hand["keypoints"]:

                    if point["confidence"] < 0.30:
                        continue

                    x = int(point["x"])
                    y = int(point["y"])

                    cv2.circle(
                        annotated,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1,
                    )


            image_path = os.path.join(
                OUTPUT_DIR,
                f"frame_{frame_id:06d}.jpg",
            )

            cv2.imwrite(
                image_path,
                annotated,
            )


        frame_count += 1


finally:

    cap.release()


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

elapsed = time.time() - start_time

processing_fps = (
    frame_count / elapsed
    if elapsed > 0
    else 0
)

detection_rate = (
    frames_with_hands / frame_count * 100
    if frame_count > 0
    else 0
)


print()
print("=" * 60)
print("HAND TRACKING VERIFICATION RESULT")
print("=" * 60)

print(
    f"Frames processed:    {frame_count}"
)

print(
    f"Frames with hands:   {frames_with_hands}"
)

print(
    f"Detection rate:      {detection_rate:.2f}%"
)

print(
    f"Total detections:    {total_detections}"
)

print(
    f"Unique hand IDs:     {sorted(unique_ids)}"
)

print(
    f"Processing FPS:      {processing_fps:.2f}"
)

print(
    f"Output directory:    {OUTPUT_DIR}"
)

print("=" * 60)
