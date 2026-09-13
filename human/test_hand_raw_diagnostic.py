from __future__ import annotations

import cv2
import time
import json
import os

from hand_processor import HandProcessor


CAMERA = "/dev/video0"

MODEL = "/home/hawkeye/bas-hmr/runs/hand_pose_yolo11n/weights/best.pt"

WIDTH = 640
HEIGHT = 480
FPS = 10

CONFIDENCE = 0.20

OUTPUT_DIR = "/home/hawkeye/bas-hmr/test_results/hand_raw_diagnostic"

TEST_DURATION = 20


os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("RAW HAND DETECTION DIAGNOSTIC")
print("=" * 60)


processor = HandProcessor(
    model_path=MODEL,
    device="cpu",
    confidence=CONFIDENCE,
    image_size=416,
)


cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2,
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open {CAMERA}"
    )


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


print(
    f"Camera: "
    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
)

print()
print("Move ONE hand slowly around the camera.")
print("Duration: 20 seconds")
print()


frame_id = 0
start_time = time.time()


try:

    while True:

        if time.time() - start_time >= TEST_DURATION:
            break

        ret, frame = cap.read()

        if not ret:
            continue

        timestamp = time.time()

        result = processor.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        hands = result.get("hands", [])

        print(
            f"Frame {frame_id:04d} | "
            f"Detections={len(hands)}"
        )

        for i, hand in enumerate(hands):

            bbox = hand["bbox"]

            print(
                f"    Detection {i}: "
                f"conf={hand['confidence']:.3f} "
                f"center=("
                f"{hand['center']['x']:.1f},"
                f"{hand['center']['y']:.1f}) "
                f"bbox=("
                f"{bbox['x1']:.0f},"
                f"{bbox['y1']:.0f},"
                f"{bbox['x2']:.0f},"
                f"{bbox['y2']:.0f}) "
                f"kp_conf="
                f"{hand['mean_keypoint_confidence']:.3f}"
            )

        # ----------------------------------------------------
        # Annotated image
        # ----------------------------------------------------

        annotated = frame.copy()

        for i, hand in enumerate(hands):

            bbox = hand["bbox"]

            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])

            confidence = hand["confidence"]

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

            label = (
                f"DET {i} "
                f"conf={confidence:.2f}"
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

        # Save every 5 frames
        if frame_id % 5 == 0:

            image_path = os.path.join(
                OUTPUT_DIR,
                f"frame_{frame_id:06d}.jpg",
            )

            cv2.imwrite(
                image_path,
                annotated,
            )

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

        frame_id += 1


finally:

    cap.release()


print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print(
    f"Frames processed: {frame_id}"
)
print(
    f"Results saved to: {OUTPUT_DIR}"
)
print("=" * 60)
