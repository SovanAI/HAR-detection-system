from __future__ import annotations

import cv2
import time
import json
import os

from ultralytics import YOLO

from hand_processor import HandProcessor
from person_hand_association import PersonHandAssociator


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = "/dev/video0"

PERSON_MODEL = "/home/hawkeye/bas-hmr/yolo11n.pt"

HAND_MODEL = (
    "/home/hawkeye/bas-hmr/"
    "runs/hand_pose_yolo11n/weights/best.pt"
)

WIDTH = 640
HEIGHT = 480
FPS = 10

PERSON_CONFIDENCE = 0.35
HAND_CONFIDENCE = 0.20

IMAGE_SIZE = 416

TEST_DURATION = 30

OUTPUT_DIR = (
    "/home/hawkeye/bas-hmr/"
    "test_results/person_hand_fusion"
)

SAVE_EVERY = 10


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("BAS PERSON + HAND FUSION TEST")
print("=" * 60)


# ============================================================
# LOAD PERSON YOLO
# ============================================================

print("[PERSON] Loading YOLO person model...")

person_model = YOLO(
    PERSON_MODEL
)

print("[PERSON] Model loaded")


# ============================================================
# LOAD HAND PROCESSOR
# ============================================================

hand_processor = HandProcessor(
    model_path=HAND_MODEL,
    device="cpu",
    confidence=HAND_CONFIDENCE,
    image_size=IMAGE_SIZE,
)


# ============================================================
# LOAD ASSOCIATOR
# ============================================================

associator = PersonHandAssociator(
    wrist_margin=0.35,
    center_distance_threshold=0.75,
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open camera: {CAMERA}"
    )


# IMPORTANT:
# Your webcam works reliably with MJPEG.

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
    f"Camera: "
    f"{actual_width}x"
    f"{actual_height} "
    f"@ {actual_fps:.1f} FPS"
)

print()

print(
    "Move around in front of the camera."
)

print(
    "If possible, test with multiple people."
)

print(
    f"Test duration: {TEST_DURATION} seconds"
)

print()


# ============================================================
# STATISTICS
# ============================================================

frame_count = 0

frames_with_persons = 0

frames_with_hands = 0

total_persons = 0

total_hands = 0

associated_hands = 0

unassociated_hands = 0

start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        elapsed = (
            time.time()
            - start_time
        )

        if elapsed >= TEST_DURATION:
            break


        # ----------------------------------------------------
        # CAMERA FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "[CAMERA] Failed to read frame"
            )

            continue


        frame_id = frame_count

        timestamp = time.time()


        # ====================================================
        # PERSON DETECTION
        # ====================================================

        results = person_model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=PERSON_CONFIDENCE,
            device="cpu",
            verbose=False,
        )


        persons = []


        if results:

            result = results[0]

            if result.boxes is not None:

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                classes = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                )

                for i in range(
                    len(boxes)
                ):

                    # COCO class 0 = person
                    if int(classes[i]) != 0:
                        continue


                    x1, y1, x2, y2 = (
                        boxes[i]
                    )

                    persons.append(
                        {
                            "person_id": len(
                                persons
                            ),

                            "confidence": float(
                                confidences[i]
                            ),

                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                            },
                        }
                    )


        # ====================================================
        # HAND DETECTION
        # ====================================================

        hand_result = (
            hand_processor.process_frame(
                frame=frame,
                frame_id=frame_id,
                timestamp=timestamp,
            )
        )


        hands = hand_result.get(
            "hands",
            []
        )


        # ====================================================
        # PERSON-HAND ASSOCIATION
        # ====================================================

        person_result = {

            "frame_id": frame_id,

            "timestamp": timestamp,

            "persons": persons,
        }


        fused = associator.fuse(
            person_result,
            hand_result,
        )


        # ====================================================
        # STATISTICS
        # ====================================================

        if persons:
            frames_with_persons += 1

        if hands:
            frames_with_hands += 1

        total_persons += len(
            persons
        )

        total_hands += len(
            hands
        )

        associated_count = 0

        for person in fused[
            "persons"
        ]:

            associated_count += (
                person["hand_count"]
            )

        associated_hands += (
            associated_count
        )

        unassociated_hands += len(
            fused[
                "unassociated_hands"
            ]
        )


        # ====================================================
        # PRINT STATUS
        # ====================================================

        if frame_id % 10 == 0:

            print(
                f"Frame {frame_id:04d} | "
                f"Persons={len(persons)} | "
                f"Hands={len(hands)} | "
                f"Associated={associated_count} | "
                f"Unassociated="
                f"{len(fused['unassociated_hands'])}"
            )


            for person in fused[
                "persons"
            ]:

                print(
                    f"    Person "
                    f"{person['person_id']} "
                    f"hands="
                    f"{person['hand_count']}"
                )

                for hand in person[
                    "hands"
                ]:

                    print(
                        f"        Hand "
                        f"{hand['hand_id']} "
                        f"conf="
                        f"{hand['confidence']:.3f} "
                        f"association="
                        f"{hand['association']['score']}"
                    )


            for hand in fused[
                "unassociated_hands"
            ]:

                print(
                    f"    UNASSOCIATED "
                    f"Hand "
                    f"{hand['hand_id']} "
                    f"conf="
                    f"{hand['confidence']:.3f}"
                )


        # ====================================================
        # SAVE JSON
        # ====================================================

        if frame_id % SAVE_EVERY == 0:

            json_path = os.path.join(
                OUTPUT_DIR,
                f"frame_{frame_id:06d}.json"
            )

            with open(
                json_path,
                "w"
            ) as f:

                json.dump(
                    fused,
                    f,
                    indent=2
                )


        # ====================================================
        # SAVE ANNOTATED IMAGE
        # ====================================================

        if frame_id % SAVE_EVERY == 0:

            annotated = frame.copy()


            # ------------------------------------------------
            # Draw persons
            # ------------------------------------------------

            for person in fused[
                "persons"
            ]:

                bbox = person[
                    "bbox"
                ]

                x1 = int(
                    bbox["x1"]
                )

                y1 = int(
                    bbox["y1"]
                )

                x2 = int(
                    bbox["x2"]
                )

                y2 = int(
                    bbox["y2"]
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2,
                )


                cv2.putText(
                    annotated,
                    (
                        f"Person "
                        f"{person['person_id']} "
                        f"Hands="
                        f"{person['hand_count']}"
                    ),
                    (
                        x1,
                        max(
                            20,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )


                # --------------------------------------------
                # Draw associated hands
                # --------------------------------------------

                for hand in person[
                    "hands"
                ]:

                    center = hand[
                        "center"
                    ]

                    cx = int(
                        center["x"]
                    )

                    cy = int(
                        center["y"]
                    )

                    hand_id = hand[
                        "hand_id"
                    ]

                    cv2.circle(
                        annotated,
                        (cx, cy),
                        8,
                        (0, 255, 0),
                        -1,
                    )

                    cv2.putText(
                        annotated,
                        (
                            f"H{hand_id}"
                        ),
                        (
                            cx + 8,
                            cy
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )


            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            image_path = os.path.join(
                OUTPUT_DIR,
                f"frame_{frame_id:06d}.jpg"
            )

            cv2.imwrite(
                image_path,
                annotated
            )


        frame_count += 1


finally:

    cap.release()


# ============================================================
# SUMMARY
# ============================================================

elapsed = (
    time.time()
    - start_time
)

processing_fps = (
    frame_count / elapsed
    if elapsed > 0
    else 0
)

association_rate = (
    associated_hands
    /
    total_hands
    * 100
    if total_hands > 0
    else 0
)


print()

print("=" * 60)

print("PERSON + HAND FUSION SUMMARY")

print("=" * 60)

print(
    f"Frames processed:       "
    f"{frame_count}"
)

print(
    f"Frames with persons:    "
    f"{frames_with_persons}"
)

print(
    f"Frames with hands:      "
    f"{frames_with_hands}"
)

print(
    f"Total person detections:"
    f" {total_persons}"
)

print(
    f"Total hand detections:  "
    f"{total_hands}"
)

print(
    f"Associated hands:       "
    f"{associated_hands}"
)

print(
    f"Unassociated hands:     "
    f"{unassociated_hands}"
)

print(
    f"Association rate:       "
    f"{association_rate:.2f}%"
)

print(
    f"Processing FPS:         "
    f"{processing_fps:.2f}"
)

print(
    f"Output directory:       "
    f"{OUTPUT_DIR}"
)

print("=" * 60)
