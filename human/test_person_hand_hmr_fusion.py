from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PERSON_MODEL = ROOT / "yolo11n.pt"

HAND_MODEL = (
    ROOT
    / "runs"
    / "hand_pose_yolo11n"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = ROOT / "test_results" / "person_hand_hmr_fusion"

OUTPUT_JSON = OUTPUT_DIR / "fused_human_frame.json"
OUTPUT_IMAGE = OUTPUT_DIR / "fused_human_frame.jpg"


# ============================================================
# CAMERA
# ============================================================

CAMERA = "/dev/video0"

WIDTH = 640
HEIGHT = 480
FPS = 10

# Detection settings
PERSON_CONFIDENCE = 0.35
HAND_CONFIDENCE = 0.20

IMAGE_SIZE = 416


# ============================================================
# IMPORT OUR PROCESSORS
# ============================================================
import os 

sys.path.insert(
    0,
    str(ROOT)
)
from human.hand_processor import HandProcessor
from human.person_hand_association import PersonHandAssociator
from hmr.hmr_processor import HMRProcessor


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BAS PERSON + HAND + HMR2 FUSION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    if not PERSON_MODEL.exists():
        raise FileNotFoundError(
            f"Person model not found:\n{PERSON_MODEL}"
        )

    if not HAND_MODEL.exists():
        raise FileNotFoundError(
            f"Hand model not found:\n{HAND_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load person detector
    # --------------------------------------------------------

    print()
    print("[PERSON] Loading YOLO person model...")

    person_model = YOLO(
        str(PERSON_MODEL)
    )

    print("[PERSON] Model loaded")

    # --------------------------------------------------------
    # Load hand processor
    # --------------------------------------------------------

    hand_processor = HandProcessor(
        model_path=str(HAND_MODEL),
        device="cpu",
        confidence=HAND_CONFIDENCE,
        image_size=IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Load association system
    # --------------------------------------------------------

    associator = PersonHandAssociator()

    print("[FUSION] Person-hand associator ready")

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    print()
    print("[CAMERA] Opening camera...")

    cap = cv2.VideoCapture(
        CAMERA,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera: {CAMERA}"
        )

    # IMPORTANT:
    # MJPEG is required for your stable webcam setup.

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

    time.sleep(1.0)

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
        f"[CAMERA] {actual_width}x{actual_height} "
        f"@ {actual_fps:.1f} FPS"
    )

    # --------------------------------------------------------
    # Warm-up camera
    # --------------------------------------------------------

    frame = None

    for _ in range(10):

        ok, current_frame = cap.read()

        if ok:
            frame = current_frame

    if frame is None:
        cap.release()

        raise RuntimeError(
            "Could not capture a valid camera frame."
        )

    # --------------------------------------------------------
    # Frame ID
    # --------------------------------------------------------

    frame_id = 0

    timestamp = time.time()

    print()
    print("[CAMERA] Captured test frame")
    print(
        f"[FRAME] ID={frame_id} "
        f"resolution={frame.shape[1]}x{frame.shape[0]}"
    )

    # ========================================================
    # STEP 1 — PERSON DETECTION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 1: PERSON DETECTION")
    print("=" * 70)

    person_start = time.time()

    person_results = person_model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=PERSON_CONFIDENCE,
        device="cpu",
        verbose=False,
    )

    person_time = time.time() - person_start

    persons = []
    boxes = []

    if person_results:

        result = person_results[0]

        if result.boxes is not None:

            boxes_xyxy = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            for i in range(
                len(boxes_xyxy)
            ):

                # COCO class 0 = person
                if int(classes[i]) != 0:
                    continue

                x1, y1, x2, y2 = [
                    float(v)
                    for v in boxes_xyxy[i]
                ]

                confidence = float(
                    confidences[i]
                )

                person_id = len(persons)

                person = {
                    "person_id": person_id,

                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },

                    "confidence": confidence,
                }

                persons.append(
                    person
                )

                boxes.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    )

    print(
        f"Persons detected: {len(persons)}"
    )

    print(
        f"Person inference time: "
        f"{person_time:.2f}s"
    )

    for person in persons:

        print(
            f"  Person {person['person_id']} "
            f"conf={person['confidence']:.3f} "
            f"bbox={person['bbox']}"
        )

    # ========================================================
    # STEP 2 — HAND DETECTION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 2: HAND POSE DETECTION")
    print("=" * 70)

    hand_start = time.time()

    hand_result = hand_processor.process_frame(
        frame=frame,
        frame_id=frame_id,
        timestamp=timestamp,
    )

    hand_time = time.time() - hand_start

    print(
        f"Hands detected: "
        f"{len(hand_result.get('hands', []))}"
    )

    print(
        f"Hand inference time: "
        f"{hand_time:.2f}s"
    )

    # ========================================================
    # STEP 3 — PERSON-HAND ASSOCIATION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 3: PERSON-HAND ASSOCIATION")
    print("=" * 70)

    associated_hands = associator.associate(
        persons,
        hand_result.get("hands", [])
    )

    for hand in associated_hands:

        print(
            f"  Hand {hand['hand_id']} "
            f"conf={hand['confidence']:.3f} "
            f"person={hand.get('person_id')} "
            f"score={hand['association']['score']}"
        )

    associated_count = sum(
        1
        for hand in associated_hands
        if hand.get("person_id") is not None
    )

    print(
        f"Associated hands: "
        f"{associated_count}/"
        f"{len(associated_hands)}"
    )

    # ========================================================
    # STEP 4 — HMR2
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4: HMR2 3D BODY ESTIMATION")
    print("=" * 70)

    if len(boxes) == 0:

        print(
            "[HMR] No persons detected."
        )

        hmr_result = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "model": "HMR2",
            "device": "cpu",
            "persons": [],
        }

    else:

        print(
            f"[HMR] Running HMR2 for "
            f"{len(boxes)} person(s)..."
        )

        print(
            "[HMR] WARNING: CPU inference may "
            "take several seconds per person."
        )

        hmr_start = time.time()

        hmr_processor = HMRProcessor()

        hmr_result = hmr_processor.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp,
            boxes=boxes,
        )

        hmr_time = time.time() - hmr_start

        print(
            f"[HMR] Inference time: "
            f"{hmr_time:.2f}s"
        )

    # ========================================================
    # STEP 5 — FINAL FUSION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 5: FINAL HUMAN FUSION")
    print("=" * 70)

    # --------------------------------------------------------
    # Build lookup of hand observations by person
    # --------------------------------------------------------

    hands_by_person = {}

    for hand in associated_hands:

        person_id = hand.get(
            "person_id"
        )

        if person_id is None:
            continue

        hands_by_person.setdefault(
            int(person_id),
            []
        ).append(
            hand
        )

    # --------------------------------------------------------
    # Merge HMR persons with hands
    # --------------------------------------------------------

    fused_persons = []

    for person in hmr_result.get(
        "persons",
        []
    ):

        person_id = int(
            person["person_id"]
        )

        person_copy = dict(
            person
        )

        person_hands = hands_by_person.get(
            person_id,
            []
        )

        person_copy["hands"] = (
            person_hands
        )

        person_copy["hand_count"] = (
            len(person_hands)
        )

        fused_persons.append(
            person_copy
        )

    # --------------------------------------------------------
    # Add unassociated hands
    # --------------------------------------------------------

    unassociated_hands = [
        hand
        for hand in associated_hands
        if hand.get("person_id") is None
    ]

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    fused_result = {

        "frame_id": int(
            frame_id
        ),

        "timestamp": float(
            timestamp
        ),

        "image": {
            "width": int(
                frame.shape[1]
            ),
            "height": int(
                frame.shape[0]
            ),
        },

        "persons": fused_persons,

        "unassociated_hands": (
            unassociated_hands
        ),

        "metadata": {

            "person_model": "YOLO11n",

            "hand_model": (
                "YOLO11n-Pose-custom"
            ),

            "body_model": "HMR2",

            "device": "cpu",

            "person_count": len(
                fused_persons
            ),

            "hand_count": len(
                associated_hands
            ),

            "associated_hand_count": (
                associated_count
            ),

            "unassociated_hand_count": (
                len(unassociated_hands)
            ),
        },
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            fused_result,
            f,
            indent=2
        )

    print()
    print(
        f"[OUTPUT] JSON saved:"
    )

    print(
        OUTPUT_JSON
    )

    # ========================================================
    # CREATE VISUALIZATION
    # ========================================================

    annotated = frame.copy()

    # --------------------------------------------------------
    # Draw person boxes
    # --------------------------------------------------------

    for person in persons:

        bbox = person["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"Person {person['person_id']} "
            f"{person['confidence']:.2f}"
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Draw hands
    # --------------------------------------------------------

    for hand in associated_hands:

        bbox = hand["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        person_id = hand.get(
            "person_id"
        )

        label = (
            f"Hand {hand['hand_id']} "
            f"P={person_id}"
        )

        cv2.putText(
            annotated,
            label,
            (x1, min(
                annotated.shape[0] - 5,
                y2 + 18
            )),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            2
        )

        # Draw 21 hand keypoints

        for point in hand.get(
            "keypoints",
            []
        ):

            if point["confidence"] < 0.20:
                continue

            px = int(
                point["x"]
            )

            py = int(
                point["y"]
            )

            cv2.circle(
                annotated,
                (px, py),
                3,
                (255, 0, 0),
                -1
            )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    cv2.imwrite(
        str(OUTPUT_IMAGE),
        annotated
    )

    print(
        f"[OUTPUT] Visualization saved:"
    )

    print(
        OUTPUT_IMAGE
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("BAS HUMAN FUSION SUMMARY")
    print("=" * 70)

    print(
        f"Frame ID:              {frame_id}"
    )

    print(
        f"Persons:               "
        f"{len(fused_persons)}"
    )

    print(
        f"Hands:                 "
        f"{len(associated_hands)}"
    )

    print(
        f"Associated hands:     "
        f"{associated_count}"
    )

    print(
        f"Unassociated hands:   "
        f"{len(unassociated_hands)}"
    )

    for person in fused_persons:

        print(
            f"Person {person['person_id']}: "
            f"{person.get('hand_count', 0)} hand(s), "
            f"{person.get('pose', {}).get('joint_count', 0)} "
            f"body joints"
        )

        camera = person.get(
            "camera_translation"
        )

        if camera:

            print(
                "    Camera position: "
                f"x={camera['x']:.3f}, "
                f"y={camera['y']:.3f}, "
                f"z={camera['z']:.3f}"
            )

    print("=" * 70)

    cap.release()


if __name__ == "__main__":
    main()
