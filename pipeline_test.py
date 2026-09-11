import cv2
import json
from pathlib import Path
import time

from yolo.yolo_processor import YOLOProcessor
from hmr.hmr_processor import HMRProcessor


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

IMAGE_PATH = ROOT / "input" / "astronaut_image_2.jpg"

OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

print()
print("====================================")
print(" BAS YOLO + HMR PIPELINE")
print("====================================")

print(f"[PIPELINE] Loading image:")
print(IMAGE_PATH)

frame = cv2.imread(
    str(IMAGE_PATH)
)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print(
    f"[PIPELINE] Image shape: {frame.shape}"
)


# ---------------------------------------------------------
# FRAME INFORMATION
# ---------------------------------------------------------

frame_id = 0

timestamp = time.time()

print()
print(f"[FRAME] frame_id = {frame_id}")
print(f"[FRAME] timestamp = {timestamp}")


# ---------------------------------------------------------
# INITIALIZE YOLO
# ---------------------------------------------------------

print()
print("[PIPELINE] Initializing YOLO...")

yolo = YOLOProcessor(
    model_path="yolo11n.pt",
    device="cpu"
)


# ---------------------------------------------------------
# INITIALIZE HMR
# ---------------------------------------------------------

print()
print("[PIPELINE] Initializing HMR2...")

hmr = HMRProcessor()


# ---------------------------------------------------------
# YOLO
# ---------------------------------------------------------

print()
print("====================================")
print(" STEP 1: YOLO")
print("====================================")

yolo_result = yolo.process_frame(
    frame=frame,
    frame_id=frame_id,
    timestamp=timestamp
)

print(
    f"[YOLO] Persons detected: "
    f"{len(yolo_result['persons'])}"
)


# ---------------------------------------------------------
# EXTRACT BOUNDING BOXES
# ---------------------------------------------------------

boxes = []

for person in yolo_result["persons"]:

    bbox = person["bbox"]

    boxes.append([
        bbox["x1"],
        bbox["y1"],
        bbox["x2"],
        bbox["y2"]
    ])


print()
print("[YOLO] Bounding boxes sent to HMR:")

for i, box in enumerate(boxes):

    print(
        f"Person {i}: {box}"
    )


# ---------------------------------------------------------
# HMR
# ---------------------------------------------------------

print()
print("====================================")
print(" STEP 2: HMR2")
print("====================================")

hmr_result = hmr.process_frame(
    frame=frame,
    frame_id=frame_id,
    timestamp=timestamp,
    boxes=boxes
)

print(
    f"[HMR] Persons processed: "
    f"{len(hmr_result['persons'])}"
)


# ---------------------------------------------------------
# FUSION
# ---------------------------------------------------------

print()
print("====================================")
print(" STEP 3: FUSION")
print("====================================")

fused_result = {

    "frame_id": frame_id,

    "timestamp": timestamp,

    "synchronization": {
        "status": "synchronized",
        "timestamp_difference": 0.0
    },

    "yolo": yolo_result,

    "hmr": hmr_result
}


# ---------------------------------------------------------
# SAVE FUSED JSON
# ---------------------------------------------------------

output_file = (
    OUTPUT_DIR /
    f"frame_{frame_id:06d}.json"
)

with open(output_file, "w") as f:

    json.dump(
        fused_result,
        f,
        indent=4
    )


print()
print("====================================")
print(" PIPELINE COMPLETE")
print("====================================")

print()
print(f"Frame ID: {frame_id}")

print(
    f"YOLO persons: "
    f"{len(yolo_result['persons'])}"
)

print(
    f"HMR persons: "
    f"{len(hmr_result['persons'])}"
)

print()
print("Fused JSON:")

print(output_file)
