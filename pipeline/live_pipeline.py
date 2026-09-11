import os
import sys
import time
import json
import cv2


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from ultralytics import YOLO

from hmr.hmr_processor import HMRProcessor
from fusion.synchronizer import FrameSynchronizer


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = "/dev/video0"

YOLO_MODEL = os.path.join(
    PROJECT_ROOT,
    "yolo11n.pt"
)

HMR_INTERVAL = 10

YOLO_CONFIDENCE = 0.30

MAX_SYNC_TIME = 0.050

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output"
)


# ============================================================
# LIVE PIPELINE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("              BAS LIVE PERCEPTION PIPELINE")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ========================================================
    # LOAD YOLO
    # ========================================================

    print("[1/4] Loading YOLO11n...")

    yolo = YOLO(
        YOLO_MODEL
    )

    print("[YOLO] Model loaded")
    print("[YOLO] Device: CPU")


    # ========================================================
    # LOAD HMR
    # ========================================================

    print()
    print("[2/4] Loading HMR2...")

    hmr = HMRProcessor()

    print("[HMR] Model loaded")
    print("[HMR] Device: CPU")


    # ========================================================
    # CREATE SYNCHRONIZER
    # ========================================================

    print()
    print("[3/4] Creating synchronizer...")

    synchronizer = FrameSynchronizer(
        max_buffer_size=30,
        max_time_difference=MAX_SYNC_TIME
    )

    print("[SYNC] Synchronizer ready")


    # ========================================================
    # OPEN CAMERA
    # ========================================================

    print()
    print("[4/4] Opening camera...")

    cap = cv2.VideoCapture(
        CAMERA,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera: {CAMERA}"
        )


    # Force MJPEG
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    print(
        f"[CAMERA] {width}x{height} @ {fps:.1f} FPS"
    )


    # ========================================================
    # START
    # ========================================================

    print()
    print("=" * 70)
    print("PIPELINE STARTED")
    print("=" * 70)

    print()
    print(
        f"HMR interval: every {HMR_INTERVAL} frames"
    )

    print(
        "Press Ctrl+C to stop"
    )

    print()


    # ========================================================
    # VARIABLES
    # ========================================================

    frame_id = 0

    total_frames = 0

    yolo_frames = 0

    hmr_frames = 0

    fused_frames = 0


    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Capture frame
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                print(
                    "[CAMERA] Failed to read frame"
                )

                continue


            timestamp = time.time()

            total_frames += 1


            # =================================================
            # YOLO
            # =================================================

            yolo_results = yolo(
                frame,
                device="cpu",
                classes=[0],
                conf=YOLO_CONFIDENCE,
                verbose=False
            )


            persons = []


            for result in yolo_results:

                if result.boxes is None:
                    continue


                for i, box in enumerate(
                    result.boxes
                ):

                    confidence = float(
                        box.conf[0]
                    )


                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )


                    persons.append({

                        "person_id": i,

                        "confidence": confidence,

                        "bbox": {

                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        }

                    })


            # ------------------------------------------------
            # Create YOLO JSON
            # ------------------------------------------------

            yolo_data = {

                "frame_id": frame_id,

                "timestamp": timestamp,

                "model": "YOLO11n",

                "device": "cpu",

                "image": {

                    "width": width,

                    "height": height
                },

                "persons": persons
            }


            yolo_frames += 1


            print(
                f"[YOLO] Frame {frame_id}: "
                f"{len(persons)} person(s)"
            )


            # ------------------------------------------------
            # Add YOLO to synchronizer
            # ------------------------------------------------

            fused = synchronizer.add_yolo(
                yolo_data
            )


            if fused is not None:

                save_fused_frame(
                    fused
                )

                fused_frames += 1


            # =================================================
            # HMR
            # =================================================

            if (
                frame_id % HMR_INTERVAL == 0
                and len(persons) > 0
            ):

                print(
                    f"[HMR] Processing frame "
                    f"{frame_id}..."
                )


                start_hmr = time.time()


                # Convert YOLO boxes
                # into HMR format

                boxes = []


                for person in persons:

                    bbox = person["bbox"]

                    boxes.append([

                        bbox["x1"],
                        bbox["y1"],
                        bbox["x2"],
                        bbox["y2"]

                    ])


                try:

                    hmr_data = hmr.process_frame(

                        frame=frame,

                        frame_id=frame_id,

                        timestamp=timestamp,

                        boxes=boxes

                    )


                    hmr_frames += 1


                    elapsed = (
                        time.time()
                        - start_hmr
                    )


                    print(
                        f"[HMR] Frame {frame_id} "
                        f"completed in "
                        f"{elapsed:.2f}s"
                    )


                    print(
                        f"[HMR] Persons: "
                        f"{len(hmr_data['persons'])}"
                    )


                    # ----------------------------------------
                    # Add HMR to synchronizer
                    # ----------------------------------------

                    fused = (
                        synchronizer.add_hmr(
                            hmr_data
                        )
                    )


                    # ----------------------------------------
                    # FUSED RESULT
                    # ----------------------------------------

                    if fused is not None:

                        fused_frames += 1

                        print()
                        print(
                            "******** FUSED FRAME ********"
                        )

                        print(
                            f"Frame ID: "
                            f"{fused['frame_id']}"
                        )

                        print(
                            f"YOLO persons: "
                            f"{len(fused['yolo']['persons'])}"
                        )

                        print(
                            f"HMR persons: "
                            f"{len(fused['hmr']['persons'])}"
                        )

                        print(
                            f"Time difference: "
                            f"{fused['synchronization']['timestamp_difference']:.4f}s"
                        )

                        print(
                            "*******************************"
                        )

                        print()


                        save_fused_frame(
                            fused
                        )


                except Exception as e:

                    print(
                        f"[HMR ERROR] "
                        f"Frame {frame_id}: {e}"
                    )


            frame_id += 1


    except KeyboardInterrupt:

        print()
        print()
        print("=" * 70)
        print("PIPELINE STOPPED")
        print("=" * 70)


    finally:

        cap.release()


        print()
        print(
            f"Total camera frames : {total_frames}"
        )

        print(
            f"YOLO frames         : {yolo_frames}"
        )

        print(
            f"HMR frames          : {hmr_frames}"
        )

        print(
            f"Fused frames        : {fused_frames}"
        )

        print()


# ============================================================
# SAVE FUSED FRAME
# ============================================================

def save_fused_frame(data):

    frame_id = data["frame_id"]

    filename = (
        f"live_fused_"
        f"{frame_id:06d}.json"
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    with open(
        path,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


    print(
        f"[FUSED] Saved: {filename}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
