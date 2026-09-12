import json
import os
import time

from pipeline.camera_worker import CameraWorker
from yolo.yolo_processor import YOLOProcessor
from hmr.hmr_processor import HMRProcessor
from fusion.synchronizer import FrameSynchronizer


def main():
    print("=" * 70)
    print("ONE-FRAME YOLO -> HMR -> SYNCHRONIZER TEST")
    print("=" * 70)

    camera = CameraWorker(
        camera="/dev/video0",
        width=640,
        height=480,
        fps=30,
    )

    yolo = None
    hmr = None

    try:
        # ---------------------------------------------------------
        # START CAMERA
        # ---------------------------------------------------------

        print("\n[1] Starting camera...")
        camera.start()

        # Capture exactly ONE frame
        item = camera.read()

        if item is None:
            raise RuntimeError("Failed to capture frame")

        frame_id = item["frame_id"]
        timestamp = item["timestamp"]
        frame = item["frame"]

        print(
            f"[CAMERA] Frame captured | "
            f"id={frame_id} | "
            f"timestamp={timestamp:.6f} | "
            f"shape={frame.shape}"
        )

        # ---------------------------------------------------------
        # LOAD YOLO
        # ---------------------------------------------------------

        print("\n[2] Loading YOLO...")
        yolo = YOLOProcessor(
            model_path="yolo11n.pt",
            device="cpu",
        )

        # ---------------------------------------------------------
        # RUN YOLO
        # ---------------------------------------------------------

        print("\n[3] Running YOLO...")

        yolo_result = yolo.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        print(
            f"[YOLO] Frame {frame_id} | "
            f"Persons={len(yolo_result['persons'])}"
        )

        for person in yolo_result["persons"]:
            print(
                f"        Person {person['person_id']} | "
                f"confidence={person['confidence']:.3f} | "
                f"bbox={person['bbox']}"
            )

        # ---------------------------------------------------------
        # CHECK PERSON DETECTION
        # ---------------------------------------------------------

        if len(yolo_result["persons"]) == 0:
            print("\n[WARNING] YOLO found no person.")
            print("[WARNING] HMR will not be executed.")
            print("[WARNING] Keep a person clearly visible and run again.")

            return

        # ---------------------------------------------------------
        # EXTRACT YOLO BOUNDING BOXES
        # ---------------------------------------------------------

        boxes = []

        for person in yolo_result["persons"]:
            bbox = person["bbox"]

            boxes.append([
                bbox["x1"],
                bbox["y1"],
                bbox["x2"],
                bbox["y2"],
            ])

        print(
            f"\n[YOLO -> HMR] Sending "
            f"{len(boxes)} person bounding box(es)"
        )

        # ---------------------------------------------------------
        # LOAD HMR
        # ---------------------------------------------------------

        print("\n[4] Loading HMR2...")
        print("[HMR] CPU-only inference may take several seconds.")

        hmr = HMRProcessor()

        # ---------------------------------------------------------
        # RUN HMR
        # ---------------------------------------------------------

        print("\n[5] Running HMR2...")

        hmr_start = time.time()

        hmr_result = hmr.process_frame(
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp,
            boxes=boxes,
        )

        hmr_elapsed = time.time() - hmr_start

        print(
            f"[HMR] Completed in {hmr_elapsed:.2f} seconds"
        )

        print(
            f"[HMR] Frame {hmr_result['frame_id']} | "
            f"Persons={len(hmr_result['persons'])}"
        )

        for person in hmr_result["persons"]:
            print(
                f"        Person {person['person_id']} | "
                f"joints={person['pose']['joint_count']}"
            )

        # ---------------------------------------------------------
        # SYNCHRONIZATION
        # ---------------------------------------------------------

        print("\n[6] Running FrameSynchronizer...")

        synchronizer = FrameSynchronizer(
            max_buffer_size=30,
            max_time_difference=0.050,
        )

        fused_result = synchronizer.add_yolo(yolo_result)

        if fused_result is None:
            fused_result = synchronizer.add_hmr(hmr_result)

        if fused_result is None:
            print("\n[SYNC] Synchronization FAILED")
            print(
                f"[SYNC] YOLO frame_id = "
                f"{yolo_result['frame_id']}"
            )
            print(
                f"[SYNC] HMR frame_id = "
                f"{hmr_result['frame_id']}"
            )

            print(
                f"[SYNC] YOLO timestamp = "
                f"{yolo_result['timestamp']:.6f}"
            )
            print(
                f"[SYNC] HMR timestamp = "
                f"{hmr_result['timestamp']:.6f}"
            )

            return

        # ---------------------------------------------------------
        # SUCCESS
        # ---------------------------------------------------------

        print("\n" + "=" * 70)
        print("SYNCHRONIZATION SUCCESS")
        print("=" * 70)

        print(
            f"Frame ID              : "
            f"{fused_result['frame_id']}"
        )

        print(
            f"Timestamp difference  : "
            f"{fused_result['synchronization']['timestamp_difference']:.6f} s"
        )

        print(
            f"YOLO persons          : "
            f"{len(fused_result['yolo']['persons'])}"
        )

        print(
            f"HMR persons           : "
            f"{len(fused_result['hmr']['persons'])}"
        )

        # ---------------------------------------------------------
        # SAVE FUSED JSON
        # ---------------------------------------------------------

        os.makedirs("output", exist_ok=True)

        output_path = (
            f"output/frame_{frame_id:06d}_fused.json"
        )

        with open(output_path, "w") as f:
            json.dump(
                fused_result,
                f,
                indent=2,
            )

        print(
            f"\n[SAVED] {output_path}"
        )

        print("\n" + "=" * 70)
        print("ONE-FRAME TEST COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")

    except Exception as e:
        print("\n[ERROR] Test failed")
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise

    finally:
        print("\n[CLEANUP] Stopping camera...")

        try:
            camera.stop()
        except Exception as e:
            print(f"[CAMERA] Stop warning: {e}")

        print("[CLEANUP] Complete")


if __name__ == "__main__":
    main()
