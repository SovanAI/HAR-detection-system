import time
import queue
import json
import os

from pipeline.camera_worker import CameraWorker
from pipeline.yolo_worker import YOLOWorker
from pipeline.hmr_worker import HMRWorker, LatestFrameQueue
from fusion.synchronizer import FrameSynchronizer


def save_fused_frame(fused_data, output_dir):
    frame_id = fused_data["frame_id"]

    filename = os.path.join(
        output_dir,
        f"live_fused_{frame_id:06d}.json",
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            fused_data,
            file,
            indent=2,
        )

    print(
        f"[FUSED] Saved: {filename}"
    )


def main():

    print("=" * 70)
    print("YOLO + HMR SYNCHRONIZATION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Queues
    # ---------------------------------------------------------

    camera_to_yolo = queue.Queue(
        maxsize=2
    )

    yolo_output = queue.Queue(
        maxsize=30
    )

    hmr_input = LatestFrameQueue()

    hmr_output = queue.Queue(
        maxsize=10
    )

    # ---------------------------------------------------------
    # Synchronizer
    # ---------------------------------------------------------

    synchronizer = FrameSynchronizer(
        max_buffer_size=100,
        max_time_difference=0.050,
    )

    # ---------------------------------------------------------
    # Camera
    # ---------------------------------------------------------

    camera = CameraWorker(
        camera="/dev/video0",
        width=640,
        height=480,
        fps=30,
    )

    # ---------------------------------------------------------
    # YOLO
    # ---------------------------------------------------------

    yolo = YOLOWorker(
        input_queue=camera_to_yolo,
        output_queue=yolo_output,
        model_path="yolo11n.pt",
        device="cpu",
    )

    # ---------------------------------------------------------
    # HMR
    # ---------------------------------------------------------

    hmr = HMRWorker(
        input_queue=hmr_input,
        output_queue=hmr_output,
    )

    # ---------------------------------------------------------
    # Frame cache
    # ---------------------------------------------------------

    frame_cache = {}

    max_frame_cache = 100

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    frames_captured = 0
    frames_sent_to_yolo = 0
    yolo_results = 0
    hmr_inputs = 0
    hmr_results = 0
    synchronized_frames = 0

    # ---------------------------------------------------------
    # Control
    # ---------------------------------------------------------

    hmr_frame_sent = False
    sync_success = False

    start_time = time.time()

    output_dir = "output"

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    try:

        # =====================================================
        # START SYSTEM
        # =====================================================

        print()
        print("[SYSTEM] Starting camera...")
        camera.start()

        print("[SYSTEM] Starting YOLO...")
        yolo.start()

        print("[SYSTEM] Starting HMR...")
        hmr.start()

        print()
        print("[SYSTEM] All workers started")
        print()
        print(
            "[SYSTEM] Waiting for YOLO detection..."
        )
        print(
            "[SYSTEM] After one frame is sent to HMR, "
            "the test will wait for HMR."
        )
        print()

        # =====================================================
        # PHASE 1
        # Capture frames and run YOLO
        # =====================================================

        while time.time() - start_time < 15:

            item = camera.read()

            if item is not None:

                frame_id = item["frame_id"]

                frame_cache[frame_id] = item

                frames_captured += 1

                if len(frame_cache) > max_frame_cache:

                    oldest_frame = min(
                        frame_cache.keys()
                    )

                    del frame_cache[
                        oldest_frame
                    ]

                try:

                    camera_to_yolo.put_nowait(
                        item
                    )

                    frames_sent_to_yolo += 1

                except queue.Full:

                    pass

            # -------------------------------------------------
            # Read YOLO results
            # -------------------------------------------------

            while True:

                try:

                    yolo_result = (
                        yolo_output.get_nowait()
                    )

                except queue.Empty:

                    break

                yolo_results += 1

                frame_id = (
                    yolo_result["frame_id"]
                )

                persons = yolo_result.get(
                    "persons",
                    [],
                )

                print(
                    f"[YOLO] Frame {frame_id} | "
                    f"Persons: {len(persons)}"
                )

                # -------------------------------------------------
                # Put YOLO result into synchronizer.
                # -------------------------------------------------

                fused = synchronizer.add_yolo(
                    yolo_result
                )

                if fused is not None:

                    synchronized_frames += 1

                    save_fused_frame(
                        fused,
                        output_dir,
                    )

                    sync_success = True

                # -------------------------------------------------
                # If HMR has not received a frame yet,
                # send this detected frame to HMR.
                # -------------------------------------------------

                if (
                    not hmr_frame_sent
                    and len(persons) > 0
                ):

                    camera_item = frame_cache.get(
                        frame_id
                    )

                    if camera_item is not None:

                        boxes = []

                        for person in persons:

                            bbox = person["bbox"]

                            boxes.append(
                                [
                                    bbox["x1"],
                                    bbox["y1"],
                                    bbox["x2"],
                                    bbox["y2"],
                                ]
                            )

                        if boxes:

                            hmr_input.put(
                                {
                                    "frame":
                                        camera_item["frame"],

                                    "frame_id":
                                        frame_id,

                                    "timestamp":
                                        camera_item["timestamp"],

                                    "boxes":
                                        boxes,
                                }
                            )

                            hmr_inputs += 1
                            hmr_frame_sent = True

                            print()
                            print(
                                "[ADAPTER] Sending ONE frame "
                                "to HMR"
                            )

                            print(
                                f"[ADAPTER] Frame: {frame_id}"
                            )

                            print(
                                f"[ADAPTER] Persons: "
                                f"{len(boxes)}"
                            )

                            print()

                yolo_output.task_done()

            # -------------------------------------------------
            # Check HMR output
            # -------------------------------------------------

            while True:

                try:

                    hmr_result = (
                        hmr_output.get_nowait()
                    )

                except queue.Empty:

                    break

                if hmr_result is None:

                    break

                hmr_results += 1

                hmr_frame_id = (
                    hmr_result["frame_id"]
                )

                persons = hmr_result.get(
                    "persons",
                    [],
                )

                print()
                print("=" * 70)
                print("[HMR RESULT]")
                print(
                    f"Frame ID : {hmr_frame_id}"
                )
                print(
                    f"Persons  : {len(persons)}"
                )
                print("=" * 70)

                # -------------------------------------------------
                # Add HMR result to synchronizer.
                # -------------------------------------------------

                fused = synchronizer.add_hmr(
                    hmr_result
                )

                if fused is not None:

                    synchronized_frames += 1
                    sync_success = True

                    save_fused_frame(
                        fused,
                        output_dir,
                    )

                    print()
                    print("=" * 70)
                    print("[SYNC SUCCESS]")
                    print("=" * 70)

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
                        "Timestamp difference: "
                        f"{fused['synchronization']['timestamp_difference']:.6f} s"
                    )

                    print(
                        "Status: "
                        f"{fused['synchronization']['status']}"
                    )

                    print("=" * 70)
                    print()

                hmr_output.task_done()

            time.sleep(0.001)

        # =====================================================
        # PHASE 2
        # Wait for HMR to finish
        # =====================================================

        if hmr_frame_sent and not sync_success:

            print()
            print("=" * 70)
            print("[SYSTEM] YOLO frame sent to HMR")
            print("[SYSTEM] Waiting for HMR CPU inference...")
            print("=" * 70)
            print()

            hmr_wait_start = time.time()

            while (
                time.time() - hmr_wait_start < 60
                and not sync_success
            ):

                while True:

                    try:

                        hmr_result = (
                            hmr_output.get_nowait()
                        )

                    except queue.Empty:

                        break

                    if hmr_result is None:

                        break

                    hmr_results += 1

                    hmr_frame_id = (
                        hmr_result["frame_id"]
                    )

                    print()
                    print(
                        f"[HMR] Completed frame "
                        f"{hmr_frame_id}"
                    )

                    # ---------------------------------------------
                    # Synchronize HMR result.
                    # ---------------------------------------------

                    fused = synchronizer.add_hmr(
                        hmr_result
                    )

                    if fused is not None:

                        synchronized_frames += 1
                        sync_success = True

                        save_fused_frame(
                            fused,
                            output_dir,
                        )

                        print()
                        print(
                            "=" * 70
                        )
                        print(
                            "[SYNC SUCCESS]"
                        )
                        print(
                            "=" * 70
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
                            "Timestamp difference: "
                            f"{fused['synchronization']['timestamp_difference']:.6f} s"
                        )

                        print(
                            "Status: "
                            f"{fused['synchronization']['status']}"
                        )

                        print(
                            "=" * 70
                        )

                    hmr_output.task_done()

                if sync_success:

                    break

                time.sleep(0.1)

            if not sync_success:

                print()
                print(
                    "[SYSTEM] HMR did not produce a "
                    "synchronized result within 60 seconds."
                )

    except KeyboardInterrupt:

        print()
        print("[SYSTEM] Interrupted by user")

    except Exception as error:

        print()
        print("[SYSTEM] ERROR")
        print(error)

    finally:

        print()
        print("[SYSTEM] Stopping HMR...")
        hmr.stop()

        print("[SYSTEM] Stopping YOLO...")
        yolo.stop()

        print("[SYSTEM] Stopping camera...")
        camera.stop()

    # =========================================================
    # FINAL STATISTICS
    # =========================================================

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("SYNCHRONIZATION TEST COMPLETE")
    print("=" * 70)

    print(
        f"Runtime                : {elapsed:.2f} s"
    )

    print(
        f"Camera frames captured : {frames_captured}"
    )

    print(
        f"Frames sent to YOLO    : {frames_sent_to_yolo}"
    )

    print(
        f"YOLO results           : {yolo_results}"
    )

    print(
        f"HMR inputs             : {hmr_inputs}"
    )

    print(
        f"HMR results            : {hmr_results}"
    )

    print(
        f"Synchronized frames    : {synchronized_frames}"
    )

    print(
        f"YOLO buffer remaining  : "
        f"{synchronizer.yolo_buffer.size()}"
    )

    print(
        f"HMR buffer remaining   : "
        f"{synchronizer.hmr_buffer.size()}"
    )

    if sync_success:

        print()
        print(
            "STATUS: SYNCHRONIZATION SUCCESSFUL"
        )

    else:

        print()
        print(
            "STATUS: SYNCHRONIZATION NOT COMPLETED"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
