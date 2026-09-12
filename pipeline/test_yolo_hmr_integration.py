import time
import queue

from pipeline.camera_worker import CameraWorker
from pipeline.yolo_worker import YOLOWorker
from pipeline.hmr_worker import HMRWorker, LatestFrameQueue


def main():
    print("=" * 70)
    print("YOLO -> HMR INTEGRATION TEST")
    print("=" * 70)

    camera_to_yolo = queue.Queue(maxsize=2)
    yolo_output = queue.Queue(maxsize=30)

    hmr_input = LatestFrameQueue()
    hmr_output = queue.Queue(maxsize=10)

    camera = CameraWorker(
        camera="/dev/video0",
        width=640,
        height=480,
        fps=30,
    )

    yolo = YOLOWorker(
        input_queue=camera_to_yolo,
        output_queue=yolo_output,
        model_path="yolo11n.pt",
        device="cpu",
    )

    hmr = HMRWorker(
        input_queue=hmr_input,
        output_queue=hmr_output,
    )

    frame_cache = {}
    max_cache_size = 30

    frames_captured = 0
    frames_sent_to_yolo = 0
    yolo_results = 0
    hmr_inputs = 0
    hmr_results = 0

    try:
        print()
        print("[TEST] Starting camera...")
        camera.start()

        print("[TEST] Starting YOLO worker...")
        yolo.start()

        print("[TEST] Starting HMR worker...")
        hmr.start()

        print()
        print("[TEST] All workers started")
        print("[TEST] Running integration test...")
        print("[TEST] HMR is CPU-only, so processing will be slow.")
        print()

        start_time = time.time()

        while time.time() - start_time < 35:

            # =================================================
            # CAMERA
            # =================================================

            item = camera.read()

            if item is not None:
                frame_id = item["frame_id"]

                frame_cache[frame_id] = item
                frames_captured += 1

                if len(frame_cache) > max_cache_size:
                    oldest_frame = min(frame_cache.keys())
                    del frame_cache[oldest_frame]

                try:
                    camera_to_yolo.put_nowait(item)
                    frames_sent_to_yolo += 1
                except queue.Full:
                    pass

            # =================================================
            # YOLO -> HMR
            # =================================================

            while True:
                try:
                    yolo_result = yolo_output.get_nowait()
                except queue.Empty:
                    break

                yolo_results += 1

                yolo_frame_id = yolo_result["frame_id"]
                persons = yolo_result.get("persons", [])

                print(
                    f"[YOLO RESULT] Frame {yolo_frame_id} | "
                    f"Persons: {len(persons)}"
                )

                camera_item = frame_cache.get(yolo_frame_id)

                if camera_item is None:
                    print(
                        f"[ADAPTER] Frame {yolo_frame_id} "
                        "no longer in cache"
                    )

                    yolo_output.task_done()
                    continue

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

                if not boxes:
                    print(
                        f"[ADAPTER] Frame {yolo_frame_id}: "
                        "No persons for HMR"
                    )

                    yolo_output.task_done()
                    continue

                hmr_input_data = {
                    "frame": camera_item["frame"],
                    "frame_id": yolo_frame_id,
                    "timestamp": camera_item["timestamp"],
                    "boxes": boxes,
                }

                hmr_input.put(hmr_input_data)
                hmr_inputs += 1

                print(
                    f"[ADAPTER] Frame {yolo_frame_id} -> HMR | "
                    f"persons={len(boxes)}"
                )

                yolo_output.task_done()

            # =================================================
            # HMR OUTPUT
            # =================================================

            while True:
                try:
                    hmr_result = hmr_output.get_nowait()
                except queue.Empty:
                    break

                if hmr_result is None:
                    break

                hmr_results += 1

                result_frame_id = hmr_result.get(
                    "frame_id",
                    -1,
                )

                persons = hmr_result.get(
                    "persons",
                    [],
                )

                print()
                print("=" * 50)
                print("[HMR RESULT]")
                print(f"Frame ID : {result_frame_id}")
                print(f"Persons  : {len(persons)}")

                for person in persons:
                    person_id = person.get(
                        "person_id",
                        -1,
                    )

                    joints = person.get(
                        "joints_3d",
                        [],
                    )

                    camera_translation = person.get(
                        "camera_translation",
                        {},
                    )

                    print(
                        f"Person {person_id}: "
                        f"{len(joints)} joints"
                    )

                    print(
                        f"Camera translation: "
                        f"{camera_translation}"
                    )

                print("=" * 50)
                print()

                hmr_output.task_done()

            time.sleep(0.001)

    except KeyboardInterrupt:
        print()
        print("[TEST] Interrupted by user")

    except Exception as error:
        print()
        print("[TEST] ERROR")
        print(error)

    finally:
        print()
        print("[TEST] Stopping HMR...")
        hmr.stop()

        print("[TEST] Stopping YOLO...")
        yolo.stop()

        print("[TEST] Stopping camera...")
        camera.stop()

    print()
    print("=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)

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
        f"Frames remaining cache : {len(frame_cache)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
