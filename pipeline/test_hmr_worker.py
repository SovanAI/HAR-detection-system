import time
import queue

from pipeline.camera_worker import CameraWorker
from pipeline.hmr_worker import HMRWorker, LatestFrameQueue


def main():
    print("=" * 60)
    print("HMR WORKER TEST")
    print("=" * 60)

    hmr_input_queue = LatestFrameQueue()
    hmr_output_queue = queue.Queue(maxsize=30)

    camera = CameraWorker(
    camera="/dev/video0",
    width=640,
    height=480,
    fps=30,
)

    hmr = HMRWorker(
        input_queue=hmr_input_queue,
        output_queue=hmr_output_queue,
    )

    frames_sent = 0
    hmr_results = 0

    try:
        print()
        print("[TEST] Starting camera...")
        camera.start()

        print("[TEST] Starting HMR worker...")
        hmr.start()

        print()
        print("[TEST] Camera + HMR worker started")
        print("[TEST] Running for 15 seconds...")
        print()

        start_time = time.time()

        while time.time() - start_time < 15:
            item = camera.read()

            if item is None:
                time.sleep(0.01)
                continue

            frame = item["frame"]
            frame_id = item["frame_id"]
            timestamp = item["timestamp"]

            height, width = frame.shape[:2]

            # Temporary bounding box covering the whole image.
            # Later this will come from YOLO.
            boxes = [
                [0, 0, width, height]
            ]

            hmr_input = {
                "frame": frame,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "boxes": boxes,
            }

            hmr_input_queue.put(hmr_input)
            frames_sent += 1

            while True:
                try:
                    result = hmr_output_queue.get_nowait()
                except queue.Empty:
                    break

                if result is None:
                    break

                hmr_results += 1

                frame_result_id = result.get("frame_id", -1)
                persons = result.get("persons", [])

                print(
                    f"[RESULT] Frame: {frame_result_id} | "
                    f"Persons: {len(persons)}"
                )

                for person in persons:
                    person_id = person.get("person_id", -1)

                    joints = person.get(
                        "joints_3d",
                        []
                    )

                    camera_translation = person.get(
                        "camera_translation",
                        {}
                    )

                    print(
                        f"         Person {person_id} | "
                        f"Joints: {len(joints)} | "
                        f"Camera: {camera_translation}"
                    )

                hmr_output_queue.task_done()

            time.sleep(0.001)

    except KeyboardInterrupt:
        print()
        print("[TEST] Interrupted by user")

    except Exception as error:
        print()
        print(f"[TEST] ERROR: {error}")

    finally:
        print()
        print("[TEST] Stopping HMR worker...")
        hmr.stop()

        print("[TEST] Stopping camera...")
        camera.stop()

    print()
    print("=" * 60)
    print("HMR WORKER TEST COMPLETE")
    print("=" * 60)

    print(
        f"Camera frames supplied : {frames_sent}"
    )

    print(
        f"HMR results received   : {hmr_results}"
    )

    print(
        f"HMR results remaining  : "
        f"{hmr_output_queue.qsize()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
