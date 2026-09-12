import queue
import time

from pipeline.camera_worker import CameraWorker
from pipeline.yolo_worker import YOLOWorker


def main():

    print("=" * 60)
    print("YOLO WORKER TEST")
    print("=" * 60)

    # Queues
    camera_queue = queue.Queue(maxsize=10)
    yolo_output_queue = queue.Queue(maxsize=30)

    # Camera
    camera = CameraWorker(
        camera="/dev/video0"
    )

    # YOLO
    yolo = YOLOWorker(
        input_queue=camera_queue,
        output_queue=yolo_output_queue,
        model_path="yolo11n.pt",
        device="cpu",
    )

    camera.start()
    yolo.start()

    print()
    print("[TEST] Camera + YOLO worker started")
    print("[TEST] Processing for 10 seconds...")
    print()

    start = time.time()

    frames_sent = 0

    try:

        while time.time() - start < 10:

            # Get camera frame
            item = camera.read()

            if item is None:
                continue

            # Don't allow queue to grow forever
            if camera_queue.full():

                try:
                    camera_queue.get_nowait()
                    camera_queue.task_done()
                except queue.Empty:
                    pass

            camera_queue.put(item)

            frames_sent += 1

            # Read YOLO results
            while True:

                try:
                    result = yolo_output_queue.get_nowait()
                except queue.Empty:
                    break

                print(
                    f"[RESULT] Frame "
                    f"{result['frame_id']} | "
                    f"Persons: "
                    f"{len(result['persons'])}"
                )

                yolo_output_queue.task_done()

    except KeyboardInterrupt:

        print("\n[TEST] Interrupted")

    finally:

        yolo.stop()
        camera.stop()

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    print(f"Camera frames sent: {frames_sent}")
    print(f"YOLO results remaining: {yolo_output_queue.qsize()}")


if __name__ == "__main__":
    main()
