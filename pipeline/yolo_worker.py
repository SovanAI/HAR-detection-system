import threading
import queue
import time

from yolo.yolo_processor import YOLOProcessor


class YOLOWorker:
    """
    Independent YOLO worker.

    Receives frames from CameraWorker and processes them
    independently of the HMR worker.
    """

    def __init__(
        self,
        input_queue,
        output_queue,
        model_path="yolo11n.pt",
        device="cpu",
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.model_path = model_path
        self.device = device

        self.processor = None

        self.running = False
        self.thread = None

        self.processed_frames = 0
        self.start_time = None

    def start(self):
        """Start YOLO worker thread."""

        print("[YOLO WORKER] Starting...")

        # Load model once
        self.processor = YOLOProcessor(
            model_path=self.model_path,
            device=self.device,
        )

        self.running = True
        self.start_time = time.time()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

        print("[YOLO WORKER] Started")

    def _run(self):
        """Main YOLO worker loop."""

        while self.running:

            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break

            frame_id = item["frame_id"]
            timestamp = item["timestamp"]
            frame = item["frame"]

            try:
                result = self.processor.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )

                self.output_queue.put(result)

                self.processed_frames += 1

            except Exception as e:

                print(
                    f"[YOLO WORKER] Error on frame "
                    f"{frame_id}: {e}"
                )

            finally:
                self.input_queue.task_done()

    def stop(self):
        """Stop YOLO worker."""

        print("[YOLO WORKER] Stopping...")

        self.running = False

        # Wake worker if waiting
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass

        if self.thread is not None:
            self.thread.join(timeout=2)

        elapsed = time.time() - self.start_time if self.start_time else 0

        fps = (
            self.processed_frames / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"[YOLO WORKER] Processed: "
            f"{self.processed_frames} frames"
        )

        print(
            f"[YOLO WORKER] Average FPS: "
            f"{fps:.2f}"
        )

        print("[YOLO WORKER] Stopped")
