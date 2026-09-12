import threading
import time

from hmr.hmr_processor import HMRProcessor


class LatestFrameQueue:
    """
    Thread-safe latest-frame queue.

    Only the newest item is retained.

    This prevents HMR from processing stale camera frames
    when HMR inference is slower than the camera.
    """

    def __init__(self):
        self._item = None
        self._lock = threading.Lock()

    def put(self, item):
        """
        Store the newest item.

        Any previous waiting item is replaced.
        """
        with self._lock:
            self._item = item

    def get_latest(self):
        """
        Return and remove the newest item.

        Returns None if no item is available.
        """
        with self._lock:
            item = self._item
            self._item = None
            return item

    def empty(self):
        with self._lock:
            return self._item is None


class HMRWorker:
    """
    Independent HMR2 worker.

    Input item:

        {
            "frame_id": int,
            "timestamp": float,
            "frame": numpy.ndarray,
            "boxes": [
                [x1, y1, x2, y2],
                ...
            ]
        }

    Output:

        HMRProcessor result dictionary.
    """

    def __init__(
        self,
        input_queue,
        output_queue,
    ):

        self.input_queue = input_queue
        self.output_queue = output_queue

        self.processor = None

        self.running = False
        self.thread = None

        self.processed_frames = 0
        self.start_time = None

        self.last_frame_id = None

    # =========================================================
    # START
    # =========================================================

    def start(self):

        if self.running:
            print("[HMR WORKER] Already running")
            return

        print("[HMR WORKER] Starting...")
        print("[HMR WORKER] Loading HMR2 model...")

        # Your HMRProcessor.__init__ takes no arguments.
        self.processor = HMRProcessor()

        print("[HMR WORKER] HMR2 model loaded")

        self.running = True
        self.start_time = time.time()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="HMRWorker",
        )

        self.thread.start()

        print("[HMR WORKER] Started")

    # =========================================================
    # WORKER LOOP
    # =========================================================

    def _run(self):

        while self.running:

            item = self.input_queue.get_latest()

            # No new frame
            if item is None:
                time.sleep(0.005)
                continue

            frame_id = item["frame_id"]
            timestamp = item["timestamp"]
            frame = item["frame"]
            boxes = item["boxes"]

            self.last_frame_id = frame_id

            processing_start = time.time()

            try:

                # -------------------------------------------------
                # No person detected
                # -------------------------------------------------

                if boxes is None or len(boxes) == 0:

                    print(
                        f"[HMR WORKER] Frame "
                        f"{frame_id}: no persons"
                    )

                    result = self.processor.process_frame(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        boxes=[],
                    )

                # -------------------------------------------------
                # Persons detected
                # -------------------------------------------------

                else:

                    print(
                        f"[HMR WORKER] Processing "
                        f"frame {frame_id} | "
                        f"persons={len(boxes)}"
                    )

                    result = self.processor.process_frame(
                        frame=frame,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        boxes=boxes,
                    )

                # -------------------------------------------------
                # Timing
                # -------------------------------------------------

                processing_time = (
                    time.time() - processing_start
                )

                result["worker"] = {
                    "processing_time": processing_time,
                }

                # -------------------------------------------------
                # Send result
                # -------------------------------------------------

                self.output_queue.put(result)

                self.processed_frames += 1

                print(
                    f"[HMR WORKER] Frame "
                    f"{frame_id} completed | "
                    f"time={processing_time:.2f}s | "
                    f"persons="
                    f"{len(result.get('persons', []))}"
                )

            except Exception as e:

                print(
                    f"[HMR WORKER] ERROR | "
                    f"frame={frame_id} | "
                    f"{type(e).__name__}: {e}"
                )

    # =========================================================
    # STOP
    # =========================================================

    def stop(self):

        if not self.running:
            return

        print("[HMR WORKER] Stopping...")

        self.running = False

        # HMR inference may currently be running.
        # Give it enough time to finish cleanly.
        if self.thread is not None:

            self.thread.join(timeout=10)

            if self.thread.is_alive():

                print(
                    "[HMR WORKER] Warning: "
                    "worker still running."
                )

        elapsed = 0

        if self.start_time is not None:

            elapsed = (
                time.time() - self.start_time
            )

        fps = 0

        if elapsed > 0:

            fps = (
                self.processed_frames /
                elapsed
            )

        print(
            f"[HMR WORKER] Processed: "
            f"{self.processed_frames} frames"
        )

        print(
            f"[HMR WORKER] Average FPS: "
            f"{fps:.2f}"
        )

        print("[HMR WORKER] Stopped")
