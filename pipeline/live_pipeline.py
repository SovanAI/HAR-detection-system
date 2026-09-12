import os
import sys
import time
import json
import threading
import queue

import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from pipeline.camera_worker import CameraWorker
from yolo.yolo_processor import YOLOProcessor
from hmr.hmr_processor import HMRProcessor
from fusion.synchronizer import FrameSynchronizer


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = "/dev/video0"

YOLO_MODEL = os.path.join(
    PROJECT_ROOT,
    "yolo11n.pt",
)

YOLO_CONFIDENCE = 0.30

MAX_SYNC_TIME = 0.050

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output",
)

# Save every synchronized frame as JSON.
SAVE_FUSED_JSON = True


# ============================================================
# LATEST FRAME SLOT
# ============================================================

class LatestHMRQueue:
    """
    Keeps only the newest frame waiting for HMR.

    HMR is extremely slow on CPU, so we never allow
    old frames to accumulate.
    """

    def __init__(self):
        self._item = None
        self._lock = threading.Lock()

    def put_latest(self, item):
        with self._lock:
            self._item = item

    def get_latest(self):
        with self._lock:
            item = self._item
            self._item = None
            return item

    def clear(self):
        with self._lock:
            self._item = None


# ============================================================
# HMR WORKER
# ============================================================

class BackgroundHMR:
    """
    Background HMR processor.

    Only one HMR inference runs at a time.
    A newer frame replaces any older waiting frame.
    """

    def __init__(self, output_callback):
        self.output_callback = output_callback

        self.queue = LatestHMRQueue()

        self.running = False
        self.thread = None

        self.processor = None

        self.processed_frames = 0
        self.start_time = None

    def start(self):
        print("[HMR BG] Starting...")
        print("[HMR BG] Loading HMR2...")

        self.processor = HMRProcessor()

        print("[HMR BG] HMR2 loaded")
        print("[HMR BG] Device: CPU")

        self.running = True
        self.start_time = time.time()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

        print("[HMR BG] Started")

    def submit(
        self,
        frame,
        frame_id,
        timestamp,
        boxes,
        yolo_result,
    ):
        """
        Submit newest frame for HMR.

        The newest frame replaces the previous waiting frame.
        """

        item = {
            "frame": frame,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "boxes": boxes,
            "yolo_result": yolo_result,
        }

        self.queue.put_latest(item)

    def _run(self):
        while self.running:

            item = self.queue.get_latest()

            if item is None:
                time.sleep(0.01)
                continue

            frame = item["frame"]
            frame_id = item["frame_id"]
            timestamp = item["timestamp"]
            boxes = item["boxes"]
            yolo_result = item["yolo_result"]

            try:
                print(
                    f"[HMR BG] Processing frame "
                    f"{frame_id} | persons={len(boxes)}"
                )

                start = time.time()

                result = self.processor.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    boxes=boxes,
                )

                elapsed = time.time() - start

                self.processed_frames += 1

                print(
                    f"[HMR BG] Frame {frame_id} completed | "
                    f"time={elapsed:.2f}s | "
                    f"persons={len(result.get('persons', []))}"
                )

                # Send result back to main pipeline.
                self.output_callback(
                    yolo_result,
                    result,
                )

            except Exception as e:
                print(
                    f"[HMR BG] Error on frame "
                    f"{frame_id}: {type(e).__name__}: {e}"
                )

    def stop(self):
        print("[HMR BG] Stopping...")

        self.running = False
        self.queue.clear()

        if self.thread is not None:
            self.thread.join(timeout=30)

        elapsed = (
            time.time() - self.start_time
            if self.start_time is not None
            else 0
        )

        fps = (
            self.processed_frames / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"[HMR BG] Processed: "
            f"{self.processed_frames} frames"
        )

        print(
            f"[HMR BG] Average FPS: "
            f"{fps:.4f}"
        )

        print("[HMR BG] Stopped")


# ============================================================
# MAIN PIPELINE
# ============================================================

class BASLivePipeline:

    def __init__(self):

        self.camera = None
        self.yolo = None
        self.hmr = None
        self.synchronizer = None

        self.running = False

        self.total_frames = 0
        self.yolo_frames = 0
        self.hmr_submitted = 0
        self.fused_frames = 0

    # --------------------------------------------------------
    # HMR RESULT CALLBACK
    # --------------------------------------------------------

    def handle_hmr_result(
        self,
        yolo_result,
        hmr_result,
    ):
        """
        Called automatically when HMR finishes.
        """

        fused = self.synchronizer.add_yolo(
            yolo_result
        )

        if fused is None:
            fused = self.synchronizer.add_hmr(
                hmr_result
            )

        if fused is None:
            return

        self.fused_frames += 1

        frame_id = fused["frame_id"]

        print()
        print(
            f"[FUSED] Frame {frame_id} synchronized"
        )

        print(
            f"[FUSED] YOLO persons: "
            f"{len(fused['yolo']['persons'])}"
        )

        print(
            f"[FUSED] HMR persons: "
            f"{len(fused['hmr']['persons'])}"
        )

        print(
            f"[FUSED] Timestamp difference: "
            f"{fused['synchronization']['timestamp_difference']:.6f}s"
        )

        if SAVE_FUSED_JSON:
            output_path = os.path.join(
                OUTPUT_DIR,
                f"frame_{frame_id:06d}_fused.json",
            )

            try:
                with open(
                    output_path,
                    "w",
                ) as f:

                    json.dump(
                        fused,
                        f,
                        indent=2,
                    )

                print(
                    f"[FUSED] Saved: {output_path}"
                )

            except Exception as e:
                print(
                    f"[FUSED] JSON save error: {e}"
                )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self):

        print()
        print("=" * 70)
        print("           BAS CPU-SAFE LIVE PIPELINE")
        print("=" * 70)
        print()

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        print("[1/4] Starting camera...")

        self.camera = CameraWorker(
            camera=CAMERA,
            width=640,
            height=480,
            fps=30,
        )

        self.camera.start()

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        print()
        print("[2/4] Loading YOLO...")

        self.yolo = YOLOProcessor(
            model_path=YOLO_MODEL,
            device="cpu",
        )

        print("[YOLO] Ready")

        # ----------------------------------------------------
        # Synchronizer
        # ----------------------------------------------------

        print()
        print("[3/4] Creating synchronizer...")

        self.synchronizer = FrameSynchronizer(
            max_buffer_size=30,
            max_time_difference=MAX_SYNC_TIME,
        )

        print("[SYNC] Ready")

        # ----------------------------------------------------
        # HMR
        # ----------------------------------------------------

        print()
        print("[4/4] Starting background HMR...")

        self.hmr = BackgroundHMR(
            output_callback=self.handle_hmr_result,
        )

        self.hmr.start()

        self.running = True

        print()
        print("=" * 70)
        print("PIPELINE STARTED")
        print("=" * 70)

        print()
        print("Architecture:")
        print("Camera -> YOLO -> Latest HMR -> Synchronizer")
        print()
        print("HMR queue size: 1")
        print("Old waiting frames are discarded.")
        print("Press Ctrl+C to stop.")
        print()

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(self):

        self.start()

        last_status_time = time.time()

        try:

            while self.running:

                # ------------------------------------------------
                # CAPTURE FRAME
                # ------------------------------------------------

                item = self.camera.read()

                if item is None:
                    continue

                frame = item["frame"]
                frame_id = item["frame_id"]
                timestamp = item["timestamp"]

                self.total_frames += 1

                # ------------------------------------------------
                # YOLO
                # ------------------------------------------------

                yolo_result = self.yolo.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )

                self.yolo_frames += 1

                persons = yolo_result.get(
                    "persons",
                    [],
                )

                # ------------------------------------------------
                # NO PERSON
                # ------------------------------------------------

                if len(persons) == 0:

                    # Nothing useful to send to HMR.
                    continue

                # ------------------------------------------------
                # CREATE HMR BOXES
                # ------------------------------------------------

                boxes = []

                for person in persons:

                    bbox = person["bbox"]

                    boxes.append([
                        bbox["x1"],
                        bbox["y1"],
                        bbox["x2"],
                        bbox["y2"],
                    ])

                # ------------------------------------------------
                # SUBMIT NEWEST FRAME TO HMR
                # ------------------------------------------------

                self.hmr.submit(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    boxes=np.asarray(
                        boxes,
                        dtype=np.float32,
                    ),
                    yolo_result=yolo_result,
                )

                self.hmr_submitted += 1

                # ------------------------------------------------
                # STATUS
                # ------------------------------------------------

                now = time.time()

                if now - last_status_time >= 5:

                    print()
                    print(
                        "[STATUS]"
                    )

                    print(
                        f"Camera frames : "
                        f"{self.total_frames}"
                    )

                    print(
                        f"YOLO frames   : "
                        f"{self.yolo_frames}"
                    )

                    print(
                        f"HMR submitted : "
                        f"{self.hmr_submitted}"
                    )

                    print(
                        f"Fused frames  : "
                        f"{self.fused_frames}"
                    )

                    last_status_time = now

        except KeyboardInterrupt:

            print()
            print("[SYSTEM] Ctrl+C received")

        finally:

            self.stop()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self):

        if not self.running:
            return

        self.running = False

        print()
        print("=" * 70)
        print("STOPPING PIPELINE")
        print("=" * 70)

        # ----------------------------------------------------
        # HMR
        # ----------------------------------------------------

        if self.hmr is not None:

            try:
                self.hmr.stop()
            except Exception as e:
                print(
                    f"[HMR] Stop warning: {e}"
                )

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        if self.camera is not None:

            try:
                self.camera.stop()
            except Exception as e:
                print(
                    f"[CAMERA] Stop warning: {e}"
                )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("PIPELINE STATISTICS")
        print("=" * 70)

        print(
            f"Camera frames captured : "
            f"{self.total_frames}"
        )

        print(
            f"YOLO frames processed  : "
            f"{self.yolo_frames}"
        )

        print(
            f"HMR frames submitted   : "
            f"{self.hmr_submitted}"
        )

        print(
            f"Fused frames           : "
            f"{self.fused_frames}"
        )

        print("=" * 70)
        print("PIPELINE STOPPED")
        print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    pipeline = BASLivePipeline()

    pipeline.run()


if __name__ == "__main__":
    main()