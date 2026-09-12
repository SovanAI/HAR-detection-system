import cv2
import time
import threading


class CameraWorker:
    """
    Camera capture worker for the BAS-HMR pipeline.

    Camera:
        Lenovo FHD Webcam

    Configuration:
        Device: /dev/video0
        Backend: V4L2
        Format: MJPG
        Resolution: 640x480
        FPS: 30
    """

    def __init__(
        self,
        camera="/dev/video0",
        width=640,
        height=480,
        fps=30,
    ):
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None

        self.running = False
        self.lock = threading.Lock()

        self.frame_id = 0

        self.start_time = None
        self.frames_captured = 0

    # ---------------------------------------------------------
    # START CAMERA
    # ---------------------------------------------------------

    def start(self):
        """Open and initialize the camera."""

        if self.running:
            print("[CAMERA] Already running")
            return

        print("[CAMERA] Opening camera...")

        # -----------------------------------------------------
        # Open using V4L2
        # -----------------------------------------------------

        self.cap = cv2.VideoCapture(
            self.camera,
            cv2.CAP_V4L2
        )

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None

            raise RuntimeError(
                f"Could not open camera: {self.camera}"
            )

        # -----------------------------------------------------
        # Configure camera
        # -----------------------------------------------------

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            self.fps
        )

        # -----------------------------------------------------
        # Read actual configuration
        # -----------------------------------------------------

        actual_width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        actual_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        print(
            f"[CAMERA] {actual_width}x"
            f"{actual_height} @ "
            f"{actual_fps:.1f} FPS"
        )

        # -----------------------------------------------------
        # Validate camera
        # -----------------------------------------------------

        ret, frame = self.cap.read()

        if not ret or frame is None:

            self.cap.release()
            self.cap = None

            raise RuntimeError(
                "Camera opened but failed to capture "
                "the first frame."
            )

        print(
            f"[CAMERA] First frame OK | "
            f"shape={frame.shape}"
        )

        # -----------------------------------------------------
        # Initialize state
        # -----------------------------------------------------

        self.running = True
        self.start_time = time.time()
        self.frame_id = 0
        self.frames_captured = 0

    # ---------------------------------------------------------
    # READ FRAME
    # ---------------------------------------------------------

    def read(self):
        """
        Capture one frame.

        Returns:

        {
            "frame_id": int,
            "timestamp": float,
            "frame": numpy.ndarray
        }

        Returns None if capture fails.
        """

        if not self.running:
            return None

        with self.lock:

            if self.cap is None:
                return None

            ret, frame = self.cap.read()

        if not ret or frame is None:
            print("[CAMERA] Frame capture failed")
            return None

        timestamp = time.time()

        current_frame_id = self.frame_id

        self.frame_id += 1
        self.frames_captured += 1

        return {
            "frame_id": current_frame_id,
            "timestamp": timestamp,
            "frame": frame,
        }

    # ---------------------------------------------------------
    # GET STATISTICS
    # ---------------------------------------------------------

    def get_stats(self):
        """Return camera statistics."""

        elapsed = 0

        if self.start_time is not None:
            elapsed = time.time() - self.start_time

        fps = 0

        if elapsed > 0:
            fps = self.frames_captured / elapsed

        return {
            "frames_captured": self.frames_captured,
            "elapsed_seconds": elapsed,
            "fps": fps,
        }

    # ---------------------------------------------------------
    # STOP CAMERA
    # ---------------------------------------------------------

    def stop(self):
        """Safely stop camera and release resources."""

        if not self.running and self.cap is None:
            return

        print("[CAMERA] Stopping...")

        self.running = False

        # -----------------------------------------------------
        # Release camera safely
        # -----------------------------------------------------

        with self.lock:

            if self.cap is not None:

                try:
                    self.cap.release()

                except Exception as e:

                    print(
                        f"[CAMERA] Release warning: {e}"
                    )

                finally:

                    self.cap = None

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        stats = self.get_stats()

        print(
            f"[CAMERA] Frames captured: "
            f"{stats['frames_captured']}"
        )

        print(
            f"[CAMERA] Average FPS: "
            f"{stats['fps']:.2f}"
        )

        print("[CAMERA] Camera released")

    # ---------------------------------------------------------
    # CONTEXT MANAGER
    # ---------------------------------------------------------

    def __enter__(self):
        """Allow usage with 'with CameraWorker()'."""

        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        """Automatically release camera."""

        self.stop()
