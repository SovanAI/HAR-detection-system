import cv2
import time


class CameraWorker:

    def __init__(
        self,
        camera="/dev/video0",
        width=640,
        height=480,
        fps=30
    ):

        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None


    def start(self):

        print("[CAMERA] Opening camera...")

        self.cap = cv2.VideoCapture(
            self.camera,
            cv2.CAP_V4L2
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Could not open camera: "
                f"{self.camera}"
            )


        # Force MJPEG
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            )
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


        actual_width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        actual_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )


        print(
            f"[CAMERA] "
            f"{actual_width}x{actual_height} "
            f"@ {actual_fps:.1f} FPS"
        )


    def read(self):

        if self.cap is None:

            raise RuntimeError(
                "Camera has not been started"
            )


        ret, frame = self.cap.read()


        if not ret:

            return None


        timestamp = time.time()


        return {
            "frame_id": None,
            "timestamp": timestamp,
            "frame": frame
        }


    def stop(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None

            print("[CAMERA] Camera released")
