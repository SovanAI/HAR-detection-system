import cv2
import time


class FrameManager:

    def __init__(self, camera_index="/dev/video0"):

        self.camera_index = camera_index

        print("[CAMERA] Opening camera...")
        print(f"[CAMERA] Device: {camera_index}")

        self.cap = cv2.VideoCapture(
            camera_index,
            cv2.CAP_V4L2
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[CAMERA] Could not open {camera_index}"
            )

        # Force MJPG
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        # Force resolution
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        # Force FPS
        self.cap.set(
            cv2.CAP_PROP_FPS,
            30
        )

        # Read back actual configuration
        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        fourcc = int(
            self.cap.get(cv2.CAP_PROP_FOURCC)
        )

        fourcc_text = "".join(
            [
                chr((fourcc >> 0) & 0xFF),
                chr((fourcc >> 8) & 0xFF),
                chr((fourcc >> 16) & 0xFF),
                chr((fourcc >> 24) & 0xFF),
            ]
        )

        print(
            f"[CAMERA] Resolution: "
            f"{self.width}x{self.height}"
        )

        print(
            f"[CAMERA] FPS: {self.fps}"
        )

        print(
            f"[CAMERA] Format: {fourcc_text}"
        )

        self.frame_id = 0

    def read_frame(self):

        success, frame = self.cap.read()

        if not success:
            return None

        # Validate frame
        if frame is None:
            return None

        if frame.size == 0:
            return None

        timestamp = time.time()

        current_frame_id = self.frame_id

        self.frame_id += 1

        return (
            current_frame_id,
            timestamp,
            frame
        )

    def release(self):

        print("[CAMERA] Releasing camera...")

        self.cap.release()


if __name__ == "__main__":

    print("====================================")
    print(" CAMERA FRAME MANAGER TEST")
    print("====================================")

    camera = FrameManager(
        camera_index="/dev/video0"
    )

    print()
    print("Capturing frames...")
    print("Press Ctrl+C to stop.")
    print()

    try:

        while True:

            result = camera.read_frame()

            if result is None:

                print(
                    "[CAMERA] Failed to read frame."
                )

                break

            frame_id, timestamp, frame = result

            print(
                f"[CAMERA] Frame "
                f"{frame_id:06d} | "
                f"timestamp={timestamp:.3f} | "
                f"shape={frame.shape}"
            )

            # Save the first good frame
            if frame_id == 10:

                cv2.imwrite(
                    "input/live_test_mjpg.jpg",
                    frame
                )

                print(
                    "[CAMERA] Saved "
                    "input/live_test_mjpg.jpg"
                )

    except KeyboardInterrupt:

        print(
            "\n[CAMERA] Test interrupted."
        )

    finally:

        camera.release()

        print(
            "[CAMERA] Camera test completed."
        )
