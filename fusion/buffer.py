import threading
import time


class ModelBuffer:

    def __init__(self, name, max_size=30):
        self.name = name
        self.max_size = max_size

        # Stores frames using frame_id as the key
        self.buffer = {}

        # Makes the buffer safe when YOLO and HMR
        # access it from different threads
        self.lock = threading.Lock()

    def add(self, frame_id, data):
        """
        Add model output to the buffer.
        """

        with self.lock:

            # If buffer is full, remove the oldest frame
            if len(self.buffer) >= self.max_size:

                oldest_frame = min(self.buffer.keys())

                del self.buffer[oldest_frame]

                print(
                    f"[{self.name}] "
                    f"Dropped old frame {oldest_frame}"
                )

            # Store the data
            self.buffer[frame_id] = {
                "data": data,
                "received_at": time.time()
            }

            print(
                f"[{self.name}] "
                f"Buffered frame {frame_id}"
            )

    def get(self, frame_id):
        """
        Get a specific frame from the buffer.
        """

        with self.lock:

            item = self.buffer.get(frame_id)

            if item is None:
                return None

            return item["data"]

    def remove(self, frame_id):
        """
        Remove a frame from the buffer.
        """

        with self.lock:

            if frame_id in self.buffer:
                del self.buffer[frame_id]

    def contains(self, frame_id):
        """
        Check whether a frame exists.
        """

        with self.lock:
            return frame_id in self.buffer

    def size(self):
        """
        Return number of buffered frames.
        """

        with self.lock:
            return len(self.buffer)

    def frames(self):
        """
        Return all frame IDs currently in the buffer.
        """

        with self.lock:
            return list(self.buffer.keys())
