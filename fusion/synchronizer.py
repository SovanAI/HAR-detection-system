from fusion.buffer import ModelBuffer


class FrameSynchronizer:

    def __init__(
        self,
        max_buffer_size=30,
        max_time_difference=0.050
    ):
        """
        Synchronizes YOLO and HMR results
        using frame_id and timestamp.
        """

        self.yolo_buffer = ModelBuffer(
            name="YOLO",
            max_size=max_buffer_size
        )

        self.hmr_buffer = ModelBuffer(
            name="HMR",
            max_size=max_buffer_size
        )

        # Maximum allowed timestamp difference.
        # 0.050 = 50 milliseconds.
        self.max_time_difference = max_time_difference


    def add_yolo(self, data):
        """
        Add YOLO result to the buffer
        and check whether HMR is already available.
        """

        frame_id = data["frame_id"]

        self.yolo_buffer.add(
            frame_id,
            data
        )

        return self.try_sync(frame_id)


    def add_hmr(self, data):
        """
        Add HMR result to the buffer
        and check whether YOLO is already available.
        """

        frame_id = data["frame_id"]

        self.hmr_buffer.add(
            frame_id,
            data
        )

        return self.try_sync(frame_id)


    def try_sync(self, frame_id):
        """
        Try to synchronize YOLO and HMR
        for the same frame_id.
        """

        # Get YOLO result
        yolo_data = self.yolo_buffer.get(
            frame_id
        )

        # Get HMR result
        hmr_data = self.hmr_buffer.get(
            frame_id
        )

        # If either result is missing,
        # synchronization cannot happen yet.
        if yolo_data is None or hmr_data is None:

            return None


        # -----------------------------------------
        # Compare timestamps
        # -----------------------------------------

        yolo_timestamp = yolo_data["timestamp"]

        hmr_timestamp = hmr_data["timestamp"]

        timestamp_difference = abs(
            yolo_timestamp - hmr_timestamp
        )


        # -----------------------------------------
        # Check timing
        # -----------------------------------------

        if (
            timestamp_difference
            > self.max_time_difference
        ):

            print(
                f"[SYNC] Frame {frame_id} rejected"
            )

            print(
                f"[SYNC] Timestamp difference: "
                f"{timestamp_difference:.4f}s"
            )

            return None


        # -----------------------------------------
        # Create fused frame
        # -----------------------------------------

        fused_data = {

            "frame_id": frame_id,

            "timestamp": max(
                yolo_timestamp,
                hmr_timestamp
            ),

            "synchronization": {

                "status": "synchronized",

                "timestamp_difference":
                    timestamp_difference
            },

            "yolo": yolo_data,

            "hmr": hmr_data
        }


        # -----------------------------------------
        # Remove synchronized frames
        # -----------------------------------------

        self.yolo_buffer.remove(
            frame_id
        )

        self.hmr_buffer.remove(
            frame_id
        )


        print(
            f"[SYNC] Frame {frame_id} synchronized"
        )


        return fused_data
