from camera_worker import CameraWorker


def main():

    camera = CameraWorker()

    camera.start()

    print("[TEST] Reading 10 frames...")


    for frame_id in range(10):

        data = camera.read()

        if data is None:

            print(
                "[TEST] Failed to read frame"
            )

            continue


        data["frame_id"] = frame_id


        frame = data["frame"]


        print(
            f"[TEST] Frame {frame_id} "
            f"| shape={frame.shape} "
            f"| timestamp={data['timestamp']}"
        )


    camera.stop()

    print("[TEST] Camera worker OK")


if __name__ == "__main__":
    main()
