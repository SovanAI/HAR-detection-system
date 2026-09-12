import time

from pipeline.camera_worker import CameraWorker


def main():

    print("=" * 60)
    print("CAMERA WORKER TEST")
    print("=" * 60)

    camera = CameraWorker(
        camera="/dev/video0",
        width=640,
        height=480,
        fps=30,
    )

    try:

        camera.start()

        print()
        print("[TEST] Camera started")
        print("[TEST] Capturing for 10 seconds...")
        print()

        start = time.time()

        while time.time() - start < 10:

            data = camera.read()

            if data is None:
                continue

            print(
                f"[FRAME] "
                f"id={data['frame_id']} | "
                f"time={data['timestamp']:.3f} | "
                f"shape={data['frame'].shape}"
            )

    except KeyboardInterrupt:

        print()
        print("[TEST] Interrupted by user")

    except Exception as e:

        print()
        print(f"[TEST] ERROR: {e}")

    finally:

        camera.stop()

    print()
    print("=" * 60)
    print("CAMERA WORKER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
