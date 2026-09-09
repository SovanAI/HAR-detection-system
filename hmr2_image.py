import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
HMR2_ROOT = ROOT / "4D-Humans"

sys.path.insert(0, str(HMR2_ROOT))

from hmr2.models import load_hmr2
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils import recursive_to


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHECKPOINT = (
    Path.home()
    / ".cache"
    / "4DHumans"
    / "logs"
    / "train"
    / "multiruns"
    / "hmr2"
    / "0"
    / "checkpoints"
    / "epoch=35-step=1000000.ckpt"
)

OUTPUT_DIR = ROOT / "output"

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Get image from command line
    # -----------------------------------------------------

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python hmr2_image.py <image_path>")
        print()
        print("Example:")
        print("  python hmr2_image.py input/astronaut.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1]).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"HMR2 checkpoint not found:\n{CHECKPOINT}"
        )

    print("=" * 60)
    print("BAS HUMAN 3D POSE PIPELINE")
    print("=" * 60)

    print(f"[1] Image      : {image_path}")
    print(f"[2] Device     : {DEVICE}")

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    img_cv2 = cv2.imread(str(image_path))

    if img_cv2 is None:
        raise RuntimeError(
            f"OpenCV could not read:\n{image_path}"
        )

    print(f"[3] Image shape: {img_cv2.shape}")

    # -----------------------------------------------------
    # YOLO
    # -----------------------------------------------------

    print("\n[4] Loading YOLO...")

    yolo = YOLO("yolo11n.pt")

    print("[5] Detecting people...")

    results = yolo(
        img_cv2,
        device="cpu",
        classes=[0],
        verbose=False
    )

    boxes = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0].cpu().numpy()
            )

            if confidence < 0.5:
                continue

            xyxy = box.xyxy[0].cpu().numpy()

            boxes.append(xyxy)

            print(
                f"    Person | "
                f"confidence={confidence:.3f} | "
                f"box={xyxy.astype(int)}"
            )

    if not boxes:
        print("\nNo person detected.")
        return

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    )

    print(f"\n[6] Persons detected: {len(boxes)}")

    # -----------------------------------------------------
    # Load HMR2
    # -----------------------------------------------------

    print("\n[7] Loading HMR2...")

    model, model_cfg = load_hmr2(
        str(CHECKPOINT)
    )

    model = model.to(DEVICE)
    model.eval()

    print("[8] HMR2 loaded.")

    # -----------------------------------------------------
    # HMR2 preprocessing
    # -----------------------------------------------------

    print("\n[9] Preparing person crops...")

    dataset = ViTDetDataset(
        model_cfg,
        img_cv2,
        boxes
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # HMR2 inference
    # -----------------------------------------------------

    print("\n[10] Running 3D pose estimation...")

    all_joints = []
    all_vertices = []
    all_camera = []

    with torch.no_grad():

        for person_id, batch in enumerate(dataloader):

            batch = recursive_to(
                batch,
                DEVICE
            )

            output = model(batch)

            joints = (
                output["pred_keypoints_3d"]
                .cpu()
                .numpy()
            )

            vertices = (
                output["pred_vertices"]
                .cpu()
                .numpy()
            )

            camera = (
                output["pred_cam_t"]
                .cpu()
                .numpy()
            )

            all_joints.append(joints)
            all_vertices.append(vertices)
            all_camera.append(camera)

            print(
                f"    Person {person_id}: "
                f"joints={joints.shape}, "
                f"vertices={vertices.shape}"
            )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    joints = np.concatenate(
        all_joints,
        axis=0
    )

    vertices = np.concatenate(
        all_vertices,
        axis=0
    )

    camera = np.concatenate(
        all_camera,
        axis=0
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR / "hmr2_result.npz"
    )

    np.savez(
        output_file,
        boxes=boxes,
        joints_3d=joints,
        vertices=vertices,
        camera_translation=camera
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(f"Persons       : {len(boxes)}")
    print(f"3D joints     : {joints.shape}")
    print(f"SMPL vertices : {vertices.shape}")
    print(f"Camera        : {camera.shape}")

    print(f"\nSaved:")
    print(output_file)


if __name__ == "__main__":
    main()
