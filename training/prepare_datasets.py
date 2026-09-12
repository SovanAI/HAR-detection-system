from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path("/home/hawkeye/bas-hmr")

SOURCE = ROOT / "datasets/source"

CARD_SOURCE = (
    SOURCE / "Cardboard Box.v1-carboard-dataset1.coco"
)

HAND_SOURCE = (
    SOURCE / "Hand.v8i.coco"
)

CARD_OUT = (
    ROOT / "datasets/cardboard_yolo"
)

HAND_OUT = (
    ROOT / "datasets/hand_pose"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

HAND_VAL_RATIO = 0.10


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def reset_directory(path: Path) -> None:

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )


def ensure_dir(path: Path) -> None:

    path.mkdir(
        parents=True,
        exist_ok=True
    )


def write_text(
    path: Path,
    text: str
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        text,
        encoding="utf-8"
    )


# ============================================================
# COCO BBOX -> YOLO BBOX
# ============================================================

def bbox_to_yolo(
    bbox,
    image_width,
    image_height
):

    x = float(bbox[0])
    y = float(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    center_x = x + w / 2.0
    center_y = y + h / 2.0

    return (
        center_x / image_width,
        center_y / image_height,
        w / image_width,
        h / image_height
    )


# ============================================================
# PREPARE CARDBOARD DATASET
# ============================================================

def prepare_cardboard():

    print()
    print("=" * 60)
    print("PREPARING CARDBOARD DATASET")
    print("=" * 60)

    if not CARD_SOURCE.exists():

        raise FileNotFoundError(
            f"Cardboard dataset not found:\n{CARD_SOURCE}"
        )

    reset_directory(
        CARD_OUT
    )

    for split in [
        "train",
        "valid",
        "test"
    ]:

        source_split = (
            CARD_SOURCE / split
        )

        if not source_split.exists():

            print(
                f"[WARNING] Missing split: "
                f"{source_split}"
            )

            continue

        if split == "valid":
            output_split = "val"
        else:
            output_split = split

        image_out = (
            CARD_OUT
            / "images"
            / output_split
        )

        label_out = (
            CARD_OUT
            / "labels"
            / output_split
        )

        ensure_dir(image_out)
        ensure_dir(label_out)

        annotation_file = (
            source_split
            / "_annotations.coco.json"
        )

        if not annotation_file.exists():

            raise FileNotFoundError(
                f"Missing annotation file:\n"
                f"{annotation_file}"
            )

        with annotation_file.open(
            "r",
            encoding="utf-8"
        ) as file:

            coco = json.load(file)

        images = {
            image["id"]: image
            for image in coco["images"]
        }

        annotations_by_image = {}

        for annotation in coco["annotations"]:

            image_id = annotation["image_id"]

            annotations_by_image.setdefault(
                image_id,
                []
            ).append(
                annotation
            )

        image_count = 0
        annotation_count = 0

        for image_id, image_info in images.items():

            filename = image_info["file_name"]

            source_image = (
                source_split / filename
            )

            if not source_image.exists():

                print(
                    f"[WARNING] Missing image: "
                    f"{source_image}"
                )

                continue

            destination_image = (
                image_out / filename
            )

            shutil.copy2(
                source_image,
                destination_image
            )

            width = float(
                image_info["width"]
            )

            height = float(
                image_info["height"]
            )

            label_lines = []

            for annotation in annotations_by_image.get(
                image_id,
                []
            ):

                # Both cardboard categories are
                # named "box".
                #
                # We intentionally collapse them
                # into one YOLO class:
                #
                # 0 = box

                class_id = 0

                (
                    center_x,
                    center_y,
                    box_width,
                    box_height
                ) = bbox_to_yolo(
                    annotation["bbox"],
                    width,
                    height
                )

                label_lines.append(
                    (
                        f"{class_id} "
                        f"{center_x:.6f} "
                        f"{center_y:.6f} "
                        f"{box_width:.6f} "
                        f"{box_height:.6f}"
                    )
                )

                annotation_count += 1

            label_file = (
                label_out
                / f"{Path(filename).stem}.txt"
            )

            write_text(
                label_file,
                "\n".join(label_lines)
                + (
                    "\n"
                    if label_lines
                    else ""
                )
            )

            image_count += 1

        print(
            f"[{output_split}] "
            f"images={image_count} "
            f"annotations={annotation_count}"
        )

    # --------------------------------------------------------
    # YOLO data.yaml
    # --------------------------------------------------------

    yaml_text = """path: /home/hawkeye/bas-hmr/datasets/cardboard_yolo

train: images/train
val: images/val
test: images/test

names:
  0: box
"""

    write_text(
        CARD_OUT / "data.yaml",
        yaml_text
    )

    print(
        "[OK] Cardboard dataset ready."
    )


# ============================================================
# PREPARE HAND DATASET
# ============================================================

def prepare_hand():

    print()
    print("=" * 60)
    print("PREPARING HAND POSE DATASET")
    print("=" * 60)

    if not HAND_SOURCE.exists():

        raise FileNotFoundError(
            f"Hand dataset not found:\n{HAND_SOURCE}"
        )

    train_source = (
        HAND_SOURCE / "train"
    )

    annotation_file = (
        train_source
        / "_annotations.coco.json"
    )

    if not annotation_file.exists():

        raise FileNotFoundError(
            f"Missing hand annotation file:\n"
            f"{annotation_file}"
        )

    reset_directory(
        HAND_OUT
    )

    with annotation_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        coco = json.load(file)

    # --------------------------------------------------------
    # Find the hand category.
    # --------------------------------------------------------

    hand_category_ids = {
        category["id"]
        for category in coco["categories"]
        if category["name"].lower() == "hand"
    }

    if not hand_category_ids:

        raise RuntimeError(
            "No 'hand' category found in COCO dataset."
        )

    print(
        f"[INFO] Hand category IDs: "
        f"{hand_category_ids}"
    )

    images = list(
        coco["images"]
    )

    annotations_by_image = {}

    for annotation in coco["annotations"]:

        if (
            annotation["category_id"]
            not in hand_category_ids
        ):
            continue

        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(
            annotation
        )

    # --------------------------------------------------------
    # Create validation split.
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        images
    )

    val_count = max(
        1,
        int(
            len(images)
            * HAND_VAL_RATIO
        )
    )

    val_images = images[
        :val_count
    ]

    train_images = images[
        val_count:
    ]

    print(
        f"[INFO] Train images: "
        f"{len(train_images)}"
    )

    print(
        f"[INFO] Validation images: "
        f"{len(val_images)}"
    )

    # ========================================================
    # PROCESS SPLIT
    # ========================================================

    def process_split(
        split_name,
        split_images
    ):

        image_out = (
            HAND_OUT
            / "images"
            / split_name
        )

        label_out = (
            HAND_OUT
            / "labels"
            / split_name
        )

        ensure_dir(
            image_out
        )

        ensure_dir(
            label_out
        )

        image_count = 0
        hand_count = 0

        for image_info in split_images:

            filename = image_info["file_name"]

            source_image = (
                train_source / filename
            )

            if not source_image.exists():

                print(
                    f"[WARNING] Missing image: "
                    f"{source_image}"
                )

                continue

            destination_image = (
                image_out / filename
            )

            shutil.copy2(
                source_image,
                destination_image
            )

            width = float(
                image_info["width"]
            )

            height = float(
                image_info["height"]
            )

            label_lines = []

            for annotation in annotations_by_image.get(
                image_info["id"],
                []
            ):

                bbox = annotation["bbox"]

                (
                    center_x,
                    center_y,
                    box_width,
                    box_height
                ) = bbox_to_yolo(
                    bbox,
                    width,
                    height
                )

                keypoints = annotation.get(
                    "keypoints",
                    []
                )

                if len(keypoints) != 63:

                    print(
                        f"[WARNING] "
                        f"{filename}: "
                        f"expected 63 keypoint "
                        f"values, got "
                        f"{len(keypoints)}"
                    )

                    continue

                values = [

                    # class
                    "0",

                    # bbox
                    f"{center_x:.6f}",
                    f"{center_y:.6f}",
                    f"{box_width:.6f}",
                    f"{box_height:.6f}",
                ]

                # ------------------------------------------------
                # 21 keypoints
                #
                # x
                # y
                # visibility
                # ------------------------------------------------

                for index in range(21):

                    x = float(
                        keypoints[
                            index * 3
                        ]
                    )

                    y = float(
                        keypoints[
                            index * 3 + 1
                        ]
                    )

                    visibility = int(
                        keypoints[
                            index * 3 + 2
                        ]
                    )

                    values.append(
                        f"{x / width:.6f}"
                    )

                    values.append(
                        f"{y / height:.6f}"
                    )

                    values.append(
                        str(
                            visibility
                        )
                    )

                label_lines.append(
                    " ".join(values)
                )

                hand_count += 1

            label_path = (
                label_out
                / f"{Path(filename).stem}.txt"
            )

            write_text(
                label_path,
                "\n".join(label_lines)
                + (
                    "\n"
                    if label_lines
                    else ""
                )
            )

            image_count += 1

        print(
            f"[{split_name}] "
            f"images={image_count} "
            f"hands={hand_count}"
        )

    process_split(
        "train",
        train_images
    )

    process_split(
        "val",
        val_images
    )

    # ========================================================
    # HAND DATA YAML
    # ========================================================

    yaml_text = """path: /home/hawkeye/bas-hmr/datasets/hand_pose

train: images/train
val: images/val

kpt_shape: [21, 3]

flip_idx:
  [0, 1, 2, 4, 3,
   10, 11, 12, 13, 14,
   5, 6, 7, 8, 9,
   15, 16, 17, 18, 19, 20]

names:
  0: hand

kpt_names:
  0:
    - wrist
    - thumb_cmc
    - thumb_mcp
    - thumb_ip
    - thumb_tip
    - index_mcp
    - index_pip
    - index_dip
    - index_tip
    - middle_mcp
    - middle_pip
    - middle_dip
    - middle_tip
    - ring_mcp
    - ring_pip
    - ring_dip
    - ring_tip
    - pinky_mcp
    - pinky_pip
    - pinky_dip
    - pinky_tip
"""

    write_text(
        HAND_OUT / "data.yaml",
        yaml_text
    )

    print(
        "[OK] Hand pose dataset ready."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "BAS-HMR DATASET PREPARATION"
    )
    print("=" * 60)

    prepare_cardboard()

    prepare_hand()

    print()
    print("=" * 60)
    print(
        "DATASET PREPARATION COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
