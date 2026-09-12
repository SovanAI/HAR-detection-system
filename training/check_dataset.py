from pathlib import Path
import yaml


ROOT = Path(
    "/home/hawkeye/bas-hmr/datasets/bas_objects"
)

DATA_YAML = ROOT / "data.yaml"


def load_config():

    with DATA_YAML.open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def check_split(
    split_name: str,
    config: dict,
):

    image_dir = (
        ROOT
        / config[split_name]
    )

    label_dir = (
        ROOT
        / config[split_name]
        .replace(
            "images",
            "labels",
            1,
        )
    )

    print()
    print("=" * 60)
    print(
        f"CHECKING {split_name.upper()}"
    )
    print("=" * 60)

    if not image_dir.exists():

        print(
            f"[ERROR] Missing image directory:"
            f" {image_dir}"
        )

        return

    if not label_dir.exists():

        print(
            f"[ERROR] Missing label directory:"
            f" {label_dir}"
        )

        return

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = [
        p
        for p in image_dir.iterdir()
        if p.suffix.lower()
        in image_extensions
    ]

    print(
        f"Images: {len(images)}"
    )

    missing_labels = []
    invalid_labels = []

    class_count = {
        i: 0
        for i in range(
            int(config["nc"])
        )
    }

    for image in images:

        label = (
            label_dir
            / f"{image.stem}.txt"
        )

        if not label.exists():

            missing_labels.append(
                image.name
            )

            continue

        try:

            with label.open(
                "r",
                encoding="utf-8",
            ) as file:

                lines = file.readlines()

            for line_number, line in enumerate(
                lines,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) != 5:

                    invalid_labels.append(
                        (
                            image.name,
                            line_number,
                            "Expected 5 values",
                        )
                    )

                    continue

                class_id = int(
                    values[0]
                )

                x = float(
                    values[1]
                )

                y = float(
                    values[2]
                )

                width = float(
                    values[3]
                )

                height = float(
                    values[4]
                )

                if (
                    class_id < 0
                    or class_id >= int(config["nc"])
                ):

                    invalid_labels.append(
                        (
                            image.name,
                            line_number,
                            f"Invalid class {class_id}",
                        )
                    )

                    continue

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                ):

                    invalid_labels.append(
                        (
                            image.name,
                            line_number,
                            "Coordinates outside valid range",
                        )
                    )

                    continue

                class_count[
                    class_id
                ] += 1

        except Exception as error:

            invalid_labels.append(
                (
                    image.name,
                    0,
                    str(error),
                )
            )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    if missing_labels:

        print()
        print(
            f"[WARNING] Missing labels: "
            f"{len(missing_labels)}"
        )

        for name in missing_labels[:20]:

            print(
                f"  - {name}"
            )

    else:

        print(
            "[OK] Every image has a label."
        )

    if invalid_labels:

        print()
        print(
            f"[ERROR] Invalid label entries: "
            f"{len(invalid_labels)}"
        )

        for item in invalid_labels[:20]:

            print(
                f"  - {item}"
            )

    else:

        print(
            "[OK] All label formats valid."
        )

    print()
    print("Class distribution:")

    names = config["names"]

    for class_id, count in class_count.items():

        class_name = names.get(
            class_id,
            f"class_{class_id}",
        )

        print(
            f"  {class_id}: "
            f"{class_name} = {count}"
        )


def main():

    print()
    print("=" * 60)
    print(
        " BAS OBJECT DATASET CHECKER"
    )
    print("=" * 60)

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"Missing {DATA_YAML}"
        )

    config = load_config()

    print()
    print(
        f"Dataset: {ROOT}"
    )

    print(
        f"Classes: {config['nc']}"
    )

    print(
        f"Names: {config['names']}"
    )

    check_split(
        "train",
        config,
    )

    check_split(
        "val",
        config,
    )

    print()
    print("=" * 60)
    print(
        "DATASET CHECK COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
