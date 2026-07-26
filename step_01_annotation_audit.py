from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2


ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "license_plate"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


def yolo_to_xyxy(
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO cx, cy, width, height to OpenCV pixel corners."""
    x1 = round((center_x - box_width / 2) * image_width)
    y1 = round((center_y - box_height / 2) * image_height)
    x2 = round((center_x + box_width / 2) * image_width)
    y2 = round((center_y + box_height / 2) * image_height)

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))
    return x1, y1, x2, y2


def read_yolo_labels(label_path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not label_path.exists():
        return boxes

    for line_no, raw_line in enumerate(label_path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no} has {len(parts)} columns, expected 5")
        class_id = int(float(parts[0]))
        center_x, center_y, width, height = map(float, parts[1:])
        boxes.append(YoloBox(class_id, center_x, center_y, width, height))
    return boxes


def draw_yolo_boxes(image_path: Path, label_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]
    for box in read_yolo_labels(label_path):
        x1, y1, x2, y2 = yolo_to_xyxy(
            box.center_x,
            box.center_y,
            box.width,
            box.height,
            image_width,
            image_height,
        )
        label = CLASS_NAMES.get(box.class_id, str(box.class_id))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def blur_yolo_boxes(image_path: Path, label_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]
    for box in read_yolo_labels(label_path):
        x1, y1, x2, y2 = yolo_to_xyxy(
            box.center_x,
            box.center_y,
            box.width,
            box.height,
            image_width,
            image_height,
        )
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        min_side = min(roi.shape[:2])
        kernel = max(3, min(99, min_side // 2 * 2 + 1))
        image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def image_files(split: str) -> list[Path]:
    split_dir = ROOT / "images" / split
    return sorted(path for path in split_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def label_files(split: str) -> list[Path]:
    split_dir = ROOT / "labels" / split
    return sorted(path for path in split_dir.iterdir() if path.suffix.lower() == ".txt")


def matching_label_path(image_path: Path, split: str) -> Path:
    return ROOT / "labels" / split / f"{image_path.stem}.txt"


def audit_split(split: str) -> dict:
    images = image_files(split)
    labels = label_files(split)
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    widths: list[float] = []
    heights: list[float] = []
    classes: dict[str, int] = {}
    invalid_rows: list[dict] = []
    total_boxes = 0

    for label_path in labels:
        for box in read_yolo_labels(label_path):
            total_boxes += 1
            widths.append(box.width)
            heights.append(box.height)
            classes[str(box.class_id)] = classes.get(str(box.class_id), 0) + 1
            valid = (
                0 <= box.center_x <= 1
                and 0 <= box.center_y <= 1
                and 0 < box.width <= 1
                and 0 < box.height <= 1
            )
            if not valid:
                invalid_rows.append({"file": label_path.name, "box": box.__dict__})

    return {
        "split": split,
        "image_count": len(images),
        "label_count": len(labels),
        "box_count": total_boxes,
        "missing_label_count": len(image_stems - label_stems),
        "orphan_label_count": len(label_stems - image_stems),
        "missing_label_examples": sorted(image_stems - label_stems)[:10],
        "orphan_label_examples": sorted(label_stems - image_stems)[:10],
        "class_counts": classes,
        "invalid_row_count": len(invalid_rows),
        "invalid_row_examples": invalid_rows[:10],
        "normalized_box_width": {
            "min": min(widths) if widths else None,
            "mean": mean(widths) if widths else None,
            "max": max(widths) if widths else None,
        },
        "normalized_box_height": {
            "min": min(heights) if heights else None,
            "mean": mean(heights) if heights else None,
            "max": max(heights) if heights else None,
        },
    }


def write_split_files() -> None:
    split_dir = ROOT / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        rows: list[str] = []
        for image_path in image_files(split):
            if matching_label_path(image_path, split).exists():
                rows.append(image_path.relative_to(ROOT).as_posix())

        output_name = "train_labeled.txt" if split == "train" else f"{split}.txt"
        (split_dir / output_name).write_text("\n".join(rows) + "\n")


def create_visual_samples(sample_count: int) -> None:
    output_dir = ROOT / "outputs" / "step_01_annotation_check"
    for split in ("train", "val", "test"):
        created = 0
        for image_path in image_files(split):
            label_path = matching_label_path(image_path, split)
            if not label_path.exists():
                continue
            created += 1
            overlay_path = output_dir / "overlays" / split / f"{image_path.stem}_boxes.jpg"
            blurred_path = output_dir / "blurred" / split / f"{image_path.stem}_blurred.jpg"
            draw_yolo_boxes(image_path, label_path, overlay_path)
            blur_yolo_boxes(image_path, label_path, blurred_path)
            if created >= sample_count:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO license plate labels and create visual checks.")
    parser.add_argument("--samples", type=int, default=6, help="Number of sample images per split to render.")
    args = parser.parse_args()

    report = {split: audit_split(split) for split in ("train", "val", "test")}
    write_split_files()
    create_visual_samples(args.samples)

    output_dir = ROOT / "outputs" / "step_01_annotation_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "annotation_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    for split, details in report.items():
        print(
            f"{split}: images={details['image_count']}, labels={details['label_count']}, "
            f"boxes={details['box_count']}, missing_labels={details['missing_label_count']}, "
            f"orphan_labels={details['orphan_label_count']}, invalid_rows={details['invalid_row_count']}"
        )
    print(f"Saved report: {report_path}")
    print(f"Saved visual samples under: {output_dir}")


if __name__ == "__main__":
    main()
