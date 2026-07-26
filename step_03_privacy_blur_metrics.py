"""Measure privacy protection success for YOLOv8 license plate blurring.

Detection metrics such as precision, recall, and mAP are useful model metrics,
but a privacy system also needs a direct answer: did every real plate get
covered by a blur region?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "runs" / "yolov8s_plate_detection-2" / "weights" / "best.pt"
DEFAULT_SPLIT_FILE = PROJECT_DIR / "splits" / "test.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "model_outputs" / "privacy_metrics"


def read_split_images(split_file: Path) -> list[Path]:
    image_paths: list[Path] = []
    for raw_line in split_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        image_path = Path(line)
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path
        if image_path.exists():
            image_paths.append(image_path)
    return image_paths


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = parts.index("images")
        parts[image_index] = "labels"
    except ValueError:
        return PROJECT_DIR / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
    return Path(*parts).with_suffix(".txt")


def yolo_to_xyxy(row: str, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    _, cx, cy, width, height = [float(value) for value in row.split()]
    x1 = (cx - width / 2) * image_width
    y1 = (cy - height / 2) * image_height
    x2 = (cx + width / 2) * image_width
    y2 = (cy + height / 2) * image_height
    return x1, y1, x2, y2


def read_ground_truth_boxes(image_path: Path) -> list[tuple[float, float, float, float]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    image_height, image_width = image.shape[:2]
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            boxes.append(yolo_to_xyxy(line, image_width, image_height))
    return boxes


def intersection_area(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clip_box(x1, y1, x2, y2, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, image_width: int, image_height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, image_width, image_height)
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(round(box_width * padding))
    pad_y = int(round(box_height * padding))
    return clip_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, image_width, image_height)


def max_gt_coverage(gt_box, predicted_boxes) -> float:
    gt_area = area(gt_box)
    if gt_area == 0 or len(predicted_boxes) == 0:
        return 0.0
    return max(intersection_area(gt_box, pred_box) / gt_area for pred_box in predicted_boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate privacy blur success metrics.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.80,
        help="A ground-truth plate is protected if this fraction of its area is covered by a predicted blur box.",
    )
    parser.add_argument("--padding", type=float, default=0.15, help="Padding added around each predicted blur box.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR))

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    image_paths = read_split_images(args.split_file)
    args.output.mkdir(parents=True, exist_ok=True)

    image_rows = []
    plate_rows = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image_height, image_width = image.shape[:2]
        gt_boxes = read_ground_truth_boxes(image_path)
        prediction = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device="cpu",
            verbose=False,
        )[0]
        raw_pred_boxes = prediction.boxes.xyxy.cpu().numpy() if len(prediction.boxes) else []
        pred_boxes = [
            expand_box(*box, image_width=image_width, image_height=image_height, padding=args.padding)
            for box in raw_pred_boxes
        ]

        protected_count = 0
        for gt_index, gt_box in enumerate(gt_boxes):
            coverage = max_gt_coverage(gt_box, pred_boxes)
            is_protected = coverage >= args.coverage_threshold
            protected_count += int(is_protected)
            plate_rows.append(
                {
                    "image": str(image_path.relative_to(PROJECT_DIR)),
                    "plate_index": gt_index,
                    "max_gt_coverage": coverage,
                    "protected": is_protected,
                }
            )

        image_pass = len(gt_boxes) > 0 and protected_count == len(gt_boxes)
        image_rows.append(
            {
                "image": str(image_path.relative_to(PROJECT_DIR)),
                "ground_truth_plates": len(gt_boxes),
                "predicted_blur_boxes": len(pred_boxes),
                "protected_plates": protected_count,
                "missed_plates": len(gt_boxes) - protected_count,
                "image_privacy_pass": image_pass,
            }
        )

    plate_df = pd.DataFrame(plate_rows)
    image_df = pd.DataFrame(image_rows)

    total_plates = int(image_df["ground_truth_plates"].sum())
    protected_plates = int(image_df["protected_plates"].sum())
    missed_plates = total_plates - protected_plates
    evaluated_images = int((image_df["ground_truth_plates"] > 0).sum())
    passed_images = int(image_df["image_privacy_pass"].sum())

    summary = {
        "confidence_threshold": args.conf,
        "coverage_threshold": args.coverage_threshold,
        "padding": args.padding,
        "evaluated_images": evaluated_images,
        "total_ground_truth_plates": total_plates,
        "protected_plates": protected_plates,
        "missed_plates": missed_plates,
        "plate_miss_rate": missed_plates / total_plates if total_plates else 0.0,
        "privacy_protection_rate": protected_plates / total_plates if total_plates else 0.0,
        "image_privacy_pass_rate": passed_images / evaluated_images if evaluated_images else 0.0,
        "images_with_all_plates_blurred": passed_images,
    }

    plate_df.to_csv(args.output / "plate_level_privacy_results.csv", index=False)
    image_df.to_csv(args.output / "image_level_privacy_results.csv", index=False)
    (args.output / "privacy_metrics_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
