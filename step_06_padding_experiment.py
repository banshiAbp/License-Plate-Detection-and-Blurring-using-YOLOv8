"""Evaluate privacy padding around predicted license plate boxes.

The model may predict a tight box that misses plate borders or edge characters.
This script expands predicted boxes by different padding percentages and measures
how often ground-truth plates are covered by the final blur regions.
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
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "model_outputs" / "padding_experiment"


def image_files(split: str) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path
        for path in (PROJECT_DIR / "images" / split).glob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def read_yolo_labels(label_path: Path):
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            class_id, cx, cy, width, height = [float(value) for value in line.split()]
            rows.append((int(class_id), cx, cy, width, height))
    return rows


def yolo_to_xyxy(cx: float, cy: float, width: float, height: float, image_width: int, image_height: int):
    x1 = (cx - width / 2) * image_width
    y1 = (cy - height / 2) * image_height
    x2 = (cx + width / 2) * image_width
    y2 = (cy + height / 2) * image_height
    return x1, y1, x2, y2


def clip_box(x1, y1, x2, y2, image_width: int, image_height: int):
    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, image_width: int, image_height: int, padding: float):
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, image_width, image_height)
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(round(box_width * padding))
    pad_y = int(round(box_height * padding))
    return clip_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, image_width, image_height)


def area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def max_gt_coverage(gt_box, predicted_boxes) -> float:
    gt_area = area(gt_box)
    if gt_area == 0 or len(predicted_boxes) == 0:
        return 0.0
    return max(intersection_area(gt_box, pred_box) / gt_area for pred_box in predicted_boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate blur padding settings.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--padding-values", nargs="+", type=float, default=[0.00, 0.05, 0.08, 0.10, 0.15])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR))

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for padding in args.padding_values:
        summary_rows.append(
            {
                "padding": padding,
                "evaluated_images": 0,
                "total_ground_truth_plates": 0,
                "protected_plates": 0,
                "images_with_all_plates_blurred": 0,
            }
        )

    for image_path in image_files("test"):
        image = cv2.imread(str(image_path))
        image_height, image_width = image.shape[:2]
        label_path = PROJECT_DIR / "labels" / "test" / f"{image_path.stem}.txt"
        gt_boxes = [
            yolo_to_xyxy(cx, cy, width, height, image_width, image_height)
            for _, cx, cy, width, height in read_yolo_labels(label_path)
        ]

        result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)[0]
        base_pred_boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else []

        for row in summary_rows:
            padding = row["padding"]
            padded_pred_boxes = [
                expand_box(*box, image_width=image_width, image_height=image_height, padding=padding)
                for box in base_pred_boxes
            ]

            protected = 0
            for gt_box in gt_boxes:
                protected += int(max_gt_coverage(gt_box, padded_pred_boxes) >= args.coverage_threshold)

            row["evaluated_images"] += int(len(gt_boxes) > 0)
            row["total_ground_truth_plates"] += len(gt_boxes)
            row["protected_plates"] += protected
            row["images_with_all_plates_blurred"] += int(len(gt_boxes) > 0 and protected == len(gt_boxes))

    for row in summary_rows:
        total_plates = row["total_ground_truth_plates"]
        evaluated_images = row["evaluated_images"]
        row["missed_plates"] = total_plates - row["protected_plates"]
        row["privacy_protection_rate"] = row["protected_plates"] / total_plates if total_plates else 0
        row["plate_miss_rate"] = row["missed_plates"] / total_plates if total_plates else 0
        row["image_privacy_pass_rate"] = (
            row["images_with_all_plates_blurred"] / evaluated_images if evaluated_images else 0
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output / "padding_experiment_summary.csv", index=False)

    selected = summary_df.sort_values(["privacy_protection_rate", "padding"], ascending=[False, True]).iloc[0].to_dict()
    (args.output / "selected_padding.json").write_text(json.dumps(selected, indent=2))

    print(summary_df.to_string(index=False))
    print("\nSelected padding:")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
