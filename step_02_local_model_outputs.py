"""Create local CPU evaluation artifacts from the trained YOLOv8 plate model.

This script uses the downloaded best.pt checkpoint to create:
- prediction images with detected boxes
- blurred-output examples where only detected plate regions are blurred
- a CSV summary of detections

Full test-set validation is optional because it can take longer on CPU.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "runs" / "yolov8s_plate_detection-2" / "weights" / "best.pt"
DEFAULT_DATA = PROJECT_DIR / "data_eval.yaml"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "model_outputs"


def resolve_split_images(split_file: Path, limit: int | None) -> list[Path]:
    """Read YOLO split file and return existing image paths."""
    image_paths: list[Path] = []
    for raw_line in split_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        path = Path(line)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if path.exists():
            image_paths.append(path)
        if limit is not None and len(image_paths) >= limit:
            break
    return image_paths


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp model box coordinates to valid image pixels."""
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    return left, top, right, bottom


def expand_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    """Expand a predicted box before clipping it to image boundaries."""
    left, top, right, bottom = clamp_box(x1, y1, x2, y2, width, height)
    box_width = right - left
    box_height = bottom - top
    pad_x = int(round(box_width * padding))
    pad_y = int(round(box_height * padding))
    return clamp_box(left - pad_x, top - pad_y, right + pad_x, bottom + pad_y, width, height)


def blur_detected_boxes(image, boxes_xyxy, padding: float = 0.15) -> object:
    """Apply Gaussian blur only inside predicted license plate boxes."""
    blurred_image = image.copy()
    height, width = image.shape[:2]

    for x1, y1, x2, y2 in boxes_xyxy:
        left, top, right, bottom = expand_box(x1, y1, x2, y2, width, height, padding)
        if right <= left or bottom <= top:
            continue

        roi = blurred_image[top:bottom, left:right]
        kernel_w = max(15, ((right - left) // 2) * 2 + 1)
        kernel_h = max(15, ((bottom - top) // 2) * 2 + 1)
        blurred_image[top:bottom, left:right] = cv2.GaussianBlur(roi, (kernel_w, kernel_h), 0)

    return blurred_image


def write_detection_outputs(model, image_paths: list[Path], output_dir: Path, imgsz: int, conf: float) -> None:
    """Save prediction and blurred images for selected examples."""
    predictions_dir = output_dir / "predictions"
    blurred_dir = output_dir / "blurred"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    blurred_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            device="cpu",
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else []
        confidences = boxes.conf.cpu().numpy().tolist() if len(boxes) else []

        plotted = result.plot()
        cv2.imwrite(str(predictions_dir / image_path.name), plotted)

        image = cv2.imread(str(image_path))
        blurred = blur_detected_boxes(image, boxes_xyxy)
        cv2.imwrite(str(blurred_dir / image_path.name), blurred)

        summary_rows.append(
            {
                "image": str(image_path.relative_to(PROJECT_DIR)),
                "detections": len(boxes),
                "confidences": ";".join(f"{score:.4f}" for score in confidences),
            }
        )

    with (output_dir / "detection_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "detections", "confidences"])
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local YOLOv8 prediction and blur outputs.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-file", type=Path, default=PROJECT_DIR / "splits" / "test.txt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=30, help="Number of test images to process for examples.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--validate", action="store_true", help="Also run full test-set validation on CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR))

    from ultralytics import YOLO

    if not args.weights.exists():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    image_paths = resolve_split_images(args.split_file, args.limit)
    if not image_paths:
        raise ValueError(f"No images found from split file: {args.split_file}")

    write_detection_outputs(model, image_paths, args.output, args.imgsz, args.conf)
    print(f"Saved prediction images to: {args.output / 'predictions'}")
    print(f"Saved blurred examples to: {args.output / 'blurred'}")
    print(f"Saved detection summary to: {args.output / 'detection_summary.csv'}")

    if args.validate:
        model.val(
            data=str(args.data),
            split="test",
            imgsz=args.imgsz,
            batch=1,
            device="cpu",
            project=str(args.output),
            name="test_validation_cpu",
        )


if __name__ == "__main__":
    main()
