"""Run YOLOv8 license-plate detection and privacy blurring on images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "demo_outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect license plates and save detected and privacy-blurred images."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="An image file or directory containing images.",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Fractional padding added around each detected plate before blurring.",
    )
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0.")
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"No supported image file or directory found: {source}")


def expanded_box(
    box: np.ndarray,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(float)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(image_width, int(round(x2 + pad_x))),
        min(image_height, int(round(y2 + pad_y))),
    )


def blur_detected_plates(
    image: np.ndarray,
    boxes: np.ndarray,
    padding: float,
) -> np.ndarray:
    blurred_image = image.copy()
    image_height, image_width = blurred_image.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = expanded_box(
            box,
            image_width,
            image_height,
            padding,
        )
        if x2 <= x1 or y2 <= y1:
            continue
        region = blurred_image[y1:y2, x1:x2]
        sigma = max(3.0, min(region.shape[:2]) / 3.0)
        blurred_image[y1:y2, x1:x2] = cv2.GaussianBlur(
            region,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
    return blurred_image


def main() -> None:
    args = parse_args()
    source_images = image_paths(args.source)
    if not args.weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {args.weights}\n"
            "Copy your trained best.pt to weights/best.pt, or pass --weights."
        )

    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(PROJECT_DIR / ".ultralytics"),
    )
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    for source_path in source_images:
        image = cv2.imread(str(source_path))
        if image is None:
            print(f"Skipped unreadable image: {source_path}")
            continue

        result = model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        boxes = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None
            else np.empty((0, 4))
        )

        detected_path = args.output / f"{source_path.stem}_detected.jpg"
        blurred_path = args.output / f"{source_path.stem}_blurred.jpg"
        cv2.imwrite(str(detected_path), result.plot())
        cv2.imwrite(
            str(blurred_path),
            blur_detected_plates(image, boxes, args.padding),
        )
        print(
            f"{source_path.name}: {len(boxes)} plate(s) detected -> "
            f"{blurred_path}"
        )

    print(f"Saved outputs to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
