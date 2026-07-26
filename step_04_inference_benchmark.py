"""Benchmark YOLOv8 license plate detection and blurring latency.

This measures the local execution environment rather than assuming real-time
performance. It reports model prediction latency, blur post-processing latency,
end-to-end latency, and approximate FPS.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "runs" / "yolov8s_plate_detection-2" / "weights" / "best.pt"
DEFAULT_IMAGE = PROJECT_DIR / "images" / "test" / "003a5aaf6d17c917.jpg"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "model_outputs" / "benchmark"


def blur_boxes(image: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    output = image.copy()
    image_height, image_width = output.shape[:2]

    for x1, y1, x2, y2 in boxes_xyxy:
        left = max(0, min(image_width - 1, int(round(x1))))
        top = max(0, min(image_height - 1, int(round(y1))))
        right = max(0, min(image_width, int(round(x2))))
        bottom = max(0, min(image_height, int(round(y2))))
        if right <= left or bottom <= top:
            continue

        roi = output[top:bottom, left:right]
        kernel_w = max(15, ((right - left) // 2) * 2 + 1)
        kernel_h = max(15, ((bottom - top) // 2) * 2 + 1)
        output[top:bottom, left:right] = cv2.GaussianBlur(roi, (kernel_w, kernel_h), 0)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local YOLOv8 inference and blur speed.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR))

    import torch
    from ultralytics import YOLO

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read benchmark image: {args.image}")

    model = YOLO(str(args.weights))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for _ in range(args.warmup):
        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4))
        blur_boxes(image, boxes)

    predict_times = []
    blur_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    detection_counts = []

    start_all = time.perf_counter()
    for _ in range(args.runs):
        start_predict = time.perf_counter()
        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
        predict_times.append(time.perf_counter() - start_predict)

        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4))
        detection_counts.append(len(boxes))

        start_blur = time.perf_counter()
        blur_boxes(image, boxes)
        blur_times.append(time.perf_counter() - start_blur)

        preprocess_times.append(float(result.speed.get("preprocess", 0.0)))
        inference_times.append(float(result.speed.get("inference", 0.0)))
        postprocess_times.append(float(result.speed.get("postprocess", 0.0)))

    total_elapsed = time.perf_counter() - start_all
    average_predict_ms = float(np.mean(predict_times) * 1000)
    average_blur_ms = float(np.mean(blur_times) * 1000)
    average_end_to_end_ms = average_predict_ms + average_blur_ms

    summary = {
        "hardware": platform.processor() or platform.platform(),
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model": "YOLOv8s best.pt",
        "image": str(args.image.relative_to(PROJECT_DIR)),
        "image_shape": list(image.shape),
        "input_size": args.imgsz,
        "confidence_threshold": args.conf,
        "warmup_runs": args.warmup,
        "benchmark_runs": args.runs,
        "average_preprocess_ms": float(np.mean(preprocess_times)),
        "average_inference_ms": float(np.mean(inference_times)),
        "average_postprocess_ms": float(np.mean(postprocess_times)),
        "average_model_predict_latency_ms": average_predict_ms,
        "average_blur_processing_ms": average_blur_ms,
        "average_end_to_end_blur_latency_ms": average_end_to_end_ms,
        "approx_fps_end_to_end": 1000 / average_end_to_end_ms if average_end_to_end_ms else 0.0,
        "total_elapsed_seconds": total_elapsed,
        "average_detections_per_frame": float(np.mean(detection_counts)),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inference_benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
