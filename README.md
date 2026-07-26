# License Plate Detection and Blurring using YOLOv8

This project detects vehicle license plates with YOLOv8 and blurs only the detected plate region to protect sensitive personal information in traffic or surveillance images.

## Submission Workflow

The main working notebook is:

```text
license_plate_detection_case_study.ipynb
```

The final submission should be exported from the completed notebook as a PDF under 20 MB and 50 pages.

## Project Files

- `Original_files/`: preserved original image and label zip archives.
- `images/` and `labels/`: extracted YOLO dataset folders.
- `data.yaml`: YOLOv8 dataset configuration.
- `step_01_annotation_audit.py`: helper script for label audit, coordinate conversion, box overlays, and blur samples.
- `build_case_study_notebook.py`: recreates the notebook in the same step-by-step case-study style as previous projects.
- `license_plate_detection_case_study.ipynb`: main notebook for analysis and final PDF export.

## Step 01: Annotation Quality Check

YOLO labels store boxes in normalized format:

```text
class_id center_x center_y width height
```

OpenCV needs pixel corner coordinates:

```python
x1 = (center_x - width / 2) * image_width
y1 = (center_y - height / 2) * image_height
x2 = (center_x + width / 2) * image_width
y2 = (center_y + height / 2) * image_height
```

Drawing these boxes on raw images is critical before training because it catches shifted boxes, wrong sizes, missing labels, duplicate images, or labels that point outside the image. If the model learns from poor boxes, the final blur will also be poor because the blur region comes directly from the predicted bounding box.

Run the audit:

```powershell
cd C:\python\ML-POC\License_Plate_Detection
$env:YOLO_CONFIG_DIR='C:\python\ML-POC\License_Plate_Detection'
..\.venv\Scripts\python.exe step_01_annotation_audit.py
```

Outputs are saved to:

```text
outputs/step_01_annotation_check/
```

## Next Notebook Steps

The notebook now also includes:

- YOLOv8 `data.yaml` validation.
- A controlled training cell for `YOLOv8s` with `imgsz=960`.
- Evaluation metrics for precision, recall, mAP50, mAP50-95, and visual blur quality.

Full model training is intentionally disabled in the notebook by default:

```python
RUN_FULL_TRAINING = False
```

When you are ready to train the model, change it to `True` inside the notebook and run that cell.
