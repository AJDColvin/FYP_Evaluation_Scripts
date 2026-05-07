#!/usr/bin/env python3
"""
YOLOE Auto-Labeling Tool
========================
Uses Ultralytics YOLOE with a text prompt to detect objects in images and
produce a COCO-format dataset (_annotations.coco.json + predictions.coco.json).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLOE

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}




def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label images using YOLOE with a text prompt (bounding boxes only)."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to a folder containing input images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yoloe-26l-seg.pt",
        help="Path to the YOLOE model weights file (default: yoloe-26l-seg.pt).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Text prompt describing the target object (e.g. "Horse").',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_yoloe",
        help="Output directory (default: ./output).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.80,
        help="Confidence threshold for predictions (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g. 'mps', 'cuda', 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Optional path to a ground truth COCO JSON. When provided, prediction "
             "image IDs are aligned to match the GT (matched by file_name).",
    )
    return parser.parse_args()


def collect_images(images_dir: str) -> list[Path]:
    """Return a sorted list of image file paths from the given directory."""
    root = Path(images_dir)
    if not root.is_dir():
        print(f"Error: '{images_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    paths = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print(f"Error: no images found in '{images_dir}'.", file=sys.stderr)
        sys.exit(1)

    return paths


def init_model(model_path: str, prompt: str) -> YOLOE:
    """Initialise the YOLOE model and set the text prompt classes."""
    model = YOLOE(model_path)
    model.set_classes([prompt])
    return model


def predict_boxes(
    model: YOLOE,
    image_path: Path,
    conf: float,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, object | None]:
    """
    Run YOLOE on a single image and return bounding boxes, confidence scores,
    and the raw Ultralytics Result object (for annotated image rendering).

    Returns
    -------
    boxes : numpy.ndarray of shape (N, 4) in xyxy format, or empty (0, 4).
    scores : numpy.ndarray of shape (N,), or empty (0,).
    result : ultralytics Result object, or None if no results.
    """
    predict_kwargs: dict = dict(
        source=str(image_path),
        conf=conf,
        verbose=False,
        imgsz=1080
    )
    if device is not None:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            return boxes, scores, result

    return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), None


def xyxy_to_xywh(box: np.ndarray) -> list[float]:
    """Convert a single [x1, y1, x2, y2] box to COCO [x, y, w, h]."""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]




def build_coco_json(
    images_meta: list[dict],
    annotations: list[dict],
    category_name: str,
) -> dict:
    """Assemble a full COCO-format dictionary."""
    return {
        "info": {
            "description": f"Auto-labelled dataset — prompt: '{category_name}'",
            "version": "1.0",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images_meta,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": "none",
            }
        ],
    }


def main() -> None:
    args = parse_args()

    # Prepare output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images & init model
    image_paths = collect_images(args.images)
    print(f"Found {len(image_paths)} image(s) in '{args.images}'.")

    model = init_model(args.model, args.prompt)
    print(f"YOLOE model initialised  (model={args.model}, conf={args.conf}, device={args.device or 'auto'}).")
    print(f"Text prompt: '{args.prompt}'\n")

    # Load GT image-ID mapping (if provided)
    gt_name_to_id: dict[str, int] | None = None
    if args.gt:
        with open(args.gt, "r") as f:
            gt_data = json.load(f)
        gt_name_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
        print(f"Loaded GT mapping with {len(gt_name_to_id)} image(s) from '{args.gt}'.\n")

    # Process each image
    images_meta: list[dict] = []
    all_annotations: list[dict] = []
    annotation_id = 1
    skipped_no_gt = 0

    for seq_id, img_path in enumerate(image_paths, start=0):
        # Resolve image ID: use GT mapping if available, else sequential
        if gt_name_to_id is not None:
            # Local filenames include RF hash — match directly against GT file_name
            image_id = gt_name_to_id.get(img_path.name)
            if image_id is None:
                skipped_no_gt += 1
                print(f"[{seq_id}/{len(image_paths)}] SKIPPED {img_path.name} (not in GT)")
                continue
        else:
            image_id = seq_id

        print(f"[{seq_id}/{len(image_paths)}] Processing {img_path.name} (id={image_id}) ... ", end="")

        # Read image for dimensions
        image = cv2.imread(str(img_path))
        if image is None:
            print("SKIPPED (could not read)")
            continue
        h, w = image.shape[:2]

        # Run YOLOE inference
        boxes, scores, result = predict_boxes(model, img_path, args.conf, args.device)
        num_boxes = len(boxes)
        print(f"{num_boxes} detection(s)")

        # Record image metadata
        images_meta.append({
            "id": image_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        # Record annotations
        for box, score in zip(boxes, scores):
            bbox_xywh = xyxy_to_xywh(box)
            area = bbox_xywh[2] * bbox_xywh[3]
            all_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox_xywh,
                "area": float(area),
                "score": float(score),
                "iscrowd": 0,
            })
            annotation_id += 1

        # Save annotated image with bounding boxes drawn
        if result is not None:
            annotated = result.plot()
            cv2.imwrite(str(output_dir / img_path.name), annotated)


    # Write COCO JSON (full dataset format)
    coco = build_coco_json(images_meta, all_annotations, args.prompt)
    coco_path = output_dir / "_annotations.coco.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    # Write predictions JSON (flat COCOeval results format)
    predictions = [
        {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "score": ann["score"],
        }
        for ann in all_annotations
    ]
    pred_path = output_dir / "predictions.coco.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # Summary
    total_detections = annotation_id - 1
    print(f"\n{'='*50}")
    print(f"Done!  {len(images_meta)} image(s) processed, {total_detections} detection(s) total.")
    print(f"  COCO annotations : {coco_path}")
    print(f"  Predictions      : {pred_path}")
    print(f"  Annotated images : {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
