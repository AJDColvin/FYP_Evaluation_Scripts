#!/usr/bin/env python3
"""
SAM3 Auto-Labeling Tool
========================
Uses Ultralytics SAM3 (SAM3SemanticPredictor) with a text prompt to detect
objects in images and produce a COCO-format dataset (_annotations.coco.json + images).
Annotated images are automatically saved by the Ultralytics library.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ultralytics.models.sam import SAM3SemanticPredictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label images using SAM3 with a text prompt (bounding boxes only)."
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
        default="/Volumes/USB Drive/models/sam3.pt",
        help="Path to the SAM3 model weights file (e.g. sam3.pt).",
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
        default="./output",
        help="Output directory (default: ./output).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
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
    parser.add_argument(
        "--individuals",
        type=int,
        default=None,
        help="Optional maximum number of predictions to keep per image (keeps highest confidence).",
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

def init_predictor(model_path: str, conf: float, device: str | None = None):
    """Initialise the SAM3 semantic predictor."""
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=model_path,
        imgsz=640,
        half=True,
    )
    if device is not None:
        overrides["device"] = device
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor

def predict_boxes(
    predictor: SAM3SemanticPredictor,
    image_path: Path,
    prompt: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SAM3 on a single image with a text prompt and return bounding boxes
    and confidence scores.

    Returns
    boxes : numpy.ndarray of shape (N, 4) in xyxy format, or empty (0, 4).
    scores : numpy.ndarray of shape (N,), or empty (0,).
    """
    predictor.set_image(str(image_path))
    results = predictor(text=[prompt])

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            return boxes, scores

    return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)


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

    predictor = init_predictor(args.model, args.conf, args.device)
    print(f"SAM3 predictor initialised  (model={args.model}, conf={args.conf}, device={args.device or 'auto'}).")
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

        # Run SAM3 inference
        boxes, scores = predict_boxes(predictor, img_path, args.prompt)
        
        if args.individuals is not None and len(boxes) > args.individuals:
            sorted_indices = np.argsort(scores)[::-1]
            top_indices = sorted_indices[:args.individuals]
            boxes = boxes[top_indices]
            scores = scores[top_indices]

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
    print(f"  Dataset images   : {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
