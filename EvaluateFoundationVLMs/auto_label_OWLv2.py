#!/usr/bin/env python3
"""
OWLv2 Auto-Labeling Tool
======================================
Uses HuggingFace transformers OWLv2 model with a text prompt to
detect objects in images and produce a COCO-format dataset
(_annotations.coco.json + predictions.coco.json).
Annotated images with bounding boxes are saved to the output directory.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Annotation drawing settings
# ---------------------------------------------------------------------------
BOX_COLOR = (0, 255, 0)       # Green (BGR)
TEXT_COLOR = (255, 255, 255)   # White (BGR)
TEXT_BG_COLOR = (0, 255, 0)    # Green background for label
BOX_THICKNESS = 2
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label images using OWLv2 with a text prompt (bounding boxes only)."
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
        default="google/owlv2-large-patch14-ensemble",
        help="HuggingFace model ID or local path (default: google/owlv2-large-patch14-ensemble).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Text prompt describing the target object (e.g. "horse").',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_owlv2",
        help="Output directory (default: ./output_owlv2).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,  # OWLv2 generally needs slightly lower threshold bounds than GDINO
        help="Confidence threshold for predictions (default: 0.10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g. 'mps', 'cuda', 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/Volumes/USB Drive/models",
        help="Directory to cache downloaded HuggingFace models. Defaults.",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Optional path to a ground truth COCO JSON. When provided, prediction "
             "image IDs are aligned to match the GT (matched by file_name).",
    )
    parser.add_argument(
        "--clip-format",
        action="store_true",
        help="If set, wraps the prompt in 'a photo of a {prompt}' for better CLIP alignment.",
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


def get_device(requested: str | None) -> torch.device:
    """Resolve the compute device."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_model(
    model_id: str,
    device: torch.device,
    cache_dir: str | None = None,
) -> tuple[Owlv2Processor, Owlv2ForObjectDetection]:
    """Load the OWLv2 processor and model from HuggingFace."""
    processor = Owlv2Processor.from_pretrained(model_id, cache_dir=cache_dir)
    model = Owlv2ForObjectDetection.from_pretrained(
        model_id, cache_dir=cache_dir
    ).to(device)
    return processor, model


def predict_boxes(
    processor: Owlv2Processor,
    model: Owlv2ForObjectDetection,
    image: Image.Image,
    prompt: str,
    threshold: float,
    device: torch.device,
    use_clip_format: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run OWLv2 on a single PIL image with a text prompt
    and return bounding boxes and confidence scores.

    Returns
    -------
    boxes : numpy.ndarray of shape (N, 4) in xyxy format, or empty (0, 4).
    scores : numpy.ndarray of shape (N,), or empty (0,).
    """
    clean_prompt = prompt.replace('.', '').strip()
    if use_clip_format:
        texts = [[f"a photo of a {clean_prompt}"]]
    else:
        texts = [[clean_prompt]]

    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    width, height = image.size
    # OWLv2 API specifies target_sizes in tensor format
    target_sizes = torch.Tensor([[height, width]]).to(device)
    
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )

    if results and len(results) > 0:
        result = results[0]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        if len(boxes) > 0:
            return boxes, scores

    return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)


def draw_annotations(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    label: str,
) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image (in-place). Returns the image."""
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Label text
        text = f"{label.replace('.', '').strip()} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        # Draw filled background rectangle for text
        cv2.rectangle(image_bgr, (x1, y1 - th - baseline - 4), (x1 + tw, y1), TEXT_BG_COLOR, -1)
        cv2.putText(image_bgr, text, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return image_bgr


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
    clean_cat = category_name.replace('.', '').strip()
    return {
        "info": {
            "description": f"Auto-labelled dataset — prompt: '{clean_cat}' (OWLv2)",
            "version": "1.0",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images_meta,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": clean_cat,
                "supercategory": "none",
            }
        ],
    }


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect images & init model
    # ------------------------------------------------------------------
    image_paths = collect_images(args.images)
    print(f"Found {len(image_paths)} image(s) in '{args.images}'.")

    device = get_device(args.device)
    processor, model = init_model(args.model, device, args.cache_dir)
    cache_info = args.cache_dir or "~/.cache/huggingface/hub"
    print(f"OWLv2 model initialised  (model={args.model}, device={device}).")
    print(f"  threshold={args.threshold}")
    print(f"  model cache : {cache_info}")
    print(f"Text prompt: '{args.prompt}'\n")

    # ------------------------------------------------------------------
    # Load GT image-ID mapping (if provided)
    # ------------------------------------------------------------------
    gt_name_to_id: dict[str, int] | None = None
    if args.gt:
        with open(args.gt, "r") as f:
            gt_data = json.load(f)
        gt_name_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
        print(f"Loaded GT mapping with {len(gt_name_to_id)} image(s) from '{args.gt}'.\n")

    # ------------------------------------------------------------------
    # Process each image
    # ------------------------------------------------------------------
    images_meta: list[dict] = []
    all_annotations: list[dict] = []
    annotation_id = 1
    skipped_no_gt = 0
    frame_times: list[float] = []

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

        # Load image (PIL for model, cv2 for dimensions and annotation saving)
        pil_image = Image.open(img_path).convert("RGB")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print("SKIPPED (could not read)")
            continue
        h, w = image_bgr.shape[:2]

        # Run OWLv2 inference (timed)
        t_start = time.perf_counter()
        boxes, scores = predict_boxes(
            processor, model, pil_image, args.prompt,
            args.threshold, device, args.clip_format
        )
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        frame_times.append(elapsed)

        num_boxes = len(boxes)
        print(f"{num_boxes} detection(s)  ({elapsed:.3f}s)")

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
        if num_boxes > 0:
            annotated = draw_annotations(image_bgr, boxes, scores, args.prompt)
            cv2.imwrite(str(output_dir / img_path.name), annotated)
        else:
            # Save the original image even if no detections
            cv2.imwrite(str(output_dir / img_path.name), image_bgr)


    # ------------------------------------------------------------------
    # Write COCO JSON (full dataset format)
    # ------------------------------------------------------------------
    coco = build_coco_json(images_meta, all_annotations, args.prompt)
    coco_path = output_dir / "_annotations.coco.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Write predictions JSON (flat COCOeval results format)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_detections = annotation_id - 1
    avg_time = sum(frame_times) / len(frame_times) if frame_times else 0.0
    print(f"\n{'='*50}")
    print(f"Done!  {len(images_meta)} image(s) processed, {total_detections} detection(s) total.")
    print(f"  COCO annotations : {coco_path}")
    print(f"  Predictions      : {pred_path}")
    print(f"  Annotated images : {output_dir}")
    print(f"  Avg time / frame : {avg_time:.3f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
