import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate COCO JSON predictions using a YOLO model.")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to YOLO model weights (e.g. yolov8n.pt).")
    parser.add_argument("--images", type=str, required=True, 
                        help="Filepath to folder containing images.")
    parser.add_argument("--class", dest="class_name", type=str, required=True, 
                        help='Class to target (e.g., "Horse"). All other detected classes will be ignored.')
    parser.add_argument("--conf", type=float, required=True, 
                        help="Confidence threshold for detections.")
    parser.add_argument("--output", type=str, default="./yolo_outputs", 
                        help="Output directory for predictions and annotated images.")
    parser.add_argument("--gt", type=str, default=None, 
                        help="Optional COCO Ground Truth JSON file to map image IDs properly.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize the YOLO model
    print(f"Loading YOLO model from {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # 2. Map target class to YOLO class ID
    target_class_id = None
    target_cls_lower = args.class_name.lower().strip()
    
    for class_id, class_name in model.names.items():
        if class_name.lower() == target_cls_lower:
            target_class_id = class_id
            break

    if target_class_id is None:
        print(f"Error: Target class '{args.class_name}' not found in the model's class names.")
        print(f"Available classes: {list(model.names.values())}")
        sys.exit(1)
        
    print(f"Mapped target class '{args.class_name}' to class ID {target_class_id}")

    # 3. Collect images
    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"Error: Images directory '{args.images}' does not exist.")
        sys.exit(1)
        
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("._")])
    print(f"Found {len(image_paths)} images in '{args.images}'.")
    
    if len(image_paths) == 0:
        print("No images found to process. Exiting.")
        sys.exit(0)

    # 3.5 Load GT image-ID mapping (if provided)
    gt_name_to_id = None
    if args.gt:
        with open(args.gt, "r") as f:
            gt_data = json.load(f)
        gt_name_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
        print(f"Loaded GT mapping with {len(gt_name_to_id)} image(s) from '{args.gt}'.\n")

    # 4. Run inference and generate COCO predictions
    predictions = []
    
    for idx, img_path in enumerate(image_paths):
        if gt_name_to_id is not None:
            image_id = gt_name_to_id.get(img_path.name)
            if image_id is None:
                continue
        else:
            image_id = idx + 1
        
        # Determine if we should save the annotated image (1 out of every 10)
        # Note: (idx % 10 == 0) means every 10th image (0th, 10th, 20th...)
        save_annotated_image = (idx % 10 == 0)
        
        # Run inference
        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        result = results[0]
        
        # Load image for drawing bounding boxes if needed
        img_bgr = None
        if save_annotated_image:
            img_bgr = cv2.imread(str(img_path))
            # Just in case cv2 fails to read the image
            if img_bgr is None:
                save_annotated_image = False
        
        # Process detections
        for box in result.boxes:
            cls_idx = int(box.cls[0].item())
            
            # Filter out ignored classes
            if cls_idx != target_class_id:
                continue
                
            conf = float(box.conf[0].item())
            
            # COCO bbox format is [x_min, y_min, width, height]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            
            # Typically for PyCOCOTools evaluation, category_id = 1 for the first target class
            # or you can use the YOLO class_id depending on your GT mapping. We'll set it to 1.
            predictions.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "score": conf
            })
            
            # Draw bounding box if saving this image
            if save_annotated_image and img_bgr is not None:
                cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label_text = f"{args.class_name}: {conf:.2f}"
                cv2.putText(img_bgr, label_text, (int(x1), max(10, int(y1) - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save annotated image
        if save_annotated_image and img_bgr is not None:
            save_path = output_dir / f"annotated_{img_path.name}"
            cv2.imwrite(str(save_path), img_bgr)
            
        print(f"[{idx + 1}/{len(image_paths)}] Processed {img_path.name} | "
              f"Detections: {len(result.boxes)} | "
              f"Annotated Image Saved: {'Yes' if save_annotated_image else 'No'}")

    # 5. Save the JSON predictions
    pred_json_path = output_dir / "predictions.json"
    with open(pred_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    print(f"\n--- Processing Complete ---")
    print(f"Total predictions found: {len(predictions)}")
    print(f"Predictions saved to: {pred_json_path}")
    print(f"Annotated sample images saved to: {output_dir}")

if __name__ == "__main__":
    main()
