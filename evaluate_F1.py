#!/usr/bin/env python3
"""
F1 Evaluation Script
======================
Computes the F1-Confidence curve and max F1 score for bounding-box predictions 
against a ground truth COCO JSON file. 

By passing in predictions generated at a very low confidence (e.g. 0.05),
this script simulates filtering them down over higher thresholds to find
the optimal threshold that maximizes both Precision and Recall.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate F1 Score and plot F1-Confidence curve."
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to the ground truth _annotations.coco.json file.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the predictions.coco.json file.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for considering a detection a True Positive (default: 0.5)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="f1_confidence_curve.png",
        help="Path to save the F1-Confidence curve image (default: f1_confidence_curve.png)."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    gt_path = Path(args.gt).resolve()
    pred_path = Path(args.predictions).resolve()

    if not gt_path.exists():
        print(f"Error: ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not pred_path.exists():
        print(f"Error: predictions file not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading ground truth …")
    coco_gt = COCO(str(gt_path))

    print("Loading predictions …")
    coco_dt = coco_gt.loadRes(str(pred_path))

    if len(coco_dt.anns) == 0:
        print("Error: No predictions found. The predictions JSON is empty.")
        sys.exit(1)

    print(f"Running exact evaluation at IoU={args.iou} …")
    
    # Init evaluators
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = [args.iou]  # Evaluate only at desired IoU
    # Allow a high maximum detections number to properly sweep the tail
    coco_eval.params.maxDets = [2000]
    
    from contextlib import redirect_stdout
    import os
    # Suppress raw pycocotools prints for a cleaner terminal output
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        coco_eval.evaluate()
        coco_eval.accumulate()

    # Extract all matchings across all images dataset-wide
    dt_scores = []
    dt_matches = []
    total_gt = 0

    for e in coco_eval.evalImgs:
        if e is None:
            continue
        
        # Matches against GT at the selected IoU threshold
        matches = e['dtMatches'][0] 
        scores = e['dtScores']
        
        # We need to drop ignored detections (they don't count as FP or TP)
        ignore = e['dtIgnore'][0]
        valid_idx = (ignore == 0)
        
        matches = matches[valid_idx]
        scores = np.array(scores)[valid_idx]
        
        # If match ID is > 0, it's a True Positive!
        tp = (matches > 0).astype(int)
        
        dt_scores.extend(scores.tolist())
        dt_matches.extend(tp.tolist())
        
        # Count total valid ground truth objects to evaluate Recall against
        gt_ignore = e['gtIgnore']
        total_gt += len(gt_ignore) - sum(gt_ignore)

    if len(dt_scores) == 0:
        print("No valid detections evaluated.")
        sys.exit(1)

    if total_gt == 0:
        print("No valid ground truth objects found.")
        sys.exit(1)

    dt_scores = np.array(dt_scores)
    dt_matches = np.array(dt_matches)

    # Sort descending by confidence score representing the slider moving left to right
    sort_idx = np.argsort(dt_scores)[::-1]
    dt_scores = dt_scores[sort_idx]
    dt_matches = dt_matches[sort_idx]

    # Calculate cumulative metrics (simulating higher thresholds progressively retaining less)
    cum_tp = np.cumsum(dt_matches)
    cum_fp = np.cumsum(1 - dt_matches)

    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-16)
    
    # Calculate F1
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-16)

    # Find the threshold that produced the best F1
    max_f1_idx = np.argmax(f1)
    max_f1 = f1[max_f1_idx]
    best_conf = dt_scores[max_f1_idx]
    best_p = precision[max_f1_idx]
    best_r = recall[max_f1_idx]

    print(f"\n{'='*50}")
    print(f"Optimal F1 Score: {max_f1:.4f}")
    print(f"Achieved at Confidence: {best_conf:.4f}")
    print(f"  --> Precision at optimal: {best_p:.4f}")
    print(f"  --> Recall at optimal:    {best_r:.4f}")
    print(f"{'='*50}\n")

    # Plot & Save
    plt.figure(figsize=(8, 6))
    plt.plot(dt_scores, f1, label='F1 Score', color='blue', linewidth=2)
    plt.axvline(x=best_conf, color='red', linestyle='--', label=f'Best Threshold ({best_conf:.2f})')
    plt.axhline(y=max_f1, color='green', linestyle=':', alpha=0.5)
    
    plt.title('F1 Score vs. Confidence Threshold')
    plt.xlabel('Confidence Threshold (Inferred)')
    plt.ylabel('F1 Score')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    
    out_path = Path(args.output).resolve()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved F1-Confidence curve image to {out_path}")

if __name__ == "__main__":
    main()
