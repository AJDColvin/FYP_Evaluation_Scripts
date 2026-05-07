#!/usr/bin/env python3
"""
Multi-F1 Curve Plotter
======================
Evaluates multiple predictions.coco.json files against a single ground truth
and plots all F1-Confidence curves on the same graph for visual comparison.
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
        description="Compare multiple F1-Confidence curves on a single plot."
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to the ground truth _annotations.coco.json file.",
    )
    parser.add_argument(
        "--predictions",
        nargs='+',
        type=str,
        required=True,
        help="List of paths to predictions.coco.json files (space separated).",
    )
    parser.add_argument(
        "--labels",
        nargs='+',
        type=str,
        required=True,
        help="List of labels for each predictions file, corresponding to the order provided.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_f1_curves.png",
        help="Output image path (default: combined_f1_curves.png).",
    )
    return parser.parse_args()


def get_f1_curve(coco_gt: COCO, pred_path: Path, iou: float):
    """Evaluates the predictions and returns the F1 curve arrays."""
    print(f"--> Evaluating: {pred_path.name}...")
    coco_dt = coco_gt.loadRes(str(pred_path))

    if len(coco_dt.anns) == 0:
        return np.array([]), np.array([]), 0.0, 0.0

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = [iou]
    coco_eval.params.maxDets = [2000]

    import os
    from contextlib import redirect_stdout
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        coco_eval.evaluate()
        coco_eval.accumulate()

    dt_scores = []
    dt_matches = []
    total_gt = 0

    for e in coco_eval.evalImgs:
        if e is None:
            continue
        valid_idx = (e['dtIgnore'][0] == 0)
        matches = e['dtMatches'][0][valid_idx]
        scores = np.array(e['dtScores'])[valid_idx]

        tp = (matches > 0).astype(int)
        dt_scores.extend(scores.tolist())
        dt_matches.extend(tp.tolist())

        total_gt += len(e['gtIgnore']) - sum(e['gtIgnore'])

    if len(dt_scores) == 0 or total_gt == 0:
        return np.array([]), np.array([]), 0.0, 0.0

    dt_scores = np.array(dt_scores)
    dt_matches = np.array(dt_matches)

    sort_idx = np.argsort(dt_scores)[::-1]
    dt_scores = dt_scores[sort_idx]
    dt_matches = dt_matches[sort_idx]

    cum_tp = np.cumsum(dt_matches)
    cum_fp = np.cumsum(1 - dt_matches)

    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-16)

    f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-16)

    max_f1_idx = np.argmax(f1)
    return dt_scores, f1, f1[max_f1_idx], dt_scores[max_f1_idx]


def main() -> None:
    args = parse_args()

    if len(args.predictions) != len(args.labels):
        print("Error: The number of --predictions must match the number of --labels.", file=sys.stderr)
        sys.exit(1)

    gt_path = Path(args.gt).resolve()
    if not gt_path.exists():
        print(f"Error: ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading Ground Truth...")
    coco_gt = COCO(str(gt_path))

    plt.figure(figsize=(10, 7))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']

    for i, (pred_file, label) in enumerate(zip(args.predictions, args.labels)):
        pred_path = Path(pred_file).resolve()
        if not pred_path.exists():
            print(f"Warning: {pred_path} not found. Skipping.")
            continue

        dt_scores, f1, best_f1, best_conf = get_f1_curve(coco_gt, pred_path, args.iou)
        
        if len(dt_scores) == 0:
            print(f"Warning: No valid data found for {label}. Skipping plot line.")
            continue

        c = colors[i % len(colors)]
        
        # Plot curve
        plt.plot(dt_scores, f1, label=f"{label} (Max: {best_f1:.4f} @ {best_conf:.2f})", color=c, linewidth=2)
        
        # Plot a scatter point at the peak
        plt.scatter(best_conf, best_f1, color=c, s=50, zorder=5, marker='x')

    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend(loc='lower center')

    out_path = Path(args.output).resolve()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n=============================================")
    print(f"Saved combined F1-Confidence curve to {out_path}")
    print(f"=============================================")


if __name__ == "__main__":
    main()
