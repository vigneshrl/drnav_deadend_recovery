#!/usr/bin/env python3
"""Inference / visualization runner for DR.Nav model.

Usage:
    python Scripts/inference_runner.py --model_path /path/to/model_best.pth --data_root /path/to/test_bags --output_dir /path/to/out --batch_size 8 --num_samples 50
"""
import argparse
import os
import sys
import torch

# Ensure Scripts/ is on sys.path so we can import the model package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model.data_loader import visualize_test_results


def parse_args():
    p = argparse.ArgumentParser(description="Run inference and visualize predictions")
    p.add_argument("--model_path", required=True, help="Path to saved model weights (.pth)")
    p.add_argument("--data_root", required=True, help="Path to dataset root containing bag folders")
    p.add_argument("--output_dir", required=True, help="Directory to save visualizations and predictions.json")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Running inference on device: {device}")
    print(f"Using model: {args.model_path}")
    print(f"Data root: {args.data_root}")

    results = visualize_test_results(
        model_path=args.model_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=device,
        output_dir=args.output_dir,
    )

    if results is None:
        print("Inference finished with errors (see logs above)")
    else:
        print(f"Inference finished. Saved artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
