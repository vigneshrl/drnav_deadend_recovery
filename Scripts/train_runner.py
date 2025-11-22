#!/usr/bin/env python3
"""Training runner for DR.Nav model.

Usage:
    python Scripts/train_runner.py --data_root /path/to/train_bags --save_dir /path/to/saved_models --batch_size 4 --epochs 20
"""
import argparse
import os
import sys
import torch

# Ensure Scripts/ is on sys.path so we can import the model package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model.data_loader import get_memory_efficient_data_loaders, train_model
from model.model_CA import DeadEndDetectionModel


def parse_args():
    p = argparse.ArgumentParser(description="Train DeadEndDetectionModel")
    p.add_argument("--data_root", required=True, help="Path to dataset root containing bag folders")
    p.add_argument("--save_dir", required=True, help="Directory where models/checkpoints will be saved")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading data from: {args.data_root}")

    train_loader, val_loader = get_memory_efficient_data_loaders(
        args.data_root, batch_size=args.batch_size
    )

    model = DeadEndDetectionModel()

    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
    )

    print("Training finished. Best model (if any) saved to:", args.save_dir)


if __name__ == "__main__":
    main()
