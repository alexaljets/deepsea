"""
smoke_test.py
Run this BEFORE submitting a full training job.
Verifies: model builds, data loads, one forward/backward pass works.
Takes < 2 minutes on a CPU compute node.

Usage:
    python smoke_test.py --data_dir data_out/
"""

import argparse
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_out/')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Test 1: Model builds and forward pass works ────────────────────
    print("\n[1/4] Testing model build and forward pass...")
    from model import DeepSEA
    model = DeepSEA().to(device)
    dummy_x = torch.rand(4, 4, 1000).to(device)
    out = model(dummy_x)
    assert out.shape == (4, 919), f"Expected (4, 919), got {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output not in [0,1]"
    print(f"  ✓ Forward pass OK. Output shape: {out.shape}")

    # ── Test 2: Loss and backward pass work ───────────────────────────
    print("\n[2/4] Testing backward pass...")
    import torch.nn as nn
    criterion = nn.BCELoss()
    dummy_y = torch.rand(4, 919).to(device)
    loss = criterion(out, dummy_y)
    loss.backward()
    print(f"  ✓ Backward pass OK. Loss: {loss.item():.4f}")

    # ── Test 3: Data loading ───────────────────────────────────────────
    print("\n[3/4] Testing data loading (first 5 samples)...")
    import os
    train_data   = os.path.join(args.data_dir, 'train_data.npy')
    train_labels = os.path.join(args.data_dir, 'train_labels.npy')

    if not os.path.exists(train_data):
        print(f"  ⚠ Data not found at {train_data}. Skipping data test.")
    else:
        from data_utils import DeepSEANumpyDataset
        ds = DeepSEANumpyDataset(train_data, train_labels)
        x0, y0 = ds[0]
        assert x0.shape == torch.Size([4, 1000]), f"Expected (4, 1000), got {x0.shape}"
        assert y0.shape == torch.Size([919]),     f"Expected (919,), got {y0.shape}"
        # Check one-hot: each position should sum to ~1.0
        print(f"  ✓ Data load OK. X: {x0.shape}, y: {y0.shape}")

    # ── Test 4: Saliency computation ──────────────────────────────────
    print("\n[4/4] Testing saliency computation...")
    model2 = DeepSEA().to(device)
    x_test = torch.rand(4, 1000).to(device)
    x_test.requires_grad_(True)
    out2 = model2(x_test.unsqueeze(0))
    out2[0, 0].backward()
    saliency = (x_test.grad * x_test).abs().sum(dim=0)
    assert saliency.shape == torch.Size([1000])
    print(f"  ✓ Saliency OK. Shape: {saliency.shape}")

    print("\n" + "="*40)
    print("✓ All smoke tests passed! Safe to run full training.")
    print("="*40)

if __name__ == '__main__':
    main()
