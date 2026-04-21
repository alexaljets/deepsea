"""
train.py
Training loop for the DeepSEA CNN.

Usage:
    python train.py --data_dir data_out/ --max_train_samples 500000
    python train.py --data_dir data_out/  # uses all 4.4M (slow)
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score

from model import DeepSEA
from data_utils import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train DeepSEA CNN')
    parser.add_argument('--data_dir',           type=str,   default='data_out/')
    parser.add_argument('--output_dir',         type=str,   default='checkpoints/')
    parser.add_argument('--epochs',             type=int,   default=60)
    parser.add_argument('--batch_size',         type=int,   default=128)
    parser.add_argument('--lr',                 type=float, default=1e-3)
    parser.add_argument('--num_workers',        type=int,   default=4)
    parser.add_argument('--log_every',          type=int,   default=200)
    parser.add_argument('--max_train_samples',  type=int,   default=500000,
                        help='Cap training set size. 500000 = ~15-20min/epoch on V100. '
                             'Use None/0 for full 4.4M dataset.')
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    aurocs, auprcs = [], []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        try:
            aurocs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            auprcs.append(average_precision_score(y_true[:, i], y_pred[:, i]))
        except Exception:
            pass
    return np.mean(aurocs), np.mean(auprcs)


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=200):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            batches_per_sec = (batch_idx + 1) / elapsed
            remaining = (len(loader) - batch_idx - 1) / batches_per_sec
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Elapsed: {elapsed/60:.1f}m | "
                  f"ETA: {remaining/60:.1f}m")

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, compute_auc=True):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x)
        loss = criterion(preds, y)
        total_loss += loss.item()
        if compute_auc:
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    if compute_auc and all_preds:
        auroc, auprc = compute_metrics(np.vstack(all_labels), np.vstack(all_preds))
    else:
        auroc, auprc = 0.0, 0.0

    return avg_loss, auroc, auprc


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs/', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    max_samples = args.max_train_samples if args.max_train_samples > 0 else None
    print(f"\nLoading data (max_train_samples={max_samples})...")
    train_loader, valid_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=max_samples,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DeepSEA().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auroc = 0.0
    log_lines = []
    print(f"\nStarting training: {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        print(f"── Epoch {epoch}/{args.epochs} ──────────────────────────────")
        t_epoch = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.log_every
        )
        val_loss, val_auroc, val_auprc = evaluate(
            model, valid_loader, criterion, device, compute_auc=True
        )
        scheduler.step(val_loss)

        epoch_mins = (time.time() - t_epoch) / 60
        print(f"  Epoch time: {epoch_mins:.1f}m")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | AUROC: {val_auroc:.4f} | AUPRC: {val_auprc:.4f}")

        log_lines.append(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_auroc:.6f},{val_auprc:.6f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'val_auprc': val_auprc,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ★ New best AUROC! Saved checkpoint.")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.output_dir, 'latest_model.pt'))

    # ── Save log ──────────────────────────────────────────────────────────────
    with open('logs/training_log.csv', 'w') as f:
        f.write("epoch,train_loss,val_loss,val_auroc,val_auprc\n")
        f.write('\n'.join(log_lines))

    # ── Final test eval ───────────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_auroc, test_auprc = evaluate(
        model, test_loader, criterion, device, compute_auc=True
    )
    print(f"\n=== Final Test Results ===")
    print(f"  Test Loss:  {test_loss:.4f}")
    print(f"  Test AUROC: {test_auroc:.4f}  (paper reports ~0.933)")
    print(f"  Test AUPRC: {test_auprc:.4f}")
    print("=" * 30)


if __name__ == '__main__':
    args = parse_args()
    train(args)
