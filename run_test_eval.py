"""
run_test_eval.py
Loads the best checkpoint and runs final test set evaluation.
"""

import torch
import torch.nn as nn
from model import DeepSEA
from data_utils import get_dataloaders
from train import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load model
model = DeepSEA().to(device)
ckpt = torch.load('checkpoints/best_model.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded checkpoint from epoch {ckpt['epoch']} | Val AUROC: {ckpt['val_auroc']:.4f}")

# Load test data
_, _, test_loader = get_dataloaders('data_out/', batch_size=128, num_workers=4)

# Evaluate
criterion = nn.BCELoss()
test_loss, test_auroc, test_auprc = evaluate(model, test_loader, criterion, device)

print(f"\n=== Final Test Results ===")
print(f"  Test Loss:  {test_loss:.4f}")
print(f"  Test AUROC: {test_auroc:.4f}  (paper reports ~0.933)")
print(f"  Test AUPRC: {test_auprc:.4f}")
print("=" * 30)
