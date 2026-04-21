"""
evaluate.py
Post-training evaluation and biological sanity checks.

Covers:
  - Milestone 4: AUROC / AUPRC on test set (detailed, per-feature)
  - Milestone 5: Saliency maps + model randomization test

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from model import DeepSEA
from data_utils import DeepSEANumpyDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--n_saliency', type=int, default=50,
                        help='Number of sequences to use for saliency maps')
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Which of the 919 features to visualize saliency for')
    return parser.parse_args()


# ── Milestone 4: Detailed Metrics ─────────────────────────────────────────────

def evaluate_test_set(model, test_dataset, device, batch_size=512):
    """
    Compute per-feature AUROC and AUPRC on the full test set.
    Returns arrays of shape (919,) for auroc and auprc.
    """
    model.eval()
    from torch.utils.data import DataLoader
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())

    y_pred = np.vstack(all_preds)   # (N_test, 919)
    y_true = np.vstack(all_labels)  # (N_test, 919)

    aurocs = np.zeros(919)
    auprcs = np.zeros(919)

    for i in range(919):
        if y_true[:, i].sum() == 0:
            aurocs[i] = np.nan
            auprcs[i] = np.nan
            continue
        aurocs[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
        auprcs[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    print(f"\n=== Test Set Metrics (919 features) ===")
    print(f"  Mean AUROC: {np.nanmean(aurocs):.4f}  (paper target: ~0.933)")
    print(f"  Mean AUPRC: {np.nanmean(auprcs):.4f}")
    print(f"  Median AUROC: {np.nanmedian(aurocs):.4f}")
    print(f"  Features with AUROC > 0.9: {(aurocs > 0.9).sum()} / 919")
    print(f"  Features with AUROC > 0.8: {(aurocs > 0.8).sum()} / 919")

    # Save per-feature results
    np.save('logs/per_feature_auroc.npy', aurocs)
    np.save('logs/per_feature_auprc.npy', auprcs)
    print("  Saved per-feature scores to logs/")

    return aurocs, auprcs, y_true, y_pred


# ── Milestone 5a: Saliency Maps ───────────────────────────────────────────────

def compute_saliency(model, x: torch.Tensor, feature_idx: int, device: str):
    """
    Compute input × gradient saliency map for one feature.

    The saliency map shows which nucleotide positions most influence
    the model's prediction for a specific chromatin feature.
    If trained correctly, high-saliency positions should overlap
    with known TF binding motifs (e.g., the TATA box: TATAAA).

    Args:
        x:           (1, 4, 1000) input sequence
        feature_idx: which of the 919 outputs to compute saliency for

    Returns:
        saliency: (1000,) importance score per position
    """
    model.eval()
    x = x.to(device).unsqueeze(0)  # (1, 4, 1000)
    x.requires_grad_(True)

    output = model(x)  # (1, 919)
    score = output[0, feature_idx]
    score.backward()

    # Gradient × input — highlights positions where both the input
    # signal is strong AND changing the input would change the output
    saliency = (x.grad * x).abs().sum(dim=1).squeeze()  # (1000,)
    return saliency.detach().cpu().numpy()


def plot_saliency(saliency: np.ndarray, title: str, save_path: str = None):
    """Plot a saliency map over the 1000-bp input sequence."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(range(len(saliency)), saliency, alpha=0.7, color='steelblue')
    ax.set_xlabel('Sequence position (bp)')
    ax.set_ylabel('Saliency score')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saliency plot saved to {save_path}")
    plt.show()


def generate_saliency_maps(model, test_dataset, feature_idx, n_samples, device):
    """
    Generate and plot average saliency map across n_samples sequences
    for a specific chromatin feature.
    """
    print(f"\nComputing saliency maps for feature {feature_idx} "
          f"over {n_samples} sequences...")

    # Only use positive-label sequences for this feature (more interesting)
    pos_indices = []
    for i in range(len(test_dataset)):
        _, y = test_dataset[i]
        if y[feature_idx] == 1:
            pos_indices.append(i)
        if len(pos_indices) >= n_samples:
            break

    if len(pos_indices) == 0:
        print(f"  No positive examples found for feature {feature_idx}")
        return

    print(f"  Found {len(pos_indices)} positive sequences")

    saliency_maps = []
    for idx in pos_indices[:n_samples]:
        x, _ = test_dataset[idx]
        sal = compute_saliency(model, x, feature_idx, device)
        saliency_maps.append(sal)

    avg_saliency = np.mean(saliency_maps, axis=0)

    plot_saliency(
        avg_saliency,
        title=f"Average Saliency Map — Feature {feature_idx} (n={len(pos_indices)} positive seqs)",
        save_path=f"logs/saliency_feature_{feature_idx}.png"
    )
    return avg_saliency


# ── Milestone 5b: Model Randomization Test ────────────────────────────────────

def model_randomization_test(model, test_dataset, feature_idx, device, n=200):
    """
    Sanity check: Randomly reinitialize all model weights and compare
    saliency maps to the trained model.

    If randomized saliency ≈ trained saliency → bug! Model isn't learning.
    If randomized saliency is noise and trained saliency shows patterns → good!

    Reference: Adebayo et al., NeurIPS 2018 'Sanity Checks for Saliency Maps'
    """
    import copy

    print("\n=== Model Randomization Test ===")
    print("Computing saliency for trained model...")
    trained_saliencies = []
    for i in range(n):
        x, y = test_dataset[i]
        if y[feature_idx] == 1:
            sal = compute_saliency(model, x, feature_idx, device)
            trained_saliencies.append(sal)

    if not trained_saliencies:
        print("No positive examples found. Skipping randomization test.")
        return

    trained_avg = np.mean(trained_saliencies, axis=0)

    # Now randomize all weights
    print("Randomizing model weights...")
    rand_model = copy.deepcopy(model)
    for layer in rand_model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    rand_model = rand_model.to(device)

    print("Computing saliency for randomized model...")
    rand_saliencies = []
    for i in range(n):
        x, y = test_dataset[i]
        if y[feature_idx] == 1:
            sal = compute_saliency(rand_model, x, feature_idx, device)
            rand_saliencies.append(sal)

    rand_avg = np.mean(rand_saliencies, axis=0)

    # Plot side by side
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].fill_between(range(1000), trained_avg, alpha=0.7, color='steelblue')
    axes[0].set_title(f'Trained Model — Feature {feature_idx}')
    axes[0].set_ylabel('Saliency')

    axes[1].fill_between(range(1000), rand_avg, alpha=0.7, color='salmon')
    axes[1].set_title('Randomized Model (sanity check — should look like noise)')
    axes[1].set_ylabel('Saliency')
    axes[1].set_xlabel('Sequence position (bp)')

    plt.tight_layout()
    plt.savefig(f'logs/randomization_test_feature_{feature_idx}.png', dpi=150)
    print("  Randomization test plot saved to logs/")

    # Correlation between trained and random: should be near 0
    corr = np.corrcoef(trained_avg, rand_avg)[0, 1]
    print(f"  Correlation (trained vs random): {corr:.4f}")
    print(f"  (Should be near 0 if model is learning data-dependent features)")
    print("=================================\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    import os
    os.makedirs('logs/', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = DeepSEA().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")

    # Load test dataset
    test_dataset = DeepSEANumpyDataset(f"{args.data_dir}/test_data.npy", f"{args.data_dir}/test_labels.npy")

    # Milestone 4: Full metrics
    aurocs, auprcs, y_true, y_pred = evaluate_test_set(model, test_dataset, device)

    # Milestone 5a: Saliency maps
    generate_saliency_maps(model, test_dataset,
                           feature_idx=args.feature_idx,
                           n_samples=args.n_saliency,
                           device=device)

    # Milestone 5b: Randomization test
    model_randomization_test(model, test_dataset,
                             feature_idx=args.feature_idx,
                             device=device)
