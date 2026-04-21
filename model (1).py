"""
model.py
DeepSEA CNN architecture in PyTorch.

Architecture (from Zhou & Troyanskaya, 2015):
  Input: (batch, 4, 1000)  — one-hot DNA sequence
  Conv1 -> MaxPool -> Dropout
  Conv2 -> MaxPool -> Dropout
  Conv3 -> Dropout
  Flatten
  FC (2003 -> 925)
  FC (925 -> 919)
  Sigmoid output

Output: (batch, 919) probabilities for 919 chromatin features
"""

import torch
import torch.nn as nn


class DeepSEA(nn.Module):
    """
    Faithful PyTorch reproduction of the DeepSEA CNN.

    Key design choices:
    - 3 convolutional layers with increasing filter counts (320, 480, 960)
    - Max pooling after conv1 and conv2 to reduce spatial dimension
    - Dropout after each conv for regularization (genomic data overfits fast)
    - Two fully connected layers to integrate global motif information
    - Sigmoid (not softmax) because this is multi-label classification:
      each of the 919 outputs is an independent binary prediction
    """

    def __init__(
        self,
        sequence_length: int = 1000,
        n_features: int = 919,
        conv1_filters: int = 320,
        conv2_filters: int = 480,
        conv3_filters: int = 960,
        conv_kernel_size: int = 8,
        pool_size: int = 4,
        pool_stride: int = 4,
        fc_size: int = 925,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.n_features = n_features

        # ── Convolutional blocks ──────────────────────────────────────────
        # Conv1d input shape: (batch, channels_in, length)
        # After conv: length changes to (L - kernel + 1)
        # After max pool: length changes to floor((L - pool_size) / pool_stride + 1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=conv1_filters,
                      kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride),
            nn.Dropout(p=dropout_rate),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters,
                      kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride),
            nn.Dropout(p=dropout_rate),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv2_filters, out_channels=conv3_filters,
                      kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # ── Calculate flattened size after all conv/pool layers ───────────
        # Run a dummy forward pass to figure out the exact FC input size.
        # This avoids hardcoding and works even if you change kernel sizes.
        self._flat_size = self._get_flat_size(sequence_length)

        # ── Fully connected layers ────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, fc_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_size, n_features),
            nn.Sigmoid(),  # Multi-label: each output is independent probability
        )

    def _get_flat_size(self, sequence_length: int) -> int:
        """Compute the flattened feature size by passing a dummy tensor."""
        with torch.no_grad():
            dummy = torch.zeros(1, 4, sequence_length)
            out = self.conv1(dummy)
            out = self.conv2(out)
            out = self.conv3(out)
            return out.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 4, 1000) one-hot encoded DNA sequences
        Returns:
            out: (batch, 919) sigmoid probabilities for each chromatin feature
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out


def build_model(device: str = 'cuda') -> DeepSEA:
    """Instantiate model and move to device. Print architecture summary."""
    model = DeepSEA()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeepSEA model built on {device}")
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  FC input size: {model._flat_size}")
    print(f"  Output: {model.n_features} chromatin features")
    return model


# ── Quick architecture sanity check ──────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(device)

    # Dummy forward pass — checks shapes are all correct
    batch_size = 4
    dummy_seq = torch.zeros(batch_size, 4, 1000).to(device)
    output = model(dummy_seq)
    print(f"\nDummy forward pass:")
    print(f"  Input:  {dummy_seq.shape}")
    print(f"  Output: {output.shape}")   # Should be (4, 919)
    assert output.shape == (batch_size, 919), "Shape mismatch!"
    print("  ✓ Shape check passed")

    # Check output is valid probabilities
    assert output.min() >= 0 and output.max() <= 1, "Sigmoid output out of [0,1]!"
    print("  ✓ Probability range check passed")
