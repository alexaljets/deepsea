"""
data_utils.py
Utilities for loading the DeepSEA dataset.

Supports two formats:
  - .npy files (from jakublipinski rebuild script)
      train_data.npy:   (N, 1000, 4)
      train_labels.npy: (N, 919)

  - .mat HDF5 files (original Princeton format)
      trainxdata: (1000, 4, N)
      traindata:  (919, N)

Both are normalized to (4, 1000) per sample for PyTorch Conv1d.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── .npy Dataset (use this — it's what you have) ──────────────────────────────

class DeepSEANumpyDataset(Dataset):
    """
    Loads from the .npy files produced by the rebuild script.

    Input shapes on disk:
      data:   (N, 1000, 4)
      labels: (N, 919)

    __getitem__ returns:
      x: (4, 1000)  float32  — transposed for Conv1d
      y: (919,)     float32
    """

    def __init__(self, data_path: str, labels_path: str, max_samples: int = None):
        # mmap_mode='r' means the array is NOT loaded into RAM —
        # numpy reads slices directly from disk. Essential for 17GB files.
        self.X = np.load(data_path, mmap_mode='r')    # (N, 1000, 4)
        self.y = np.load(labels_path, mmap_mode='r')  # (N, 919)

        if max_samples is not None:
            self.X = self.X[:max_samples]
            self.y = self.y[:max_samples]

        assert self.X.shape[0] == self.y.shape[0], "Data/label sample count mismatch"
        assert self.X.shape[1] == 1000, f"Expected 1000bp sequences, got {self.X.shape[1]}"
        assert self.X.shape[2] == 4,    f"Expected 4 channels, got {self.X.shape[2]}"
        assert self.y.shape[1] == 919,  f"Expected 919 features, got {self.y.shape[1]}"

        print(f"Loaded {len(self.X):,} samples | "
              f"seq shape: (4, 1000) | labels: {self.y.shape[1]} features")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Disk: (1000, 4) -> transpose to (4, 1000) for Conv1d
        x = self.X[idx]                                    # (1000, 4) numpy
        x = torch.tensor(x, dtype=torch.float32).T        # -> (4, 1000)
        y = torch.tensor(self.y[idx], dtype=torch.float32) # (919,)
        return x, y


# ── .mat HDF5 Dataset (kept for reference, not needed right now) ───────────────

class DeepSEAMatDataset(Dataset):
    """
    Loads from the original Princeton .mat (HDF5) files.
    Only use this if you have the original Princeton bundle.

    Input shapes on disk:
      trainxdata: (1000, 4, N)
      traindata:  (919, N)
    """

    SEQ_KEYS   = {'train': 'trainxdata', 'valid': 'validxdata', 'test': 'testxdata'}
    LABEL_KEYS = {'train': 'traindata',  'valid': 'validdata',  'test': 'testdata'}

    def __init__(self, mat_path: str, split: str = 'train'):
        assert split in ('train', 'valid', 'test')
        self._file = h5py.File(mat_path, 'r')
        self._X = self._file[self.SEQ_KEYS[split]]    # (1000, 4, N)
        self._y = self._file[self.LABEL_KEYS[split]]  # (919, N)
        self.n_samples = self._X.shape[-1]
        print(f"[{split}] Loaded {self.n_samples:,} samples from .mat")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self._X[:, :, idx]           # (1000, 4)
        x = np.transpose(x, (1, 0))      # (4, 1000)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self._y[:, idx], dtype=torch.float32)
        return x, y

    def close(self):
        self._file.close()


# ── DataLoader factory (uses .npy by default) ─────────────────────────────────

def get_dataloaders(data_dir: str, batch_size: int = 128, num_workers: int = 4, max_train_samples: int = None):
    """
    Returns train, valid, test DataLoaders from .npy files.

    Expected files in data_dir:
      train_data.npy, train_labels.npy
      valid_data.npy, valid_labels.npy
      test_data.npy,  test_labels.npy
    """
    loaders = {}
    for split in ('train', 'valid', 'test'):
        dataset = DeepSEANumpyDataset(
            data_path   = os.path.join(data_dir, f'{split}_data.npy'),
            labels_path = os.path.join(data_dir, f'{split}_labels.npy'),
            max_samples = max_train_samples if split == 'train' else None,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders['train'], loaders['valid'], loaders['test']


# ── Quick sanity check ────────────────────────────────────────────────────────

def quick_data_check(data_dir: str, n_check: int = 100):
    print(f"\n=== Data Check: {data_dir} ===")

    ds = DeepSEANumpyDataset(
        os.path.join(data_dir, 'train_data.npy'),
        os.path.join(data_dir, 'train_labels.npy'),
    )

    xs, ys = [], []
    for i in range(min(n_check, len(ds))):
        x, y = ds[i]
        xs.append(x.numpy())
        ys.append(y.numpy())

    X = np.stack(xs)  # (n, 4, 1000)
    Y = np.stack(ys)  # (n, 919)

    col_sums = X.sum(axis=1)  # sum over 4 channels -> (n, 1000), should be ~1.0
    print(f"One-hot col sums — min: {col_sums.min():.3f}, max: {col_sums.max():.3f} (expect ~1.0)")
    print(f"Label positive rate: {Y.mean():.4f} (expect ~0.03-0.05)")
    print("=== Check complete ===\n")


if __name__ == '__main__':
    import sys
    quick_data_check(sys.argv[1] if len(sys.argv) > 1 else 'data_out/')
