"""
Phase 4 — Dataset Loader & PyTorch Dataset/DataLoader
======================================================
Production-grade data pipeline with:
  - Participant-aware train/val/test splitting (avoid data leakage)
  - Per-participant normalization (generalization across users)
  - Augmentation for small datasets
  - Weighted sampling for label imbalance
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Optional, List, Dict
import warnings


# Constants
FEATURE_COLUMNS = [
    "ear", "gaze_pitch", "gaze_yaw",
    "head_pitch", "head_yaw", "head_roll",
    "eyebrow_tension", "eye_openness"
]
N_FEATURES = len(FEATURE_COLUMNS)   # 8
SEQ_LEN = 150                        # 5s @ 30fps


# Augmentation
def augment_sequence(x: np.ndarray) -> np.ndarray:
    """
    Lightweight augmentation for small datasets.
    Applied ONLY during training.

    Techniques chosen for physiological time-series:
    1. Gaussian noise: simulates sensor jitter / lighting variation
    2. Temporal shift: simulates slight temporal misalignment in windows
    3. Feature dropout: simulates partial occlusion / face-tracking loss
    4. Magnitude scaling: simulates inter-user physiological variance
    """
    x = x.copy()

    # 1. Additive Gaussian noise (σ = 0.01, i.e. 1% of normalized range)
    if np.random.rand() < 0.5:
        x += np.random.normal(0, 0.01, x.shape).astype(np.float32)

    # 2. Magnitude scaling (±10%)
    if np.random.rand() < 0.4:
        scale = np.random.uniform(0.90, 1.10, (1, N_FEATURES)).astype(np.float32)
        x *= scale

    # 3. Random feature dropout (0–2 features zeroed for up to 10 frames)
    if np.random.rand() < 0.3:
        n_drop = np.random.randint(1, 3)
        feat_idx = np.random.choice(N_FEATURES, n_drop, replace=False)
        start = np.random.randint(0, SEQ_LEN - 20)
        length = np.random.randint(5, 20)
        x[start:start + length, feat_idx] = 0.0

    # 4. Temporal shift (roll by ±15 frames to simulate window boundary variation)
    if np.random.rand() < 0.3:
        shift = np.random.randint(-15, 15)
        x = np.roll(x, shift, axis=0)

    return x.astype(np.float32)


# Normalization
class FeatureScaler:
    """
    Per-feature standardization (z-score).
    IMPORTANT: fit only on TRAINING data, then transform val/test.
    Saved as .npz alongside model checkpoints.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        """X: (N, T, F)"""
        self.mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, F)
        self.std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str):
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
        return self


# PyTorch Dataset
class CognitiveLoadDataset(Dataset):
    """
    PyTorch Dataset for cognitive load temporal sequences.

    Supports:
    - Label smoothing (reduces overconfidence on pseudo-labels)
    - Soft labels (float targets instead of hard 0/1)
    - Training augmentation
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        augment: bool = False,
        label_smoothing: float = 0.05,
    ):
        """
        Args:
            X: (N, seq_len, features) — normalized sequences
            Y: (N, 1)                 — continuous cognitive load labels [0, 1]
            augment: apply augmentation (training only)
            label_smoothing: shrinks labels toward 0.5 to handle pseudo-label noise.
                             Formula: y_smooth = y * (1 - ε) + 0.5 * ε
        """
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.augment = augment

        # Label smoothing
        if label_smoothing > 0:
            self.Y = self.Y * (1.0 - label_smoothing) + 0.5 * label_smoothing

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.Y[idx]

        if self.augment:
            x = augment_sequence(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Data Loading Utilities
def load_raw_data(
    processed_dir: str,
    dataset_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads preprocessed X/Y tensors.
    Also tries to reconstruct participant IDs for proper group-based splitting.

    Returns:
        X: (N, T, F)
        Y: (N, 1)
        groups: list of participant IDs (length N) — used for GroupShuffleSplit
    """
    X_path = os.path.join(processed_dir, "X_sequences.npy")
    Y_path = os.path.join(processed_dir, "Y_labels.npy")

    if not os.path.exists(X_path) or not os.path.exists(Y_path):
        raise FileNotFoundError(
            f"Processed data not found in {processed_dir}. "
            "Run preprocess.py first."
        )

    X = np.load(X_path)
    Y = np.load(Y_path)

    # Try to reconstruct group labels from raw dataset directory
    groups = _reconstruct_groups(X, Y, dataset_dir) if dataset_dir else ["unknown"] * len(X)

    print(f"[DataLoader] Loaded X={X.shape}, Y={Y.shape}")
    print(f"[DataLoader] Participants: {len(set(groups))}")
    print(f"[DataLoader] Label stats — mean={Y.mean():.3f}, std={Y.std():.3f}, "
          f"min={Y.min():.3f}, max={Y.max():.3f}")

    return X, Y, groups


def _reconstruct_groups(X: np.ndarray, Y: np.ndarray, dataset_dir: str) -> List[str]:
    """
    Attempts to map sequence windows back to participant IDs.
    Since preprocess.py loses participant info, we do a best-effort reconstruction.
    """
    import pandas as pd
    FEATURE_COLUMNS_LOCAL = [
        "ear", "gaze_pitch", "gaze_yaw",
        "head_pitch", "head_yaw", "head_roll",
        "eyebrow_tension", "eye_openness"
    ]
    STRIDE = 30

    groups = []
    sessions = sorted(glob.glob(f"{dataset_dir}/*/*"))

    for session_path in sessions:
        csv_path = os.path.join(session_path, "features.csv")
        meta_path = os.path.join(session_path, "metadata.json")

        if not os.path.exists(csv_path) or not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        participant_id = meta.get("participant_id", "unknown")
        df = pd.read_csv(csv_path)
        num_frames = len(df)
        num_windows = max(0, (num_frames - SEQ_LEN) // STRIDE + 1)

        for _ in range(num_windows):
            groups.append(participant_id)

    # If mismatch (e.g., new sessions added), pad or trim
    if len(groups) != len(X):
        warnings.warn(
            f"Group reconstruction mismatch: {len(groups)} groups vs {len(X)} sequences. "
            "Falling back to single-group mode (random split). "
            "This may cause data leakage if multiple participants exist."
        )
        return ["all"] * len(X)

    return groups


def create_dataloaders(
    processed_dir: str,
    dataset_dir: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,         # 0 = main process; safer on macOS
    label_smoothing: float = 0.05,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureScaler]:
    """
    Creates train/val/test DataLoaders with proper group-based splitting.

    Critical design decisions:
    - GroupShuffleSplit ensures no participant appears in both train and val/test.
      Without this, the model memorizes individual face patterns, not load patterns.
    - Scaler is fit ONLY on training data.
    - WeightedRandomSampler upsamples rare label ranges to address class imbalance.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, Y, groups = load_raw_data(processed_dir, dataset_dir)
    N = len(X)

    # Group-aware train / (val+test) split
    unique_groups = list(set(groups))
    n_unique = len(unique_groups)

    if n_unique >= 3:
        # Full group-aware split (no data leakage across participants)
        gss_outer = GroupShuffleSplit(
            n_splits=1,
            test_size=round(1 - train_ratio, 2),
            random_state=seed,
        )
        train_idx, temp_idx = next(gss_outer.split(X, Y, groups))
        temp_groups = [groups[i] for i in temp_idx]
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        try:
            val_rel, test_rel = next(gss_inner.split(X[temp_idx], Y[temp_idx], temp_groups))
            val_idx  = temp_idx[val_rel]
            test_idx = temp_idx[test_rel]
        except Exception:
            mid = len(temp_idx) // 2
            val_idx, test_idx = temp_idx[:mid], temp_idx[mid:]
    else:
        # ≤2 participants — use stratified random split (risk of mild leakage, but
        # unavoidable with tiny datasets; warn the user)
        if n_unique < 3:
            warnings.warn(
                f"Only {n_unique} unique participant(s). Using random split. "
                "Record sessions with more participants for proper LOPO validation."
            )
        np.random.seed(seed)
        idx = np.random.permutation(N)
        n_train = max(1, int(N * train_ratio))
        n_val   = max(1, int(N * val_ratio))
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

    print(f"[Split] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Normalization (fit on train only)
    scaler = FeatureScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]

    # Datasets
    train_ds = CognitiveLoadDataset(X_train, Y_train, augment=True, label_smoothing=label_smoothing)
    val_ds = CognitiveLoadDataset(X_val, Y_val, augment=False, label_smoothing=0.0)
    test_ds = CognitiveLoadDataset(X_test, Y_test, augment=False, label_smoothing=0.0)

    # Weighted Sampler (label balance)
    if use_weighted_sampler and len(train_idx) > 10:
        sampler = _build_weighted_sampler(Y_train)
        shuffle_train = False
    else:
        sampler = None
        shuffle_train = True

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,              # Avoid batch-size-1 batches (bad for LayerNorm)
        persistent_workers=False,    # Safe default for macOS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,   # Larger batches OK for eval (no grad)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, scaler


def _build_weighted_sampler(Y: np.ndarray, n_bins: int = 5) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that upsamples under-represented label ranges.
    Divides [0,1] into n_bins equal-width buckets and assigns inverse-frequency weights.
    """
    y_flat = Y.flatten()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_flat, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    bin_counts = np.where(bin_counts == 0, 1, bin_counts)  # avoid div-by-zero
    weights = 1.0 / bin_counts[bin_ids]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(Y),
        replacement=True,
    )
    return sampler
