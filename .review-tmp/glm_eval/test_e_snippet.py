import numpy as np


def split_indices(
    n_samples: int, n_splits: int, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/test indices for n_splits cross-validation folds.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of CV folds.
        seed: RNG seed for reproducibility.

    Returns:
        List of (train_idx, test_idx) tuples, one per fold.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    fold_size = n_samples // n_splits
    splits = []
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        splits.append((train_idx, test_idx))
    return splits
