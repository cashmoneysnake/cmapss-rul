"""Summary features for tree models: collapse each window into a flat row."""
import numpy as np


def window_features(X: np.ndarray) -> np.ndarray:
    """(n, window, f) -> (n, 4*f): last value, mean, std, and change (last - first)."""
    last = X[:, -1, :]
    return np.concatenate([last, X.mean(axis=1), X.std(axis=1), last - X[:, 0, :]], axis=1)
