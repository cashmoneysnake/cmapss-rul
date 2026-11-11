"""Turn one engine's sensor history into fixed-length windows for a sequence model."""
import numpy as np


def make_windows(features: np.ndarray, window: int = 30) -> np.ndarray:
    """Slide a window one cycle at a time over one engine.

    features: shape (n_cycles, n_features)
    returns:  shape (n_windows, window, n_features), n_windows = n_cycles - window + 1
    """
    n_cycles = len(features)
    if n_cycles < window:
        raise ValueError(f"engine has {n_cycles} cycles, fewer than window={window}")
    return np.stack([features[start:start + window] for start in range(n_cycles - window + 1)])


def make_targets(rul: np.ndarray, window: int = 30, cap: int = 125) -> np.ndarray:
    """Target for each window = RUL at its last cycle, capped.

    rul: shape (n_cycles,), true RUL at every cycle of one training engine
    returns: shape (n_windows,), matching make_windows row for row
    """
    return np.minimum(rul[window - 1:], cap)


def pad_start(features: np.ndarray, window: int = 30) -> np.ndarray:
    """Repeat the first row at the start until the engine has `window` cycles."""
    n_missing = window - len(features)
    if n_missing <= 0:
        return features
    padding = np.repeat(features[:1], n_missing, axis=0)
    return np.concatenate([padding, features])


def last_window(features: np.ndarray, window: int = 30) -> np.ndarray:
    """The single window used to score one test engine (padded if it is short).

    returns: shape (window, n_features)
    """
    return pad_start(features, window)[-window:]
