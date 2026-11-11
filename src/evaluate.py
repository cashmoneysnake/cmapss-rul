"""Metrics and last-window test scoring (see docs/evaluation_protocol.md)."""
import numpy as np

from src.data.windowing import last_window


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """NASA asymmetric score, summed over engines. Lower is better.

    d = predicted - true.  Early (d < 0): exp(-d/13) - 1.  Late (d >= 0): exp(d/10) - 1.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def score_test_set(predict_fn, test_engines: dict, true_rul, window: int = 30) -> dict:
    """Score a final model on the official test set, using each engine's last window.

    predict_fn:   takes an array (n_engines, window, n_features), returns (n_engines,)
    test_engines: {unit: array (n_cycles, n_features)} as built by engine_arrays
    true_rul:     NASA's RUL_FD00X values, in unit order
    """
    units = sorted(test_engines)
    X = np.stack([last_window(test_engines[u], window) for u in units])
    y_pred = np.asarray(predict_fn(X), dtype=float)
    return {"rmse": rmse(true_rul, y_pred), "nasa_score": nasa_score(true_rul, y_pred)}
