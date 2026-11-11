import numpy as np
import pytest

from src.evaluate import nasa_score, rmse, score_test_set


def test_worked_example():
    assert rmse([40, 40], [30, 50]) == pytest.approx(10)
    assert nasa_score([40, 40], [30, 50]) == pytest.approx(2.876, abs=1e-3)


def test_perfect_predictions_score_zero():
    assert rmse([10, 20], [10, 20]) == 0
    assert nasa_score([10, 20], [10, 20]) == 0


def test_late_is_punished_more_than_early_but_rmse_is_equal():
    early, late = nasa_score([40], [30]), nasa_score([40], [50])
    assert late > early
    assert rmse([40], [30]) == rmse([40], [50])


def test_score_test_set_uses_last_window_per_engine_in_unit_order():
    engines = {2: np.arange(1, 6).reshape(-1, 1).astype(float),   # last value 5
               1: np.arange(1, 41).reshape(-1, 1).astype(float)}  # last value 40
    seen = {}

    def predict(X):
        seen["shape"] = X.shape
        return X[:, -1, 0]  # "predict" the last reading

    out = score_test_set(predict, engines, true_rul=[40, 5], window=30)
    assert seen["shape"] == (2, 30, 1)   # short engine 2 was padded
    assert out["rmse"] == 0              # unit 1 -> 40, unit 2 -> 5, matches true_rul order
