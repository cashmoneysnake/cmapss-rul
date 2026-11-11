import numpy as np
import pandas as pd
import pytest

from src.data.loading import (COLUMNS, RAW_DIR, SENSOR_COLS, add_rul, constant_sensors,
                              engine_arrays, load_split, load_test_rul, split_units)

needs_data = pytest.mark.skipif(not (RAW_DIR / "train_FD001.txt").exists(),
                                reason="raw data not downloaded (run make data)")


def tiny_df():
    return pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2],
                         "s_1": [5.0] * 5, "s_2": [1.0, 2.0, 3.0, 4.0, 5.0]})


def test_add_rul_counts_down_to_zero_per_engine():
    out = add_rul(tiny_df())
    assert out["rul"].tolist() == [2, 1, 0, 1, 0]


def test_constant_sensors_finds_zero_variance_only():
    df = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in SENSOR_COLS})
    df["s_4"] = 14.62  # never changes
    assert constant_sensors(df) == ["s_4"]


def test_split_units_is_by_engine_and_has_no_overlap():
    tr, va = split_units(range(1, 101), val_frac=0.2, seed=0)
    assert len(va) == 20 and len(tr) == 80
    assert set(tr).isdisjoint(set(va))


def test_split_units_is_reproducible():
    a = split_units(range(1, 101), seed=3)
    b = split_units(range(1, 101), seed=3)
    assert (a[1] == b[1]).all()


def test_engine_arrays_shapes():
    arrs = engine_arrays(tiny_df(), ["s_2"])
    assert arrs[1].shape == (3, 1) and arrs[2].shape == (2, 1)


@needs_data
def test_fd001_loads_with_26_columns_and_100_engines():
    df = load_split("FD001", "train")
    assert list(df.columns) == COLUMNS and df["unit"].nunique() == 100
    assert len(load_test_rul("FD001")) == 100
