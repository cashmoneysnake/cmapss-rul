"""Raw files -> scaled, windowed train/validation arrays plus the test engines."""
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.loading import (SENSOR_COLS, add_rul, constant_sensors, engine_arrays,
                              load_split, load_test_rul, split_units)
from src.data.windowing import make_targets, make_windows


@dataclass
class Datasets:
    X_train: np.ndarray       # (n_windows, window, n_features)
    y_train: np.ndarray       # capped RUL
    X_val: np.ndarray
    y_val: np.ndarray
    test_engines: dict        # {unit: (n_cycles, n_features)}, scaled with train statistics
    test_rul: np.ndarray      # NASA's true RUL, in unit order
    feature_cols: list


def _windows_for(arrays: dict, rul_by_unit: dict, window: int, cap: int):
    X, y = [], []
    for u, feats in arrays.items():
        X.append(make_windows(feats, window))
        y.append(make_targets(rul_by_unit[u], window, cap))
    return np.concatenate(X), np.concatenate(y)


def build_datasets(subset: str, window: int = 30, cap: int = 125,
                   seed: int = 0, val_frac: float = 0.2) -> Datasets:
    train_df = add_rul(load_split(subset, "train"))
    test_df = load_split(subset, "test")

    feature_cols = [c for c in SENSOR_COLS if c not in constant_sensors(train_df)]

    # split by whole engine, then fit the scaler on TRAIN engines only
    train_units, val_units = split_units(train_df["unit"].unique(), val_frac, seed)
    tr_df = train_df[train_df["unit"].isin(train_units)]
    scaler = StandardScaler().fit(tr_df[feature_cols])

    def scaled(df):
        out = df.copy()
        out[feature_cols] = scaler.transform(df[feature_cols])
        return out

    train_s, test_s = scaled(train_df), scaled(test_df)
    rul_by_unit = {u: g.sort_values("cycle")["rul"].to_numpy()
                   for u, g in train_df.groupby("unit")}

    X_tr, y_tr = _windows_for(engine_arrays(train_s, feature_cols, train_units),
                              rul_by_unit, window, cap)
    X_va, y_va = _windows_for(engine_arrays(train_s, feature_cols, val_units),
                              rul_by_unit, window, cap)
    return Datasets(X_tr, y_tr, X_va, y_va,
                    engine_arrays(test_s, feature_cols), load_test_rul(subset), feature_cols)
