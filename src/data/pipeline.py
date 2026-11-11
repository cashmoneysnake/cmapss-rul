"""Raw files -> scaled, windowed train/validation arrays plus the test engines."""
from dataclasses import dataclass

import numpy as np

from src.data.loading import (SENSOR_COLS, add_rul, constant_sensors, engine_arrays,
                              load_split, load_test_rul, split_units)
from src.data.scaling import ConditionScaler
from src.data.windowing import make_targets, make_windows


N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass
class Datasets:
    X_train: np.ndarray       # (n_windows, window, n_features)
    y_train: np.ndarray       # capped RUL
    X_val: np.ndarray
    y_val: np.ndarray
    val_end_cycle: np.ndarray  # cycle number each validation window ends at
    test_engines: dict        # {unit: (n_cycles, n_features)}, scaled with train statistics
    test_rul: np.ndarray      # NASA's true RUL, in unit order
    feature_cols: list


def _windows_for(arrays: dict, rul_by_unit: dict, window: int, cap: int):
    X, y, end = [], [], []
    for u, feats in arrays.items():
        X.append(make_windows(feats, window))
        y.append(make_targets(rul_by_unit[u], window, cap))
        end.append(np.arange(window, len(feats) + 1))   # window i ends at cycle window + i
    return np.concatenate(X), np.concatenate(y), np.concatenate(end)


def build_datasets(subset: str, window: int = 30, cap: int = 125,
                   seed: int = 0, val_frac: float = 0.2) -> Datasets:
    train_df = add_rul(load_split(subset, "train"))
    test_df = load_split(subset, "test")

    feature_cols = [c for c in SENSOR_COLS if c not in constant_sensors(train_df)]

    # split by whole engine, then fit the scaler on TRAIN engines only
    train_units, val_units = split_units(train_df["unit"].unique(), val_frac, seed)
    tr_df = train_df[train_df["unit"].isin(train_units)]
    scaler = ConditionScaler(N_CONDITIONS[subset]).fit(tr_df, feature_cols)

    train_s, test_s = scaler.transform(train_df), scaler.transform(test_df)

    # drop sensors with zero variance AFTER per-condition scaling (judged on train engines only).
    # In FD002/FD004 some sensors only encode the operating condition and are flat inside each one.
    train_part = train_s[train_s["unit"].isin(train_units)]
    feature_cols = [c for c in feature_cols if train_part[c].std() > 1e-6]
    rul_by_unit = {u: g.sort_values("cycle")["rul"].to_numpy()
                   for u, g in train_df.groupby("unit")}

    X_tr, y_tr, _ = _windows_for(engine_arrays(train_s, feature_cols, train_units),
                              rul_by_unit, window, cap)
    X_va, y_va, end_va = _windows_for(engine_arrays(train_s, feature_cols, val_units),
                              rul_by_unit, window, cap)
    return Datasets(X_tr, y_tr, X_va, y_va, end_va,
                    engine_arrays(test_s, feature_cols), load_test_rul(subset), feature_cols)
