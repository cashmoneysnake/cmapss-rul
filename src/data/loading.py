"""Load raw C-MAPSS files and prepare them for windowing."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS  # 26 columns


def load_split(subset: str, split: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read train_FD00X.txt or test_FD00X.txt (split is 'train' or 'test')."""
    path = raw_dir / f"{split}_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(subset: str, raw_dir: Path = RAW_DIR) -> np.ndarray:
    """True RUL at the last cycle of each test engine, ordered by unit number."""
    return np.loadtxt(raw_dir / f"RUL_{subset}.txt")


def add_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Add the true (uncapped) RUL: cycles left until each engine's last cycle.

    The cap is applied later, in windowing.make_targets.
    """
    out = train_df.copy()
    out["rul"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    return out


def constant_sensors(train_df: pd.DataFrame) -> list[str]:
    """Sensors that never change in the training data (zero variance, no information).

    Uses "only one distinct value" rather than std() == 0, because floating-point
    noise makes the std of a constant column come out as ~1e-15, not exactly 0.
    """
    return [c for c in SENSOR_COLS if train_df[c].nunique() <= 1]


def split_units(units, val_frac: float = 0.2, seed: int = 0):
    """Split whole engines (never rows or windows) into train and validation."""
    units = np.array(sorted(units))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = int(round(len(units) * val_frac))
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def engine_arrays(df: pd.DataFrame, feature_cols: list[str], units=None) -> dict:
    """{unit: array of shape (n_cycles, n_features)}, rows in cycle order."""
    if units is not None:
        df = df[df["unit"].isin(units)]
    return {u: g.sort_values("cycle")[feature_cols].to_numpy(dtype=float)
            for u, g in df.groupby("unit")}
