"""Per-operating-condition scaling (needed for FD002 and FD004, harmless for FD001/FD003)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data.loading import SETTING_COLS


class ConditionScaler:
    """Scale each sensor reading against other cycles from the SAME operating condition.

    fit() learns, from TRAINING data only:
      1. which condition cluster each settings triple belongs to (KMeans), and
      2. the mean and std of every sensor inside each cluster.
    transform() reuses those saved numbers on any data (train, validation, test).
    With n_conditions=1 this is plain global standardization.
    """

    def __init__(self, n_conditions: int = 1, seed: int = 0):
        self.n_conditions = n_conditions
        self.seed = seed

    def _assign(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_conditions == 1:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(df[SETTING_COLS].to_numpy())

    def fit(self, train_df: pd.DataFrame, feature_cols: list):
        self.feature_cols_ = list(feature_cols)
        if self.n_conditions > 1:
            self.kmeans_ = KMeans(self.n_conditions, n_init=10, random_state=self.seed)
            self.kmeans_.fit(train_df[SETTING_COLS].to_numpy())
        cond = self._assign(train_df)
        self.mean_ = np.zeros((self.n_conditions, len(feature_cols)))
        self.std_ = np.ones((self.n_conditions, len(feature_cols)))
        for c in range(self.n_conditions):
            rows = train_df.loc[cond == c, self.feature_cols_]
            self.mean_[c] = rows.mean().to_numpy()
            std = rows.std().to_numpy()
            # a sensor that never changes inside a condition has std ~1e-15 (float noise),
            # not exactly 0; dividing by that would blow noise up into +-1 values
            self.std_[c] = np.where(std > 1e-8, std, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cond = self._assign(df)
        out = df.copy()
        out[self.feature_cols_] = (df[self.feature_cols_].to_numpy() - self.mean_[cond]) / self.std_[cond]
        return out
