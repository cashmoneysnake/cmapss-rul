import numpy as np
import pandas as pd

from src.data.loading import SETTING_COLS
from src.data.scaling import ConditionScaler


def two_condition_df(seed=0, n=200):
    rng = np.random.default_rng(seed)
    # condition A: settings ~ (0,0,100), sensor ~ 640 ; condition B: settings ~ (40,0.8,60), sensor ~ 550
    cond = rng.integers(0, 2, n)
    settings = np.where(cond[:, None] == 0, [0.0, 0.0, 100.0], [40.0, 0.8, 60.0])
    settings = settings + rng.normal(0, 0.01, settings.shape)
    sensor = np.where(cond == 0, 640.0, 550.0) + rng.normal(0, 1.0, n)
    df = pd.DataFrame(settings, columns=SETTING_COLS)
    df["s_2"] = sensor
    return df, cond


def test_each_condition_is_centered_and_unit_scale_after_fit():
    df, cond = two_condition_df()
    out = ConditionScaler(n_conditions=2).fit(df, ["s_2"]).transform(df)
    for c in (0, 1):
        v = out.loc[cond == c, "s_2"]
        assert abs(v.mean()) < 1e-6 and abs(v.std() - 1) < 1e-6


def test_transform_reuses_training_statistics_not_new_data():
    train, _ = two_condition_df(seed=0)
    new, _ = two_condition_df(seed=1)
    new["s_2"] += 5.0   # shift the new data
    sc = ConditionScaler(2).fit(train, ["s_2"])
    shifted_mean = sc.transform(new)["s_2"].mean()
    unshifted_mean = sc.transform(new.assign(s_2=new["s_2"] - 5.0))["s_2"].mean()
    assert shifted_mean > unshifted_mean + 1   # a re-fit would have hidden the shift


def test_one_condition_is_plain_standardization():
    df, _ = two_condition_df()
    out = ConditionScaler(1).fit(df, ["s_2"]).transform(df)
    assert abs(out["s_2"].mean()) < 1e-6 and abs(out["s_2"].std() - 1) < 1e-6


def test_sensor_flat_inside_each_condition_scales_to_zero_not_noise():
    df, cond = two_condition_df()
    df["s_5"] = np.where(cond == 0, 14.62, 21.61)   # differs by condition, flat inside each
    out = ConditionScaler(2).fit(df, ["s_2", "s_5"]).transform(df)
    assert out["s_5"].abs().max() < 1e-6
