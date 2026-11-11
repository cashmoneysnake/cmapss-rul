import numpy as np
import pytest

from src.data.windowing import last_window, make_targets, make_windows, pad_start


def toy(n_cycles, n_features=2):
    # row t holds [t, 10*t] so we can see exactly which cycles ended up where
    t = np.arange(1, n_cycles + 1)
    return np.stack([t * 10**k for k in range(n_features)], axis=1)


def test_toy_example_shape_and_contents():
    w = make_windows(toy(8), window=3)
    assert w.shape == (6, 3, 2)
    assert w[0][:, 0].tolist() == [1, 2, 3]
    assert w[-1][:, 0].tolist() == [6, 7, 8]


def test_192_cycles_gives_163_windows():
    assert make_windows(toy(192), window=30).shape == (163, 30, 2)


def test_short_engine_raises():
    with pytest.raises(ValueError):
        make_windows(toy(20), window=30)


def test_targets_are_capped_rul_at_last_cycle():
    rul = 200 - np.arange(1, 201)  # RUL at cycle t is 200 - t
    y = make_targets(rul, window=30, cap=125)
    assert len(y) == 171
    assert y[0] == 125            # window ending at cycle 30 (true RUL 170)
    assert y[50 - 30] == 125      # window ending at cycle 50 (true RUL 150)
    assert y[120 - 30] == 80      # window ending at cycle 120
    assert y[-1] == 0             # window ending at cycle 200


def test_targets_and_windows_line_up():
    assert len(make_targets(np.arange(192)[::-1], 30)) == len(make_windows(toy(192), 30))


def test_padding_short_engine():
    f = toy(20)
    p = pad_start(f, window=30)
    assert p.shape == (30, 2)
    assert (p[:10] == f[0]).all()     # 10 padding rows copy the first row
    assert (p[10:] == f).all()


def test_last_window_shapes():
    assert last_window(toy(20), 30).shape == (30, 2)
    lw = last_window(toy(100), 30)
    assert lw.shape == (30, 2) and lw[-1][0] == 100
