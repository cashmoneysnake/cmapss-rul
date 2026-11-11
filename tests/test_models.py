import pytest

torch = pytest.importorskip("torch")

from src.models.factory import build_model


@pytest.mark.parametrize("name", ["lstm", "gru", "cnn", "transformer"])
def test_each_model_maps_a_batch_of_windows_to_one_number_each(name):
    model = build_model(name, n_features=15)
    out = model(torch.zeros(4, 30, 15))
    assert out.shape == (4,)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("nope", 15)


def test_transformer_handles_other_window_lengths():
    model = build_model("transformer", n_features=17)
    assert model(torch.zeros(2, 70, 17)).shape == (2,)
    assert model(torch.zeros(2, 100, 17)).shape == (2,)
