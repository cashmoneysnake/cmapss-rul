import pytest

torch = pytest.importorskip("torch")

from src.models.factory import build_model


@pytest.mark.parametrize("name", ["lstm", "gru", "cnn"])
def test_each_model_maps_a_batch_of_windows_to_one_number_each(name):
    model = build_model(name, n_features=15)
    out = model(torch.zeros(4, 30, 15))
    assert out.shape == (4,)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("nope", 15)
