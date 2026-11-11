import pytest

pytest.importorskip("torch")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import DEFAULT_DIR, RAW_COLUMNS

pytestmark = pytest.mark.skipif(not (DEFAULT_DIR / "model.pt").exists(),
                                reason="serving bundle not found")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs the startup that loads the model
        yield c


def typical_cycle(client):
    """A cycle sitting at the training mean of condition 0, so the scaled input is ~0."""
    svc = app.state.service
    row = {c: 0.0 for c in RAW_COLUMNS}
    row.update(dict(zip(["setting_1", "setting_2", "setting_3"], svc.scaler.centers_[0])))
    row.update(dict(zip(svc.feature_cols, svc.scaler.mean_[0])))
    return [row[c] for c in RAW_COLUMNS]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info_lists_the_24_input_columns(client):
    info = client.get("/model-info").json()
    assert len(info["input_columns"]) == 24 and info["window"] > 0


def test_predict_returns_rul_in_range(client):
    body = client.post("/predict", json={"cycles": [typical_cycle(client)] * 80}).json()
    assert 0 <= body["rul"] <= 125 and body["padded"] is False and body["cycles_used"] == 80


def test_short_history_is_padded(client):
    body = client.post("/predict", json={"cycles": [typical_cycle(client)] * 10}).json()
    assert body["padded"] is True


def test_wrong_number_of_values_is_rejected(client):
    assert client.post("/predict", json={"cycles": [[1.0, 2.0]]}).status_code == 422


def test_empty_history_is_rejected(client):
    assert client.post("/predict", json={"cycles": []}).status_code == 422


def test_missing_body_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422
