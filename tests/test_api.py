from fastapi.testclient import TestClient

from src.api.app import app
from src.data.synthetic import synthetic_long_signal


def test_api_health_route():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "runtime" in payload


def test_api_model_info_route():
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "class_names" in response.json()


def test_api_predict_route():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"signal": synthetic_long_signal(seed=11).astype(float).tolist(), "sampling_rate": 360},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "prediction" in payload
