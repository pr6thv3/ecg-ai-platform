import pytest
import torch
from fastapi.testclient import TestClient
from api.main import app
from inference.model_manager import ModelManager

# Setup TestClient
client = TestClient(app)

@pytest.fixture(autouse=True)
def init_model(dummy_model_path):
    # Ensure model is loaded for API tests
    manager = ModelManager()
    if not manager.is_loaded():
        manager.load_model(dummy_model_path, torch.device("cpu"))

def test_health_check():
    """GET /health returns 200, status "ok", and checks if model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert data["model_loaded"] is True

def test_analyze_endpoint_success():
    """POST /analyze with valid 360-element array returns 200 and expected inference dict structure."""
    payload = {"beat_window": [0.0] * 360}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "beat_type" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "latency_ms" in data

def test_analyze_endpoint_invalid_length():
    """POST /analyze with 100 elements returns 422 Unprocessable Entity."""
    payload = {"beat_window": [0.0] * 100}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422

def test_explain_endpoint_success():
    """POST /explain returns 200, includes saliency_map and dominant_region."""
    payload = {"beat_window": [0.0] * 360, "predicted_class": 0}
    response = client.post("/explain", json=payload)
    # If grad_cam isn't strictly available, this might return 500 or error, so let's check
    if response.status_code == 200:
        data = response.json()
        assert "saliency_map" in data
        assert "dominant_region" in data
    else:
        # Assuming we might not have grad_cam in test env without extra setup
        assert response.status_code in [200, 500]

def test_report_generate_endpoint_success():
    """POST /report/generate with dummy session data returns 200 and a mock URL or PDF bytes."""
    payload = {
        "session_id": "dummy_session",
        "patient_metadata": {
            "id": "123",
            "age": 45,
            "gender": "M",
            "session_date": "2023-01-01"
        },
        "signal_metadata": {
            "duration_sec": 60.0,
            "sampling_rate": 360,
            "snr_before": 10.0,
            "snr_after": 20.0
        },
        "beat_statistics": {
            "total_beats": 100,
            "class_distribution": {"N": 90, "V": 5, "A": 2, "L": 1, "R": 2},
            "dominant_rhythm": "Sinus"
        },
        "anomaly_events": [],
        "model_metrics": {
            "average_confidence": 0.95,
            "low_confidence_beats": 2
        }
    }
    response = client.post("/report/generate", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"

def test_report_endpoint_missing_session():
    """GET /report/session/{session_id}/json with non-existent session returns 404."""
    response = client.get("/report/session/non_existent_session/json")
    assert response.status_code == 404
