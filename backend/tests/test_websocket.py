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

def test_websocket_accepts_connection():
    """Connects to /ws/ecg-stream and receives the streaming contract."""
    with client.websocket_connect("/ws/ecg-stream") as websocket:
        response = websocket.receive_json()
        assert "timestamp" in response
        assert "bpm" in response
        assert "beat_type" in response
        assert "confidence" in response
        assert "raw_window" in response

def test_websocket_rejects_invalid_mode():
    """Invalid stream modes are rejected before the socket is accepted."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/ecg-stream?mode=invalid"):
            pass

def test_websocket_processes_valid_beat():
    """The stream emits valid classification fields."""
    with client.websocket_connect("/ws/ecg-stream") as websocket:
        response = websocket.receive_json()
        assert response["beat_type"] in {"N", "V", "A", "L", "R", "UNKNOWN"}
        assert "confidence" in response
        assert "alert" in response

def test_websocket_triggers_alert_on_arrhythmia(monkeypatch):
    """Mock model to predict 'V', assert emitted message sets alert=True."""
    def classify(_self, _window):
        return {
            "beat_type": "V",
            "confidence": 0.55,
            "probabilities": {"N": 0.01, "V": 0.55, "A": 0.0, "L": 0.0, "R": 0.0},
            "latency_ms": 1.0,
        }

    monkeypatch.setattr("inference.model_manager.ModelManager.classify", classify)
    
    with client.websocket_connect("/ws/ecg-stream") as websocket:
        response = websocket.receive_json()
        assert response["beat_type"] == "V"
        assert response["alert"]

def test_websocket_graceful_disconnect():
    """Client closes connection, server does not crash."""
    # We connect, then close. The server should handle WebSocketDisconnect gracefully.
    websocket = client.websocket_connect("/ws/ecg-stream")
    websocket.close()
    # If the server crashed, the next request would fail, or we'd get a test error.
    response = client.get("/health")
    assert response.status_code == 200
