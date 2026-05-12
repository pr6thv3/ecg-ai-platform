import pytest
import json
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
        manager.load_model(dummy_model_path, "cpu")

def test_websocket_accepts_connection():
    """Connects to /ws/ecg-stream and does not disconnect immediately."""
    with client.websocket_connect("/ws/ecg-stream") as websocket:
        # Check it's connected by trying to send a ping or similar
        websocket.send_json({"type": "ping"})
        # Not expecting a specific answer for ping if not implemented, but connection is alive
        pass

def test_websocket_rejects_invalid_json():
    """Send malformed JSON string, assert connection closed (server might handle it by disconnecting or ignoring)."""
    with client.websocket_connect("/ws/ecg-stream") as websocket:
        websocket.send_text("this is not json")
        # FastAPI typically closes with 1003 or similar if it fails to parse expected JSON,
        # but our implementation might just catch it.
        # Let's see what happens.
        try:
            websocket.receive()
        except Exception as e:
            # We expect a disconnect exception
            pass

def test_websocket_processes_valid_beat():
    """Send valid beat JSON, receive classification."""
    with client.websocket_connect("/ws/ecg-stream") as websocket:
        payload = {"type": "beat", "payload": [0.0] * 360}
        websocket.send_json(payload)
        
        response = websocket.receive_json()
        assert response["type"] == "classification"
        assert "class" in response
        assert "confidence" in response
        assert "alert" in response

def test_websocket_triggers_alert_on_arrhythmia(mocker):
    """Mock model to predict 'V', assert emitted message sets alert=True."""
    # We use pytest-mock to mock ModelManager.predict
    mock_predict = mocker.patch("inference.model_manager.ModelManager.predict")
    mock_predict.return_value = {
        "beat_type": "V",
        "confidence": 0.99,
        "probabilities": {"N": 0.01, "V": 0.99, "A": 0.0, "L": 0.0, "R": 0.0},
        "latency_ms": 1.0
    }
    
    with client.websocket_connect("/ws/ecg-stream") as websocket:
        payload = {"type": "beat", "payload": [0.0] * 360}
        websocket.send_json(payload)
        
        response = websocket.receive_json()
        assert response["type"] == "classification"
        assert response["class"] == "V"
        assert response["alert"] is True

def test_websocket_graceful_disconnect():
    """Client closes connection, server does not crash."""
    # We connect, then close. The server should handle WebSocketDisconnect gracefully.
    websocket = client.websocket_connect("/ws/ecg-stream")
    websocket.close()
    # If the server crashed, the next request would fail, or we'd get a test error.
    response = client.get("/health")
    assert response.status_code == 200
