import pytest
import numpy as np
import threading
from fastapi.testclient import TestClient

from api.main import app
from inference.model_manager import ModelManager
from utils.dsp import preprocess_signal, segment_beats

client = TestClient(app)

# --- 1. DSP & Preprocessing ---
def test_preprocess_signal_empty():
    """preprocess_signal should handle empty arrays gracefully."""
    empty_sig = np.array([])
    filtered = preprocess_signal(empty_sig)
    assert len(filtered) == 0

def test_segment_beats_out_of_bounds():
    """segment_beats should skip R-peaks too close to the edge of the array."""
    signal = np.ones(400)
    # Peak at index 10 is too close to start (requires 180 samples before peak)
    beats = segment_beats(signal, [10])
    assert len(beats) == 0

# --- 2. Inference Engine ---
@pytest.fixture
def manager():
    mgr = ModelManager()
    mgr.warmup_latency_ms = None # Reset state for test isolation
    return mgr

def test_predict_nan_input(manager):
    """predict() should raise ValueError on NaN inputs to prevent downstream math errors."""
    bad_input = np.full((360,), np.nan)
    # Mocking loaded state bypass
    manager.model = True 
    with pytest.raises(ValueError, match="NaN"):
        manager.predict(bad_input)

def test_concurrent_inference(manager, dummy_model_path):
    """ModelManager should handle concurrent predict calls cleanly via internal threading.Lock."""
    manager.load_model(dummy_model_path, "cpu")
    results = []
    
    def worker():
        res = manager.predict(np.zeros(360))
        results.append(res)
        
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    assert len(results) == 5
    assert all("confidence" in r for r in results)

# --- 3. WebSocket Router ---
def test_ws_connect_and_receive():
    """WebSocket should connect, receive valid JSON payload, and gracefully close."""
    with client.websocket_connect("/ws/ecg-stream?mode=synthetic") as websocket:
        data = websocket.receive_json()
        assert "bpm" in data
        assert "beat_type" in data
        assert "rhythm_class" in data

def test_explain_malformed_input():
    """POST /explain should return 422 Validation Error for arrays not exactly 360 length."""
    response = client.post("/explain", json={"beat_window": [0.0]*10, "predicted_class": 1})
    assert response.status_code == 422
