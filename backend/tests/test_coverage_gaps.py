import pytest
import numpy as np
import threading
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasicCredentials

from api.main import app, verify_metrics_auth
from api.report_router import verify_report_access
from config.settings import settings
from inference.model_manager import ModelManager
from monitoring.metrics import DashboardTracker
from reports.report_generator import ReportGenerator
from utils.dsp import preprocess_signal, segment_beats
from utils.stream_state import StreamStateTracker

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

def test_model_manager_unloaded_paths(manager):
    """Unloaded model paths should return explicit runtime errors instead of attribute crashes."""
    manager.model = None
    manager.onnx_session = None
    manager.grad_cam = None
    manager.warmup()

    window = np.zeros(360, dtype=np.float32)
    with pytest.raises(RuntimeError, match="PyTorch model is not loaded"):
        manager.classify(window)
    with pytest.raises(RuntimeError, match="ONNX model is not loaded"):
        manager.onnx_predict(window)
    with pytest.raises(RuntimeError, match="GradCAM"):
        manager.explain(window, 0)

    assert manager._get_dominant_region(100) == "P-wave"
    assert manager._get_dominant_region(180) == "QRS Complex"
    assert manager._get_dominant_region(230) == "ST-segment"
    assert manager._get_dominant_region(300) == "T-wave"

def test_model_hash_mismatch_clears_loaded_state(manager, tmp_path, monkeypatch):
    """A mismatched MODEL_SHA256 should reject the artifact and leave the manager unloaded."""
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"not-a-real-model")
    monkeypatch.setattr(settings, "MODEL_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="Model hash mismatch"):
        manager.load_model(str(artifact), torch.device("cpu"))

    assert manager.is_loaded() is False
    monkeypatch.setattr(settings, "MODEL_SHA256", None)

def test_model_missing_file_clears_loaded_state(manager):
    """Missing model artifacts should not leave stale singleton state behind."""
    manager.model = True
    manager.onnx_session = None
    with pytest.raises(FileNotFoundError):
        manager.load_model("/tmp/does-not-exist.onnx", torch.device("cpu"))
    assert manager.is_loaded() is False

def test_metrics_auth_branches(monkeypatch):
    """Metrics auth is disabled by default and enforced when a token is configured."""
    monkeypatch.setattr(settings, "METRICS_AUTH_TOKEN", None)
    assert verify_metrics_auth(None) is True

    monkeypatch.setattr(settings, "METRICS_AUTH_TOKEN", "secret")
    with pytest.raises(HTTPException):
        verify_metrics_auth(None)
    with pytest.raises(HTTPException):
        verify_metrics_auth(HTTPBasicCredentials(username="admin", password="wrong"))
    assert verify_metrics_auth(HTTPBasicCredentials(username="admin", password="secret")) is True

def test_report_auth_branches(monkeypatch):
    """Report endpoints can be public for demos or protected by a bearer token."""
    monkeypatch.setattr(settings, "REPORT_AUTH_TOKEN", None)
    assert verify_report_access(None) is True

    monkeypatch.setattr(settings, "REPORT_AUTH_TOKEN", "report-secret")
    with pytest.raises(HTTPException):
        verify_report_access(None)
    assert verify_report_access(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials="report-secret")
    ) is True

def test_dashboard_tracker_and_stream_state_edges():
    """Metric and stream state helpers should summarize edge cases deterministically."""
    tracker = DashboardTracker()
    assert tracker.get_summary(active_connections=2)["active_connections"] == 2
    tracker.record_inference(latency_ms=12.5, is_alert=True)
    summary = tracker.get_summary(active_connections=1)
    assert summary["inference_count_60s"] == 1
    assert summary["alert_count_60s"] == 1

    stream = StreamStateTracker()
    stream.add_beat(rr_interval_sec=2.0, confidence=0.5, is_pvc=True)
    stream.add_beat(rr_interval_sec=2.0, confidence=0.5, is_pvc=True)
    stream.add_beat(rr_interval_sec=2.0, confidence=0.5, is_pvc=True)
    alerts = stream.check_alerts(confidence=0.5)
    assert any("VTach" in alert for alert in alerts)
    assert any("Low Confidence" in alert for alert in alerts)

def test_report_generator_event_and_image_branches():
    """Report generation should handle event overflow and invalid embedded images gracefully."""
    generator = ReportGenerator()
    assert generator._get_risk_color(0.7, is_percentage=True)
    assert generator._get_risk_color(0.85, is_percentage=True)
    assert generator._get_risk_color(0.95, is_percentage=True)
    assert generator._get_risk_color(0.1, is_percentage=False)

    events = [
        {
            "timestamp": float(index),
            "beat_type": "V",
            "confidence": 0.55,
            "alert_message": "WARN: Low Confidence Beat Classification",
        }
        for index in range(21)
    ]
    payload = {
        "patient_metadata": {"id": "QA", "age": 0, "gender": "unspecified", "session_date": "2026-05-13"},
        "signal_metadata": {"duration_sec": 60, "sampling_rate": 360, "snr_before": 0, "snr_after": 0},
        "beat_statistics": {
            "total_beats": 21,
            "class_distribution": {"N": 1, "V": 20, "A": 0, "L": 0, "R": 0},
            "dominant_rhythm": "V",
        },
        "anomaly_events": events,
        "model_metrics": {"average_confidence": 0.75, "low_confidence_beats": 21},
        "waveform_b64": "not-valid-image-data",
        "confusion_matrix_b64": "not-valid-image-data",
    }
    pdf = generator.generate_pdf(payload)
    assert len(pdf.getvalue()) > 0

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
