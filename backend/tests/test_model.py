import pytest
import numpy as np
import threading
import torch
from inference.model_manager import ModelManager

def test_model_loads_without_error(dummy_model_path):
    """ModelManager.load_model() completes successfully and model is not None."""
    manager = ModelManager()
    # Reset singleton state if necessary for clean test
    manager.model = None
    manager.load_model(dummy_model_path, torch.device("cpu"))
    assert manager.is_loaded() is True
    assert manager.model is not None

def test_predict_returns_valid_structure(setup_model_manager):
    """Output dict has keys beat_type, confidence, probabilities, latency_ms."""
    manager = setup_model_manager
    window = np.zeros(360, dtype=np.float32)
    result = manager.predict(window)
    
    assert "beat_type" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "latency_ms" in result

def test_predict_output_shape(setup_model_manager):
    """Probabilities dict has exactly 5 keys (N, V, A, L, R)."""
    manager = setup_model_manager
    window = np.zeros(360, dtype=np.float32)
    result = manager.predict(window)
    
    probs = result["probabilities"]
    assert len(probs) == 5
    for key in ["N", "V", "A", "L", "R"]:
        assert key in probs

def test_predict_confidence_range(setup_model_manager):
    """Confidence is a float in [0.0, 1.0]."""
    manager = setup_model_manager
    window = np.random.randn(360).astype(np.float32)
    result = manager.predict(window)
    
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0

def test_predict_probabilities_sum_to_one(setup_model_manager):
    """Sum of all probabilities ≈ 1.0 (within 1e-5)."""
    manager = setup_model_manager
    window = np.random.randn(360).astype(np.float32)
    result = manager.predict(window)
    
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-5

def test_predict_invalid_input_shape(setup_model_manager):
    """Passing shape (1, 400) raises ValueError with clear message."""
    manager = setup_model_manager
    window = np.zeros(400, dtype=np.float32)
    
    with pytest.raises(ValueError) as excinfo:
        manager.predict(window)
    assert "shape" in str(excinfo.value).lower()

def test_predict_nan_input(setup_model_manager):
    """Passing NaN-filled tensor raises ValueError, does not crash."""
    manager = setup_model_manager
    window = np.full(360, np.nan, dtype=np.float32)
    
    with pytest.raises(ValueError) as excinfo:
        manager.predict(window)
    assert "nan" in str(excinfo.value).lower()

def test_concurrent_inference(setup_model_manager):
    """Launch 10 threads calling predict() simultaneously, assert all return valid results with no race conditions."""
    manager = setup_model_manager
    results = []
    exceptions = []
    
    def worker():
        try:
            window = np.random.randn(360).astype(np.float32)
            res = manager.predict(window)
            results.append(res)
        except Exception as e:
            exceptions.append(e)
            
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    assert len(exceptions) == 0, f"Exceptions occurred during concurrent inference: {exceptions}"
    assert len(results) == 10
    for res in results:
        assert "beat_type" in res
