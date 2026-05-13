import pytest
import os
import torch
import numpy as np
from fastapi.testclient import TestClient
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.train import ECGNet
from inference.model_manager import ModelManager
from api.main import app

@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory):
    model = ECGNet(num_classes=5)
    fn = tmp_path_factory.mktemp("models") / "dummy_model.pth"
    torch.save(model.state_dict(), str(fn))
    return str(fn)

@pytest.fixture(autouse=True)
def setup_model_manager(dummy_model_path):
    manager = ModelManager()
    manager.load_model(dummy_model_path, torch.device("cpu"))
    yield manager

@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def synthetic_signal():
    """Generates a 10-second MIT-BIH-like synthetic signal at 360 Hz with known peaks."""
    # 360 Hz * 10 seconds = 3600 samples
    signal = np.random.normal(0, 0.1, 3600)
    
    # Inject exactly 10 R-peaks at known intervals (every 360 samples starting at 180)
    peaks = list(range(180, 3600, 360))
    for p in peaks:
        # Create a sharp spike (QRS complex)
        signal[p-5:p+5] = np.array([-1, -2, 2, 5, 8, -3, -1, 0, 0, 0])
        
    return signal.tolist(), peaks
