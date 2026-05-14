import numpy as np
import pytest

from src.config import load_config
from src.inference import InferencePipeline


def test_nan_signal_rejected():
    config = load_config("configs/default.yaml")
    signal = np.ones(360, dtype=np.float32)
    signal[12] = np.nan
    with pytest.raises(ValueError, match="NaN|Inf"):
        InferencePipeline(config).predict_signal(signal, sampling_rate=360)


def test_sampling_rate_mismatch_rejected():
    config = load_config("configs/default.yaml")
    with pytest.raises(ValueError, match="Sampling rate mismatch"):
        InferencePipeline(config).predict_signal(np.ones(360, dtype=np.float32), sampling_rate=250)
