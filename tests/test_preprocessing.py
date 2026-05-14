import numpy as np
import pytest

from src.config import load_config
from src.data.synthetic import synthetic_long_signal
from src.preprocessing import SignalValidationError, segment_signal, validate_signal


def test_segment_signal_returns_fixed_windows():
    config = load_config("configs/default.yaml")
    signal = synthetic_long_signal(length=1440, seed=7)
    windows = segment_signal(signal, config)
    assert windows.ndim == 2
    assert windows.shape[1] == config["preprocessing"]["window_size"]


def test_short_signal_rejected():
    config = load_config("configs/default.yaml")
    with pytest.raises(SignalValidationError, match="too short"):
        validate_signal(np.ones(10), 360, config)
