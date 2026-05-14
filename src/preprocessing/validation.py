from __future__ import annotations

from typing import Any

import numpy as np


class SignalValidationError(ValueError):
    """Raised when ECG input cannot safely enter the inference pipeline."""


def validate_signal(signal: np.ndarray, sampling_rate: int, config: dict[str, Any]) -> np.ndarray:
    preprocessing = config["preprocessing"]
    expected_rate = int(preprocessing.get("expected_sampling_rate", config["dataset"]["sampling_rate"]))
    if sampling_rate != expected_rate:
        raise SignalValidationError(
            f"Sampling rate mismatch: received {sampling_rate} Hz, expected {expected_rate} Hz."
        )

    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise SignalValidationError("ECG signal is empty.")
    if arr.size < int(preprocessing["min_signal_length"]):
        raise SignalValidationError(
            f"ECG signal is too short: {arr.size} samples, minimum is {preprocessing['min_signal_length']}."
        )

    finite_mask = np.isfinite(arr)
    nan_fraction = 1.0 - float(finite_mask.mean())
    if nan_fraction > float(preprocessing.get("max_nan_fraction", 0.0)):
        raise SignalValidationError(f"ECG signal contains NaN/Inf values ({nan_fraction:.2%} invalid).")
    if not np.any(np.abs(arr) > 1e-8):
        raise SignalValidationError("ECG signal is flat or all zeros.")
    return arr
