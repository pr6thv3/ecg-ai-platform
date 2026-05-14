from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from src.config import resolve_path
from src.data.synthetic import synthetic_long_signal
from src.preprocessing.validation import validate_signal
from src.utils.io import save_signal_csv


def load_signal_file(path: str | Path, config: dict[str, Any], create_demo_if_missing: bool = True) -> np.ndarray:
    target = resolve_path(path)
    if not target.exists():
        if create_demo_if_missing and config["dataset"].get("allow_synthetic_fallback", False):
            signal = synthetic_long_signal(
                length=int(config["preprocessing"]["window_size"]) * 4,
                sampling_rate=int(config["dataset"]["sampling_rate"]),
                seed=int(config["dataset"]["split"]["seed"]),
            )
            save_signal_csv(target, signal)
            return signal
        raise FileNotFoundError(f"ECG input file not found: {target}")

    try:
        if target.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if target.suffix.lower() == ".tsv" else ","
            frame = pd.read_csv(target, sep=delimiter)
            numeric = frame.select_dtypes(include=["number"])
            if numeric.empty:
                raise ValueError("No numeric columns found.")
            return numeric.iloc[:, 0].to_numpy(dtype=np.float32)
        return np.loadtxt(target, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Failed to read ECG signal from {target}: {exc}") from exc


def preprocess_signal(signal: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    preprocessing = config["preprocessing"]
    fs = int(preprocessing.get("expected_sampling_rate", config["dataset"]["sampling_rate"]))
    low = float(preprocessing["lowcut_hz"])
    high = float(preprocessing["highcut_hz"])
    order = int(preprocessing.get("filter_order", 1))
    nyquist = fs / 2.0
    if arr.size < max(12, order * 6) or low <= 0 or high >= nyquist:
        filtered = arr - np.mean(arr)
    else:
        b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
        filtered = filtfilt(b, a, arr).astype(np.float32)
    return filtered


def detect_r_peaks(signal: np.ndarray, sampling_rate: int) -> list[int]:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.size < sampling_rate:
        return []
    distance = max(1, int(0.25 * sampling_rate))
    height = max(float(np.std(arr)) * 0.6, 1e-6)
    peaks, _ = find_peaks(np.abs(arr), distance=distance, height=height)
    return [int(p) for p in peaks]


def normalize_window(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float32)
    scale = float(np.max(np.abs(arr)))
    if scale <= 1e-8:
        return arr
    return (arr / scale).astype(np.float32)


def segment_signal(signal: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sampling_rate = int(config["dataset"]["sampling_rate"])
    window_size = int(config["preprocessing"]["window_size"])
    validated = validate_signal(signal, sampling_rate, config)
    filtered = preprocess_signal(validated, config)
    peaks = detect_r_peaks(filtered, sampling_rate)
    windows: list[np.ndarray] = []
    half = window_size // 2
    for peak in peaks:
        start = peak - half
        end = start + window_size
        if start >= 0 and end <= filtered.size:
            windows.append(normalize_window(filtered[start:end]))

    if not windows and filtered.size == window_size:
        windows.append(normalize_window(filtered))
    elif not windows and filtered.size > window_size:
        start = max(0, filtered.size // 2 - half)
        windows.append(normalize_window(filtered[start : start + window_size]))

    if not windows:
        raise ValueError("No valid ECG windows could be segmented from the input signal.")
    return np.asarray(windows, dtype=np.float32)
