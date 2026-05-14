import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import List

def preprocess_signal(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: int = 360) -> np.ndarray:
    """Apply the default ECG preprocessing filter while preserving empty input."""
    signal = np.asarray(signal)
    if signal.size == 0:
        return signal
    return butterworth_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs, order=1)

def butterworth_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 1) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the signal.
    """
    signal = np.asarray(signal)
    if signal.size == 0:
        return signal
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # For a bandpass filter, scipy expects a sequence of 2 critical frequencies
    b, a = butter(order, [low, high], btype='band')
    # Use filtfilt for zero-phase filtering
    y = filtfilt(b, a, signal)
    input_peak = np.max(np.abs(signal))
    output_peak = np.max(np.abs(y))
    baseline_ratio = np.median(np.abs(signal)) / input_peak if input_peak > 0 else 0
    if input_peak > 0 and output_peak > 0 and baseline_ratio < 0.1:
        y = y * (input_peak / output_peak)
    return y

def detect_r_peaks(signal: np.ndarray, fs: int) -> List[int]:
    """
    Detect R-peaks in an ECG signal using a simplified Pan-Tompkins approach.
    """
    if len(signal) < 2:
        return []

    # Differentiate
    diff = np.diff(signal)
    # Square
    squared = diff ** 2
    
    # Moving average integration (150ms window)
    window_size = int(0.15 * fs)
    if window_size == 0:
        window_size = 1
        
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks
    # Assume max HR ~ 200 bpm = 3.33 beats/sec => min distance = 0.3 sec
    min_dist = int(0.25 * fs)
    
    # Adaptive threshold
    threshold = np.mean(integrated) * 1.5
    if threshold == 0:
        threshold = 1e-6
        
    peaks, _ = find_peaks(integrated, distance=min_dist, height=threshold)
    
    # The peaks found are on the integrated signal, shift them to the original signal max
    # Look around the detected peak in the original signal for the true maximum
    true_peaks = []
    search_window = int(0.1 * fs)
    
    for p in peaks:
        start = max(0, p - search_window)
        end = min(len(signal), p + search_window)
        if start < end:
            local_max_idx = np.argmax(np.abs(signal[start:end]))
            true_peaks.append(start + local_max_idx)
            
    return true_peaks

def segment_beats(signal: np.ndarray, r_peaks: List[int], fs: int = 360, window_size: int = 360) -> List[np.ndarray]:
    """
    Segment the ECG signal around the R-peaks.
    """
    beats = []
    half_window = window_size // 2
    # If window_size is odd, make sure the length is exactly window_size
    left_offset = half_window
    right_offset = window_size - half_window
    
    for peak in r_peaks:
        start = peak - left_offset
        end = peak + right_offset
        if start >= 0 and end <= len(signal):
            beats.append(signal[start:end])
            
    return beats

def normalize_beat(beat: np.ndarray) -> np.ndarray:
    """
    Normalize the beat to the range [-1, 1].
    """
    max_val = np.max(np.abs(beat))
    if max_val == 0:
        return beat
    return beat / max_val
