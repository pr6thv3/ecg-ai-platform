import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# In a real TDD workflow, these functions might not exist yet if they haven't been implemented.
# We assume their signatures for these tests.
try:
    from preprocessing.dsp import butterworth_filter, detect_r_peaks, segment_beats, normalize_beat
except ImportError:
    # Dummy implementations to allow tests to run and fail properly if not implemented
    def butterworth_filter(signal, lowcut, highcut, fs, order): return signal
    def detect_r_peaks(signal, fs): return []
    def segment_beats(signal, r_peaks, fs, window_size): return []
    def normalize_beat(beat): return beat

def test_butterworth_filter_removes_baseline_wander(synthetic_signal):
    """Assert filtered signal has lower low-freq power when DC offset and noise are injected."""
    signal, _ = synthetic_signal
    # Inject baseline wander (low freq sine wave)
    t = np.linspace(0, 10, len(signal))
    wander = 2.0 * np.sin(2 * np.pi * 0.1 * t)
    noisy_signal = np.array(signal) + wander
    
    filtered = butterworth_filter(noisy_signal, lowcut=0.5, highcut=40.0, fs=360, order=1)
    
    # Calculate power < 0.5Hz using FFT
    fft_noisy = np.abs(np.fft.rfft(noisy_signal))
    fft_filtered = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(signal), 1/360)
    
    low_freq_idx = np.where(freqs < 0.5)[0]
    power_noisy = np.sum(fft_noisy[low_freq_idx])
    power_filtered = np.sum(fft_filtered[low_freq_idx])
    
    assert power_filtered < power_noisy * 0.1 # Should reduce by at least 90%

def test_butterworth_filter_preserves_qrs(synthetic_signal):
    """Verify peak amplitude is preserved within 10% after filtering."""
    signal, peaks = synthetic_signal
    filtered = butterworth_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=1)
    
    for p in peaks:
        original_peak_val = np.max(np.abs(signal[p-5:p+5]))
        filtered_peak_val = np.max(np.abs(filtered[p-5:p+5]))
        assert abs(original_peak_val - filtered_peak_val) < 0.1 * original_peak_val

def test_r_peak_detection_on_clean_signal(synthetic_signal):
    """Use a synthetic signal with known peaks at exact indices, assert detected peaks match within ±5 samples."""
    signal, true_peaks = synthetic_signal
    detected_peaks = detect_r_peaks(signal, fs=360)
    
    assert len(detected_peaks) == len(true_peaks)
    for detected, true in zip(detected_peaks, true_peaks):
        assert abs(detected - true) <= 5

def test_r_peak_detection_on_noisy_signal(synthetic_signal):
    """Add Gaussian noise (SNR=10dB), assert at least 90% of true peaks detected."""
    signal, true_peaks = synthetic_signal
    # SNR = 10 * log10(P_signal / P_noise) -> P_noise = P_signal / 10^(SNR/10)
    signal = np.array(signal)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (10 / 10))
    noisy_signal = signal + np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    detected_peaks = detect_r_peaks(noisy_signal, fs=360)
    
    # Check that at least 90% of true peaks have a matching detected peak within 10 samples
    matches = 0
    for true in true_peaks:
        if any(abs(detected - true) <= 10 for detected in detected_peaks):
            matches += 1
            
    assert matches >= 0.9 * len(true_peaks)

def test_beat_segmentation_window_size(synthetic_signal):
    """Assert every segmented beat has exactly 360 samples."""
    signal, peaks = synthetic_signal
    beats = segment_beats(signal, peaks, fs=360, window_size=360)
    
    assert len(beats) > 0
    for beat in beats:
        assert len(beat) == 360

def test_beat_segmentation_boundary_handling(synthetic_signal):
    """Test behavior when an R-peak is within 180 samples of signal start/end."""
    signal = np.zeros(1000)
    peaks = [50, 500, 950] # 50 is near start, 950 is near end (180 window radius)
    beats = segment_beats(signal, peaks, fs=360, window_size=360)
    
    # Should only return the middle beat (500), skipping out-of-bounds beats
    assert len(beats) == 1

def test_normalization_range(synthetic_signal):
    """Assert all segmented beats are normalized to [-1, 1]."""
    beat = np.array([0, 5, 10, -5, 2])
    normalized = normalize_beat(beat)
    
    assert np.max(normalized) <= 1.0 + 1e-5
    assert np.min(normalized) >= -1.0 - 1e-5
    assert np.isclose(np.max(np.abs(normalized)), 1.0)
