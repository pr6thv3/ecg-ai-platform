import asyncio
import numpy as np
import time
from typing import AsyncGenerator
import wfdb

class ECGSimulator:
    """
    Simulates real-time ECG telemetry streaming. Capable of generating mathematical 
    synthetic waves or replaying actual MIT-BIH records beat-by-beat.
    """
    def __init__(self, sample_rate: int = 360, window_size: int = 360):
        self.sample_rate = sample_rate
        self.window_size = window_size
        
    async def stream_synthetic(self, pattern: str = "normal") -> AsyncGenerator[dict, None]:
        """
        Streams a synthetic ECG signal at real-time speeds.
        Patterns: 'normal', 'pvc_burst', 'apb'
        """
        t = np.linspace(0, 1, self.window_size)
        
        while True:
            # Simulate real-time arrival of beats at ~72 BPM
            await asyncio.sleep(60.0 / 72.0)
            
            beat_type = "N"
            confidence = np.random.uniform(0.85, 0.99)
            
            # Dynamically alter the waveform mathematically based on the injected condition
            if pattern == "pvc_burst" and np.random.rand() < 0.3:
                beat_type = "V" # PVC
                confidence = np.random.uniform(0.7, 0.9)
                # PVCs are typically wider and taller than normal QRS complexes
                qrs = np.exp(-((t - 0.5) ** 2) / 0.005) * 1.5 
            elif pattern == "apb" and np.random.rand() < 0.15:
                beat_type = "A" # APB
                qrs = np.exp(-((t - 0.5) ** 2) / 0.001)
            else:
                qrs = np.exp(-((t - 0.5) ** 2) / 0.001)
                
            # Add realistic baseline wander and noise
            noise = np.random.normal(0, 0.02, self.window_size)
            signal = qrs + noise
            
            yield {
                "timestamp": time.time(),
                "raw_window": signal.tolist(),
                "true_type": beat_type,
                "confidence": confidence
            }

    async def stream_mitbih(self, record_path: str) -> AsyncGenerator[dict, None]:
        """
        Replays a real MIT-BIH record at its true temporal frequency.
        """
        try:
            sig, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            signal = sig[:, 0]
            
            last_sample = 0
            for symbol, sample in zip(annotation.symbol, annotation.sample):
                # Throttle the generator to strictly match biological real-time
                samples_diff = sample - last_sample
                sleep_time = samples_diff / self.sample_rate
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                left = sample - self.window_size // 2
                right = sample + self.window_size // 2
                
                if left >= 0 and right < len(signal):
                    window = signal[left:right]
                    yield {
                        "timestamp": time.time(),
                        "raw_window": window.tolist(),
                        "true_type": symbol,
                        "confidence": 1.0
                    }
                last_sample = sample
        except Exception as e:
            print(f"Failed to load MIT-BIH record from {record_path}: {e}")
