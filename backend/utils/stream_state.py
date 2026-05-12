from collections import deque
from typing import List
import numpy as np

class StreamStateTracker:
    """
    Tracks rolling state of the ECG stream to calculate BPM, rhythm types, 
    and identify temporal anomalies (e.g., VTach).
    """
    def __init__(self):
        self.rr_intervals = deque(maxlen=10) # Stores last 10 R-to-R intervals in seconds
        self.anomaly_scores = deque(maxlen=20) # Stores last 20 (1 - confidence) anomaly scores
        self.consecutive_pvcs = 0

    def add_beat(self, rr_interval_sec: float, confidence: float, is_pvc: bool):
        self.rr_intervals.append(rr_interval_sec)
        self.anomaly_scores.append(1.0 - confidence)
        
        if is_pvc:
            self.consecutive_pvcs += 1
        else:
            self.consecutive_pvcs = 0

    @property
    def current_bpm(self) -> float:
        if len(self.rr_intervals) == 0:
            return 0.0
        avg_rr = sum(self.rr_intervals) / len(self.rr_intervals)
        return 60.0 / avg_rr if avg_rr > 0 else 0.0

    @property
    def anomaly_score(self) -> float:
        if len(self.anomaly_scores) == 0:
            return 0.0
        return sum(self.anomaly_scores) / len(self.anomaly_scores)

    @property
    def rhythm_classification(self) -> str:
        bpm = self.current_bpm
        if bpm == 0:
            return "Analyzing"
        if bpm < 40:
            return "Bradycardia"
        if bpm > 150:
            return "Tachycardia"
            
        # Check for irregularity via Standard Deviation of RR intervals
        if len(self.rr_intervals) > 1:
            variance = np.var(self.rr_intervals)
            if variance > 0.05: # Threshold for irregular rhythm
                return "Irregular"
                
        return "Regular"

    def check_alerts(self, confidence: float) -> List[str]:
        """Evaluates clinical alerts based on continuous history."""
        alerts = []
        if self.consecutive_pvcs >= 3:
            alerts.append("CRITICAL: VTach Warning (3+ PVCs)")
        if confidence < 0.6:
            alerts.append("WARN: Low Confidence Beat Classification")
            
        bpm = self.current_bpm
        if bpm > 150:
            alerts.append("WARN: Extreme Tachycardia (>150 BPM)")
        elif 0 < bpm < 40:
            alerts.append("WARN: Extreme Bradycardia (<40 BPM)")
            
        return alerts
