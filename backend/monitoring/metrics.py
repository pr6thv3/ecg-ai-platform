from prometheus_client import Counter, Histogram, Gauge
import time
import threading
from collections import deque

class DashboardTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque() # tuples of (timestamp, latency_ms, is_alert)

    def record_inference(self, latency_ms: float, is_alert: bool):
        now = time.time()
        with self.lock:
            self.history.append((now, latency_ms, is_alert))
            self._cleanup(now)

    def _cleanup(self, now: float):
        while self.history and now - self.history[0][0] > 60:
            self.history.popleft()

    def get_summary(self, active_connections: int) -> dict:
        now = time.time()
        with self.lock:
            self._cleanup(now)
            count = len(self.history)
            if count == 0:
                avg_latency = 0.0
            else:
                avg_latency = sum(lat for _, lat, _ in self.history) / count
            
            alerts = sum(1 for _, _, is_alert in self.history if is_alert)
            
        return {
            "inference_count_60s": count,
            "average_latency_ms_60s": round(avg_latency, 2),
            "alert_count_60s": alerts,
            "active_connections": active_connections
        }

dashboard_tracker = DashboardTracker()

# WebSocket Metrics
WS_CONNECTIONS = Gauge(
    "ecg_ws_connections_active",
    "Number of active WebSocket connections"
)
WS_MESSAGES_RECEIVED = Counter(
    "ecg_ws_messages_received_total",
    "Total number of messages received over WebSocket"
)
WS_ERRORS = Counter(
    "ecg_ws_errors_total",
    "Total number of WebSocket errors"
)

# Inference Metrics
INFERENCE_LATENCY = Histogram(
    "ecg_inference_latency_seconds",
    "Latency of ECG beat classification",
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500]
)
BEATS_CLASSIFIED = Counter(
    "ecg_beats_classified_total",
    "Total number of ECG beats classified",
    labelnames=["beat_type"]
)
INFERENCE_ERRORS = Counter(
    "ecg_inference_errors_total",
    "Total number of inference failures"
)

# API Metrics
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status"]
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    labelnames=["method", "endpoint"]
)
