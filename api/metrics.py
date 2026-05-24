"""
In-memory counters for /health and basic monitoring.
For real production, replace with Prometheus client.
"""
import time
from threading import Lock
from typing import Dict


class Metrics:
    """Thread-safe in-memory counters."""
    def __init__(self):
        self._lock = Lock()
        self.startup_time = time.time()
        self.requests_total = 0
        self.predictions_success = 0
        self.predictions_failed = 0
        self.validation_errors = 0
        self.acoustic_rejections = 0 
        self.inference_errors = 0
        self.rate_limit_hits = 0
        self.total_inference_time_ms = 0.0
        self.class_predictions: Dict[str, int] = {}

    def inc(self, attr: str, by: int = 1):
        with self._lock:
            setattr(self, attr, getattr(self, attr) + by)

    def add_inference_time(self, ms: float):
        with self._lock:
            self.total_inference_time_ms += ms

    def record_prediction(self, class_label: str):
        with self._lock:
            self.class_predictions[class_label] = self.class_predictions.get(class_label, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            uptime = time.time() - self.startup_time
            n_success = self.predictions_success
            avg_inference = (self.total_inference_time_ms / n_success) if n_success else 0.0
            return {
                "uptime_seconds": int(uptime),
                "requests_total": self.requests_total,
                "predictions_success": n_success,
                "predictions_failed": self.predictions_failed,
                "validation_errors": self.validation_errors,
                "acoustic_rejections": self.acoustic_rejections,
                "inference_errors": self.inference_errors,
                "rate_limit_hits": self.rate_limit_hits,
                "avg_inference_ms": round(avg_inference, 1),
                "class_predictions": dict(self.class_predictions),
            }


metrics = Metrics()