"""
Prometheus Metrics instrumentation for CallTrust Engine.

Tracks:
  - calltrust_inference_latency_seconds (Histogram)
  - calltrust_real_time_factor (Gauge)
  - calltrust_queue_depth (Gauge)
  - calltrust_model_confidence (Histogram)
  - calltrust_error_count (Counter)
  - calltrust_active_sessions (Gauge)
"""

from __future__ import annotations

import logging
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

INFERENCE_LATENCY = Histogram(
    "calltrust_inference_latency_seconds",
    "Time taken for audio feature extraction and ensemble inference",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

REAL_TIME_FACTOR = Gauge(
    "calltrust_real_time_factor",
    "Real-time factor (inference time / window duration)",
)

QUEUE_DEPTH = Gauge(
    "calltrust_audio_queue_depth",
    "Number of pending audio windows in buffer waiting for inference",
)

MODEL_CONFIDENCE = Histogram(
    "calltrust_model_confidence",
    "Distribution of calibrated model confidence values",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
)

ERROR_COUNT = Counter(
    "calltrust_errors_total",
    "Total count of exceptions and processing failures",
    ["type"],
)

ACTIVE_SESSIONS = Gauge(
    "calltrust_active_sessions",
    "Number of active WebSocket audio streaming sessions",
)
