"""
Application configuration loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


def _int(val: str | None, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _float(val: str | None, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings derived from environment."""

    # General
    app_name: str = "CallTrust Engine"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # WebSocket
    ws_ping_interval: int = 30
    ws_max_connections: int = 100

    # Database
    postgres_dsn: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ML
    model_dir: str = "./ml/checkpoints"
    mock_inference: bool = True
    rawnet2_onnx_path: str = ""
    aasist_onnx_path: str = ""
    ssl_classifier_onnx_path: str = ""
    channel_detector_onnx_path: str = ""
    speaker_encoder_onnx_path: str = ""

    # Audio Pipeline
    sample_rate: int = 16000
    window_size_sec: float = 3.0
    window_stride_sec: float = 1.0
    vad_threshold: float = 0.3
    update_interval_ms: int = 500

    # Trust Score Thresholds
    threshold_safe: int = 75
    threshold_caution: int = 50
    threshold_high_risk: int = 25

    # Privacy
    privacy_mode: str = "strict"
    persist_raw_audio: bool = False
    audit_log_enabled: bool = True
    session_ttl_minutes: int = 60

    # Prometheus
    prometheus_enabled: bool = True

    # Security
    secret_key: str = "dev-secret-key-change-in-production"
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:5173"])

    @property
    def risk_thresholds(self) -> dict:
        return {
            "safe": self.threshold_safe,
            "caution": self.threshold_caution,
            "high_risk": self.threshold_high_risk,
        }


def load_settings() -> Settings:
    """Create Settings from environment variables."""
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db = os.getenv("POSTGRES_DB", "calltrust")
    pg_user = os.getenv("POSTGRES_USER", "calltrust")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "changeme")

    cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")

    return Settings(
        app_name=os.getenv("APP_NAME", "CallTrust Engine"),
        debug=_bool(os.getenv("DEBUG"), False),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=_int(os.getenv("BACKEND_PORT"), 8000),
        workers=_int(os.getenv("BACKEND_WORKERS"), 1),
        ws_ping_interval=_int(os.getenv("WS_PING_INTERVAL"), 30),
        ws_max_connections=_int(os.getenv("WS_MAX_CONNECTIONS"), 100),
        postgres_dsn=f"postgresql+asyncpg://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}",
        redis_url=f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}",
        model_dir=os.getenv("MODEL_DIR", "./ml/checkpoints"),
        mock_inference=_bool(os.getenv("MOCK_INFERENCE"), True),
        rawnet2_onnx_path=os.getenv("RAWNET2_ONNX_PATH", ""),
        aasist_onnx_path=os.getenv("AASIST_ONNX_PATH", ""),
        ssl_classifier_onnx_path=os.getenv("SSL_CLASSIFIER_ONNX_PATH", ""),
        channel_detector_onnx_path=os.getenv("CHANNEL_DETECTOR_ONNX_PATH", ""),
        speaker_encoder_onnx_path=os.getenv("SPEAKER_ENCODER_ONNX_PATH", ""),
        sample_rate=_int(os.getenv("SAMPLE_RATE"), 16000),
        window_size_sec=_float(os.getenv("WINDOW_SIZE_SEC"), 3.0),
        window_stride_sec=_float(os.getenv("WINDOW_STRIDE_SEC"), 1.0),
        vad_threshold=_float(os.getenv("VAD_THRESHOLD"), 0.3),
        update_interval_ms=_int(os.getenv("UPDATE_INTERVAL_MS"), 500),
        threshold_safe=_int(os.getenv("THRESHOLD_SAFE"), 75),
        threshold_caution=_int(os.getenv("THRESHOLD_CAUTION"), 50),
        threshold_high_risk=_int(os.getenv("THRESHOLD_HIGH_RISK"), 25),
        privacy_mode=os.getenv("PRIVACY_MODE", "strict"),
        persist_raw_audio=_bool(os.getenv("PERSIST_RAW_AUDIO"), False),
        audit_log_enabled=_bool(os.getenv("AUDIT_LOG_ENABLED"), True),
        session_ttl_minutes=_int(os.getenv("SESSION_TTL_MINUTES"), 60),
        prometheus_enabled=_bool(os.getenv("PROMETHEUS_ENABLED"), True),
        secret_key=os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
        cors_origins=[o.strip() for o in cors_raw.split(",") if o.strip()],
    )


# Singleton settings instance
settings = load_settings()
