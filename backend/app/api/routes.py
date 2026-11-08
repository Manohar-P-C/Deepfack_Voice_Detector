"""
REST API routes for health, configuration, and session management.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import settings
from .websocket import active_connections, active_sessions

router = APIRouter(prefix="/api", tags=["api"])


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    mock_inference: bool = True
    active_sessions: int = 0


class ConfigResponse(BaseModel):
    sample_rate: int
    window_size_sec: float
    window_stride_sec: float
    update_interval_ms: int
    threshold_safe: int
    threshold_caution: int
    threshold_high_risk: int
    privacy_mode: str
    mock_inference: bool


class ScenariosResponse(BaseModel):
    scenarios: list[str]


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        mock_inference=settings.mock_inference,
        active_sessions=len(active_connections),
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    return ConfigResponse(
        sample_rate=settings.sample_rate,
        window_size_sec=settings.window_size_sec,
        window_stride_sec=settings.window_stride_sec,
        update_interval_ms=settings.update_interval_ms,
        threshold_safe=settings.threshold_safe,
        threshold_caution=settings.threshold_caution,
        threshold_high_risk=settings.threshold_high_risk,
        privacy_mode=settings.privacy_mode,
        mock_inference=settings.mock_inference,
    )


@router.get("/scenarios", response_model=ScenariosResponse)
async def get_scenarios():
    """List available demo scenarios."""
    return ScenariosResponse(
        scenarios=["genuine", "clone_gradual", "replay", "noisy_genuine", "mid_call_swap"]
    )
