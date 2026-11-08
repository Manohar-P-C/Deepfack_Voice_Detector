from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RiskLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


class Explanation(BaseModel):
    """A single human-readable reason contributing to the trust score."""
    signal: str = Field(..., description="Short signal name, e.g. 'spectral_artifact'")
    label: str = Field(..., description="Human-readable description")
    impact: float = Field(..., ge=-1.0, le=1.0, description="Impact on trust score (-1 lowers, +1 raises)")
    confidence: float = Field(..., ge=0.0, le=1.0)


class RecommendedAction(BaseModel):
    """Action suggestion shown to the user."""
    severity: RiskLevel
    message: str
    icon: str = "shield"


class BranchScore(BaseModel):
    """Score from an individual detection branch."""
    name: str
    synthetic_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    """Complete analysis payload sent every update interval."""
    # Identifiers
    session_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    window_start_sec: float
    window_end_sec: float

    # Core scores
    trust_score: float = Field(..., ge=0.0, le=100.0, description="0=definitely spoofed, 100=definitely genuine")
    risk_level: RiskLevel
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)

    # Branch scores
    synthetic_probability: float = Field(..., ge=0.0, le=1.0)
    channel_risk_probability: float = Field(..., ge=0.0, le=1.0)
    behavioral_risk_probability: float = Field(..., ge=0.0, le=1.0)
    speaker_consistency_score: float = Field(..., ge=0.0, le=1.0)

    # Detailed branch outputs
    branch_scores: List[BranchScore] = []

    # Explanations
    explanations: List[Explanation] = []
    recommended_action: RecommendedAction

    # Speech activity
    speech_active: bool = True
    speech_text: Optional[str] = None  # Optional live transcript snippet

    # Liveness
    liveness_challenge_triggered: bool = False
    liveness_phrase: Optional[str] = None


class SessionStart(BaseModel):
    """Sent when a new analysis session begins."""
    session_id: str
    mode: str = "demo"  # "demo" | "live"
    sample_rate: int = 16000
    started_at: datetime = Field(default_factory=_utc_now)


class SessionEnd(BaseModel):
    """Sent when a session ends."""
    session_id: str
    ended_at: datetime = Field(default_factory=_utc_now)
    total_duration_sec: float = 0.0
    average_trust_score: float = 0.0
    min_trust_score: float = 0.0
    risk_events_count: int = 0


class AudioChunkMeta(BaseModel):
    """Metadata for an incoming audio chunk."""
    session_id: str
    sequence: int
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm_f32le"
    duration_ms: float = 0.0
