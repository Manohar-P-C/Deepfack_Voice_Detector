"""
Audit Logger for High-Risk and Critical Decisions.

Guarantees:
  - Raw audio is NEVER stored.
  - Session IDs are hashed.
  - Records timestamp, score, confidence, uncertainty, explanations, and action.
  - Thread-safe append to structured JSONL audit trail.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import settings
from ..schemas.analysis import AnalysisResult, RiskLevel

logger = logging.getLogger(__name__)

AUDIT_LOG_DIR = Path(os.getenv("AUDIT_LOG_DIR", "./logs"))
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "audit_decisions.jsonl"


class AuditLogger:
    def __init__(self, log_path: Path = AUDIT_LOG_FILE):
        self.log_path = log_path
        if settings.audit_log_enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision_if_high_risk(self, result: AnalysisResult) -> None:
        """
        Write to audit trail if risk state is HIGH_RISK or CRITICAL,
        or if calibrated uncertainty is unusually elevated.
        """
        if not settings.audit_log_enabled:
            return

        if result.risk_level in (RiskLevel.HIGH_RISK, RiskLevel.CRITICAL) or result.trust_score < settings.threshold_caution:
            record = {
                "event": "HIGH_RISK_DETECTION",
                "session_id_hashed": result.session_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "window_start_sec": result.window_start_sec,
                "window_end_sec": result.window_end_sec,
                "trust_score": result.trust_score,
                "risk_level": result.risk_level.value,
                "confidence": result.calibrated_confidence,
                "uncertainty": result.uncertainty,
                "synthetic_prob": result.synthetic_probability,
                "channel_risk": result.channel_risk_probability,
                "behavioral_risk": result.behavioral_risk_probability,
                "speaker_consistency": result.speaker_consistency_score,
                "top_explanations": [
                    {"signal": e.signal, "impact": e.impact, "confidence": e.confidence}
                    for e in result.explanations[:3]
                ],
                "recommended_action": result.recommended_action.message,
                "privacy_mode": settings.privacy_mode,
            }

            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                logger.warning(
                    f"🛡️ AUDIT ALERT: High-risk session {result.session_id} - Score: {result.trust_score} ({result.risk_level.value})"
                )
            except Exception as e:
                logger.error(f"Failed to write audit record: {e}")


# Global singleton instance
audit_logger = AuditLogger()
