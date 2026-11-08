"""
TrustEngine: orchestrates the full audio → score pipeline for a single session.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import AsyncIterator, List, Optional

import numpy as np

from ..config import settings
from ..schemas.analysis import AnalysisResult, SessionEnd, SessionStart
from .audio_pipeline import AudioBuffer, FeatureExtractor, normalize_loudness, simple_energy_vad
from .audit import audit_logger
from .detector import EnsembleDetector, MockEnsembleDetector
from .metrics import (
    ACTIVE_SESSIONS,
    ERROR_COUNT,
    INFERENCE_LATENCY,
    MODEL_CONFIDENCE,
    QUEUE_DEPTH,
    REAL_TIME_FACTOR,
)

logger = logging.getLogger(__name__)


class TrustEngine:
    """
    Orchestrates the full analysis pipeline for one call session.

    Lifecycle:
        1. create() → TrustEngine instance
        2. ingest_chunk(pcm_bytes) → buffer audio
        3. process() → yields AnalysisResult for each ready window
        4. close() → secure cleanup

    Privacy: audio stays in memory only; raw samples are zeroed on close.
    """

    def __init__(
        self,
        session_id: str,
        detector: EnsembleDetector,
        mode: str = "demo",
        sample_rate: int = settings.sample_rate,
    ):
        self.session_id = session_id
        self.mode = mode
        self.sample_rate = sample_rate
        self.detector = detector

        self._buffer = AudioBuffer(
            sample_rate=sample_rate,
            window_sec=settings.window_size_sec,
            stride_sec=settings.window_stride_sec,
        )
        self._feature_extractor = FeatureExtractor(sample_rate=sample_rate)

        self._started_at = time.time()
        self._results: List[AnalysisResult] = []
        self._closed = False

    @classmethod
    def create(
        cls,
        mode: str = "demo",
        scenario: str = "genuine",
        sample_rate: int = settings.sample_rate,
    ) -> "TrustEngine":
        """Factory method for creating a new engine session."""
        session_id = hashlib.sha256(
            f"{uuid.uuid4()}-{time.time()}".encode()
        ).hexdigest()[:16]

        if settings.mock_inference:
            detector = MockEnsembleDetector(scenario=scenario)
        else:
            # TODO: plug in real ONNX ensemble here
            detector = MockEnsembleDetector(scenario=scenario)

        engine = cls(
            session_id=session_id,
            detector=detector,
            mode=mode,
            sample_rate=sample_rate,
        )
        logger.info(f"TrustEngine created: session={session_id}, mode={mode}, scenario={scenario}")
        return engine

    def get_session_start(self) -> SessionStart:
        return SessionStart(
            session_id=self.session_id,
            mode=self.mode,
            sample_rate=self.sample_rate,
        )

    def ingest_chunk(self, pcm_bytes: bytes, fmt: str = "pcm_f32le") -> None:
        """Add raw audio bytes to the buffer."""
        if self._closed:
            raise RuntimeError("Engine is closed")
        self._buffer.ingest(pcm_bytes, fmt=fmt)

    async def process(self) -> List[AnalysisResult]:
        """
        Process all available windows and return analysis results.

        Called at the update interval (e.g., every 500ms).
        Runs inference asynchronously to avoid blocking the WebSocket.
        """
        if self._closed:
            return []

        windows = self._buffer.get_windows()
        if not windows:
            return []

        results: List[AnalysisResult] = []
        tasks = []

        for audio_window, start_sec, end_sec in windows:
            # Normalize
            normalized = normalize_loudness(audio_window)

            # VAD
            speech_active = simple_energy_vad(normalized, self.sample_rate)

            # Extract features
            features = self._feature_extractor.extract_all(normalized)

            # Schedule async inference
            task = self.detector.analyze(
                features=features,
                session_id=self.session_id,
                window_start=start_sec,
                window_end=end_sec,
                speech_active=speech_active,
            )
            tasks.append(task)

        # Run all window analyses concurrently
        start_t = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed_sec = time.perf_counter() - start_t

        # Instrument Prometheus metrics
        INFERENCE_LATENCY.observe(elapsed_sec)
        window_duration = settings.window_size_sec
        rtf = elapsed_sec / max(window_duration, 1e-4)
        REAL_TIME_FACTOR.set(rtf)
        QUEUE_DEPTH.set(len(self._buffer._buffer) / max(self._buffer.window_samples, 1))

        for r in results:
            MODEL_CONFIDENCE.observe(r.calibrated_confidence)
            # Log decision to security audit trail if high-risk
            audit_logger.log_decision_if_high_risk(r)

        self._results.extend(results)
        return list(results)

    def set_scenario(self, scenario: str) -> None:
        """Switch mock scenario (demo mode only)."""
        if isinstance(self.detector, MockEnsembleDetector):
            self.detector.set_scenario(scenario)

    def get_session_end(self) -> SessionEnd:
        duration = time.time() - self._started_at
        scores = [r.trust_score for r in self._results]
        risk_events = sum(1 for r in self._results if r.risk_level.value in ("high_risk", "critical"))

        return SessionEnd(
            session_id=self.session_id,
            total_duration_sec=round(duration, 2),
            average_trust_score=round(float(np.mean(scores)) if scores else 0.0, 1),
            min_trust_score=round(float(np.min(scores)) if scores else 0.0, 1),
            risk_events_count=risk_events,
        )

    def close(self) -> SessionEnd:
        """Securely tear down the session."""
        if self._closed:
            return self.get_session_end()

        session_end = self.get_session_end()
        self._buffer.clear()
        self._closed = True
        logger.info(f"TrustEngine closed: session={self.session_id}, duration={session_end.total_duration_sec}s")
        return session_end
