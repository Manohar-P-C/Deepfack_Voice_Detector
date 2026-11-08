"""
Detector interface and mock implementation.

Defines the abstract DetectorBranch and EnsembleDetector contracts so
real ONNX models can be swapped in without changing any other code.
The MockDetector generates realistic, temporally coherent synthetic
probabilities for full UI development before model training is complete.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schemas.analysis import (
    AnalysisResult,
    BranchScore,
    Explanation,
    RecommendedAction,
    RiskLevel,
)

logger = logging.getLogger(__name__)


# ──────────────────── Abstract Detector Interface ─────────────────────
@dataclass
class DetectionOutput:
    """Output from a single detection branch."""
    synthetic_probability: float  # 0–1, higher = more likely spoofed
    confidence: float  # How certain the model is
    uncertainty: float  # Complementary uncertainty measure
    features: Dict[str, float] = field(default_factory=dict)  # Interpretable sub-signals


class DetectorBranch(abc.ABC):
    """
    Abstract base for a detection branch.

    Each branch receives extracted features and returns a DetectionOutput.
    Implementations may wrap ONNX Runtime, PyTorch, or mock logic.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    async def predict(self, features: dict) -> DetectionOutput:
        """Run inference on extracted features."""
        ...

    def load(self, model_path: str) -> None:
        """Load model weights. Override in real implementations."""
        pass


class EnsembleDetector(abc.ABC):
    """
    Abstract ensemble that fuses branch outputs into a unified trust score.
    """

    @abc.abstractmethod
    async def analyze(
        self,
        features: dict,
        session_id: str,
        window_start: float,
        window_end: float,
        speech_active: bool,
    ) -> AnalysisResult:
        """Run full ensemble analysis and return a complete result."""
        ...


# ──────────────────── Mock Detector Implementation ─────────────────────
class _ScenarioEngine:
    """
    Generates temporally coherent mock signals that simulate realistic
    call scenarios: genuine speech, gradual voice cloning onset, replay
    attacks, noisy genuine audio, and mid-call voice swap.
    """

    SCENARIOS = {
        "genuine": {
            "base_synth": 0.08,
            "variance": 0.04,
            "channel_risk": 0.05,
            "behavioral_risk": 0.03,
        },
        "clone_gradual": {
            "base_synth": 0.15,
            "ramp_to": 0.85,
            "ramp_start_sec": 5.0,
            "ramp_end_sec": 20.0,
            "channel_risk": 0.15,
            "behavioral_risk": 0.25,
        },
        "replay": {
            "base_synth": 0.70,
            "variance": 0.08,
            "channel_risk": 0.65,
            "behavioral_risk": 0.30,
        },
        "noisy_genuine": {
            "base_synth": 0.20,
            "variance": 0.10,
            "channel_risk": 0.35,
            "behavioral_risk": 0.08,
        },
        "mid_call_swap": {
            "base_synth": 0.08,
            "swap_at_sec": 12.0,
            "swap_synth": 0.78,
            "channel_risk": 0.10,
            "behavioral_risk": 0.40,
        },
    }

    def __init__(self, scenario: str = "genuine"):
        self.scenario = scenario
        self.config = self.SCENARIOS.get(scenario, self.SCENARIOS["genuine"])

    def get_signals(self, t: float) -> Dict[str, float]:
        """Return mock signal values for time t (seconds into the call)."""
        cfg = self.config
        rng = np.random.RandomState(int(t * 100) % (2**31))

        if self.scenario == "clone_gradual":
            ramp_start = cfg.get("ramp_start_sec", 5.0)
            ramp_end = cfg.get("ramp_end_sec", 20.0)
            if t < ramp_start:
                synth = cfg["base_synth"]
            elif t > ramp_end:
                synth = cfg["ramp_to"]
            else:
                progress = (t - ramp_start) / (ramp_end - ramp_start)
                synth = cfg["base_synth"] + progress * (cfg["ramp_to"] - cfg["base_synth"])
            synth += rng.normal(0, 0.03)
        elif self.scenario == "mid_call_swap":
            swap_at = cfg.get("swap_at_sec", 12.0)
            if t < swap_at:
                synth = cfg["base_synth"] + rng.normal(0, 0.03)
            else:
                synth = cfg["swap_synth"] + rng.normal(0, 0.05)
        else:
            synth = cfg.get("base_synth", 0.08) + rng.normal(0, cfg.get("variance", 0.04))

        synth = float(np.clip(synth, 0.0, 1.0))
        channel = float(np.clip(cfg.get("channel_risk", 0.05) + rng.normal(0, 0.03), 0.0, 1.0))
        behavioral = float(np.clip(cfg.get("behavioral_risk", 0.03) + rng.normal(0, 0.02), 0.0, 1.0))

        # Speaker consistency (lower when spoofed)
        if synth > 0.5:
            speaker = float(np.clip(0.4 + rng.normal(0, 0.1), 0.0, 1.0))
        else:
            speaker = float(np.clip(0.9 + rng.normal(0, 0.05), 0.0, 1.0))

        return {
            "synthetic": synth,
            "channel": channel,
            "behavioral": behavioral,
            "speaker": speaker,
        }


class MockRawNetBranch(DetectorBranch):
    """Mock RawNet2-style waveform branch."""

    @property
    def name(self) -> str:
        return "rawnet2_waveform"

    async def predict(self, features: dict) -> DetectionOutput:
        audio = features.get("raw_waveform", np.zeros((1, 16000)))
        # Generate deterministic mock output from audio statistics
        seed = int(np.abs(np.sum(audio[:, :100])) * 10000) % (2**31)
        rng = np.random.RandomState(seed)
        synth = float(np.clip(rng.beta(2, 5), 0, 1))
        return DetectionOutput(
            synthetic_probability=synth,
            confidence=float(np.clip(0.7 + rng.normal(0, 0.1), 0.3, 0.99)),
            uncertainty=float(np.clip(0.15 + rng.normal(0, 0.05), 0.01, 0.5)),
            features={"spectral_flatness": float(rng.uniform(0.1, 0.8))},
        )


class MockSpectrogramBranch(DetectorBranch):
    """Mock AASIST/LCNN spectrogram branch."""

    @property
    def name(self) -> str:
        return "aasist_spectrogram"

    async def predict(self, features: dict) -> DetectionOutput:
        mel = features.get("log_mel_spectrogram", np.zeros((80, 100)))
        seed = int(np.abs(np.sum(mel[:10, :10])) * 10000) % (2**31)
        rng = np.random.RandomState(seed)
        synth = float(np.clip(rng.beta(2, 5), 0, 1))
        return DetectionOutput(
            synthetic_probability=synth,
            confidence=float(np.clip(0.75 + rng.normal(0, 0.08), 0.3, 0.99)),
            uncertainty=float(np.clip(0.12 + rng.normal(0, 0.04), 0.01, 0.5)),
            features={
                "formant_stability": float(rng.uniform(0.3, 0.95)),
                "high_freq_energy": float(rng.uniform(0.1, 0.7)),
            },
        )


class MockSSLBranch(DetectorBranch):
    """Mock self-supervised embedding branch."""

    @property
    def name(self) -> str:
        return "ssl_embedding"

    async def predict(self, features: dict) -> DetectionOutput:
        emb = features.get("ssl_embedding", np.zeros(768))
        seed = int(np.abs(np.sum(emb[:50])) * 10000) % (2**31)
        rng = np.random.RandomState(seed)
        synth = float(np.clip(rng.beta(2, 6), 0, 1))
        return DetectionOutput(
            synthetic_probability=synth,
            confidence=float(np.clip(0.8 + rng.normal(0, 0.07), 0.3, 0.99)),
            uncertainty=float(np.clip(0.10 + rng.normal(0, 0.04), 0.01, 0.5)),
            features={
                "embedding_anomaly": float(rng.uniform(0.05, 0.5)),
            },
        )


class MockEnsembleDetector(EnsembleDetector):
    """
    Mock ensemble that combines three branch outputs with scenario-aware
    signal generation for realistic demo behavior.

    The fusion uses weighted averaging with calibrated confidence and
    uncertainty estimation via mock temperature scaling.
    """

    BRANCH_WEIGHTS = {
        "rawnet2_waveform": 0.30,
        "aasist_spectrogram": 0.35,
        "ssl_embedding": 0.35,
    }

    CHANNEL_WEIGHT = 0.15
    BEHAVIORAL_WEIGHT = 0.10
    SPEAKER_WEIGHT = 0.10

    def __init__(self, scenario: str = "genuine"):
        self.branches: List[DetectorBranch] = [
            MockRawNetBranch(),
            MockSpectrogramBranch(),
            MockSSLBranch(),
        ]
        self.scenario_engine = _ScenarioEngine(scenario)
        self._call_count = 0

    def set_scenario(self, scenario: str) -> None:
        """Switch the active mock scenario."""
        self.scenario_engine = _ScenarioEngine(scenario)
        self._call_count = 0

    async def analyze(
        self,
        features: dict,
        session_id: str,
        window_start: float,
        window_end: float,
        speech_active: bool,
    ) -> AnalysisResult:
        self._call_count += 1
        t = (window_start + window_end) / 2.0

        # Get scenario-driven signals
        signals = self.scenario_engine.get_signals(t)

        # Run branches concurrently (mock but async-compatible)
        branch_tasks = [branch.predict(features) for branch in self.branches]
        branch_outputs = await asyncio.gather(*branch_tasks)

        # Blend branch outputs with scenario signals for realism
        branch_scores: List[BranchScore] = []
        weighted_synth = 0.0
        total_weight = 0.0

        for branch, output in zip(self.branches, branch_outputs):
            # Blend branch prediction with scenario signal
            blended_synth = 0.4 * output.synthetic_probability + 0.6 * signals["synthetic"]
            weight = self.BRANCH_WEIGHTS.get(branch.name, 0.33)
            weighted_synth += blended_synth * weight
            total_weight += weight

            branch_scores.append(BranchScore(
                name=branch.name,
                synthetic_probability=round(blended_synth, 4),
                confidence=round(output.confidence, 4),
                uncertainty=round(output.uncertainty, 4),
            ))

        synthetic_prob = weighted_synth / max(total_weight, 1e-8)
        channel_risk = signals["channel"]
        behavioral_risk = signals["behavioral"]
        speaker_consistency = signals["speaker"]

        # Fused score: combine all signals
        combined_risk = (
            synthetic_prob * (1.0 - self.CHANNEL_WEIGHT - self.BEHAVIORAL_WEIGHT - self.SPEAKER_WEIGHT)
            + channel_risk * self.CHANNEL_WEIGHT
            + behavioral_risk * self.BEHAVIORAL_WEIGHT
            + (1.0 - speaker_consistency) * self.SPEAKER_WEIGHT
        )

        # Trust score: 100 = fully trusted, 0 = definitely spoofed
        trust_score = round((1.0 - combined_risk) * 100.0, 1)
        trust_score = float(np.clip(trust_score, 0.0, 100.0))

        # Calibrated confidence (mock temperature scaling)
        avg_confidence = np.mean([bs.confidence for bs in branch_scores])
        calibrated_confidence = round(float(np.clip(avg_confidence * 0.95, 0.3, 0.99)), 4)

        # Uncertainty estimation (mock conformal-style)
        avg_uncertainty = np.mean([bs.uncertainty for bs in branch_scores])
        uncertainty = round(float(np.clip(avg_uncertainty * 1.1, 0.01, 0.5)), 4)

        # Risk level
        risk_level = self._compute_risk_level(trust_score)

        # Generate explanations
        explanations = self._generate_explanations(
            synthetic_prob, channel_risk, behavioral_risk, speaker_consistency, trust_score
        )

        # Recommended action
        action = self._recommended_action(risk_level, trust_score, explanations)

        # Liveness challenge
        liveness_triggered = risk_level == RiskLevel.CRITICAL and self._call_count % 5 == 0
        liveness_phrase = self._generate_liveness_phrase() if liveness_triggered else None

        return AnalysisResult(
            session_id=session_id,
            window_start_sec=round(window_start, 3),
            window_end_sec=round(window_end, 3),
            trust_score=trust_score,
            risk_level=risk_level,
            calibrated_confidence=calibrated_confidence,
            uncertainty=uncertainty,
            synthetic_probability=round(synthetic_prob, 4),
            channel_risk_probability=round(channel_risk, 4),
            behavioral_risk_probability=round(behavioral_risk, 4),
            speaker_consistency_score=round(speaker_consistency, 4),
            branch_scores=branch_scores,
            explanations=explanations,
            recommended_action=action,
            speech_active=speech_active,
            liveness_challenge_triggered=liveness_triggered,
            liveness_phrase=liveness_phrase,
        )

    def _compute_risk_level(self, trust_score: float) -> RiskLevel:
        from ..config import settings
        if trust_score >= settings.threshold_safe:
            return RiskLevel.SAFE
        elif trust_score >= settings.threshold_caution:
            return RiskLevel.CAUTION
        elif trust_score >= settings.threshold_high_risk:
            return RiskLevel.HIGH_RISK
        else:
            return RiskLevel.CRITICAL

    def _generate_explanations(
        self,
        synth: float,
        channel: float,
        behavioral: float,
        speaker: float,
        trust: float,
    ) -> List[Explanation]:
        explanations = []

        if synth > 0.4:
            explanations.append(Explanation(
                signal="synthetic_voice",
                label="Voice patterns suggest possible AI generation",
                impact=round(-synth * 0.8, 3),
                confidence=round(min(synth + 0.2, 0.99), 3),
            ))

        if channel > 0.3:
            explanations.append(Explanation(
                signal="channel_artifacts",
                label="Unusual audio channel characteristics detected",
                impact=round(-channel * 0.5, 3),
                confidence=round(min(channel + 0.1, 0.95), 3),
            ))

        if behavioral > 0.25:
            explanations.append(Explanation(
                signal="behavioral_anomaly",
                label="Speech patterns show unusual timing or repetition",
                impact=round(-behavioral * 0.4, 3),
                confidence=round(min(behavioral + 0.15, 0.90), 3),
            ))

        if speaker < 0.6:
            explanations.append(Explanation(
                signal="speaker_inconsistency",
                label="Speaker voice characteristics changed during the call",
                impact=round(-(1 - speaker) * 0.6, 3),
                confidence=round(min((1 - speaker) + 0.2, 0.95), 3),
            ))

        if trust > 80 and not explanations:
            explanations.append(Explanation(
                signal="natural_speech",
                label="Voice characteristics consistent with natural speech",
                impact=0.3,
                confidence=round(trust / 120.0, 3),
            ))

        if not explanations:
            explanations.append(Explanation(
                signal="monitoring",
                label="Analyzing call audio — insufficient data for high confidence",
                impact=0.0,
                confidence=0.3,
            ))

        return sorted(explanations, key=lambda e: abs(e.impact), reverse=True)

    def _recommended_action(
        self,
        risk_level: RiskLevel,
        trust_score: float,
        explanations: List[Explanation],
    ) -> RecommendedAction:
        if risk_level == RiskLevel.SAFE:
            return RecommendedAction(
                severity=RiskLevel.SAFE,
                message="Call appears genuine. Continue normally.",
                icon="shield-check",
            )
        elif risk_level == RiskLevel.CAUTION:
            return RecommendedAction(
                severity=RiskLevel.CAUTION,
                message="Some anomalies detected. Avoid sharing sensitive information until verified.",
                icon="alert-triangle",
            )
        elif risk_level == RiskLevel.HIGH_RISK:
            return RecommendedAction(
                severity=RiskLevel.HIGH_RISK,
                message="High risk of voice manipulation. Do not share OTPs, passwords, or financial details.",
                icon="shield-alert",
            )
        else:
            return RecommendedAction(
                severity=RiskLevel.CRITICAL,
                message="Critical: Likely deepfake detected. End the call and call back using a saved number.",
                icon="phone-off",
            )

    def _generate_liveness_phrase(self) -> str:
        phrases = [
            "The weather today is sunny",
            "My favorite color is blue",
            "I had coffee this morning",
            "The capital of France is Paris",
            "Seven plus three equals ten",
        ]
        idx = int(time.time()) % len(phrases)
        return phrases[idx]
