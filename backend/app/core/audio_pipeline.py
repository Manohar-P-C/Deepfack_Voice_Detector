"""
Audio ingestion pipeline: VAD → normalization → sliding windows → feature extraction.

Processes raw PCM audio chunks arriving from the WebSocket, accumulates them
into a rolling buffer, applies voice activity detection, creates overlapping
analysis windows, and extracts multi-modal features for the ensemble detector.
"""

from __future__ import annotations

import logging
import struct
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────── Constants ────────────────────────────
SAMPLE_RATE = settings.sample_rate
WINDOW_SAMPLES = int(settings.window_size_sec * SAMPLE_RATE)
STRIDE_SAMPLES = int(settings.window_stride_sec * SAMPLE_RATE)
VAD_FRAME_MS = 30  # 30ms frames for VAD
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)


class AudioBuffer:
    """
    Rolling audio buffer that accumulates PCM chunks and yields analysis windows.

    Thread-safe by design: each WebSocket session gets its own buffer instance.
    Audio is never persisted—kept only in memory per privacy requirements.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_sec: float = settings.window_size_sec,
        stride_sec: float = settings.window_stride_sec,
        max_buffer_sec: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self.stride_samples = int(stride_sec * sample_rate)
        self.max_samples = int(max_buffer_sec * sample_rate)

        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._total_samples_ingested: int = 0
        self._next_window_start: int = 0

    @property
    def duration_sec(self) -> float:
        return len(self._buffer) / self.sample_rate

    @property
    def total_ingested_sec(self) -> float:
        return self._total_samples_ingested / self.sample_rate

    def ingest(self, pcm_bytes: bytes, fmt: str = "pcm_f32le") -> None:
        """
        Append raw PCM bytes to the buffer.

        Supported formats:
            - pcm_f32le: 32-bit float little-endian (Web Audio API default)
            - pcm_s16le: 16-bit signed integer little-endian
        """
        if fmt == "pcm_f32le":
            samples = np.frombuffer(pcm_bytes, dtype=np.float32)
        elif fmt == "pcm_s16le":
            raw = np.frombuffer(pcm_bytes, dtype=np.int16)
            samples = raw.astype(np.float32) / 32768.0
        else:
            raise ValueError(f"Unsupported audio format: {fmt}")

        self._buffer = np.concatenate([self._buffer, samples])
        self._total_samples_ingested += len(samples)

        # Trim oldest samples if buffer exceeds max
        if len(self._buffer) > self.max_samples:
            trim = len(self._buffer) - self.max_samples
            self._buffer = self._buffer[trim:]
            self._next_window_start = max(0, self._next_window_start - trim)

    def get_windows(self) -> List[Tuple[np.ndarray, float, float]]:
        """
        Extract all available overlapping windows since last call.

        Returns:
            List of (audio_window, start_sec, end_sec) tuples.
        """
        windows: List[Tuple[np.ndarray, float, float]] = []
        buf_len = len(self._buffer)

        while self._next_window_start + self.window_samples <= buf_len:
            start = self._next_window_start
            end = start + self.window_samples
            window = self._buffer[start:end].copy()

            global_start = (self._total_samples_ingested - buf_len + start) / self.sample_rate
            global_end = (self._total_samples_ingested - buf_len + end) / self.sample_rate

            windows.append((window, global_start, global_end))
            self._next_window_start += self.stride_samples

        return windows

    def clear(self) -> None:
        """Securely clear buffer contents."""
        if len(self._buffer) > 0:
            self._buffer[:] = 0  # Overwrite before discarding
        self._buffer = np.array([], dtype=np.float32)
        self._next_window_start = 0


# ──────────────────────── Voice Activity Detection ────────────────────────
def simple_energy_vad(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = VAD_FRAME_MS,
    threshold: float = settings.vad_threshold,
) -> bool:
    """
    Simple energy-based voice activity detection.

    Returns True if the audio window likely contains speech.
    This is a lightweight placeholder; production should use silero-vad or webrtcvad.
    """
    if len(audio) == 0:
        return False

    frame_samples = int(sample_rate * frame_ms / 1000)
    n_frames = len(audio) // frame_samples

    if n_frames == 0:
        rms = np.sqrt(np.mean(audio ** 2))
        return float(rms) > threshold * 0.01

    active_frames = 0
    for i in range(n_frames):
        frame = audio[i * frame_samples : (i + 1) * frame_samples]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > threshold * 0.01:
            active_frames += 1

    speech_ratio = active_frames / n_frames
    return speech_ratio > 0.3


# ──────────────────────── Audio Normalization ─────────────────────────
def normalize_loudness(audio: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    """
    Loudness normalization to approximate EBU R128 target level.

    Prevents clipping and ensures consistent feature extraction
    regardless of input volume.
    """
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-8:
        return audio

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio

    current_db = 20 * np.log10(rms + 1e-8)
    gain_db = target_db - current_db
    gain = 10.0 ** (gain_db / 20.0)

    normalized = audio * gain

    # Soft clipping to prevent distortion
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)

    return normalized


# ──────────────────────── Feature Extraction ─────────────────────────
class FeatureExtractor:
    """
    Extracts multi-modal features from an audio window for the ensemble detector.

    Features extracted:
        - Raw waveform (for RawNet2-style branch)
        - Log-mel spectrogram (for AASIST/LCNN branch)
        - LFCC features (for traditional spoof detection)
        - CQCC features (constant-Q cepstral coefficients)
        - SSL embeddings placeholder (WavLM/wav2vec 2.0)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = 80, n_fft: int = 512, hop_length: int = 160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_all(self, audio: np.ndarray) -> dict:
        """Extract all feature types from an audio window."""
        return {
            "raw_waveform": self._raw_waveform(audio),
            "log_mel_spectrogram": self._log_mel_spectrogram(audio),
            "lfcc": self._lfcc(audio),
            "cqcc": self._cqcc(audio),
            "ssl_embedding": self._ssl_embedding(audio),
        }

    def _raw_waveform(self, audio: np.ndarray) -> np.ndarray:
        """Return raw waveform reshaped for model input [1, T]."""
        return audio.reshape(1, -1)

    def _log_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute log-mel spectrogram using numpy FFT.

        For production, replace with librosa.feature.melspectrogram or
        torchaudio.transforms.MelSpectrogram for GPU acceleration.
        """
        # STFT
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        if n_frames <= 0:
            return np.zeros((self.n_mels, 1), dtype=np.float32)

        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, self.n_fft),
            strides=(audio.strides[0] * self.hop_length, audio.strides[0]),
        ).copy()

        # Apply Hann window
        window = np.hanning(self.n_fft).astype(np.float32)
        frames *= window

        # FFT
        spectrum = np.abs(np.fft.rfft(frames, n=self.n_fft)) ** 2

        # Create mel filterbank
        mel_fb = self._mel_filterbank(self.n_fft, self.sample_rate, self.n_mels)

        # Apply filterbank
        mel_spec = np.dot(spectrum, mel_fb.T)
        log_mel = np.log(mel_spec + 1e-9)

        return log_mel.T.astype(np.float32)  # [n_mels, n_frames]

    def _mel_filterbank(self, n_fft: int, sr: int, n_mels: int) -> np.ndarray:
        """Create a simple mel filterbank matrix."""
        fmax = sr / 2
        mel_low = 2595 * np.log10(1 + 0 / 700)
        mel_high = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_left = bins[m - 1]
            f_center = bins[m]
            f_right = bins[m + 1]

            for k in range(f_left, f_center):
                if f_center != f_left:
                    fb[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right != f_center:
                    fb[m - 1, k] = (f_right - k) / (f_right - f_center)

        return fb

    def _lfcc(self, audio: np.ndarray, n_coeffs: int = 20) -> np.ndarray:
        """
        Linear Frequency Cepstral Coefficients.

        Simplified implementation; production should use proper LFCC
        with pre-emphasis and liftering.
        """
        mel_spec = self._log_mel_spectrogram(audio)
        # DCT approximation via real FFT
        if mel_spec.shape[1] == 0:
            return np.zeros((n_coeffs,), dtype=np.float32)
        avg_spec = np.mean(mel_spec, axis=1)
        coeffs = np.fft.rfft(avg_spec).real[:n_coeffs]
        return coeffs.astype(np.float32)

    def _cqcc(self, audio: np.ndarray, n_coeffs: int = 20) -> np.ndarray:
        """
        Constant-Q Cepstral Coefficients (simplified).

        Uses log-spaced frequency bins instead of linear.
        Production should use a proper CQT implementation.
        """
        # Placeholder: use mel features as approximation
        return self._lfcc(audio, n_coeffs)

    def _ssl_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Self-supervised learning embedding placeholder.

        In production, replace with:
            - WavLM: microsoft/wavlm-base-plus
            - wav2vec 2.0: facebook/wav2vec2-base-960h

        Returns a 768-dim embedding vector.
        """
        # Mock: generate a deterministic pseudo-embedding from audio statistics
        if len(audio) == 0:
            return np.zeros(768, dtype=np.float32)

        rng = np.random.RandomState(int(np.sum(np.abs(audio[:100]) * 1000) % (2**31)))
        embedding = rng.randn(768).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding
