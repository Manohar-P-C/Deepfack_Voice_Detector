"""
Tests for audio pipeline: buffer, VAD, normalization, and feature extraction.
"""

import numpy as np
import pytest

from app.core.audio_pipeline import (
    AudioBuffer,
    FeatureExtractor,
    normalize_loudness,
    simple_energy_vad,
)


# ─── AudioBuffer Tests ───────────────────────────────
class TestAudioBuffer:
    def test_ingest_float32(self):
        buf = AudioBuffer(sample_rate=16000, window_sec=1.0, stride_sec=0.5)
        # 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)
        buf.ingest(audio.tobytes(), fmt="pcm_f32le")
        assert buf.duration_sec == pytest.approx(1.0, abs=0.01)

    def test_ingest_int16(self):
        buf = AudioBuffer(sample_rate=16000, window_sec=1.0, stride_sec=0.5)
        audio = np.zeros(16000, dtype=np.int16)
        buf.ingest(audio.tobytes(), fmt="pcm_s16le")
        assert buf.duration_sec == pytest.approx(1.0, abs=0.01)

    def test_windows_extraction(self):
        buf = AudioBuffer(sample_rate=16000, window_sec=1.0, stride_sec=0.5)
        # 2 seconds of audio → should get 3 windows (0-1, 0.5-1.5, 1-2)
        audio = np.random.randn(32000).astype(np.float32)
        buf.ingest(audio.tobytes())
        windows = buf.get_windows()
        assert len(windows) == 3
        for w, start, end in windows:
            assert len(w) == 16000
            assert end - start == pytest.approx(1.0, abs=0.01)

    def test_no_windows_until_full(self):
        buf = AudioBuffer(sample_rate=16000, window_sec=3.0, stride_sec=1.0)
        # Only 1 second → no windows yet
        audio = np.random.randn(16000).astype(np.float32)
        buf.ingest(audio.tobytes())
        assert len(buf.get_windows()) == 0

    def test_buffer_trimming(self):
        buf = AudioBuffer(sample_rate=16000, window_sec=1.0, stride_sec=0.5, max_buffer_sec=5.0)
        # Ingest 10 seconds
        for _ in range(10):
            audio = np.random.randn(16000).astype(np.float32)
            buf.ingest(audio.tobytes())
        assert buf.duration_sec <= 5.0 + 0.1

    def test_clear(self):
        buf = AudioBuffer()
        audio = np.random.randn(16000).astype(np.float32)
        buf.ingest(audio.tobytes())
        buf.clear()
        assert buf.duration_sec == 0.0


# ─── VAD Tests ───────────────────────────────────────
class TestVAD:
    def test_silence_returns_false(self):
        silence = np.zeros(16000, dtype=np.float32)
        assert simple_energy_vad(silence) is False

    def test_loud_signal_returns_true(self):
        # 1kHz tone
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
        assert simple_energy_vad(tone) is True

    def test_empty_returns_false(self):
        assert simple_energy_vad(np.array([], dtype=np.float32)) is False


# ─── Normalization Tests ─────────────────────────────
class TestNormalization:
    def test_silent_unchanged(self):
        silence = np.zeros(16000, dtype=np.float32)
        result = normalize_loudness(silence)
        assert np.allclose(result, silence)

    def test_output_not_clipped(self):
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = normalize_loudness(audio)
        assert np.max(np.abs(result)) <= 1.0

    def test_quiet_audio_boosted(self):
        quiet = np.random.randn(16000).astype(np.float32) * 0.001
        result = normalize_loudness(quiet)
        assert np.sqrt(np.mean(result ** 2)) > np.sqrt(np.mean(quiet ** 2))


# ─── Feature Extraction Tests ────────────────────────
class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor(sample_rate=16000)
        self.audio = np.random.randn(48000).astype(np.float32) * 0.3  # 3 seconds

    def test_extract_all_keys(self):
        features = self.extractor.extract_all(self.audio)
        expected_keys = {"raw_waveform", "log_mel_spectrogram", "lfcc", "cqcc", "ssl_embedding"}
        assert set(features.keys()) == expected_keys

    def test_raw_waveform_shape(self):
        features = self.extractor.extract_all(self.audio)
        assert features["raw_waveform"].shape == (1, 48000)

    def test_mel_spectrogram_shape(self):
        features = self.extractor.extract_all(self.audio)
        mel = features["log_mel_spectrogram"]
        assert mel.shape[0] == 80  # n_mels
        assert mel.shape[1] > 0   # n_frames

    def test_lfcc_shape(self):
        features = self.extractor.extract_all(self.audio)
        assert features["lfcc"].shape == (20,)

    def test_ssl_embedding_shape(self):
        features = self.extractor.extract_all(self.audio)
        assert features["ssl_embedding"].shape == (768,)

    def test_ssl_embedding_normalized(self):
        features = self.extractor.extract_all(self.audio)
        norm = np.linalg.norm(features["ssl_embedding"])
        assert norm == pytest.approx(1.0, abs=0.01)
