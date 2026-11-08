"""
ASVspoof Dataset loader and telephone/replay augmentations.

Simulates telephony conditions:
  - G.711 mu-law / a-law compression
  - Burst and random packet loss (jitter buffer dropouts)
  - Acoustic replay simulation (loudspeaker + room impulse response)
  - Additive background noise (crowd, street, room)
  - Sample rate degradation (8kHz narrowband telephone codec)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ─────────────────────── Telephony Augmentations ───────────────────────
class AudioAugmentor:
    """
    Applies realistic telephony channel and replay distortions to raw audio.
    Used during training to prevent the model from overfitting to pristine studio audio.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def apply_telephony_channel(
        self,
        audio: np.ndarray,
        snr_db: Optional[float] = None,
        packet_loss_rate: float = 0.03,
    ) -> np.ndarray:
        """Apply complete telephony degradation pipeline."""
        x = audio.copy()

        # 1. Bandpass filter 300Hz - 3400Hz (narrowband telephony simulation)
        x = self.bandpass_telephone(x)

        # 2. G.711 mu-law quantization
        x = self.mulaw_quantization(x)

        # 3. Packet loss simulation
        if packet_loss_rate > 0:
            x = self.simulate_packet_loss(x, loss_rate=packet_loss_rate)

        # 4. Additive call background noise
        if snr_db is not None or random.random() < 0.7:
            target_snr = snr_db if snr_db is not None else random.uniform(10.0, 30.0)
            x = self.add_background_noise(x, snr_db=target_snr)

        return x.astype(np.float32)

    def bandpass_telephone(self, audio: np.ndarray) -> np.ndarray:
        """Simulate PSTN/cellular bandpass: cut frequencies outside 300-3400Hz."""
        if len(audio) == 0:
            return audio
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / self.sample_rate)

        # Attenuate frequencies below 300Hz and above 3400Hz
        mask = np.ones_like(freqs)
        mask[freqs < 300] = np.linspace(0.01, 1.0, np.sum(freqs < 300))
        mask[freqs > 3400] = np.linspace(1.0, 0.01, np.sum(freqs > 3400))

        filtered_fft = fft * mask
        return np.fft.irfft(filtered_fft, n=len(audio))

    def mulaw_quantization(self, audio: np.ndarray, mu: int = 255) -> np.ndarray:
        """G.711 mu-law non-linear companding quantization."""
        if len(audio) == 0:
            return audio
        peak = np.max(np.abs(audio)) + 1e-9
        norm_audio = np.clip(audio / peak, -1.0, 1.0)

        # Compand
        sign = np.sign(norm_audio)
        companded = sign * np.log(1 + mu * np.abs(norm_audio)) / np.log(1 + mu)

        # Quantize to 8-bit
        quantized = np.round((companded + 1.0) / 2.0 * mu) / mu * 2.0 - 1.0

        # Expand
        expanded = np.sign(quantized) * (1.0 / mu) * ((1 + mu) ** np.abs(quantized) - 1.0)
        return expanded * peak

    def simulate_packet_loss(
        self,
        audio: np.ndarray,
        loss_rate: float = 0.03,
        packet_duration_ms: float = 20.0,
    ) -> np.ndarray:
        """Simulate VoIP/RTP packet drops and concealment artifacts."""
        packet_samples = int(self.sample_rate * (packet_duration_ms / 1000.0))
        num_packets = len(audio) // packet_samples
        if num_packets == 0:
            return audio

        out = audio.copy()
        for p in range(num_packets):
            if random.random() < loss_rate:
                start = p * packet_samples
                end = min(start + packet_samples, len(audio))
                # Dropout: zero out or fade to simulate packet loss
                out[start:end] = 0.0

        return out

    def add_background_noise(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add Gaussian white or pinkish background noise at specified SNR."""
        signal_power = np.mean(audio ** 2)
        if signal_power < 1e-9:
            return audio

        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), size=len(audio))
        return audio + noise

    def simulate_acoustic_replay(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulate a physical replay attack:
        Speech played from a low-grade speaker into a microphone in a reverberant room.
        """
        # 1. Speaker frequency non-linear distortion (cabinet resonance)
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / self.sample_rate)

        # Resonance peak around 2.5kHz and cut below 200Hz
        eq = np.ones_like(freqs)
        eq[freqs < 200] *= 0.2
        peak_idx = (freqs > 2000) & (freqs < 3000)
        eq[peak_idx] *= 1.4

        replayed_fft = fft * eq
        distorted = np.fft.irfft(replayed_fft, n=len(audio))

        # 2. Synthetic room impulse response (simple comb delay filter)
        delays = [int(0.015 * self.sample_rate), int(0.035 * self.sample_rate)]
        decays = [0.25, 0.12]
        echo = np.zeros_like(distorted)
        for d, g in zip(delays, decays):
            if len(distorted) > d:
                echo[d:] += distorted[:-d] * g

        replayed = distorted + echo
        # Add slight microphone self-noise
        replayed = self.add_background_noise(replayed, snr_db=35.0)

        return replayed.astype(np.float32)


# ────────────────────── Synthetic Data Generator ───────────────────────
@dataclass
class AudioSample:
    audio: np.ndarray
    label: int  # 0 = bona fide (genuine), 1 = spoofed (deepfake)
    attack_type: str  # 'bonafide', 'tts', 'vc', 'replay', 'hybrid'


class SyntheticDataset:
    """
    In-memory synthetic dataset generator for quick local testing and training validation.
    Generates synthetic bona fide speech and mock spoofed speech with various artifacts.
    """

    def __init__(self, size: int = 500, sample_rate: int = 16000, duration_sec: float = 3.0):
        self.size = size
        self.sample_rate = sample_rate
        self.samples_per_clip = int(sample_rate * duration_sec)
        self.augmentor = AudioAugmentor(sample_rate=sample_rate)

    def generate(self) -> List[AudioSample]:
        dataset = []
        for i in range(self.size):
            is_spoof = (i % 2 == 1)
            t = np.linspace(0, 3.0, self.samples_per_clip, dtype=np.float32)

            # Base fundamental frequency
            f0 = np.random.uniform(100.0, 220.0)

            if not is_spoof:
                # Bona fide: natural continuous pitch variation and breath noise
                pitch_contour = f0 + 20.0 * np.sin(2 * np.pi * 1.5 * t)
                audio = 0.4 * np.sin(2 * np.pi * pitch_contour * t)
                audio += 0.2 * np.sin(2 * np.pi * pitch_contour * 2.0 * t)
                audio += 0.1 * np.sin(2 * np.pi * pitch_contour * 3.0 * t)
                audio += np.random.normal(0, 0.02, size=len(t))  # Natural breath
                attack = "bonafide"
            else:
                # Spoofed: overly flat pitch or phase discontinuity (vocoder artifact)
                attack_choice = random.choice(["tts", "vc", "replay"])
                if attack_choice == "tts":
                    # Unnatural flat pitch
                    audio = 0.4 * np.sin(2 * np.pi * f0 * t)
                    audio += 0.25 * np.sin(2 * np.pi * (f0 * 2.0) * t)
                    # High frequency buzz (neural vocoder artifact)
                    audio += 0.08 * np.sin(2 * np.pi * 4200.0 * t)
                    attack = "tts"
                elif attack_choice == "vc":
                    # Voice conversion: formant mismatch & phase jitter
                    audio = 0.35 * np.sin(2 * np.pi * (f0 + 5.0 * np.sin(2 * np.pi * 0.5 * t)) * t)
                    audio += np.random.normal(0, 0.04, size=len(t))
                    attack = "vc"
                else:
                    # Replay
                    base = 0.4 * np.sin(2 * np.pi * f0 * t)
                    audio = self.augmentor.simulate_acoustic_replay(base)
                    attack = "replay"

            # Random telephony augmentation
            if random.random() < 0.6:
                audio = self.augmentor.apply_telephony_channel(audio)

            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.8

            dataset.append(AudioSample(
                audio=audio.astype(np.float32),
                label=1 if is_spoof else 0,
                attack_type=attack,
            ))

        return dataset
