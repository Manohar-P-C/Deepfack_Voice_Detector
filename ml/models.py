"""
PyTorch Deepfake Voice Detection Ensemble Architecture.

Ensemble branches:
  - Branch A (RawNet2Style): 1D Sinc/Conv front-end operating directly on raw waveform.
  - Branch B (LCNNStyle): Light-CNN with Max-Feature-Map (MFM) activations on spectrograms.
  - Branch C (SSLClassifier): Multilayer perceptron classifier head on top of WavLM / wav2vec 2.0 features.
  - CalibratedFusion: Logistic regression / ensemble fusion with temperature scaling calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────── Branch A: RawNet2-Style ───────────────────────
class SincConvBlock(nn.Module):
    """Initial bandpass / 1D conv front-end for raw waveforms."""

    def __init__(self, out_channels: int = 64, kernel_size: int = 128, stride: int = 16):
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        return self.pool(self.leaky(self.bn(self.conv(x))))


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.leaky(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.leaky(out + res)


class RawNet2Style(nn.Module):
    """
    Branch A: Raw waveform spoof detector.
    Captures vocoder phase discontinuities and high-frequency sinc artifacts.
    """

    def __init__(self, in_samples: int = 48000, feat_dim: int = 128):
        super().__init__()
        self.front = SincConvBlock(out_channels=64, kernel_size=128, stride=16)
        self.res1 = ResidualBlock1D(64)
        self.res2 = ResidualBlock1D(64)
        self.conv_down = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_down = nn.BatchNorm1d(128)
        self.res3 = ResidualBlock1D(128)
        self.gru = nn.GRU(128, feat_dim, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(feat_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, T] or [B, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.front(x)
        h = self.res1(h)
        h = self.res2(h)
        h = F.leaky_relu(self.bn_down(self.conv_down(h)), 0.2)
        h = self.res3(h)

        # Transpose for GRU: [B, C, L] -> [B, L, C]
        h = h.transpose(1, 2)
        _, h_n = self.gru(h)
        feat = h_n.squeeze(0)  # [B, feat_dim]
        logits = self.classifier(feat)
        return logits, feat


# ─────────────────────── Branch B: LCNN Spectrogram ────────────────────
class MaxFeatureMap2D(nn.Module):
    """Max-Feature-Map (MFM) non-linearity from LCNN paper."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split along channel dimension in half and take maximum
        c = x.size(1)
        assert c % 2 == 0, "Channels must be even for MFM"
        return torch.max(x[:, : c // 2, :, :], x[:, c // 2 :, :, :])


class LCNNStyle(nn.Module):
    """
    Branch B: Spectrogram-based spoof detector using Light CNN.
    Excels at detecting spectral smearing, checkerboard artifacts, and formant anomalies.
    """

    def __init__(self, in_channels: int = 1, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MaxFeatureMap2D()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.mfm2 = MaxFeatureMap2D()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MaxFeatureMap2D()
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Linear(64 * 4 * 4, feat_dim)
        self.classifier = nn.Linear(feat_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, n_mels, n_frames]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.pool1(self.mfm1(self.conv1(x)))
        h = self.pool2(self.mfm2(self.conv2(h)))
        h = self.pool3(self.mfm3(self.conv3(h)))
        h = h.flatten(1)
        feat = F.relu(self.fc(h))
        logits = self.classifier(feat)
        return logits, feat


# ─────────────────────── Branch C: SSL Classifier ─────────────────────
class SSLClassifier(nn.Module):
    """
    Branch C: Self-Supervised Learning (WavLM / wav2vec 2.0) Classifier.
    Takes 768-dimensional frozen SSL embeddings and classifies them.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 768]
        return self.net(x)


# ─────────────────────── Ensemble Calibrated Fusion ────────────────────
class CalibratedEnsemble(nn.Module):
    """
    Fuses predictions from Branch A, B, and C with learned weights
    and a temperature scaling parameter for well-calibrated probabilistic output.
    """

    def __init__(self):
        super().__init__()
        self.rawnet = RawNet2Style()
        self.lcnn = LCNNStyle()
        self.ssl = SSLClassifier()

        # Fusion layer: 2 logits from each branch + channel_prob + behavior_prob = 8 inputs
        self.fusion = nn.Linear(8, 2)

        # Temperature parameter for calibration: logits / temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(
        self,
        raw_waveform: torch.Tensor,
        spectrogram: torch.Tensor,
        ssl_embedding: torch.Tensor,
        channel_risk: torch.Tensor,
        behavioral_risk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits_raw, _ = self.rawnet(raw_waveform)
        logits_lcnn, _ = self.lcnn(spectrogram)
        logits_ssl = self.ssl(ssl_embedding)

        # Ensure channel & behavior are 2D tensors [B, 1]
        if channel_risk.dim() == 1:
            channel_risk = channel_risk.unsqueeze(1)
        if behavioral_risk.dim() == 1:
            behavioral_risk = behavioral_risk.unsqueeze(1)

        fused_input = torch.cat([logits_raw, logits_lcnn, logits_ssl, channel_risk, behavioral_risk], dim=1)
        uncalibrated_logits = self.fusion(fused_input)

        # Apply temperature scaling
        calibrated_logits = uncalibrated_logits / torch.clamp(self.temperature, min=0.1)
        probs = F.softmax(calibrated_logits, dim=-1)

        # Synthetic probability is class 1
        synth_prob = probs[:, 1]

        # Uncertainty: Shannon entropy normalized to [0, 1]
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) / 0.693147  # log(2)

        return {
            "logits": calibrated_logits,
            "synthetic_probability": synth_prob,
            "probs": probs,
            "uncertainty": entropy,
            "branch_rawnet_prob": F.softmax(logits_raw, dim=-1)[:, 1],
            "branch_lcnn_prob": F.softmax(logits_lcnn, dim=-1)[:, 1],
            "branch_ssl_prob": F.softmax(logits_ssl, dim=-1)[:, 1],
        }
