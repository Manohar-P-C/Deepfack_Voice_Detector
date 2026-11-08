# CallTrust Engine Architecture Specification

## 1. High-Level System Overview

```
Call Audio (WebRTC / Mic / Telephony Stream)
                      │
                      ▼
     ┌──────────────────────────────────┐
     │      WebSocket Ingestion         │  (Float32 / Int16 PCM, 16kHz mono)
     └──────────────────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────┐
     │    VAD & Overlapping Windows     │  (1–3s windows, 0.5–1s stride)
     │   Loudness Normalization (EBU)   │
     └──────────────────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────┐
     │     Multi-Modal Feature Extr.    │
     │  ├── Raw Waveform                │
     │  ├── Log-Mel Spectrogram (80 b)  │
     │  ├── LFCC / CQCC Coefficients    │
     │  └── SSL Embeddings (WavLM 768d) │
     └──────────────────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────┐
     │     Modular Ensemble Detector    │
     │  ├── Branch A: RawNet2           │  (SincConv + ResNet + GRU)
     │  ├── Branch B: LCNN / AASIST     │  (Max-Feature-Map + Graph Attn)
     │  └── Branch C: SSL Head (WavLM)  │  (Linear projection)
     └──────────────────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────┐
     │  Calibration & Uncertainty       │  (Temperature scaling + Shannon entropy)
     └──────────────────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────┐
     │    Multi-Factor Score Fusion     │
     │  ├── Synthetic Probability       │
     │  ├── Codec & Replay Artifacts    │
     │  ├── Behavioral / Latency Drift  │
     │  └── Speaker Consistency Match   │
     └──────────────────────────────────┘
                      │
                      ▼
     ┌──────────────────────────────────┐
     │      Rolling Trust Score (0-100) │  (Safe / Caution / High Risk / Critical)
     │   + Human-Readable Explanations  │
     │   + Prescriptive Actions         │
     └──────────────────────────────────┘
                      │
                      ▼
   React Dashboard / Telephony Hook / SIEM
```

---

## 2. Audio Ingestion & Buffer Management

1. **Protocol**: The client opens a binary WebSocket connection sending raw Float32 PCM audio chunks at 16kHz mono.
2. **Buffering**: `AudioBuffer` retains up to 30 seconds of rolling audio in volatile memory. Audio samples are zeroed out in RAM upon session completion.
3. **Voice Activity Detection (VAD)**: Short-time energy and zero-crossing rates filter out silence and non-speech background to prevent inference on dead air.
4. **Loudness Normalization**: EBU R128 soft-limiting ensures feature extraction operates within standard dynamic ranges regardless of device microphone sensitivity.

---

## 3. Ensemble Detection Branches

### Branch A: RawNet2-Style Waveform Detector
- **Input**: Raw 1D waveform `[1, 48000]`.
- **Purpose**: Detects high-frequency phase discontinuities, unnatural pitch periods, and neural vocoder artifacts that are lost during spectrogram conversion.
- **Layers**: SincConv bandpass front-end, residual 1D convolutional blocks, and GRU sequence aggregation.

### Branch B: LCNN / AASIST Spectrogram Detector
- **Input**: Log-Mel spectrogram `[80, 200]`.
- **Purpose**: Identifies checkerboard artifacts, formant smearing, and spectral energy drop-offs characteristic of TTS acoustic models.
- **Layers**: Light-CNN with Max-Feature-Map (MFM) activations to suppress noise and emphasize dominant spectral peaks.

### Branch C: Self-Supervised Learning (SSL) Embedding Classifier
- **Input**: 768-dimensional embeddings from pre-trained WavLM Base+ or wav2vec 2.0.
- **Purpose**: Leverages rich representations of phonetic transitions and prosodic naturalness learned from tens of thousands of hours of speech.

---

## 4. Multi-Factor Score Fusion

The Trust Score (0–100) is calculated via a calibrated multi-factor equation:

$$\text{TrustScore} = 100 \times \left(1 - \left[ w_{\text{synth}} P_{\text{synth}} + w_{\text{chan}} P_{\text{chan}} + w_{\text{behav}} P_{\text{behav}} + w_{\text{spk}} (1 - S_{\text{spk}}) \right] \right)$$

Where:
- $P_{\text{synth}}$: Calibrated synthetic probability from temperature-scaled ensemble.
- $P_{\text{chan}}$: Replay and loudspeaker acoustic transfer artifacts.
- $P_{\text{behav}}$: Latency hesitation, unnatural pause distribution, and monotone prosody.
- $S_{\text{spk}}$: Cosine similarity against enrolled caller speaker embedding.

---

## 5. Latency & Real-Time Factor (RTF) Budget

| Pipeline Stage | Target Latency | Max Allowed |
|---|---|---|
| Audio chunk ingestion & buffering | 2 ms | 5 ms |
| VAD & EBU Normalization | 4 ms | 10 ms |
| Mel & LFCC Feature Extraction | 12 ms | 25 ms |
| Ensemble ONNX Inference | 25 ms | 50 ms |
| Fusion, Calibration & Explainability | 3 ms | 8 ms |
| WebSocket JSON Transmission | 2 ms | 5 ms |
| **Total Pipeline Roundtrip** | **~48 ms** | **< 105 ms** |
| **Real-Time Factor (RTF)** | **~0.03** | **< 0.10** |
