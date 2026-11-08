# CallTrust Engine 🛡️

**Real-Time Deepfake Voice & Audio Spoof Detection for Live Calls**

CallTrust Engine continuously analyzes live or streamed call audio via WebSockets, calculates an animated **Trust Score (0–100)** every 500ms, and provides human-readable explanations ("Why did the score change?") and prescriptive security actions.

Designed to be **privacy-first**, **low-latency (< 50ms inference)**, and **decision-supportive** (never accusing the caller with a raw binary label, but guiding defensive actions and interactive liveness checks).

---

## 🏛️ System Architecture

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
     │    VAD & Overlapping Windows     │  (1–3s sliding windows, 0.5–1s stride)
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
   React Dashboard / Telephony Hook / Contact Center
```

---

## ✨ Key Features

1. **Animated Circular Trust Score Gauge (0–100)**:
   - Dynamic 4-state risk progression: **Safe** (75–100), **Caution** (50–74), **High Risk** (25–49), and **Critical** (0–24).
   - Calibrated confidence and uncertainty intervals ($\pm\%$).
2. **"Why?" Evidence & Interpretability Panel**:
   - Signal decomposition: neural vocoder buzz, spectral smearing, phase anomalies, and channel transfer functions.
   - Evidence chips with positive/negative point impacts.
3. **Rolling Risk Timeline**:
   - Real-time Recharts timeline tracking Trust Score, Synthetic Probability, and Speaker Consistency over the call.
4. **Prescriptive Action Guidance**:
   - Context-aware recommendations: *"Call appears genuine"*, *"Avoid sharing OTPs"*, or *"End call and call back on a verified number"*.
5. **Interactive Liveness Challenge**:
   - When risk is Critical, triggers a randomized prompt challenge (*"The purple dragonfly lands on the emerald lake"*) to test dynamic voice synthesis latency.
6. **Two Operational Modes**:
   - **Demo Mode**: Simulate live calls with 5 pre-built attack scenarios or upload arbitrary `.wav`/`.mp3` audio.
   - **Live Mic Mode**: Zero-latency microphone capture via Web Audio API, resampled in-browser to 16kHz Float32 PCM.
7. **Privacy-First (Zero Persistent Storage)**:
   - Audio processed exclusively in RAM.
   - In-memory audio buffers zeroed out upon session close.
   - SHA-256 hashed session identifiers.

---

## 📁 Repository Structure

```
deepfake-voice/
├── backend/
│   ├── app/
│   │   ├── api/             # WebSocket (/ws/analyze) & REST endpoints (/api)
│   │   ├── core/            # AudioBuffer, VAD, FeatureExtractor, TrustEngine, Metrics, Audit
│   │   ├── schemas/         # Pydantic v2 data models for analysis payloads
│   │   ├── config.py        # Environment settings with defaults
│   │   └── main.py          # FastAPI application factory
│   ├── tests/               # Pytest suite (audio pipeline, score fusion, WebSocket)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/      # TrustGauge, RiskTimeline, ExplanationPanel, SignalBreakdown, etc.
│   │   ├── hooks/           # useWebSocket, useAudioCapture
│   │   ├── types/           # TypeScript analysis interfaces
│   │   ├── App.tsx          # Full dashboard layout
│   │   └── index.css        # Tailwind v4 theme & glassmorphic design system
│   ├── Dockerfile
│   └── package.json
├── ml/
│   ├── dataset.py           # ASVspoof data loader & telephone augmentations (mu-law, packet loss, replay)
│   ├── models.py            # PyTorch models: RawNet2, LCNN, SSL Classifier, CalibratedEnsemble
│   ├── train.py             # Training loop with temperature scaling calibration
│   ├── evaluate.py          # Benchmark metrics: EER, minDCF, Accuracy, F1, ECE, RTF latency
│   ├── robustness.py        # OOD stress suite (unseen generators, noise, short utterances)
│   └── onnx_exporter.py     # Script to export PyTorch models to ONNX
├── infra/
│   ├── prometheus.yml       # Prometheus scrape configuration
│   └── nginx.conf           # Reverse proxy configuration
├── docs/
│   ├── ARCHITECTURE.md      # Detailed system design & latency budget
│   ├── THREAT_MODEL.md      # Threat analysis & defensive countermeasures
│   └── API.md               # WebSocket & REST protocol specification
├── docker-compose.yml       # Full stack container orchestration
└── .env.example             # Configuration template
```

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

Run the full stack (Backend, Frontend, Redis, PostgreSQL, Prometheus) with one command:

```bash
docker compose up --build
```

- **Dashboard UI**: http://localhost:3000 (or `localhost:5173` in local dev)
- **FastAPI Backend**: http://localhost:8000
- **Prometheus Metrics**: http://localhost:9090
- **API Documentation**: http://localhost:8000/docs (when `DEBUG=true`)

---

### Option 2: Local Development

#### 1. Backend Setup (Python 3.11+)

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup (Node.js 20+)

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 to interact with the dashboard.

---

## 🎮 Demo Mode Scenarios

In **Demo Mode**, you can switch between 5 realistic call simulation scenarios from the top dropdown:

| Scenario | Behavior | Expected System Response |
|---|---|---|
| **✅ Genuine Call** | Natural pitch contour and acoustic room reflection. | Trust Score ~90–98 (Safe). |
| **🎭 Gradual Voice Clone** | Begins genuine; after 5s, synthetic voice probability ramps up. | Trust Score drops from 92 to 24 (High Risk / Critical). |
| **🔁 Replay Attack** | Double acoustic channel and loudspeaker resonance detected. | Channel Risk ~65%, Trust Score ~35 (High Risk). |
| **🔊 Noisy Genuine** | High background noise and packet drops. | Flags Channel Risk, but Synthetic Prob stays low (Caution). |
| **⚡ Mid-Call Voice Swap** | Call identity abruptly changes at 12s. | Speaker Consistency dips, triggering Liveness Challenge. |

You can also click **"Upload WAV / MP3"** to stream any custom audio file into the real-time pipeline.

---

## 🧪 Testing & Validation

### 1. Backend Unit & Integration Tests

```bash
cd backend
pytest -v
```

Tests cover:
- Audio ingestion (Float32 and Int16 PCM, buffer trimming, stride extraction).
- Energy-based Voice Activity Detection (VAD) and EBU loudness normalization.
- Multi-modal feature extraction (Waveform, Mel spectrogram, LFCC, SSL embeddings).
- Multi-branch ensemble score fusion and scenario transitions.
- Non-binary label policy compliance.
- Full WebSocket streaming lifecycle.

### 2. ML Evaluation & Benchmarking

Run the benchmark evaluation script:

```bash
python -m ml.evaluate
```

Computes:
- **EER (Equal Error Rate)**
- **minDCF (Minimum Detection Cost Function)**
- **ECE (Expected Calibration Error)**
- **RTF (Real-Time Factor)**

### 3. Robustness & Out-of-Distribution Suite

```bash
python -m ml.robustness
```

Stress-tests against:
- Unseen diffusion TTS & flow-matching vocoders.
- Extreme acoustic noise (5dB SNR).
- Short utterances (< 1.0s).
- Cross-language tonal phonemes.

---

## 📈 Prometheus Metrics

The backend exposes native Prometheus metrics at `/metrics`:

- `calltrust_inference_latency_seconds`: Histogram of feature extraction + inference time.
- `calltrust_real_time_factor`: Gauge measuring processing time relative to audio duration (target < 0.20).
- `calltrust_audio_queue_depth`: Number of buffered audio windows waiting for inference.
- `calltrust_model_confidence`: Distribution of calibrated confidence values.
- `calltrust_active_sessions`: Count of active WebSocket streaming connections.

---

## 🗺️ Production Roadmap

- [x] Streaming WebSocket ingestion with memory-only sliding buffer.
- [x] Multi-branch modular ensemble detector (Waveform, Spectrogram, SSL).
- [x] Realistic mock scenario simulator with 5 telephony attack models.
- [x] Dark-mode React dashboard with animated Trust Gauge & Recharts timeline.
- [x] Interactive Liveness Challenge & Microphone Consent modals.
- [x] PyTorch model definitions & ONNX export pipeline.
- [x] ASVspoof data loader & telephone-channel augmentations.
- [x] Prometheus metrics & JSONL decision audit logger.
- [ ] Silero-VAD ONNX integration for sub-millisecond voice activity detection.
- [ ] WavLM Base+ PyTorch weights download script & pre-compiled ONNX models.
- [ ] Enterprise WebRTC SIP gateway connector for Asterisk / FreeSWITCH / Twilio Media Streams.
- [ ] Speaker verification voice-enrollment module with ECAPA-TDNN embeddings.

---

## 📄 License & Attribution

CallTrust Engine is distributed under the Apache 2.0 License. Designed for defense against synthetic audio impersonation, fraud prevention, and identity verification in mission-critical voice channels.
