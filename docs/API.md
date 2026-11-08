# CallTrust Engine API & WebSocket Specification

## 1. WebSocket Endpoint: `/ws/analyze`

### Handshake Protocol
```
Client                              Server
  │                                   │
  ├─── {"type": "start", ...} ───────>│  Initialize session in memory
  │<── {"type": "session_start"} ─────┤  Acknowledge with session_id
  │                                   │
  ├─── [Binary PCM Float32 chunk] ───>│  Stream audio continuously
  ├─── [Binary PCM Float32 chunk] ───>│
  │<── {"type": "analysis", ...} ─────┤  Emitted every 500ms
  │                                   │
  ├─── {"type": "set_scenario"} ─────>│  (Demo mode: switch attack scenario)
  │<── {"type": "scenario_changed"} ──┤
  │                                   │
  ├─── {"type": "stop"} ─────────────>│  Terminate call session
  │<── {"type": "session_end", ...} ──┤  Final summary telemetry & RAM wipe
```

### Inbound Messages

#### 1. Start Session (`JSON`)
```json
{
  "type": "start",
  "mode": "demo",        // "demo" | "live"
  "scenario": "genuine",  // "genuine" | "clone_gradual" | "replay" | "noisy_genuine" | "mid_call_swap"
  "sample_rate": 16000
}
```

#### 2. Audio Chunks (`Binary`)
Raw PCM audio bytes. Supported formats:
- `pcm_f32le`: 32-bit float, little-endian, mono, 16000Hz (default Web Audio format).
- `pcm_s16le`: 16-bit signed integer, little-endian, mono.

#### 3. Change Attack Scenario (`JSON`, Demo mode only)
```json
{
  "type": "set_scenario",
  "scenario": "replay"
}
```

#### 4. Stop Session (`JSON`)
```json
{
  "type": "stop"
}
```

---

### Outbound Messages

#### 1. Analysis Result (`JSON`)
Emitted periodically (every `UPDATE_INTERVAL_MS`, default 500ms):
```json
{
  "type": "analysis",
  "session_id": "a9f8e4b7c1234567",
  "timestamp": "2026-11-02T10:45:00.123456Z",
  "window_start_sec": 4.5,
  "window_end_sec": 7.5,
  "trust_score": 38.5,
  "risk_level": "high_risk",
  "calibrated_confidence": 0.8845,
  "uncertainty": 0.1155,
  "synthetic_probability": 0.7421,
  "channel_risk_probability": 0.4210,
  "behavioral_risk_probability": 0.2850,
  "speaker_consistency_score": 0.5210,
  "branch_scores": [
    {
      "name": "rawnet2_waveform",
      "synthetic_probability": 0.7812,
      "confidence": 0.8900,
      "uncertainty": 0.1100
    },
    {
      "name": "aasist_spectrogram",
      "synthetic_probability": 0.7104,
      "confidence": 0.8650,
      "uncertainty": 0.1350
    },
    {
      "name": "ssl_embedding",
      "synthetic_probability": 0.7348,
      "confidence": 0.8980,
      "uncertainty": 0.1020
    }
  ],
  "explanations": [
    {
      "signal": "synthetic_voice",
      "label": "Voice patterns suggest possible AI generation",
      "impact": -0.594,
      "confidence": 0.942
    },
    {
      "signal": "speaker_inconsistency",
      "label": "Speaker voice characteristics changed during the call",
      "impact": -0.287,
      "confidence": 0.679
    }
  ],
  "recommended_action": {
    "severity": "high_risk",
    "message": "High risk of voice manipulation. Do not share OTPs, passwords, or financial details.",
    "icon": "shield-alert"
  },
  "speech_active": true,
  "speech_text": null,
  "liveness_challenge_triggered": false,
  "liveness_phrase": null
}
```

---

## 2. REST Endpoints

### `GET /api/health`
Healthcheck and runtime state.
```json
{
  "status": "ok",
  "version": "0.1.0",
  "mock_inference": true,
  "active_sessions": 2
}
```

### `GET /api/config`
Active engine thresholds and pipeline parameters.

### `GET /api/scenarios`
Returns list of available demo simulation scenarios.

### `GET /metrics`
Prometheus metrics endpoint scraping real-time inference latency, RTF, queue depth, and errors.
