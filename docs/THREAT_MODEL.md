# CallTrust Engine Threat Model & Security Posture

## 1. Adversarial Threat Vectors in Live Telephony

| Threat Vector | Attack Mechanism | CallTrust Defense Strategy |
|---|---|---|
| **Zero-Shot Voice Cloning (TTS)** | Attacker samples victim's voice (e.g. from social media) and clones voice in real-time during a call. | **Branch A & B Detectors**: Identify neural vocoder phase artifacts (HiFi-GAN, BigVGAN, DiffWave) and spectral smearing. |
| **Real-Time Voice Conversion (VC)** | Attacker speaks into mic and an RVC / VC model transforms timbre to victim's voice in real-time. | **Channel & Prosody Analysis**: Detects latency hesitation (>600ms latency buffer), unnatural breathing cadence, and formant jitter. |
| **Physical Acoustic Replay** | Cloned voice played over desktop speaker into phone handset. | **Replay Detector**: Identifies double-channel transfer function, speaker cabinet harmonic resonances, and room reverberation. |
| **Mid-Call Identity Swap** | Call starts with genuine executive/relative, then transitions into automated voice agent. | **Continuous Rolling Window & Speaker Consistency**: Cosine similarity tracking flags sudden speaker vector divergence mid-call. |
| **Adversarial Audio Perturbation** | Attacker injects imperceptible adversarial noise to fool acoustic neural nets into outputting "bona fide". | **Multi-Branch Ensemble + Conformal Uncertainty**: Perturbations rarely fool raw waveforms and spectrograms simultaneously; triggers high uncertainty alert. |
| **Telephony Codec Camouflage** | Attacker intentionally degrades call quality via VoIP compression to mask vocoder artifacts. | **Telephony-Trained Models**: Training data augmented with G.711 mu-law, AMR-WB, Opus, and jitter loss to prevent false negatives. |

---

## 2. Privacy & Data Handling Guarantees

1. **Zero Raw Audio Persistence**: Audio chunks stream exclusively into RAM. Buffer memory is wiped using in-place zeroing upon WebSocket closure.
2. **Hashed Telemetry**: Session IDs are generated as one-way SHA-256 hashes (`sha256(uuid + salt)`). No voice biometric embeddings or transcripts are saved to disk unless customer explicitly configures an encrypted audit vault.
3. **Decentralized Execution**: The detector is containerized and deployable on-premises or within customer VPC; no audio leaves the host environment.
4. **Active Consent Mechanism**: Frontend enforces user permission prompts before opening the browser MediaStream audio track.

---

## 3. Product Rules & Defensive Operations

- **Decision-Support Paradigm**: The engine never outputs a binary label such as "FAKE" or "DEEPFAKE". Output is presented as a calibrated **Trust Score (0–100)** with associated confidence intervals and evidence chips.
- **Low False Positive Optimization**: In uncertain calls, the system prompts for a **Liveness Challenge** (interactive prompt-response test) rather than falsely accusing the caller or prematurely disconnecting legitimate business transactions.
