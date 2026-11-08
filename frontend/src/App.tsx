import React, { useState, useEffect, useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useAudioCapture } from './hooks/useAudioCapture';
import { TrustGauge } from './components/TrustGauge';
import { RiskTimeline } from './components/RiskTimeline';
import { ExplanationPanel } from './components/ExplanationPanel';
import { SignalBreakdown } from './components/SignalBreakdown';
import { RecommendedActionCard } from './components/RecommendedActionCard';
import { SpeechActivityIndicator } from './components/SpeechActivityIndicator';
import { LivenessModal } from './components/LivenessModal';
import { ConsentModal } from './components/ConsentModal';
import { SCENARIO_LABELS, type AppMode, type Scenario } from './types/analysis';

export function App() {
  const [mode, setMode] = useState<AppMode>('demo');
  const [scenario, setSelectedScenario] = useState<Scenario>('clone_gradual');
  const [consentOpen, setConsentOpen] = useState(false);
  const [livenessOpen, setLivenessOpen] = useState(false);
  const [challengePhrase, setChallengePhrase] = useState('');
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    status,
    sessionId,
    latestResult,
    timeline,
    sessionEnd,
    error: wsError,
    connect,
    disconnect,
    sendAudio,
    setScenario,
  } = useWebSocket();

  const {
    isRecording,
    audioLevel,
    error: audioError,
    startMicCapture,
    startFileSimulation,
    startSyntheticStreaming,
    stopAll,
  } = useAudioCapture({
    onAudioChunk: (pcmData) => sendAudio(pcmData),
    sampleRate: 16000,
  });

  // Watch for critical liveness trigger from backend
  useEffect(() => {
    if (latestResult?.liveness_challenge_triggered && latestResult.liveness_phrase) {
      setChallengePhrase(latestResult.liveness_phrase);
      setLivenessOpen(true);
    }
  }, [latestResult]);

  // Handle switching scenario during active demo call
  const handleScenarioChange = (newScenario: Scenario) => {
    setSelectedScenario(newScenario);
    if (status === 'connected') {
      setScenario(newScenario);
    }
  };

  // Start call session
  const handleStartCall = () => {
    if (mode === 'live') {
      setConsentOpen(true);
    } else {
      // Demo mode
      connect('demo', scenario);
      startSyntheticStreaming();
    }
  };

  // Stop call session
  const handleStopCall = () => {
    stopAll();
    disconnect();
  };

  // Handle consent acceptance
  const handleConsentAccept = () => {
    setConsentOpen(false);
    connect('live', 'genuine');
    startMicCapture();
  };

  // Handle audio file upload for Demo Mode
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFileName(file.name);
      connect('demo', scenario);
      startFileSimulation(file);
    }
  };

  const isCallActive = status === 'connected' || isRecording;

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* ─── Top Navigation Bar ────────────────────────────── */}
      <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-xl sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <span className="text-xl">🛡️</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
                  CallTrust Engine
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-950 border border-indigo-700/50 text-indigo-300">
                  v0.1.0-prod
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Real-Time Deepfake Voice & Audio Spoof Detection for Live Calls
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Mode switch */}
            <div className="bg-slate-900 border border-slate-800 p-1 rounded-xl flex items-center gap-1">
              <button
                disabled={isCallActive}
                onClick={() => setMode('demo')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  mode === 'demo'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                } ${isCallActive ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                🎮 Demo Mode
              </button>
              <button
                disabled={isCallActive}
                onClick={() => setMode('live')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  mode === 'live'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                } ${isCallActive ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                🎙️ Live Mic Mode
              </button>
            </div>

            {/* Call Action Button */}
            {!isCallActive ? (
              <button
                onClick={handleStartCall}
                className="flex items-center gap-2 px-5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold uppercase tracking-wider shadow-lg shadow-emerald-600/25 transition active:scale-95"
              >
                <span className="w-2 h-2 rounded-full bg-white animate-ping" />
                Start Protected Call
              </button>
            ) : (
              <button
                onClick={handleStopCall}
                className="flex items-center gap-2 px-5 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold uppercase tracking-wider shadow-lg shadow-rose-600/25 transition active:scale-95"
              >
                <span className="w-2 h-2 rounded-full bg-white" />
                End Call Session
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ─── Secondary Control & Status Bar ────────────────── */}
      <div className="border-b border-slate-800/60 bg-slate-900/30 px-4 py-2.5">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3 text-xs">
          {/* Status badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  status === 'connected'
                    ? 'bg-emerald-400 animate-pulse'
                    : status === 'connecting'
                    ? 'bg-amber-400 animate-ping'
                    : 'bg-slate-600'
                }`}
              />
              <span className="font-medium text-slate-300 capitalize">
                Status: {status}
              </span>
            </div>

            {sessionId && (
              <div className="flex items-center gap-1 font-mono text-[11px] text-slate-400 bg-slate-800/50 px-2 py-0.5 rounded border border-slate-700/60">
                <span className="text-slate-500">Session:</span>
                <span>{sessionId}</span>
              </div>
            )}
          </div>

          {/* Demo Controls */}
          {mode === 'demo' && (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Attack Scenario:</span>
                <select
                  value={scenario}
                  onChange={(e) => handleScenarioChange(e.target.value as Scenario)}
                  className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  {(Object.keys(SCENARIO_LABELS) as Scenario[]).map((sc) => (
                    <option key={sc} value={sc}>
                      {SCENARIO_LABELS[sc]}
                    </option>
                  ))}
                </select>
              </div>

              {/* Upload custom sample */}
              <input
                type="file"
                ref={fileInputRef}
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-2.5 py-1 rounded-lg border border-slate-700 hover:border-slate-500 text-[11px] text-slate-300 hover:text-white transition"
              >
                📁 {selectedFileName ? selectedFileName.slice(0, 16) + '...' : 'Upload WAV / MP3'}
              </button>
            </div>
          )}

          {/* Privacy badge */}
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <span className="text-emerald-400 font-bold">🔒</span>
            <span>Zero Persistence Mode (Memory Only)</span>
          </div>
        </div>
      </div>

      {/* ─── Main Content Grid ─────────────────────────────── */}
      <main className="max-w-7xl mx-auto px-4 py-6 flex-1 w-full space-y-6">
        {/* Error notification banner if any */}
        {(wsError || audioError) && (
          <div className="p-3 bg-red-950/50 border border-red-800 rounded-xl text-xs text-red-200 flex items-center justify-between">
            <span>⚠️ {wsError || audioError}</span>
            <span className="text-[10px] text-red-400">Ensure backend server is running on port 8000</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* ─── Left Column: Trust Gauge & Action Cards ────── */}
          <div className="lg:col-span-5 space-y-6">
            {/* Trust Gauge Card */}
            <div className="glass-card p-6 flex flex-col items-center relative overflow-hidden">
              <TrustGauge
                score={latestResult ? latestResult.trust_score : 100}
                riskLevel={latestResult ? latestResult.risk_level : 'safe'}
                confidence={latestResult ? latestResult.calibrated_confidence : 0.95}
                uncertainty={latestResult ? latestResult.uncertainty : 0.05}
              />
            </div>

            {/* Recommended Action Card */}
            <RecommendedActionCard
              action={latestResult ? latestResult.recommended_action : null}
              riskLevel={latestResult ? latestResult.risk_level : 'safe'}
            />

            {/* Speech Activity & Waveform Bar */}
            <SpeechActivityIndicator
              speechActive={latestResult ? latestResult.speech_active : false}
              audioLevel={audioLevel}
              transcriptSnippet={latestResult?.speech_text}
            />

            {/* Manual Liveness Challenge Trigger for Testing */}
            <div className="bg-slate-900/40 p-3 rounded-xl border border-slate-800/80 flex items-center justify-between text-xs">
              <span className="text-slate-400">Simulate Defensive Protocol:</span>
              <button
                onClick={() => {
                  setChallengePhrase('The purple dragonfly lands on the emerald lake.');
                  setLivenessOpen(true);
                }}
                className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-indigo-300 font-semibold transition"
              >
                Trigger Liveness Challenge
              </button>
            </div>
          </div>

          {/* ─── Right Column: Timeline, Evidence, Model Decomposition ── */}
          <div className="lg:col-span-7 space-y-6">
            {/* Timeline Card */}
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <span>📈</span> Rolling Acoustic Risk Timeline
                  </h3>
                  <p className="text-xs text-slate-400">
                    Live updates every 500 ms (rolling 15s window)
                  </p>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-slate-400">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-indigo-500" /> Trust Score
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-red-500" /> Synth Risk
                  </span>
                </div>
              </div>
              <RiskTimeline timeline={timeline} />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* "Why?" Explanation Evidence Panel */}
              <div className="glass-card p-5">
                <ExplanationPanel
                  explanations={latestResult?.explanations || []}
                  confidence={latestResult?.calibrated_confidence || 0.9}
                />
              </div>

              {/* Ensemble Model Decomposition */}
              <div className="glass-card p-5">
                <SignalBreakdown result={latestResult} />
              </div>
            </div>
          </div>
        </div>

        {/* ─── Post-Call Summary Card (if session ended) ─────── */}
        {sessionEnd && (
          <div className="glass-card p-6 border-indigo-700/60 animate-fade-in-up">
            <h3 className="text-base font-bold text-white mb-2 flex items-center gap-2">
              <span>📋</span> Call Session Summary & Audit Telemetry
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div className="bg-slate-900/60 p-3 rounded-lg">
                <span className="text-slate-400 block">Total Call Duration</span>
                <span className="font-mono text-base font-bold text-slate-100">
                  {sessionEnd.total_duration_sec}s
                </span>
              </div>
              <div className="bg-slate-900/60 p-3 rounded-lg">
                <span className="text-slate-400 block">Average Trust Score</span>
                <span className="font-mono text-base font-bold text-indigo-400">
                  {sessionEnd.average_trust_score}/100
                </span>
              </div>
              <div className="bg-slate-900/60 p-3 rounded-lg">
                <span className="text-slate-400 block">Lowest Trust Dip</span>
                <span className="font-mono text-base font-bold text-red-400">
                  {sessionEnd.min_trust_score}/100
                </span>
              </div>
              <div className="bg-slate-900/60 p-3 rounded-lg">
                <span className="text-slate-400 block">Risk Events Detected</span>
                <span className="font-mono text-base font-bold text-amber-400">
                  {sessionEnd.risk_events_count}
                </span>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ─── Footer ────────────────────────────────────────── */}
      <footer className="border-t border-slate-800/80 bg-slate-950/60 px-4 py-4 text-center text-xs text-slate-500">
        <p>
          CallTrust Engine • Decision-Support Spoof Verification System • Raw audio is never persisted • All inference runs in volatile memory
        </p>
      </footer>

      {/* ─── Interactive Modals ────────────────────────────── */}
      <LivenessModal
        isOpen={livenessOpen}
        phrase={challengePhrase}
        onClose={() => setLivenessOpen(false)}
        onSuccess={() => setLivenessOpen(false)}
      />

      <ConsentModal
        isOpen={consentOpen}
        onAccept={handleConsentAccept}
        onDecline={() => setConsentOpen(false)}
      />
    </div>
  );
}

export default App;
