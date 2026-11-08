import React from 'react';

interface ConsentModalProps {
  isOpen: boolean;
  onAccept: () => void;
  onDecline: () => void;
}

export const ConsentModal: React.FC<ConsentModalProps> = ({
  isOpen,
  onAccept,
  onDecline,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in-up">
      <div className="max-w-lg w-full bg-slate-900 border border-slate-700 rounded-2xl p-6 shadow-2xl">
        <div className="flex items-center gap-3 text-indigo-400 mb-3">
          <span className="text-3xl">🛡️</span>
          <div>
            <h3 className="text-lg font-bold text-white">Privacy & Audio Streaming Consent</h3>
            <p className="text-xs text-slate-400">Zero Persistent Storage Architecture</p>
          </div>
        </div>

        <div className="space-y-3 text-xs text-slate-300 leading-relaxed my-4">
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-start gap-2">
              <span className="text-emerald-400 font-bold">✓</span>
              <span>
                <strong>In-Memory Processing:</strong> Call audio is converted to spectral features in RAM and zeroed immediately after analysis.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-emerald-400 font-bold">✓</span>
              <span>
                <strong>No Audio Storage:</strong> Antigravity CallTrust does not store voice recordings or raw wav files.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-emerald-400 font-bold">✓</span>
              <span>
                <strong>Ephemeral Session ID:</strong> Session identifiers are SHA-256 hashed and anonymized for telemetry and security logs.
              </span>
            </div>
          </div>

          <p className="text-slate-400">
            By enabling <strong>Live Microphone Mode</strong>, you consent to streaming your microphone input to the local inference backend for real-time synthetic voice detection.
          </p>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={onDecline}
            className="flex-1 py-2.5 px-4 rounded-xl border border-slate-700 text-slate-300 text-xs font-semibold hover:bg-slate-800 transition"
          >
            Cancel
          </button>
          <button
            onClick={onAccept}
            className="flex-1 py-2.5 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition"
          >
            I Consent & Enable Microphone
          </button>
        </div>
      </div>
    </div>
  );
};
