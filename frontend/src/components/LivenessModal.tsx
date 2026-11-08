import React from 'react';

interface LivenessModalProps {
  isOpen: boolean;
  phrase: string;
  onClose: () => void;
  onSuccess: () => void;
}

export const LivenessModal: React.FC<LivenessModalProps> = ({
  isOpen,
  phrase,
  onClose,
  onSuccess,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in-up">
      <div className="max-w-md w-full bg-slate-900 border border-red-700/80 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
        {/* Glow header */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-600 via-orange-500 to-red-600" />

        <div className="flex items-center gap-3 text-red-400 mb-4">
          <span className="text-2xl">⚠️</span>
          <div>
            <h3 className="text-lg font-bold text-white">Liveness Verification Challenge</h3>
            <p className="text-xs text-red-300">High anomaly score detected in current speech stream</p>
          </div>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed mb-4">
          Synthetic audio generators and voice cloners often fail dynamic prompt-response tests.
          Ask the caller to read the randomized phrase below:
        </p>

        <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-center mb-5">
          <span className="text-[11px] text-slate-500 uppercase tracking-widest block mb-1">
            Verification Challenge Phrase
          </span>
          <span className="text-lg font-mono font-bold text-indigo-300 select-all">
            "{phrase || 'The blue river runs quiet in autumn.'}"
          </span>
        </div>

        <div className="bg-slate-800/40 p-3 rounded-lg border border-slate-700/60 text-xs text-slate-400 mb-5">
          <span className="font-semibold text-slate-300 block mb-1">What to observe:</span>
          <ul className="list-disc list-inside space-y-0.5">
            <li>Unnatural delay before repeating the prompt</li>
            <li>Robotic glitches, metallic ringing, or unnatural breaths</li>
            <li>Refusal to read or abrupt topic redirection</li>
          </ul>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2.5 px-4 rounded-xl border border-slate-700 text-slate-300 text-xs font-semibold hover:bg-slate-800 transition"
          >
            Dismiss Alert
          </button>
          <button
            onClick={onSuccess}
            className="flex-1 py-2.5 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition"
          >
            Caller Passed Test
          </button>
        </div>
      </div>
    </div>
  );
};
