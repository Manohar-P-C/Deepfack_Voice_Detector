import React from 'react';

interface SpeechActivityIndicatorProps {
  speechActive: boolean;
  audioLevel: number; // 0 to 1
  transcriptSnippet?: string;
}

export const SpeechActivityIndicator: React.FC<SpeechActivityIndicatorProps> = ({
  speechActive,
  audioLevel,
  transcriptSnippet,
}) => {
  // Generate 8 animated waveform bars
  const bars = Array.from({ length: 9 }, (_, i) => {
    // Dynamic height modulated by audioLevel
    const factor = Math.sin((i / 8) * Math.PI);
    const heightPx = Math.max(4, Math.round(audioLevel * 24 * factor + (speechActive ? 4 : 2)));
    return heightPx;
  });

  return (
    <div className="bg-slate-900/40 p-3 rounded-xl border border-slate-800/80 flex items-center justify-between gap-4">
      {/* VAD & Waveform status */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 h-6">
          {bars.map((h, idx) => (
            <div
              key={idx}
              className={`w-1 rounded-full transition-all duration-75 ${
                speechActive ? 'bg-indigo-400' : 'bg-slate-600'
              }`}
              style={{ height: `${h}px` }}
            />
          ))}
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                speechActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'
              }`}
            />
            <span className="text-xs font-semibold text-slate-300">
              {speechActive ? 'Speech Active' : 'Silence / Inactive'}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 font-mono">
            VAD Gating: {speechActive ? 'Analyzing Window' : 'Buffering...'}
          </p>
        </div>
      </div>

      {/* Transcript or context badge */}
      <div className="flex-1 max-w-sm text-right">
        {transcriptSnippet ? (
          <span className="text-xs text-slate-400 font-mono truncate block bg-slate-950/60 px-2 py-1 rounded border border-slate-800">
            "{transcriptSnippet}"
          </span>
        ) : (
          <span className="text-[11px] text-slate-600 italic">
            Live acoustic stream active (in-memory)
          </span>
        )}
      </div>
    </div>
  );
};
