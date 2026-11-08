import React from 'react';
import type { Explanation } from '../types/analysis';

interface ExplanationPanelProps {
  explanations: Explanation[];
  confidence: number;
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({
  explanations,
}) => {
  if (!explanations || explanations.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic p-4 text-center">
        Awaiting sufficient audio window to compute signal explanations...
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Detected Signals & Evidence
        </h4>
        <span className="text-[11px] text-slate-500">Live Interpretability</span>
      </div>

      <div className="space-y-2">
        {explanations.map((exp, idx) => {
          const isNegative = exp.impact < 0;
          const isNeutral = Math.abs(exp.impact) < 0.05;

          const chipBg = isNeutral
            ? 'bg-slate-800/60 border-slate-700 text-slate-300'
            : isNegative
            ? 'bg-rose-950/40 border-rose-800/60 text-rose-300'
            : 'bg-emerald-950/40 border-emerald-800/60 text-emerald-300';

          return (
            <div
              key={`${exp.signal}-${idx}`}
              className={`p-3 rounded-xl border transition-all ${chipBg}`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium font-mono uppercase tracking-wider opacity-80">
                      {exp.signal.replace(/_/g, ' ')}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/30 font-mono">
                      Conf: {(exp.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="text-sm font-semibold mt-1 text-slate-100">
                    {exp.label}
                  </p>
                </div>

                <div className="text-right">
                  <span
                    className={`text-sm font-mono font-bold ${
                      isNeutral ? 'text-slate-400' : isNegative ? 'text-rose-400' : 'text-emerald-400'
                    }`}
                  >
                    {exp.impact > 0 ? `+${(exp.impact * 100).toFixed(0)}` : `${(exp.impact * 100).toFixed(0)}`} pts
                  </span>
                </div>
              </div>

              {/* Progress bar of signal confidence */}
              <div className="w-full bg-black/40 h-1.5 rounded-full mt-2.5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    isNegative ? 'bg-rose-500' : isNeutral ? 'bg-slate-500' : 'bg-emerald-500'
                  }`}
                  style={{ width: `${Math.round(exp.confidence * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
