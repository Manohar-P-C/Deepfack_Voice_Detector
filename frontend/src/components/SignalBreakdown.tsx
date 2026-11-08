import React from 'react';
import type { AnalysisResult } from '../types/analysis';

interface SignalBreakdownProps {
  result: AnalysisResult | null;
}

export const SignalBreakdown: React.FC<SignalBreakdownProps> = ({ result }) => {
  if (!result) {
    return (
      <div className="text-xs text-slate-500 italic text-center p-4">
        Awaiting analysis frames...
      </div>
    );
  }

  const branches = [
    {
      name: 'Raw Waveform (RawNet2)',
      val: result.branch_scores.find((b) => b.name === 'rawnet2_waveform')?.synthetic_probability ?? result.synthetic_probability,
      type: 'synth',
    },
    {
      name: 'Spectrogram (AASIST/LCNN)',
      val: result.branch_scores.find((b) => b.name === 'aasist_spectrogram')?.synthetic_probability ?? result.synthetic_probability,
      type: 'synth',
    },
    {
      name: 'SSL Embeddings (WavLM)',
      val: result.branch_scores.find((b) => b.name === 'ssl_embedding')?.synthetic_probability ?? result.synthetic_probability,
      type: 'synth',
    },
    {
      name: 'Channel & Codec Artifacts',
      val: result.channel_risk_probability,
      type: 'risk',
    },
    {
      name: 'Behavioral & Prosody Anomalies',
      val: result.behavioral_risk_probability,
      type: 'risk',
    },
    {
      name: 'Speaker Consistency',
      val: result.speaker_consistency_score,
      type: 'positive',
    },
  ];

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center mb-1">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Ensemble Model Decomposition
        </h4>
        <span className="text-[11px] font-mono text-indigo-400">Fused Output</span>
      </div>

      <div className="space-y-2.5">
        {branches.map((b) => {
          const isPositive = b.type === 'positive';
          const pct = Math.round(b.val * 100);

          let barColor = 'bg-indigo-500';
          if (isPositive) {
            barColor = pct > 70 ? 'bg-emerald-500' : pct > 40 ? 'bg-amber-500' : 'bg-rose-500';
          } else {
            barColor = pct > 60 ? 'bg-rose-500' : pct > 30 ? 'bg-amber-500' : 'bg-emerald-500';
          }

          return (
            <div key={b.name} className="bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/80">
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-slate-300 font-medium">{b.name}</span>
                <span className="font-mono font-semibold text-slate-200">
                  {pct}% {isPositive ? 'Match' : 'Risk'}
                </span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${barColor}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
