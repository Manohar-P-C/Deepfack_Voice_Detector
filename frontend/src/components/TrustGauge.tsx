import React from 'react';
import { RISK_CONFIG, type RiskLevel } from '../types/analysis';

interface TrustGaugeProps {
  score: number;
  riskLevel: RiskLevel;
  confidence: number;
  uncertainty: number;
}

export const TrustGauge: React.FC<TrustGaugeProps> = ({
  score,
  riskLevel,
  confidence,
  uncertainty,
}) => {
  const config = RISK_CONFIG[riskLevel] || RISK_CONFIG.safe;
  const radius = 95;
  const circumference = 2 * Math.PI * radius;
  // Arc goes from 0 to 100
  const strokeDashoffset = circumference - (Math.min(100, Math.max(0, score)) / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center p-6 text-center relative">
      {/* Outer ambient glow */}
      <div
        className="absolute w-56 h-56 rounded-full blur-3xl opacity-20 pointer-events-none transition-all duration-700"
        style={{ backgroundColor: config.color }}
      />

      <div className="relative w-64 h-64 flex items-center justify-center">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 220 220">
          {/* Background circle track */}
          <circle
            cx="110"
            cy="110"
            r={radius}
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.06)"
            strokeWidth="14"
            strokeLinecap="round"
          />

          {/* Animated Gauge arc */}
          <circle
            cx="110"
            cy="110"
            r={radius}
            fill="transparent"
            stroke={config.color}
            strokeWidth="14"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-500 ease-out"
            style={{
              filter: `drop-shadow(0 0 10px ${config.color}66)`,
            }}
          />
        </svg>

        {/* Center score & risk badge */}
        <div className="absolute flex flex-col items-center justify-center">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-1">
            Trust Score
          </span>
          <div className="flex items-baseline">
            <span
              className="text-5xl font-extrabold tracking-tight transition-colors duration-300 font-mono"
              style={{ color: config.color }}
            >
              {Math.round(score)}
            </span>
            <span className="text-xl text-slate-500 font-medium ml-1">/100</span>
          </div>

          <div
            className="mt-2 px-3 py-1 rounded-full text-xs font-bold tracking-wide uppercase transition-all duration-300"
            style={{
              backgroundColor: config.bg,
              color: config.color,
              border: `1px solid ${config.color}40`,
            }}
          >
            {config.label}
          </div>
        </div>
      </div>

      {/* Uncertainty & Calibration Metrology */}
      <div className="grid grid-cols-2 gap-4 w-full max-w-xs mt-3 pt-3 border-t border-slate-800 text-xs">
        <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800 text-center">
          <span className="text-slate-400 block mb-0.5">Confidence</span>
          <span className="font-mono text-sm font-semibold text-slate-200">
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800 text-center">
          <span className="text-slate-400 block mb-0.5">Uncertainty</span>
          <span className="font-mono text-sm font-semibold text-slate-300">
            ±{(uncertainty * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
};
