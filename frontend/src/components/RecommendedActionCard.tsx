import React from 'react';
import type { RecommendedAction, RiskLevel } from '../types/analysis';

interface RecommendedActionCardProps {
  action: RecommendedAction | null;
  riskLevel: RiskLevel;
}

export const RecommendedActionCard: React.FC<RecommendedActionCardProps> = ({
  action,
  riskLevel,
}) => {
  const getStyle = () => {
    switch (riskLevel) {
      case 'critical':
        return {
          border: 'border-red-600',
          bg: 'bg-red-950/40',
          badge: 'bg-red-500 text-white',
          title: 'Immediate Defensive Action Required',
          icon: '🚨',
        };
      case 'high_risk':
        return {
          border: 'border-orange-600',
          bg: 'bg-orange-950/30',
          badge: 'bg-orange-500 text-white',
          title: 'High Caution Recommended',
          icon: '⚠️',
        };
      case 'caution':
        return {
          border: 'border-amber-600',
          bg: 'bg-amber-950/20',
          badge: 'bg-amber-500 text-black',
          title: 'Verification Advisory',
          icon: '🔍',
        };
      case 'safe':
      default:
        return {
          border: 'border-emerald-600/40',
          bg: 'bg-emerald-950/20',
          badge: 'bg-emerald-600 text-white',
          title: 'Normal Operational State',
          icon: '🛡️',
        };
    }
  };

  const style = getStyle();

  return (
    <div
      className={`p-4 rounded-xl border ${style.border} ${style.bg} backdrop-blur-md transition-all duration-300 relative overflow-hidden`}
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl select-none" role="img" aria-label="alert">
          {style.icon}
        </span>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
              {style.title}
            </span>
          </div>
          <p className="text-sm font-semibold text-slate-100 leading-snug">
            {action?.message || 'Listening for active call stream...'}
          </p>
        </div>
      </div>

      {riskLevel === 'critical' && (
        <div className="mt-3 pt-3 border-t border-red-800/60 flex items-center justify-between text-xs text-red-200">
          <span className="font-semibold">Protocol triggered:</span>
          <span className="font-mono underline cursor-pointer hover:text-white">
            Trigger Liveness Challenge
          </span>
        </div>
      )}
    </div>
  );
};
