import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import type { TimelineEntry } from '../types/analysis';

interface RiskTimelineProps {
  timeline: TimelineEntry[];
}

export const RiskTimeline: React.FC<RiskTimelineProps> = ({ timeline }) => {
  if (timeline.length === 0) {
    return (
      <div className="h-64 flex flex-col items-center justify-center text-slate-500 text-sm italic">
        <span>No call timeline data yet.</span>
        <span className="text-xs text-slate-600 mt-1">Start streaming audio to monitor score evolution.</span>
      </div>
    );
  }

  // Format data for chart
  const data = timeline.map((entry) => ({
    time: `${entry.timestamp.toFixed(1)}s`,
    rawTime: entry.timestamp,
    score: Math.round(entry.trust_score),
    synthRisk: Math.round(entry.synthetic_probability * 100),
    channelRisk: Math.round(entry.channel_risk * 100),
    speakerConsistency: Math.round(entry.speaker_consistency * 100),
  }));

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="time"
            stroke="#64748b"
            tick={{ fontSize: 11 }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, 100]}
            stroke="#64748b"
            tick={{ fontSize: 11 }}
            ticks={[0, 25, 50, 75, 100]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f172a',
              borderColor: '#334155',
              borderRadius: '8px',
              fontSize: '12px',
              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
            }}
            labelStyle={{ color: '#94a3b8', fontWeight: 'bold' }}
          />

          {/* Risk Level Threshold Lines */}
          <ReferenceLine y={75} stroke="#22c55e" strokeDasharray="3 3" strokeOpacity={0.4} />
          <ReferenceLine y={50} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.4} />
          <ReferenceLine y={25} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.4} />

          {/* Primary Trust Score line */}
          <Line
            type="monotone"
            dataKey="score"
            name="Trust Score"
            stroke="#6366f1"
            strokeWidth={3}
            dot={false}
            isAnimationActive={false}
          />

          {/* Synthetic Risk */}
          <Line
            type="monotone"
            dataKey="synthRisk"
            name="Synth Risk %"
            stroke="#ef4444"
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
            isAnimationActive={false}
          />

          {/* Speaker Consistency */}
          <Line
            type="monotone"
            dataKey="speakerConsistency"
            name="Speaker Consistency %"
            stroke="#10b981"
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
