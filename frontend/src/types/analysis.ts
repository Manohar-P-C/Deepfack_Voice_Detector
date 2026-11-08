/**
 * CallTrust Engine – TypeScript type definitions.
 * Mirrors the backend Pydantic schemas for full type safety.
 */

export type RiskLevel = 'safe' | 'caution' | 'high_risk' | 'critical';

export interface Explanation {
  signal: string;
  label: string;
  impact: number;
  confidence: number;
}

export interface RecommendedAction {
  severity: RiskLevel;
  message: string;
  icon: string;
}

export interface BranchScore {
  name: string;
  synthetic_probability: number;
  confidence: number;
  uncertainty: number;
}

export interface AnalysisResult {
  session_id: string;
  timestamp: string;
  window_start_sec: number;
  window_end_sec: number;

  trust_score: number;
  risk_level: RiskLevel;
  calibrated_confidence: number;
  uncertainty: number;

  synthetic_probability: number;
  channel_risk_probability: number;
  behavioral_risk_probability: number;
  speaker_consistency_score: number;

  branch_scores: BranchScore[];
  explanations: Explanation[];
  recommended_action: RecommendedAction;

  speech_active: boolean;
  speech_text?: string;

  liveness_challenge_triggered: boolean;
  liveness_phrase?: string;
}

export interface SessionStart {
  session_id: string;
  mode: string;
  sample_rate: number;
  started_at: string;
}

export interface SessionEnd {
  session_id: string;
  ended_at: string;
  total_duration_sec: number;
  average_trust_score: number;
  min_trust_score: number;
  risk_events_count: number;
}

export interface WSMessage {
  type: 'session_start' | 'analysis' | 'session_end' | 'scenario_changed' | 'error' | 'pong';
}

export interface TimelineEntry {
  timestamp: number;
  trust_score: number;
  risk_level: RiskLevel;
  synthetic_probability: number;
  channel_risk: number;
  behavioral_risk: number;
  speaker_consistency: number;
}

export type AppMode = 'demo' | 'live';

export type Scenario = 'genuine' | 'clone_gradual' | 'replay' | 'noisy_genuine' | 'mid_call_swap';

export const SCENARIO_LABELS: Record<Scenario, string> = {
  genuine: '✅ Genuine Call',
  clone_gradual: '🎭 Gradual Voice Clone',
  replay: '🔁 Replay Attack',
  noisy_genuine: '🔊 Noisy Genuine',
  mid_call_swap: '⚡ Mid-Call Voice Swap',
};

export const RISK_CONFIG: Record<RiskLevel, { color: string; bg: string; label: string; gradient: string }> = {
  safe: {
    color: '#22c55e',
    bg: 'rgba(34, 197, 94, 0.1)',
    label: 'Safe',
    gradient: 'from-emerald-500 to-green-400',
  },
  caution: {
    color: '#f59e0b',
    bg: 'rgba(245, 158, 11, 0.1)',
    label: 'Caution',
    gradient: 'from-amber-500 to-yellow-400',
  },
  high_risk: {
    color: '#ef4444',
    bg: 'rgba(239, 68, 68, 0.1)',
    label: 'High Risk',
    gradient: 'from-red-500 to-orange-400',
  },
  critical: {
    color: '#dc2626',
    bg: 'rgba(220, 38, 38, 0.15)',
    label: 'Critical',
    gradient: 'from-red-700 to-red-500',
  },
};
