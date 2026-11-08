/**
 * Custom hook for WebSocket connection to the CallTrust Engine backend.
 * Handles binary audio streaming and JSON message parsing.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { AnalysisResult, AppMode, Scenario, SessionEnd, SessionStart, TimelineEntry } from '../types/analysis';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/analyze';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface UseWebSocketReturn {
  status: ConnectionStatus;
  sessionId: string | null;
  latestResult: AnalysisResult | null;
  timeline: TimelineEntry[];
  sessionEnd: SessionEnd | null;
  error: string | null;
  connect: (mode: AppMode, scenario: Scenario) => void;
  disconnect: () => void;
  sendAudio: (pcmData: ArrayBuffer) => void;
  setScenario: (scenario: Scenario) => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [latestResult, setLatestResult] = useState<AnalysisResult | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [sessionEnd, setSessionEnd] = useState<SessionEnd | null>(null);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback((mode: AppMode, scenario: Scenario) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    setStatus('connecting');
    setError(null);
    setTimeline([]);
    setLatestResult(null);
    setSessionEnd(null);

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('connected');
      ws.send(JSON.stringify({
        type: 'start',
        mode,
        scenario,
        sample_rate: 16000,
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'session_start': {
            const start = data as SessionStart & { type: string };
            setSessionId(start.session_id);
            break;
          }
          case 'analysis': {
            const result = data as AnalysisResult & { type: string };
            setLatestResult(result);

            setTimeline((prev) => {
              const entry: TimelineEntry = {
                timestamp: result.window_end_sec,
                trust_score: result.trust_score,
                risk_level: result.risk_level,
                synthetic_probability: result.synthetic_probability,
                channel_risk: result.channel_risk_probability,
                behavioral_risk: result.behavioral_risk_probability,
                speaker_consistency: result.speaker_consistency_score,
              };
              const next = [...prev, entry];
              // Keep last 120 entries (about 2 minutes at 500ms intervals)
              return next.length > 120 ? next.slice(-120) : next;
            });
            break;
          }
          case 'session_end': {
            const end = data as SessionEnd & { type: string };
            setSessionEnd(end);
            setStatus('disconnected');
            break;
          }
          case 'error': {
            setError(data.message || 'Unknown error');
            break;
          }
        }
      } catch (err) {
        console.error('Failed to parse WS message:', err);
      }
    };

    ws.onerror = () => {
      setStatus('error');
      setError('WebSocket connection failed');
    };

    ws.onclose = () => {
      if (status !== 'error') {
        setStatus('disconnected');
      }
    };
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }));
      wsRef.current.close();
    }
    setStatus('disconnected');
  }, []);

  const sendAudio = useCallback((pcmData: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(pcmData);
    }
  }, []);

  const setScenario = useCallback((scenario: Scenario) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'set_scenario', scenario }));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    status,
    sessionId,
    latestResult,
    timeline,
    sessionEnd,
    error,
    connect,
    disconnect,
    sendAudio,
    setScenario,
  };
}
