import { useState, useEffect, useRef, useCallback } from 'react';
import { BeatEvent, beatEventSchema } from '../lib/schemas';

export type { BeatEvent };

export type ECGDataPoint = {
  time: number;
  value: number;
  isPeak: boolean;
  type?: string;
};

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

const MAX_ECG_POINTS = 1000;
const MAX_HISTORY = 100;
const SAMPLE_RATE_HZ = 360;

export function useECGStream(url: string) {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [ecgBuffer, setEcgBuffer] = useState<ECGDataPoint[]>([]);
  const [beatHistory, setBeatHistory] = useState<BeatEvent[]>([]);
  const [currentBPM, setCurrentBPM] = useState<number>(0);
  const [rhythmClass, setRhythmClass] = useState<string>('Analyzing');
  const [anomalyScore, setAnomalyScore] = useState<number>(0);
  const [latestAlert, setLatestAlert] = useState<string[] | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempt = useRef(0);
  const shouldReconnectRef = useRef(true);
  const connectRef = useRef<() => void>(() => {});

  const connect = useCallback(() => {
    if (!url) return;
    if (wsRef.current && (wsRef.current.readyState === 0 || wsRef.current.readyState === 1)) return;

    shouldReconnectRef.current = true;
    setConnectionStatus('connecting');
    setStreamError(null);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      reconnectAttempt.current = 0; // reset backoff on success
    };

    ws.onmessage = (event) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        setStreamError('Received malformed stream data.');
        return;
      }

      const result = beatEventSchema.safeParse(parsed);
      if (!result.success) {
        setStreamError('Received stream data that does not match the expected contract.');
        return;
      }

      const data = result.data;

      setCurrentBPM(data.bpm);
      setRhythmClass(data.rhythm_class);
      setAnomalyScore(data.anomaly_score);
      setStreamError(data.error ? `Backend stream warning: ${data.error}` : null);

      if (data.alert && data.alert.length > 0) {
        setLatestAlert(data.alert);
      }

      setBeatHistory(prev => {
        const updated = [data, ...prev];
        return updated.slice(0, MAX_HISTORY);
      });

      setEcgBuffer(prev => {
        const centerIdx = Math.floor(data.raw_window.length / 2);

        const newPoints: ECGDataPoint[] = data.raw_window.map((val, idx) => {
           const isCenter = idx === centerIdx;
           return {
             time: data.timestamp + ((idx - centerIdx) * (1 / SAMPLE_RATE_HZ)),
             value: val,
             isPeak: isCenter,
             type: isCenter ? data.beat_type : undefined
           };
        });

        const updated = [...prev, ...newPoints];
        return updated.length > MAX_ECG_POINTS ? updated.slice(updated.length - MAX_ECG_POINTS) : updated;
      });
    };

    ws.onclose = () => {
      setConnectionStatus('disconnected');
      if (!shouldReconnectRef.current) return;
      const timeout = Math.min(1000 * Math.pow(2, reconnectAttempt.current), 30000);
      reconnectAttempt.current += 1;
      reconnectTimeoutRef.current = setTimeout(() => connectRef.current(), timeout);
    };
    
    ws.onerror = () => {
      ws.close(); // Force trigger onclose
    };
  }, [url]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    connect();
    return () => {
      shouldReconnectRef.current = false;
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [connect]);

  return { connectionStatus, ecgBuffer, beatHistory, currentBPM, rhythmClass, anomalyScore, latestAlert, setLatestAlert, streamError };
}
