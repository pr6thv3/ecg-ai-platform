import { useState, useEffect, useRef, useCallback } from 'react';

export type BeatEvent = {
  timestamp: number;
  bpm: number;
  beat_type: string;
  confidence: number;
  rhythm_class: string;
  anomaly_score: number;
  raw_window: number[];
  alert: string[] | null;
};

export type ECGDataPoint = {
  time: number;
  value: number;
  isPeak: boolean;
  type?: string;
};

export function useECGStream(url: string) {
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [ecgBuffer, setEcgBuffer] = useState<ECGDataPoint[]>([]);
  const [beatHistory, setBeatHistory] = useState<BeatEvent[]>([]);
  const [currentBPM, setCurrentBPM] = useState<number>(0);
  const [rhythmClass, setRhythmClass] = useState<string>('Analyzing');
  const [anomalyScore, setAnomalyScore] = useState<number>(0);
  const [latestAlert, setLatestAlert] = useState<string[] | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempt = useRef(0);

  const connect = useCallback(() => {
    setConnectionStatus('connecting');
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      reconnectAttempt.current = 0; // reset backoff on success
    };

    ws.onmessage = (event) => {
      const data: BeatEvent = JSON.parse(event.data);
      
      setCurrentBPM(data.bpm);
      setRhythmClass(data.rhythm_class);
      setAnomalyScore(data.anomaly_score);
      
      if (data.alert && data.alert.length > 0) {
        setLatestAlert(data.alert);
      }

      setBeatHistory(prev => {
        // Maintain last 50 events
        const updated = [data, ...prev];
        return updated.slice(0, 50);
      });

      setEcgBuffer(prev => {
        const centerIdx = Math.floor(data.raw_window.length / 2);
        
        // Map raw 1D array to timestamped objects for Recharts
        const newPoints: ECGDataPoint[] = data.raw_window.map((val, idx) => {
           const isCenter = idx === centerIdx;
           return {
             // Align the peak precisely with the broadcast timestamp, and offset surrounding samples (360Hz)
             time: data.timestamp + ((idx - centerIdx) * (1/360)), 
             value: val,
             isPeak: isCenter,
             type: isCenter ? data.beat_type : undefined
           };
        });
        
        const updated = [...prev, ...newPoints];
        // Maintain a circular buffer of 1000 samples for a smooth rolling effect
        return updated.length > 1000 ? updated.slice(updated.length - 1000) : updated;
      });
    };

    ws.onclose = () => {
      setConnectionStatus('disconnected');
      // Exponential backoff reconnect logic
      const timeout = Math.min(1000 * Math.pow(2, reconnectAttempt.current), 30000);
      reconnectAttempt.current += 1;
      reconnectTimeoutRef.current = setTimeout(connect, timeout);
    };
    
    ws.onerror = () => {
      ws.close(); // Force trigger onclose
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [connect]);

  return { connectionStatus, ecgBuffer, beatHistory, currentBPM, rhythmClass, anomalyScore, latestAlert, setLatestAlert };
}
