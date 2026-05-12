import { renderHook, act } from '@testing-library/react';
import { useECGStream } from '../../hooks/useECGStream';

describe('useECGStream', () => {
  let mockWebSocket: any;
  let originalWebSocket: any;

  beforeEach(() => {
    jest.useFakeTimers();
    mockWebSocket = {
      onopen: jest.fn(),
      onmessage: jest.fn(),
      onclose: jest.fn(),
      onerror: jest.fn(),
      close: jest.fn(),
    };

    originalWebSocket = global.WebSocket;
    global.WebSocket = jest.fn(() => mockWebSocket) as any;
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
    global.WebSocket = originalWebSocket;
  });

  it('connects to websocket and handles open state', () => {
    const { result } = renderHook(() => useECGStream('ws://localhost/ws'));
    
    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost/ws');
    expect(result.current.connectionStatus).toBe('connecting');
    
    act(() => {
      mockWebSocket.onopen();
    });
    
    expect(result.current.connectionStatus).toBe('connected');
  });

  it('processes incoming messages correctly', () => {
    const { result } = renderHook(() => useECGStream('ws://localhost/ws'));
    
    act(() => {
      mockWebSocket.onopen();
    });
    
    const mockBeat = {
      timestamp: 1000,
      bpm: 75,
      beat_type: 'V',
      confidence: 0.95,
      rhythm_class: 'Irregular',
      anomaly_score: 0.8,
      raw_window: [0.1, 0.5, 1.0, 0.5, 0.1], // 5 points for simplicity
      alert: ['PVC Detected']
    };
    
    act(() => {
      mockWebSocket.onmessage({ data: JSON.stringify(mockBeat) });
    });
    
    expect(result.current.currentBPM).toBe(75);
    expect(result.current.rhythmClass).toBe('Irregular');
    expect(result.current.anomalyScore).toBe(0.8);
    expect(result.current.latestAlert).toEqual(['PVC Detected']);
    expect(result.current.beatHistory).toHaveLength(1);
    expect(result.current.beatHistory[0].beat_type).toBe('V');
    
    // ecgBuffer should have 5 points
    expect(result.current.ecgBuffer).toHaveLength(5);
    // the center peak is at index 2 (floor(5/2))
    expect(result.current.ecgBuffer[2].isPeak).toBe(true);
    expect(result.current.ecgBuffer[2].type).toBe('V');
  });

  it('maintains a circular buffer of 1000 samples for ecg data', () => {
    const { result } = renderHook(() => useECGStream('ws://localhost/ws'));
    
    const mockBeat = {
      timestamp: 1000,
      bpm: 75,
      beat_type: 'N',
      confidence: 0.9,
      rhythm_class: 'Regular',
      anomaly_score: 0.1,
      raw_window: new Array(600).fill(0), // 600 points per message
      alert: null
    };
    
    act(() => {
      mockWebSocket.onmessage({ data: JSON.stringify(mockBeat) }); // 600 points
      mockWebSocket.onmessage({ data: JSON.stringify(mockBeat) }); // +600 = 1200 points, but should truncate to 1000
    });
    
    expect(result.current.ecgBuffer).toHaveLength(1000);
  });

  it('attempts to reconnect on close with exponential backoff', () => {
    const { result } = renderHook(() => useECGStream('ws://localhost/ws'));
    
    expect(global.WebSocket).toHaveBeenCalledTimes(1);
    
    act(() => {
      mockWebSocket.onclose();
    });
    
    expect(result.current.connectionStatus).toBe('disconnected');
    
    // First backoff is 1000ms
    act(() => {
      jest.advanceTimersByTime(1000);
    });
    
    expect(global.WebSocket).toHaveBeenCalledTimes(2);
    
    // Second disconnect
    act(() => {
      mockWebSocket.onclose();
    });
    
    // Second backoff is 2000ms
    act(() => {
      jest.advanceTimersByTime(2000);
    });
    
    expect(global.WebSocket).toHaveBeenCalledTimes(3);
  });
  it('handles high-frequency load without exceeding buffer limits', () => {
    const { result } = renderHook(() => useECGStream('ws://localhost/ws'));
    
    act(() => {
      mockWebSocket.onopen();
    });

    const mockBeat = {
      timestamp: 1000,
      bpm: 75,
      beat_type: 'N',
      confidence: 0.9,
      rhythm_class: 'Regular',
      anomaly_score: 0.1,
      raw_window: new Array(360).fill(0), // 360 points per beat
      alert: null
    };

    // Simulate prolonged operation: 10,000 beats (approx 2.7 hours at 60 bpm)
    act(() => {
      for (let i = 0; i < 10000; i++) {
        mockWebSocket.onmessage({ data: JSON.stringify(mockBeat) });
      }
    });

    // Buffer limits defined in hook:
    // setEcgBuffer(prev => [...prev, ...newPoints].slice(-MAX_ECG_POINTS)) where MAX_ECG_POINTS is 1000
    // setBeatHistory(prev => [newBeatEvent, ...prev].slice(0, MAX_HISTORY)) where MAX_HISTORY is 100
    expect(result.current.ecgBuffer.length).toBeLessThanOrEqual(1000);
    expect(result.current.beatHistory.length).toBeLessThanOrEqual(100);
  });
});
