'use client';

import React, { useEffect, useState } from 'react';

interface MetricsProps {
  bpm: number;
  rhythm: string;
  anomaly: number;
  status: string;
}

export default function MetricsBar({ bpm, rhythm, anomaly, status }: MetricsProps) {
  const [sessionTime, setSessionTime] = useState(0);

  // Simple session duration tracker that only ticks when connected
  useEffect(() => {
    if (status !== 'connected') return;
    const interval = setInterval(() => setSessionTime(s => s + 1), 1000);
    return () => clearInterval(interval);
  }, [status]);

  const formatTime = (sec: number) => {
    const h = Math.floor(sec / 3600).toString().padStart(2, '0');
    const m = Math.floor((sec % 3600) / 60).toString().padStart(2, '0');
    const s = (sec % 60).toString().padStart(2, '0');
    return h === '00' ? `${m}:${s}` : `${h}:${m}:${s}`;
  };

  const getRhythmColor = () => {
    if (rhythm === 'Regular') return 'text-green-400';
    if (rhythm === 'Irregular') return 'text-orange-400';
    if (rhythm === 'Tachycardia' || rhythm === 'Bradycardia') return 'text-red-400';
    return 'text-slate-400';
  };

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <div className="bg-slate-800 p-5 rounded-lg border border-slate-700 shadow-md flex flex-col justify-center items-center">
        <span className="text-xs text-slate-400 uppercase font-semibold mb-1">Heart Rate</span>
        <div className="flex items-baseline gap-1">
           <span className="text-5xl font-bold text-white tracking-tighter">{bpm > 0 ? bpm.toFixed(0) : '--'}</span>
           <span className="text-sm text-slate-500 font-medium">BPM</span>
        </div>
      </div>

      <div className="bg-slate-800 p-5 rounded-lg border border-slate-700 shadow-md flex flex-col justify-center items-center">
        <span className="text-xs text-slate-400 uppercase font-semibold mb-2">Rhythm Status</span>
        <span className={`text-xl md:text-2xl font-bold ${getRhythmColor()} tracking-tight`}>
          {bpm > 0 ? rhythm : 'Analyzing...'}
        </span>
      </div>

      <div className="bg-slate-800 p-5 rounded-lg border border-slate-700 shadow-md flex flex-col justify-center items-center w-full">
        <span className="text-xs text-slate-400 uppercase font-semibold mb-3">Temporal Anomaly Score</span>
        <div className="w-full max-w-[140px] h-3 bg-slate-700 rounded-full overflow-hidden relative">
           <div 
             className={`absolute top-0 left-0 h-full ${anomaly > 0.4 ? 'bg-red-500' : 'bg-blue-500'} transition-all duration-500 ease-out`} 
             style={{ width: `${Math.min(anomaly * 100, 100)}%` }}>
           </div>
        </div>
        <span className="text-xs text-slate-300 mt-2 font-medium">{(anomaly * 100).toFixed(1)}% Divergence</span>
      </div>

      <div className="bg-slate-800 p-5 rounded-lg border border-slate-700 shadow-md flex flex-col justify-center items-center">
        <span className="text-xs text-slate-400 uppercase font-semibold mb-1">Session Duration</span>
        <div className="flex items-center gap-3">
           <div className={`w-2.5 h-2.5 rounded-full ${status === 'connected' ? 'bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-red-500'}`}></div>
           <span className="text-3xl font-bold text-white tracking-tight font-mono">{formatTime(sessionTime)}</span>
        </div>
      </div>
    </div>
  );
}
