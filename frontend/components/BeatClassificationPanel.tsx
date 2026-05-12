'use client';

import React, { useState } from 'react';
import { BeatEvent } from '../hooks/useECGStream';
import BeatInspector from './BeatInspector';

export default function BeatClassificationPanel({ history }: { history: BeatEvent[] }) {
  const [selectedBeat, setSelectedBeat] = useState<BeatEvent | null>(null);
  
  // Feed only displays the last 10 detections
  const recent = history.slice(0, 10);

  const getBadgeStyle = (type: string) => {
    if (type === 'N') return 'bg-green-500/20 text-green-400 border-green-500/50';
    if (type === 'A') return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
    if (type === 'L') return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
    if (type === 'R') return 'bg-purple-500/20 text-purple-400 border-purple-500/50';
    return 'bg-red-500/20 text-red-400 border-red-500/50';
  };

  const getLabel = (type: string) => {
    const map: Record<string, string> = { 'N': 'Normal', 'V': 'PVC', 'A': 'APB', 'L': 'LBBB', 'R': 'RBBB' };
    return map[type] || 'Unknown';
  };

  return (
    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-md flex flex-col h-full overflow-hidden">
      <style>{`
        @keyframes slideDown { 
          from { opacity: 0; transform: translateY(-15px); } 
          to { opacity: 1; transform: translateY(0); } 
        }
        .animate-slide-down { animation: slideDown 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
      `}</style>
      
      <h2 className="text-sm text-slate-400 font-semibold mb-4 uppercase tracking-wide">CNN Pipeline Detections</h2>
      
      <div className="flex-1 overflow-y-auto pr-2 space-y-2 pb-4">
        {recent.map((beat) => (
          <button 
            key={beat.timestamp} 
            onClick={() => setSelectedBeat(beat)}
            className="w-full animate-slide-down flex items-center justify-between p-3 bg-slate-900 rounded-md border border-slate-800/50 shadow-sm hover:border-slate-600 hover:bg-slate-800 transition-all cursor-pointer focus:outline-none focus:ring-1 focus:ring-teal-500"
          >
            <div className="flex items-center gap-3">
               <span className={`px-2.5 py-1 text-xs font-bold rounded shadow-sm border ${getBadgeStyle(beat.beat_type)}`}>
                  {getLabel(beat.beat_type)}
               </span>
               <span className="text-xs text-slate-500 font-mono tracking-tighter">
                  {new Date(beat.timestamp * 1000).toISOString().substr(11, 12)}
               </span>
            </div>
            
            <div className="flex items-center gap-2">
               <span className="text-xs font-medium text-slate-400 w-10 text-right">
                 {(beat.confidence * 100).toFixed(1)}%
               </span>
               <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                 <div 
                   className={`h-full ${beat.confidence > 0.8 ? 'bg-blue-500' : 'bg-yellow-500'}`} 
                   style={{ width: `${beat.confidence * 100}%` }}>
                 </div>
               </div>
            </div>
          </button>
        ))}
        {recent.length === 0 && (
          <div className="text-sm text-slate-500 text-center py-10 italic">
            No beats yet
          </div>
        )}
      </div>

      <BeatInspector beat={selectedBeat} onClose={() => setSelectedBeat(null)} />
    </div>
  );
}
