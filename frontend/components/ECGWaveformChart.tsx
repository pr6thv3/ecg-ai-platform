'use client';

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceDot } from 'recharts';
import { ECGDataPoint } from '../hooks/useECGStream';

interface ECGChartProps {
  data: ECGDataPoint[];
}

export default function ECGWaveformChart({ data }: ECGChartProps) {
  // Extract annotated R-peaks from the buffer for highlight rendering
  const peaks = data.filter(d => d.isPeak);

  return (
    <div className="w-full h-full bg-slate-800 rounded-lg p-4 border border-slate-700 shadow-md flex flex-col">
      <h2 className="text-sm text-slate-400 font-semibold mb-2 uppercase tracking-wide">Live ECG Telemetry (Lead II)</h2>
      <div className="flex-1 min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            {/* Subtle grid for clinical context */}
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            
            {/* XAxis hides visually but drives the scrolling domain dynamically */}
            <XAxis dataKey="time" hide={true} domain={['dataMin', 'dataMax']} type="number" />
            <YAxis domain={['auto', 'auto']} stroke="#64748b" tick={{fontSize: 11}} unit=" mV" />
            
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#2dd4bf" // Professional teal base for regular rhythm
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false} // Disabled animation allows instant 360Hz frame painting without React stutter
            />
            
            {/* Dynamically overlay R-peaks with clinical color coding */}
            {peaks.map((p, idx) => {
               // Green = Normal, Orange = APB, Red = PVC/Abnormalities
               const color = p.type === 'N' ? '#22c55e' : (p.type === 'A' ? '#f97316' : '#ef4444');
               return (
                 <ReferenceDot 
                    key={idx} 
                    x={p.time} 
                    y={p.value} 
                    r={5} 
                    fill={color} 
                    stroke="#1e293b" 
                    strokeWidth={2} 
                 />
               )
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
