import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { BeatEvent } from '../hooks/useECGStream';
import { getApiBaseUrl } from '../lib/config';
import { ExplainResponse, explainResponseSchema } from '../lib/schemas';

interface Prediction {
  class: number;
  label: string;
  confidence: number;
}

const CLASS_MAP: Record<string, number> = { 'N': 0, 'V': 1, 'A': 2, 'L': 3, 'R': 4 };

interface BeatInspectorProps {
  beat: BeatEvent | null;
  onClose: () => void;
}

type ExplanationState = {
  beatKey: string;
  data: ExplainResponse;
};

export default function BeatInspector({ beat, onClose }: BeatInspectorProps) {
  const [explanation, setExplanation] = useState<ExplanationState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const beatKey = beat ? `${beat.timestamp}-${beat.beat_type}` : '';

  useEffect(() => {
    if (!beat) return;

    let cancelled = false;

    const fetchExplanation = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`${getApiBaseUrl().replace(/\/$/, '')}/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            beat_window: beat.raw_window,
            predicted_class: CLASS_MAP[beat.beat_type] || 0
          })
        });
        if (!response.ok) {
          throw new Error('Explainability is unavailable for this beat.');
        }
        const parsed = explainResponseSchema.safeParse(await response.json());
        if (!parsed.success) {
          throw new Error('Explainability response did not match the expected contract.');
        }
        if (!cancelled) {
          setExplanation({ beatKey, data: parsed.data });
        }
      } catch (error) {
        if (!cancelled) {
          setError(error instanceof Error ? error.message : 'Failed to fetch explanation.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchExplanation();
    return () => {
      cancelled = true;
    };
  }, [beat, beatKey]);

  if (!beat) return null;

  const explainData = explanation?.beatKey === beatKey ? explanation.data : null;

  // Map raw window and saliency into a single data array for Recharts
  const chartData = beat.raw_window.map((val, idx) => ({
    index: idx,
    value: val,
    saliency: explainData?.saliency_map ? explainData.saliency_map[idx] : 0
  }));

  // Function to map saliency (0-1) to a cool-to-warm color
  // Low = blue (cool), High = red/orange (warm)
  const getSaliencyColor = (val: number) => {
    // 240 is blue, 0 is red in HSL
    const hue = (1 - val) * 240; 
    return `hsl(${hue}, 100%, 50%)`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-4xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        <div className="flex justify-between items-center p-4 border-b border-slate-800 bg-slate-800/50">
          <h2 className="text-lg font-bold text-slate-200 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-teal-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            Grad-CAM Interpretability
          </h2>
          <button 
            onClick={onClose}
            aria-label="Close beat inspector"
            className="text-slate-400 hover:text-white p-1 rounded hover:bg-slate-700 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          <div className="lg:col-span-2 space-y-4">
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide">Model Attention (Saliency)</h3>
            <div className="h-64 bg-slate-800/50 rounded-lg border border-slate-700/50 p-2 relative">
              {loading && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-900/50 rounded-lg">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-500"></div>
                </div>
              )}
              {error && !loading && (
                <div className="absolute inset-x-4 top-4 z-10 rounded border border-amber-500/40 bg-amber-950/80 p-3 text-sm text-amber-100">
                  {error}
                </div>
              )}
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="saliencyGrad" x1="0" y1="0" x2="1" y2="0">
                      {explainData?.saliency_map?.map((s, i) => (
                        <stop key={i} offset={`${(i / 359) * 100}%`} stopColor={getSaliencyColor(s)} stopOpacity={0.6} />
                      ))}
                      {!explainData && <stop offset="100%" stopColor="transparent" />}
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="index" hide />
                  <YAxis domain={['dataMin', 'dataMax']} hide />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#2dd4bf" 
                    strokeWidth={2}
                    fill="url(#saliencyGrad)" 
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            <div className="flex items-center justify-between px-2">
              <span className="text-xs text-slate-500">Model attention:</span>
              <div className="flex items-center gap-2">
                <span className="text-xs text-blue-400">Low</span>
                <div className="w-32 h-2 rounded-full bg-gradient-to-r from-blue-500 via-green-500 to-red-500"></div>
                <span className="text-xs text-red-400">High</span>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">Diagnostic Context</h3>
              <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
                <div className="text-sm text-slate-400 mb-1">Dominant Feature Region</div>
                <div className="text-xl font-bold text-teal-400">
                  {loading ? 'Analyzing...' : explainData?.dominant_region || 'Unavailable'}
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">Top 3 Predictions</h3>
              <div className="space-y-3">
                {explainData?.predictions?.map((pred, i) => (
                  <div key={pred.class} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className={i === 0 ? "text-white font-medium" : "text-slate-400"}>
                        {pred.label}
                      </span>
                      <span className={i === 0 ? "text-white font-mono" : "text-slate-400 font-mono"}>
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${i === 0 ? 'bg-teal-500' : 'bg-slate-500'}`}
                        style={{ width: `${pred.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
                {!explainData && !loading && (
                  <div className="text-sm text-slate-500 italic">No prediction data available.</div>
                )}
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
