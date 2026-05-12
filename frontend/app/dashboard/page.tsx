'use client';

import React from 'react';
import { useECGStream } from '../../hooks/useECGStream';
import ECGWaveformChart from '../../components/ECGWaveformChart';
import BeatClassificationPanel from '../../components/BeatClassificationPanel';
import MetricsBar from '../../components/MetricsBar';
import AlertBanner from '../../components/AlertBanner';
import { ErrorBoundary } from '../../components/ErrorBoundary';

export default function Dashboard() {
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  
  if (!mounted) return null;

  // Using the test endpoint that strictly fires PVC bursts to demonstrate the alert UI in action
  const { 
    connectionStatus, 
    ecgBuffer, 
    beatHistory, 
    currentBPM, 
    rhythmClass, 
    anomalyScore, 
    latestAlert, 
    setLatestAlert 
  } = useECGStream('ws://localhost:8000/ws/ecg-stream?mode=synthetic&pattern=pvc_burst');

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-4 md:p-8 font-sans selection:bg-teal-500/30 overflow-x-hidden">
      
      {/* Absolute Positioning Overlay for Alerts */}
      <AlertBanner alerts={latestAlert} onDismiss={() => setLatestAlert(null)} />
      
      {/* Header */}
      <header className="mb-8 flex flex-col md:flex-row md:justify-between md:items-end border-b border-slate-800 pb-5 gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">ECG AI Monitor</h1>
          <p className="text-slate-400 text-sm mt-1.5 font-medium">Real-Time Continuous Arrhythmia Telemetry</p>
        </div>
        <div className="flex items-center gap-2.5 bg-slate-800 px-4 py-2 rounded-full border border-slate-700 shadow-sm self-start md:self-end">
           <div className={`w-2.5 h-2.5 rounded-full ${
             connectionStatus === 'connected' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 
             connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'
           }`}></div>
           <span className="text-xs font-bold text-slate-300 uppercase tracking-wide">
             {connectionStatus === 'connected' ? 'Live Stream Active' : connectionStatus}
           </span>
        </div>
      </header>

      {/* Main Grid */}
      <main className="max-w-7xl mx-auto space-y-6">
        
        {/* Top KPI Metrics Row */}
        <ErrorBoundary fallback={<div className="h-24 bg-red-900/20 border border-red-500/50 rounded-xl flex items-center justify-center text-red-400">Failed to load metrics</div>}>
          <MetricsBar 
            bpm={currentBPM} 
            rhythm={rhythmClass} 
            anomaly={anomalyScore} 
            status={connectionStatus} 
          />
        </ErrorBoundary>

        {/* Core Visualization Array */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-auto lg:h-[550px]">
          
          {/* Left Column: Recharts Waveform & Pipeline Readout */}
          <div className="lg:col-span-2 flex flex-col gap-6 h-full">
            <div className="flex-1 min-h-[350px]">
              <ErrorBoundary>
                <ECGWaveformChart data={ecgBuffer} />
              </ErrorBoundary>
            </div>
            
            {/* Technical Pipeline Readout Box */}
            <div className="bg-slate-800/80 rounded-lg p-5 border border-slate-700/80 shadow-md backdrop-blur-sm shrink-0">
               <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                 <svg className="w-4 h-4 text-teal-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                 </svg>
                 Backend Pipeline Architecture
               </h3>
               <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-300">
                 <div className="bg-slate-900/50 p-3 rounded border border-slate-700/50">
                    <span className="block text-slate-500 text-[10px] uppercase font-bold tracking-widest mb-1">Preprocessing</span>
                    <span className="font-mono text-xs text-slate-200 block">Butterworth (0.5-45Hz)</span>
                 </div>
                 <div className="bg-slate-900/50 p-3 rounded border border-slate-700/50">
                    <span className="block text-slate-500 text-[10px] uppercase font-bold tracking-widest mb-1">Segmentation</span>
                    <span className="font-mono text-xs text-slate-200 block">Pan-Tompkins (360 frames)</span>
                 </div>
                 <div className="bg-slate-900/50 p-3 rounded border border-slate-700/50 border-l-2 border-l-blue-500">
                    <span className="block text-blue-400/80 text-[10px] uppercase font-bold tracking-widest mb-1">Inference Engine</span>
                    <span className="font-mono text-xs font-semibold text-white block">PyTorch 1D-CNN</span>
                 </div>
               </div>
            </div>
          </div>
          
          {/* Right Column: Scrolling CNN Classification Feed */}
          <div className="h-[400px] lg:h-full">
            <ErrorBoundary>
              <BeatClassificationPanel history={beatHistory} />
            </ErrorBoundary>
          </div>
          
        </div>
      </main>
    </div>
  );
}
