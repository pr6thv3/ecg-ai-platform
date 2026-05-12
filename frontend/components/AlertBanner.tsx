'use client';

import React, { useEffect } from 'react';

export default function AlertBanner({ alerts, onDismiss }: { alerts: string[] | null, onDismiss: () => void }) {
  useEffect(() => {
    // Automatically clear the alert after 8 seconds of rendering
    if (alerts && alerts.length > 0) {
      const timer = setTimeout(onDismiss, 8000);
      return () => clearTimeout(timer);
    }
  }, [alerts, onDismiss]);

  if (!alerts || alerts.length === 0) return null;

  return (
    <div className="fixed top-6 left-1/2 -translate-x-1/2 z-50 w-full px-4 md:px-0 md:max-w-xl">
      <style>{`
        @keyframes bannerDrop { 
          0% { opacity: 0; transform: translateY(-20px); } 
          100% { opacity: 1; transform: translateY(0); } 
        }
        .animate-banner { animation: bannerDrop 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards; }
      `}</style>
      
      <div className="animate-banner bg-red-950/90 border-l-4 border-red-500 text-red-50 p-4 rounded shadow-2xl flex items-start gap-4 backdrop-blur-md ring-1 ring-red-500/20">
        <svg className="w-7 h-7 text-red-500 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
           <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <div className="flex-1">
          <h3 className="font-bold text-lg leading-tight mb-1">Clinical Anomaly Detected</h3>
          <ul className="text-sm text-red-200 list-disc list-inside space-y-0.5 mt-2">
             {alerts.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </div>
        <button 
          onClick={onDismiss} 
          className="text-red-400 hover:text-white hover:bg-red-900/50 p-1 rounded transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}
