'use client';

import React, { useMemo, useState } from 'react';
import { BeatEvent } from '../hooks/useECGStream';
import { getApiBaseUrl } from '../lib/config';

type ReportActionsProps = {
  history: BeatEvent[];
};

const EMPTY_DISTRIBUTION = { N: 0, V: 0, A: 0, L: 0, R: 0 };

function buildReportPayload(history: BeatEvent[]) {
  const sorted = [...history].sort((a, b) => a.timestamp - b.timestamp);
  const classDistribution = history.reduce<Record<'N' | 'V' | 'A' | 'L' | 'R', number>>(
    (acc, beat) => {
      if (beat.beat_type in acc) {
        acc[beat.beat_type as keyof typeof acc] += 1;
      }
      return acc;
    },
    { ...EMPTY_DISTRIBUTION }
  );

  const dominantRhythm =
    Object.entries(classDistribution).sort(([, a], [, b]) => b - a)[0]?.[0] ?? 'N';
  const durationSec = sorted.length > 1 ? Math.max(1, sorted[sorted.length - 1].timestamp - sorted[0].timestamp) : 1;
  const averageConfidence =
    history.length > 0 ? history.reduce((sum, beat) => sum + beat.confidence, 0) / history.length : 0;

  return {
    patient_metadata: {
      id: 'demo-session',
      age: 0,
      gender: 'unspecified',
      session_date: new Date().toISOString(),
    },
    signal_metadata: {
      duration_sec: durationSec,
      sampling_rate: 360,
      snr_before: 0,
      snr_after: 0,
    },
    beat_statistics: {
      total_beats: history.length,
      class_distribution: classDistribution,
      dominant_rhythm: dominantRhythm,
    },
    anomaly_events: history
      .filter((beat) => beat.alert && beat.alert.length > 0)
      .slice(0, 100)
      .map((beat) => ({
        timestamp: beat.timestamp,
        beat_type: beat.beat_type,
        confidence: beat.confidence,
        alert_message: beat.alert?.join('; ') ?? 'Alert detected',
      })),
    model_metrics: {
      average_confidence: averageConfidence,
      low_confidence_beats: history.filter((beat) => beat.confidence < 0.6).length,
    },
  };
}

export default function ReportActions({ history }: ReportActionsProps) {
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState<string | null>(null);
  const canGenerate = history.length > 0 && status !== 'loading';

  const reportPayload = useMemo(() => buildReportPayload(history), [history]);

  const generateReport = async () => {
    if (!canGenerate) {
      setStatus('error');
      setMessage('Need live beats before a report can be generated.');
      return;
    }

    setStatus('loading');
    setMessage(null);

    try {
      const response = await fetch(`${getApiBaseUrl().replace(/\/$/, '')}/report/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reportPayload),
      });

      if (!response.ok) {
        throw new Error(response.status === 401 ? 'Report access requires backend authorization.' : 'Report generation failed.');
      }

      const blob = await response.blob();
      if (blob.size === 0) {
        throw new Error('Generated report was empty.');
      }

      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `ECG_Report_${new Date().toISOString().replace(/[:.]/g, '-')}.pdf`;
      anchor.click();
      URL.revokeObjectURL(url);

      setStatus('success');
      setMessage('PDF report generated.');
    } catch (error) {
      setStatus('error');
      setMessage(error instanceof Error ? error.message : 'Report generation failed.');
    }
  };

  return (
    <div className="flex flex-col items-start md:items-end gap-2">
      <button
        type="button"
        onClick={generateReport}
        disabled={!canGenerate}
        className="rounded-md border border-teal-500/50 bg-teal-500/15 px-4 py-2 text-xs font-bold uppercase tracking-wide text-teal-100 transition hover:bg-teal-500/25 focus:outline-none focus:ring-2 focus:ring-teal-400 disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-slate-800 disabled:text-slate-500"
      >
        {status === 'loading' ? 'Generating...' : 'Generate PDF'}
      </button>
      {message && (
        <div
          role="status"
          className={`max-w-xs text-xs ${status === 'error' ? 'text-amber-200' : 'text-teal-200'}`}
        >
          {message}
        </div>
      )}
    </div>
  );
}
