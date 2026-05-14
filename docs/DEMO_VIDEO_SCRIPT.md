# Demo Video Script

## Opening

"This is ECG AI Platform, a production-style portfolio system for real-time ECG telemetry experiments. It combines FastAPI, WebSocket streaming, ONNX Runtime inference, a Next.js dashboard, Prometheus metrics, and PDF report generation. It is a research demo, not a clinical product."

## Dashboard Walkthrough

1. Open the dashboard.
2. Show the connection badge moving from connecting to live stream active.
3. Point out the waveform chart, heart-rate card, rhythm state, anomaly score, and session timer.
4. Show the beat classification feed updating with AAMI labels.
5. Click a beat and open the Grad-CAM inspector. If PyTorch Grad-CAM is unavailable in the current artifact set, show the explicit unavailable state.
6. Click Generate PDF and confirm the browser receives a non-empty report.

## Backend Walkthrough

Show:

- `/health` with model status, model path, and artifact hash.
- `/metrics` with Basic Auth when `METRICS_AUTH_TOKEN` is configured.
- `/analyze` accepting a 360-sample beat window.
- `/report/generate` accepting the validated session payload.

## Engineering Notes To Mention

- The dashboard does not hardcode localhost; REST and WebSocket URLs are environment-driven.
- WebSocket payloads are runtime-validated in the browser with `zod`.
- The backend validates payload size and finite numeric inputs.
- The model artifact can be pinned by SHA-256.
- Missing model, failed WebSocket, malformed payload, and unavailable explainability states are visible and non-crashing.

## Description Template

```text
ECG AI Platform

A production-style portfolio system for ECG telemetry experiments using:
- FastAPI REST and WebSocket streaming
- ONNX Runtime inference for 360-sample beat windows
- Next.js dashboard with runtime schema validation
- Prometheus metrics and deployment-ready configuration
- Automated PDF report generation

GitHub: https://github.com/pr6thv3/ecg-ai-platform
Live Demo: add the deployed Vercel URL after deployment verification

This is a research and portfolio project. Not intended for clinical use.
```

## Pre-Recording Checklist

- Docker Desktop running.
- `docker compose up --build` verified.
- Dashboard loads at `http://localhost:3000`.
- WebSocket connects.
- Beat feed receives events.
- Grad-CAM inspector opens.
- PDF report action produces a visible result.
- Browser console has no uncaught runtime errors.
