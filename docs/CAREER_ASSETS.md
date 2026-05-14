# Career Assets

Use these snippets for a portfolio, resume, or project post after local and deployment verification.

## Resume Bullet

- Built an end-to-end ECG telemetry platform with FastAPI, WebSockets, ONNX Runtime, Next.js, Prometheus metrics, Docker, and PDF report generation; added runtime payload validation, model hash pinning, origin checks, and visible demo failure states.

## Short Project Summary

ECG AI Platform is a production-style portfolio project for real-time ECG telemetry experiments. The backend streams synthetic or MIT-BIH replay beats over WebSockets, classifies 360-sample windows with an ONNX Runtime model, exposes Prometheus metrics, and generates PDF session reports from validated JSON. The frontend is a Next.js dashboard with live waveform rendering, schema-validated payloads, reconnect states, Grad-CAM handling, and report generation.

This project is not a clinical diagnostic product. Public accuracy and latency claims should be added only after benchmark artifacts are generated and reviewed.

## Interview Talking Points

- Designed the backend around an ONNX-first model manager with optional PyTorch loading for Grad-CAM.
- Added SHA-256 model validation so deployment can prove which artifact is live.
- Replaced hardcoded browser URLs with environment-driven REST and WebSocket configuration.
- Added `zod` validation at the dashboard boundary so malformed stream payloads do not silently corrupt UI state.
- Hardened the public surface with CORS, WebSocket origin validation, metrics auth, report auth hooks, and report payload limits.
- Improved demo resilience for missing model artifacts, disconnected WebSockets, malformed payloads, and unavailable explainability.

## LinkedIn Draft

I built ECG AI Platform, a full-stack ECG telemetry demo that connects FastAPI, WebSocket streaming, ONNX Runtime inference, a Next.js dashboard, Prometheus metrics, Docker deployment config, and PDF report generation.

The engineering focus was product readiness: environment-driven frontend URLs, runtime schema validation, model hash pinning, graceful missing-model states, WebSocket origin checks, metrics/report access controls, and clean frontend CI gates.

This is a research and portfolio project, not a clinical diagnostic device. The next step is to publish reproducible MIT-BIH split, leakage-check, accuracy, and latency artifacts.

GitHub: https://github.com/pr6thv3/ecg-ai-platform

#MachineLearning #FastAPI #NextJS #ONNX #MLOps
