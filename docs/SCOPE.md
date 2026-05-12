# ECG AI Platform: Scope Boundary & Feature Freeze

**Date:** May 12, 2026
**Purpose:** Define the exact boundary of the ECG AI Platform for the final completion sprint. This document serves as a strict guardrail to prevent scope creep, ensuring the project reaches a verified, packaged, and deployed state.

---

## 1. IN Scope
The following 10 capabilities represent the *complete* feature set for Version 1.0. If an item is not on this list, it will not be built.

1. **Real-time ECG WebSocket stream:** 360 Hz bidirectional signal ingestion.
2. **DSP Preprocessing:** Butterworth bandpass filter and Pan-Tompkins R-peak detection.
3. **Beat Segmentation:** Windowing around the R-peak (360 samples/beat).
4. **1D CNN Inference:** Sub-millisecond execution via ONNX Runtime.
5. **5-Class Arrhythmia Classification:** AAMI standard (N, V, A, L, R).
6. **Grad-CAM Explainability:** 1D convolutional saliency mapping.
7. **Next.js Dashboard:** Real-time Recharts visualization and classification feed.
8. **PDF Report Generation:** Clinical summary downloads.
9. **Docker Deployment:** Containerized environment (`render.yaml`, `vercel.json`).
10. **Observability:** Prometheus metrics (`/metrics/dashboard`) and structured logging (Loguru).

---

## 2. OUT of Scope
The following items are explicitly excluded from the V1.0 release to maintain engineering focus:

- **Rust DSP:** Premature optimization; current Python throughput meets the 360 Hz single-stream goal.
- **Kafka:** Over-engineering for a standalone portfolio demonstration.
- **WebRTC:** Out of domain focus; WebSockets are perfectly sufficient for structured JSON telemetry.
- **12-lead ECG:** Appropriate dataset unavailable; system is strictly designed for single-lead (MIT-BIH mapping).
- **Database (PostgreSQL/MongoDB):** Not needed for a stateless, real-time inference demo.
- **Mobile App:** A completely separate project entirely.
- **LLM Chatbot:** Scope creep; distracts from the core ML/DSP signal processing value.
- **Patient Login/Auth:** HIPAA compliance complexity is unmanageable and unnecessary for this phase.
- **Hospital System Integration (HL7/FHIR):** Enterprise scope that provides no value to the immediate deployment.

---

## 3. Version 2.0 Ideas (Future Work)
Once V1.0 is strictly verified and deployed, the following out-of-scope items may be considered for future architectural evolution:
* Migration of DSP logic to a compiled Rust microservice via PyO3.
* Integration of HL7/FHIR mapping for enterprise hospital system ingestion.
* Implementation of an event-driven architecture (Kafka) to support thousands of concurrent patients.
* True-binary streaming protocol adoption via WebRTC data channels.
* Implementation of robust PostgreSQL-backed patient session persistence and auth.

---

## 4. Definition of Done Checklist
The project is declared "Done" when the following 16 criteria are met. No new features can be added until this checklist is 100% complete.

- [x] PyTorch 1D CNN is trained on MIT-BIH test split with Macro-F1 > 90%.
- [x] DSP Pipeline successfully filters raw signals and segments beats cleanly.
- [x] Model is successfully exported to and executes in ONNX Runtime.
- [x] WebSocket router streams simulated ECG data seamlessly at 360 Hz.
- [x] Inference Engine integrates smoothly with the WS router without blocking.
- [x] Grad-CAM saliency maps generate successfully via the `/explain` endpoint.
- [x] PDF report generation returns a valid, formatted file.
- [x] Next.js dashboard visualizes the ECG waveform dynamically via Recharts without crashing.
- [x] Classification UI component renders incoming beat history and rhythm badges correctly.
- [x] React ErrorBoundaries trap rendering panics cleanly.
- [x] Prometheus endpoint exposes WS, API, and Inference telemetry securely.
- [x] Loguru strictly handles JSON and request-id formatting for observability.
- [x] Deployment definitions (`render.yaml` and `vercel.json`) are finalized.
- [x] Automated testing (Pytest & Jest) hits >80% code coverage.
- [x] The complete Docker environment spins up correctly using `docker compose up`.
- [ ] Evaluation scripts execute on held-out test data to populate the final README performance metrics.

---

## 5. Feature Freeze Checklist
The following core modules are designated **STABLE**. They must not be modified during the completion sprint to prevent regression. 

| Module | Status | Notes |
| :--- | :--- | :--- |
| `backend/utils/dsp.py` | **Stable** | Pan-Tompkins + Butterworth math is locked. Do not modify. |
| `backend/inference/model_manager.py` | **Stable** | PyTorch + ONNX dual execution logic is locked. |
| `backend/api/ws_router.py` | **Stable** | 360 Hz stream and Prometheus metric hooks are locked. |
| `backend/api/report_router.py` | **Stable** | PDF generation is locked. |
| `backend/config/settings.py` | **Stable** | Environment bindings are locked. |
| `backend/monitoring/metrics.py` | **Stable** | Collectors and `DashboardTracker` are locked. |
| `backend/utils/logger.py` | **Stable** | Loguru structured JSON format is locked. |
| `frontend/components/ECGWaveformChart.tsx` | **Stable** | Recharts circular buffer optimization is locked. |
| `frontend/components/ErrorBoundary.tsx` | **Stable** | React fallback boundaries are locked. |
| `frontend/components/BeatClassificationPanel.tsx`| **Stable** | UI badge rendering is locked. |
| `frontend/app/dashboard/page.tsx` | **Stable** | Core dashboard layout and wrapper logic is locked. |
| `backend/scripts/export_onnx.py` | **Stable** | ONNX graph validation is locked. |
| `backend/models/train.py` | **Stable** | 1D CNN Architecture definition is locked. |
| `backend/scripts/evaluate_mitbih.py` | **Needs Work** | Must be executed to generate final numbers for the `README.md`. |
