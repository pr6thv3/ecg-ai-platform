# QA Verification Runbook: ECG AI Platform

**Purpose:** This document provides a step-by-step procedure for a complete cold-start end-to-end verification of the Dockerized ECG AI platform. Following this runbook precisely should take under 30 minutes.

---

## Section 1: Cold Start

### 1. Initialize the System
Run the following commands to pull the project and spin up the Docker environment:
```bash
git clone https://github.com/pr6thv3/ecg-ai-platform.git
cd ecg-ai-platform

# Ensure environment variables are set (create a .env file)
# Example .env content:
# USE_ONNX=true
# LOG_LEVEL=INFO
# METRICS_AUTH_TOKEN=supersecrettoken

docker compose up --build
```

### 2. Verify Startup Logs (Healthy State)
Watch the console output. A healthy startup should look like:
*   **Backend:** `[INFO] backend.main - Starting FastAPI server...` followed by Uvicorn announcing it is listening on `0.0.0.0:8000`.
*   **Frontend:** `ready - started server on 0.0.0.0:3000, url: http://localhost:3000`.
*   **Model Manager:** Look specifically for `[INFO] backend.inference.model_manager - Loaded ONNX model from /opt/model/ecg_model.onnx` or similar path.

### 3. Confirm Model State and Disable Mocking
*   **Check Model Loaded:** Open a new terminal and run:
    ```bash
    curl -s http://localhost:8000/health
    ```
    *Expected output:* `{"status": "ok", "model_loaded": true, "model_type": "onnx"}` (ensure `model_type` is strictly `"onnx"`, not `"mock"` or `"pytorch"`).
*   **Check for Mock Inference:** Scan the backend startup logs. Ensure there are **NO** `[WARNING]` logs stating `Model not found, using mock inference fallback`. If you see this warning, the volume mount failed or the ONNX file is missing.

---

## Section 2: Backend Endpoint Checks

Execute these commands from your terminal to verify the core APIs are responding properly.

### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```
**Expected:** HTTP 200 with `{"status": "ok", "model_loaded": true, "model_type": "onnx"}`

### 2. Prometheus Metrics
```bash
# Note: Replace 'supersecrettoken' with your actual METRICS_AUTH_TOKEN from .env
curl -X GET -u admin:supersecrettoken http://localhost:8000/metrics
```
**Expected:** HTTP 200 returning raw text in Prometheus format. You must see the metric `ecg_inferences_total` present in the output.

### 3. Grad-CAM Explainability
```bash
# Sending a flatline array of 360 zeros to test the endpoint
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"signal": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'
```
*(Note: Extend the 0 array to exactly 360 elements if the above throws a validation error).*
**Expected:** HTTP 200 returning a JSON response containing an array of exactly 360 float values representing the saliency heatmap.

### 4. Report Generation
```bash
curl -X POST http://localhost:8000/report/generate \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "QA-123", "session_duration": 60, "total_beats": 75, "abnormal_beats": {"V": 2, "A": 0, "L": 0, "R": 0}, "average_bpm": 72}' \
  --output report.pdf
```
**Expected:** HTTP 200, returning a binary stream. The file `report.pdf` should be created locally and openable as a valid PDF document.

---

## Section 3: Frontend Visual Checks

Open your browser to `http://localhost:3000`. Perform the following visual verifications:

1.  **WebSocket Status Indicator**
    *   **Look For:** A green dot or badge explicitly stating "Connected" or "Live".
    *   **Failure State:** It reads "Disconnected", "Connecting...", or is red, indicating a blocked `/ws/ecg-stream` route.
2.  **ECG Waveform Chart**
    *   **Look For:** A smoothly scrolling line chart drawing incoming signal data from right to left.
    *   **Failure State:** The chart is static, empty, or crashes to a fallback ErrorBoundary UI.
3.  **Live BPM Updates**
    *   **Look For:** The BPM stat updating periodically (roughly every 1–2 seconds) reflecting the incoming heart rate.
    *   **Failure State:** Frozen at 0, `NaN`, or not updating at all.
4.  **Beat Classification Panel**
    *   **Look For:** A feed or list component dynamically populating with detected beats labeled N, V, A, L, or R.
    *   **Failure State:** List remains empty despite moving waveform, or labels mismatch the expected AAMI classes.
5.  **Alert Banner**
    *   **Look For:** An alert notification dropping down or appearing when an abnormal beat (V, A, L, R) is detected, which then auto-dismisses after a few seconds.
    *   **Failure State:** The banner persists indefinitely blocking the UI, or never appears during an abnormal event.
6.  **Grad-CAM Visualization**
    *   **Look For:** Clicking on a specific beat event in the classification panel opens a modal or sub-view highlighting the specific segment of the waveform that triggered the classification (a heatmap overlay).
    *   **Failure State:** Clicking does nothing, the heatmap array mismatches the signal length causing rendering distortions, or an error is thrown.
7.  **PDF Report Download**
    *   **Look For:** Clicking the "Download Report" or "Generate Report" button triggers a browser file download dialog.
    *   **Failure State:** Clicking the button fails silently, or the downloaded file is a 0-byte corrupt PDF.
