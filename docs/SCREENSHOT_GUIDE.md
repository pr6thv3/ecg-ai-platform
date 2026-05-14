# ECG AI Platform Screenshot Guide

This guide details exactly how to capture the 7 essential screenshots for the project README and portfolio presence to maximize impact. 

**General Guidelines:**
- **Resolution:** Resize your browser window to exactly `1440x900` (or crop to this 16:10 aspect ratio) for consistency.
- **Cleanliness:** Hide browser tabs, URL bars, developer tools, and any OS-level notifications.
- **Theme:** Use Dark Mode for all captures as it generally presents medical/telemetry data more professionally.
- **Format:** Save all files as `.png` to preserve crisp text and chart lines without compression artifacts.

---

### 1. Dashboard Overview
- **Filename:** `screenshots/dashboard-overview.png`
- **State:** Full screen capture of the main dashboard. Ensure the WebSocket status is "Connected", a realistic BPM (e.g., 72) is visible, and the waveform has data flowing through it. Ensure at least one alert or anomaly indicator is subtly visible.
- **Caption:** *Real-time interactive dashboard visualizing high-frequency ECG streaming, active rhythm classification, and anomaly scoring.*

### 2. ECG Waveform Focus
- **Filename:** `screenshots/ecg-waveform.png`
- **State:** Cropped capture focusing strictly on the Recharts waveform component. Ensure approximately 3 seconds of signal is visible, with clear morphological waves and highlighted R-peaks (if your UI supports peak markers).
- **Caption:** *360 Hz live telemetry rendering with Pan-Tompkins R-peak detection highlighting dynamic cardiac cycles.*

### 3. Beat Classification Panel
- **Filename:** `screenshots/beat-classification.png`
- **State:** Cropped capture of the recent predictions/classification list. Wait for the simulator to fire a mix of normal (N) and abnormal (V, A, L, or R) beats. Ensure the probability/confidence progress bars are clearly visible.
- **Caption:** *Continuous beat-by-beat classification across 5 AAMI-style demo categories.*

### 4. Grad-CAM Explanation
- **Filename:** `screenshots/gradcam-explanation.png`
- **State:** Capture of the specific explanation modal or panel. Select a Premature Ventricular Contraction (PVC/V) beat to inspect. The red/blue saliency heatmap must be visibly overlaid on the beat signal, alongside the top-3 class probability breakdown.
- **Caption:** *Grad-CAM explainability view highlighting the signal regions that influenced a model prediction.*

### 5. Clinical PDF Report
- **Filename:** `screenshots/pdf-report.png`
- **State:** Open the generated PDF in your browser's PDF viewer. Capture the summary page showing the patient/session overview, total beat counts, anomaly distributions, and the sample anomaly chart.
- **Caption:** *Automated research-demo summary report generation via Pydantic-validated PDF streaming.*

### 6. Metrics & Observability
- **Filename:** `screenshots/metrics-dashboard.png`
- **State:** Capture either the in-app `MetricsBar` or the raw `/metrics` Prometheus endpoint in the browser. Highlight total inference counts, average latency metrics, and active WebSocket connections.
- **Caption:** *Hardened backend observability exposing real-time inference latency, queue depth, and memory telemetry.*

### 7. Architecture Diagram
- **Filename:** `screenshots/architecture-diagram.png`
- **State:** Export your final system architecture diagram (from Figma, Excalidraw, or equivalent) with a transparent or matching dark background.
- **Caption:** *End-to-end system architecture from high-frequency ingestion to edge-optimized ONNX CPU inference.*
