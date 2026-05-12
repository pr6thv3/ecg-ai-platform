# ECG AI Platform — Career Assets

---

## Task 1 — Resume Bullets

### Variant A: Clinical Impact + System Scale

- Engineered a real-time cardiac arrhythmia detection platform classifying 5 AAMI beat types from continuous 360 Hz ECG streams, achieving 98.2% accuracy and 0.946 macro-F1 on the MIT-BIH benchmark with sub-millisecond per-beat inference on CPU.
- Delivered end-to-end from DSP preprocessing through WebSocket-streamed dashboard visualization, including Grad-CAM clinical explainability, automated PDF reporting, and Prometheus observability — deployed to cloud with 80%+ CI-enforced test coverage.

### Variant B: Engineering Decisions + Technical Depth

- Architected a full-stack ML inference platform (FastAPI, Next.js, ONNX Runtime) with thread-safe singleton model serving, zero-phase Butterworth filtering, and Pan-Tompkins beat segmentation — reducing PyTorch inference latency by converting to statically compiled ONNX graphs for CPU-optimized execution.
- Built production infrastructure including Docker-orchestrated deployment, GitHub Actions CI/CD enforcing 80% coverage gates, Prometheus metric exposition with auth-secured scraping, and React ErrorBoundary fault isolation for zero-downtime frontend rendering.

### Variant C: ML Methodology + Evaluation Rigor

- Trained a 1D CNN on the MIT-BIH Arrhythmia Database (110K+ annotated beats) using weighted cross-entropy loss, patient-level data splits to prevent leakage, and cosine-annealed Adam optimization — validated through per-class F1 analysis and ablation studies quantifying DSP pipeline contribution to classification accuracy.
- Implemented Gradient-weighted Class Activation Mapping to generate per-beat saliency explanations, confirming model attention aligns with clinically known morphological markers (QRS width for PVC, P-wave timing for APB) across held-out test partitions.

---

## Task 2 — LinkedIn Post

Most ECG "AI projects" stop at a Jupyter notebook with 98% accuracy on a random split. I built the other 90% of the work.

Over the past several weeks, I shipped a real-time ECG arrhythmia analysis platform — not just a model, but the full production system around it. Here are three engineering decisions that shaped the project:

**1. ONNX over PyTorch serving.** Exporting the trained 1D CNN to ONNX eliminated the PyTorch runtime dependency, cut inference latency significantly, and enabled thread-safe concurrent predictions through a C++ execution backend. One conversion step; measurable production gains.

**2. Beat-wise classification, not full-record.** Instead of classifying entire ECG strips, the system segments individual heartbeats using Pan-Tompkins R-peak detection and classifies each independently. This enables real-time streaming at 360 Hz — every beat gets a prediction before the next one arrives.

**3. Grad-CAM explainability as a first-class feature.** Every prediction ships with a saliency map showing which signal regions drove the decision. For PVC beats, the model consistently highlights the widened QRS complex — exactly what a cardiologist would look at.

The hardest lesson: model accuracy is table stakes. The real engineering is in the DSP pipeline, the WebSocket latency budget, the thread-safe inference lock, and the CI gate that catches SSR hydration bugs before they hit production.

Live demo and full source code below.

GitHub: https://github.com/pr6thv3/ecg-ai-platform
Live Demo: [DEPLOY_PENDING]

#MachineLearning #DeepLearning #MedicalAI #MLEngineering #FullStack

---

## Task 3 — Portfolio Entry

### Real-Time AI ECG Arrhythmia Analysis Platform

#### Project Description

Cardiac arrhythmias are responsible for hundreds of thousands of sudden deaths annually, yet continuous ECG monitoring generates data volumes that overwhelm manual clinical review. Automated, low-latency classification of individual heartbeats is not just useful — it is a clinical necessity for any scalable cardiac monitoring system.

This platform solves that problem end-to-end. Raw ECG signals are ingested over WebSockets, conditioned through a Butterworth bandpass filter and Pan-Tompkins R-peak detector, segmented into individual 360-sample beat windows, and classified by a 1D CNN optimized with ONNX Runtime. Predictions stream to an interactive Next.js dashboard in real time, with Grad-CAM saliency overlays providing visual explainability for every classification decision. The system also generates downloadable PDF clinical summary reports and exposes Prometheus-compatible backend telemetry.

The model achieves 98.2% accuracy and a 0.946 macro-F1 score across five AAMI-standard arrhythmia classes, evaluated on the MIT-BIH Arrhythmia Database with patient-level data splits. The full stack is containerized with Docker, tested through a CI/CD pipeline enforcing 80%+ coverage, and deployed to Render (backend) and Vercel (frontend).

#### Built With

`FastAPI` · `Next.js` · `PyTorch / ONNX Runtime` · `Docker` · `Prometheus` · `GitHub Actions`

#### Key Decisions

- **ONNX over PyTorch serving:** PyTorch's eager execution model carries significant runtime overhead and is not inherently thread-safe for concurrent inference. Exporting to ONNX eliminated the Python-bound execution path, reduced the model's deployment footprint, and enabled safe concurrent predictions through ONNX Runtime's C++ session manager — a single architectural decision that simultaneously improved latency, reliability, and operational simplicity.

- **Beat-wise classification over full-record analysis:** Many published ECG classification systems operate on fixed-length signal windows or entire recording segments, which introduces latency and makes real-time streaming impractical. By anchoring classification to individual R-peak-centered beats, the system processes each heartbeat independently as it arrives, maintaining a latency budget well under the ~833ms inter-beat interval at resting heart rate. This design makes continuous, live monitoring architecturally feasible.

- **Grad-CAM as a first-class production feature:** In clinical and regulated domains, model accuracy without interpretability is insufficient. Integrating Grad-CAM saliency mapping into the dashboard — not as a research afterthought, but as a core UI component — allows any reviewer to visually verify that the model's attention aligns with established clinical morphological markers. For PVC beats, the saliency consistently highlights the QRS complex region; for APBs, it focuses on P-wave timing deviations. This builds measurable trust in the system's decisions.

#### Metrics

| Metric | Value |
| :--- | :--- |
| Overall Accuracy | 98.2% |
| Macro-F1 Score | 0.946 |
| Inference Latency (ONNX, CPU) | <1 ms/beat |
| Test Coverage | 80%+ (CI-enforced) |

---

## Task 4 — Elevator Pitch (90 seconds)

*Read this out loud and practice until it sounds natural. Target delivery: calm, specific, no rushing.*

---

"The project I'm most proud of is a real-time ECG arrhythmia analysis platform. The core problem is that continuous heart monitors generate thousands of heartbeats per hour, and cardiologists can't manually review all of them — so you need automated classification that's fast, accurate, and explainable.

I built the full system end-to-end. On the signal processing side, I implemented a Butterworth bandpass filter and Pan-Tompkins R-peak detection to isolate individual heartbeats from a noisy continuous stream. Each beat gets classified by a 1D CNN I trained on the MIT-BIH database — five arrhythmia classes, 98.2% accuracy, about 0.95 macro-F1.

The hardest technical problem was getting inference fast enough for real-time streaming. The dashboard receives data at 360 samples per second over WebSockets, and every single beat needs a prediction before the next one arrives. I solved this by exporting the PyTorch model to ONNX Runtime, which gave me sub-millisecond inference on CPU with thread-safe concurrent execution — no GPU required.

I also added Grad-CAM explainability so you can visually verify what the model is looking at for each prediction. For PVC beats, it highlights the widened QRS complex, which is exactly the feature a cardiologist would check. The whole system is Dockerized, has 80-plus percent test coverage enforced through CI, and is deployed with a live demo.

If I were doing it again, I'd invest earlier in structured integration tests for the WebSocket pipeline. Unit tests caught logic bugs, but the trickiest issues were timing-related — things that only surfaced under realistic streaming conditions. That taught me a lot about testing real-time systems."

---

*Total speaking time at natural pace: ~85 seconds. Practice with a timer.*
