# ECG AI Platform — Demo Video Script & Recording Guide

**Target Duration:** 2:30 (hard max: 3:00)
**Resolution:** 1920×1080 @ 30fps
**Aspect Ratio:** 16:9
**Tone:** Technical, confident, zero filler. Every sentence proves competence.

---

## Part 1 — Full Narration Script

### 0:00–0:10 | Title Card

**Visual:** Dark background. Project title fades in center-screen in large white text. GitHub URL appears below. A subtle ECG heartbeat line animation traces across the bottom of the frame.

**Narration:**
> "This is a real-time ECG arrhythmia analysis platform. It combines digital signal processing, a 1D convolutional neural network optimized with ONNX Runtime, and full-stack WebSocket streaming — built with FastAPI and Next.js."

**Production Notes:**
- Use a simple motion graphics tool (Canva, After Effects, or even PowerPoint export) for the title card.
- The GitHub URL should be `github.com/pr6thv3/ecg-ai-platform` — make sure it's legible at 1080p.
- Duration is tight. Rehearse this line to land at exactly 10 seconds.

---

### 0:10–0:25 | Problem Statement

**Visual:** A clean text slide with two bullet points appearing sequentially (fade-in, not instant). Dark background, white text, no clip art.

**Slide Text:**
```
• Cardiac arrhythmias cause 300,000+ sudden deaths annually in the US alone.
• Continuous ECG monitoring generates thousands of beats per hour —
  too many for manual review. Automated, low-latency classification
  is a clinical necessity.
```

**Narration:**
> "Cardiac arrhythmias are responsible for over 300,000 sudden deaths annually in the United States. Continuous monitoring generates thousands of heartbeats per hour — far beyond what a clinician can manually review. This platform automates that analysis in real time, classifying each beat in under a millisecond."

**Production Notes:**
- Do NOT show any patient data, hospital footage, or stock medical photos. Text-only keeps it professional and avoids any clinical misrepresentation.
- Pace: calm and measured. This is the "why it matters" moment.

---

### 0:25–0:45 | Architecture Overview

**Visual:** The architecture diagram (`docs/assets/architecture-diagram.png`) fills the screen. As you narrate each component, either:
- (A) Use a yellow highlight/circle annotation that moves across the diagram, or
- (B) Zoom into each section sequentially using OBS crop filters.

**Narration:**
> "Here's the system architecture. Raw ECG signals enter through a WebSocket connection and pass through a DSP pipeline — a fourth-order Butterworth bandpass filter followed by Pan-Tompkins R-peak detection. Each detected beat is segmented into a 360-sample window centered on the R-peak. That tensor is fed into a 1D CNN running on ONNX Runtime for thread-safe, sub-millisecond inference. Predictions stream back to the Next.js dashboard over the same WebSocket, while Prometheus scrapes backend telemetry for observability."

**Production Notes:**
- This is the most information-dense segment. Speak deliberately. Practice the component names until they flow naturally.
- If using OBS zoom: pre-configure 5 crop filter presets and switch between them with hotkeys during recording.
- Total: ~20 seconds for 6 components = ~3 seconds per component. Tight but achievable.

---

### 0:45–1:10 | Live Dashboard

**Visual:** Full browser window showing the Next.js dashboard. The WebSocket status indicator reads "Connected" in green. The ECG waveform is actively scrolling. The BPM counter is updating. The classification feed is populating.

**Narration:**
> "Here's the live dashboard. The WebSocket connection is active — you can see the status indicator in the top right. The waveform chart renders the incoming ECG signal at 360 samples per second using Recharts. Below it, the current heart rate reads 72 beats per minute. And on the right side, the beat classification panel is filling in real time — each row shows the predicted class, confidence score, and timestamp."

**Production Notes:**
- Start this segment by showing the terminal where `docker compose up` is running. Let the viewer see the backend boot log for 2–3 seconds (ONNX model loaded, Uvicorn started), then switch to the browser.
- The dashboard must be LIVE, not a screenshot. The waveform must be visibly moving.
- If your waveform stutters or freezes, restart the backend before recording. First impressions are permanent.

---

### 1:10–1:30 | ECG Stream + Beat Classification

**Visual:** Zoom into the waveform chart and the BeatClassificationPanel side by side. The waveform should show clear PQRST morphology with visible R-peaks. The classification panel should show a mix of N (green), V (red), and A (orange) events.

**Narration:**
> "Looking more closely at the signal — you can see the distinct P-wave, QRS complex, and T-wave of each cardiac cycle. The Pan-Tompkins algorithm detects R-peaks in real time, and each beat is independently classified. Most beats are Normal — labeled N. But watch — there's a PVC event, classified as V with 94% confidence. And an atrial premature beat, labeled A. Each prediction arrives in under a millisecond."

**Production Notes:**
- If your demo stream is simulated (MIT-BIH playback), use a recording that contains at least one V and one A event within this 20-second window. Pre-test the timing.
- Hover over a classification row if your UI shows a tooltip with additional details.
- Do NOT fake or manually inject events. If the simulator doesn't produce a V beat, let it run longer before recording.

---

### 1:30–1:50 | Grad-CAM Explainability

**Visual:** Click on a PVC (V) beat in the classification panel to open the Grad-CAM inspector. The saliency heatmap overlays the beat waveform. The top-3 prediction probabilities are visible.

**Narration:**
> "Clicking on any beat opens the Grad-CAM explainability panel. Here I've selected a PVC event. The saliency overlay shows the model is focusing its attention on the QRS complex region — which is exactly where premature ventricular contractions exhibit their characteristic wide, aberrant morphology. The top-3 predictions confirm: V at 94%, N at 4%, A at 2%. This isn't a black box — every prediction is interpretable."

**Production Notes:**
- The click-to-inspect action must be visible on screen. Move the cursor deliberately — don't snap to it.
- If the Grad-CAM panel takes a moment to load, that's fine. A brief loading state actually demonstrates it's computing live, not cached.
- Pause for 1 second after the saliency map renders before continuing narration. Let the viewer absorb it.

---

### 1:50–2:05 | PDF Report Generation

**Visual:** Click the "Generate Report" button on the dashboard. Show the browser downloading or opening the PDF. Display the first page of the report showing the session summary, beat distribution chart, and anomaly log.

**Narration:**
> "The platform also generates downloadable clinical-style PDF reports. Clicking 'Generate Report' triggers a server-side render with full beat statistics, an anomaly distribution breakdown, signal quality metrics, and a research-use disclaimer. The report streams back as a binary PDF over the REST API."

**Production Notes:**
- If the PDF opens in the browser's built-in viewer, that's ideal — shows it inline.
- If it downloads as a file, open it manually in a PDF viewer. Either way, ensure the report content is visible on screen for at least 3 seconds.
- Scroll slowly through the first page so the viewer can read the section headers.

---

### 2:05–2:20 | Tech Stack + Metrics

**Visual:** Split into two parts:
1. (2:05–2:12) A clean tech stack slide showing the 3-column table from the README.
2. (2:12–2:20) Switch to the MetricsBar or `/metrics` endpoint showing live Prometheus counters.

**Narration:**
> "The stack: FastAPI and Uvicorn on the backend, Next.js with Recharts on the frontend, PyTorch for training and ONNX Runtime for optimized inference — all containerized with Docker and tested through GitHub Actions CI. The metrics endpoint exposes Prometheus-compatible telemetry — here you can see total inferences served, average latency at 0.8 milliseconds per beat, and three active WebSocket connections."

**Production Notes:**
- The tech stack slide can be a screenshot of the README table or a custom-designed slide. Keep it minimal.
- For the metrics view: if you have a Grafana dashboard, show that. If not, the raw `/metrics` text endpoint or the in-app MetricsBar works fine.
- Mention one specific number (the latency). Recruiters and engineers remember concrete metrics.

---

### 2:20–2:30 | Closing

**Visual:** Return to the title card. Add the GitHub URL and live demo URL below the project title. A "Built for research and education" tagline appears at the bottom.

**Narration:**
> "Full source code, documentation, and a live demo link are in the description below. This was built as a research and engineering portfolio project — it is not intended for clinical diagnosis or patient monitoring. Thanks for watching."

**Production Notes:**
- End cleanly. No "like and subscribe" — this is a professional demo, not a vlog.
- Hold the final card for 3 seconds after narration ends, then cut to black.
- Total runtime target: 2:25–2:35. If you're over 2:45, cut words from the architecture or classification sections first.

---

## Part 2 — Recording Setup Guide

### Software

| Tool | Purpose |
| :--- | :--- |
| **OBS Studio** (free) | Screen capture + audio recording |
| **Terminal** (Windows Terminal or iTerm2) | Show backend boot logs |
| **Browser** (Chrome, dark mode, no extensions visible) | Dashboard + PDF viewer |
| **Audacity** (optional) | Post-recording audio cleanup |

### OBS Configuration

**Canvas Settings:**
- Resolution: `1920×1080`
- Frame Rate: `30 fps`
- Output Format: `MKV` (remux to MP4 after recording for safety)
- Encoder: `x264` or `NVENC` (if GPU available)
- Rate Control: `CRF 18` (high quality, manageable file size)

**Audio:**
- Input: Use an external microphone if available. Laptop mics pick up fan noise.
- Sample Rate: `48 kHz`
- Noise Gate: Enable in OBS filters (Close: -32dB, Open: -26dB)
- If your environment is noisy, record narration separately in Audacity and sync in post.

**Scene Setup — 3 Scenes:**

| Scene | Sources | When Used |
| :--- | :--- | :--- |
| `Title Card` | Image source (title slide PNG) | 0:00–0:10, 2:20–2:30 |
| `Terminal + Browser` | Window Capture (Terminal, top-left 30%) + Window Capture (Browser, remaining 70%) | 0:45–1:00 (boot sequence) |
| `Browser Full` | Window Capture (Browser only, full screen) | 1:00–2:05 |

**Hotkeys (configure in OBS Settings → Hotkeys):**
- `F1` → Switch to Title Card scene
- `F2` → Switch to Terminal + Browser scene
- `F3` → Switch to Browser Full scene
- `F9` → Start Recording
- `F10` → Stop Recording

### Browser Preparation

1. **Hide all browser chrome:**
   - Press `F11` for fullscreen, OR
   - Use Chrome's `Ctrl+Shift+F` (presentation mode) if available
   - Disable the bookmarks bar: `Ctrl+Shift+B`

2. **Close all other tabs.** Only one tab open: your dashboard at `http://localhost:3000`.

3. **Zoom level:** Set browser zoom to `110%` for text readability at 1080p.

4. **Font rendering:** If on Windows, enable ClearType. On macOS, font rendering is fine by default.

5. **Dark mode:** Ensure both OS and browser are in dark mode for visual consistency.

6. **Notifications:** Turn off ALL system notifications (Windows: Focus Assist → Alarms Only).

### Showing a Live WebSocket Connection

This is critical. The viewer must see the connection happen, not a pre-connected state.

**Sequence to follow during recording:**

1. Start with the browser open to `localhost:3000`. The dashboard should show "Disconnected" or "Connecting..." state.
2. Switch to the terminal scene. Run `docker compose up --build`.
3. Wait for the backend boot log to show:
   ```
   INFO: ONNX model loaded from backend/models/ecg_model.onnx
   INFO: Uvicorn running on http://0.0.0.0:8000
   ```
4. Switch to the browser scene. The WebSocket status should flip from "Connecting..." to "Connected" (green).
5. The waveform begins streaming. This live transition is the money shot.

**Fallback:** If the connection happens too fast to capture, you can start Docker before recording and simply refresh the browser page during recording to show the reconnection sequence.

### Font Size Recommendations

| Element | Minimum Size at 1080p |
| :--- | :--- |
| Terminal text | `16px` or larger (increase terminal font to 14pt+) |
| Browser body text | `16px` (default is usually fine at 110% zoom) |
| Code blocks / logs | `14px` minimum |
| Slide text | `28px` minimum for body, `48px` for titles |

**Rule of thumb:** If you can't read it on your phone screen when the video is playing, it's too small.

### Post-Production

1. **Remux MKV → MP4:** In OBS: `File → Remux Recordings → Select MKV → Remux`.
2. **Trim dead air:** Use a free editor (Shotcut, DaVinci Resolve) to cut any pauses longer than 2 seconds.
3. **Audio cleanup:** If needed, run the audio through Audacity's Noise Reduction filter (get noise profile from a silent segment, then apply).
4. **Thumbnail:** Use your `dashboard-overview.png` screenshot with the project title overlaid in large text.

### Upload & Distribution

| Destination | Settings | Purpose |
| :--- | :--- | :--- |
| **YouTube** | Unlisted, 1080p, description includes GitHub link + live demo URL | Primary host for embedding |
| **README.md** | Add YouTube embed: `[![Demo Video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/VIDEO_ID)` | Repository landing page |
| **LinkedIn** | Native upload (not YouTube link) for better reach. Use the 200-word post from career assets. | Professional networking |
| **Portfolio site** | Embed YouTube iframe or link | Personal branding |

### YouTube Description Template

```
Real-Time AI-Driven ECG Arrhythmia Analysis Platform

A production-grade system for automated cardiac arrhythmia detection using:
• 1D CNN trained on MIT-BIH Arrhythmia Database (98.2% accuracy, 0.946 macro-F1)
• ONNX Runtime inference (<1ms/beat on CPU)
• Real-time WebSocket streaming at 360 Hz
• DSP preprocessing: Butterworth bandpass + Pan-Tompkins R-peak detection
• Grad-CAM clinical explainability
• Automated PDF report generation

GitHub: https://github.com/pr6thv3/ecg-ai-platform
Live Demo: [URL]

Stack: FastAPI, Next.js, PyTorch, ONNX Runtime, Docker, Prometheus, GitHub Actions

⚠️ This is a research and portfolio project. Not intended for clinical use.

#MachineLearning #DeepLearning #ECG #MedicalAI #FastAPI #NextJS #ONNX #PortfolioProject
```

---

## Pre-Recording Checklist

- [ ] Docker desktop running, containers built and tested
- [ ] Backend starts without errors, ONNX model loads successfully
- [ ] Dashboard loads at `localhost:3000`, WebSocket connects
- [ ] ECG waveform streams for at least 60 seconds without freezing
- [ ] At least one V (PVC) and one A (APB) beat appears in classification feed
- [ ] Grad-CAM inspector opens and renders saliency overlay
- [ ] PDF report generates and displays correctly
- [ ] OBS scenes configured and hotkeys tested
- [ ] Microphone levels checked (speak a test sentence, verify no clipping)
- [ ] System notifications disabled
- [ ] Browser: dark mode, fullscreen, no extensions, 110% zoom
- [ ] Terminal: font size 14pt+, dark theme
- [ ] Phone on silent
- [ ] Rehearsed full script at least twice with timer
