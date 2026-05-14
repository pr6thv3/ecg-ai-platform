# Verification Runbook

Use this runbook for a cold-start check of the ECG AI Platform.

## 1. Start The Stack

```bash
git clone https://github.com/pr6thv3/ecg-ai-platform.git
cd ecg-ai-platform
cp .env.example .env
docker compose up --build
```

Expected:

- Backend listens on `0.0.0.0:8000`.
- Frontend listens on `0.0.0.0:3000`.
- `/health` is reachable even if the model cannot load.

## 2. Backend Checks

Health:

```bash
curl http://localhost:8000/health
```

Expected for a healthy model:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "onnx"
}
```

Expected for a missing or invalid model:

```json
{
  "status": "model_unavailable",
  "model_loaded": false
}
```

Analyze:

```bash
python - <<'PY'
import json
import urllib.request

payload = json.dumps({"beat_window": [0.0] * 360}).encode()
request = urllib.request.Request(
    "http://localhost:8000/analyze",
    data=payload,
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(request).read().decode())
PY
```

Metrics:

```bash
curl -u admin:$METRICS_AUTH_TOKEN http://localhost:8000/metrics
```

Report generation:

```bash
python - <<'PY'
import json
import urllib.request

payload = {
    "patient_metadata": {"id": "QA-123", "age": 0, "gender": "unspecified", "session_date": "2026-05-13"},
    "signal_metadata": {"duration_sec": 60, "sampling_rate": 360, "snr_before": 0, "snr_after": 0},
    "beat_statistics": {
        "total_beats": 2,
        "class_distribution": {"N": 1, "V": 1, "A": 0, "L": 0, "R": 0},
        "dominant_rhythm": "N"
    },
    "anomaly_events": [],
    "model_metrics": {"average_confidence": 0.9, "low_confidence_beats": 0}
}
request = urllib.request.Request(
    "http://localhost:8000/report/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
data = urllib.request.urlopen(request).read()
print(len(data))
PY
```

Expected: the byte count is greater than zero and the content type is `application/pdf`.

## 3. Frontend Checks

Open `http://localhost:3000`.

Expected:

- The connection badge changes from connecting to live stream active.
- The ECG chart starts drawing samples.
- The metrics row updates.
- The beat classification panel receives beat rows.
- Clicking a beat opens the Grad-CAM inspector. If PyTorch Grad-CAM is unavailable, the modal shows a clear unavailable state.
- The Generate PDF button downloads a non-empty PDF when report auth is disabled, or shows an authorization error when report auth is enabled.

## 4. CI Checks

Run:

```bash
cd frontend
npm ci
npm run lint
npm run typecheck
npm test -- --runInBand
npm run build
npm audit --audit-level=moderate
```

Run backend tests in Python 3.10:

```bash
cd backend
pytest --cov=. --cov-fail-under=80
```

## 5. Pass Criteria

- No frontend build, lint, type, audit, or unit-test failures.
- Backend health is reachable.
- ONNX model hash matches when `MODEL_SHA256` is set.
- Dashboard has visible non-crashing states for backend offline, malformed stream payload, and explainability unavailable.
- Documentation does not claim benchmark or clinical performance numbers without generated artifacts.
