# Production Deployment Checklist — ECG AI Platform

**Version:** 1.0.0
**Date:** 2026-05-12

---

## 1. Pre-Deployment Verification

| # | Check | Command / Action | Expected Result | Status |
|:--|:------|:-----------------|:----------------|:-------|
| 1 | Backend builds locally | `cd backend && pip install -r requirements.txt` | Zero install errors | [ ] |
| 2 | Frontend builds locally | `cd frontend && npm ci && npm run build` | Zero TypeScript or build errors | [ ] |
| 3 | Docker cold start | `docker compose up --build` | Both containers healthy within 60s | [ ] |
| 4 | ONNX model loads | Check backend logs for `Loaded ONNX model` | No mock/fallback warnings | [ ] |
| 5 | `/health` returns correct state | `curl localhost:8000/health` | `model_loaded: true, model_type: onnx` | [ ] |
| 6 | WebSocket streams | Open `localhost:3000`, check connection indicator | Green "Connected" status | [ ] |
| 7 | Backend tests pass | `cd backend && pytest --cov=. --cov-fail-under=80` | 80%+ coverage, zero failures | [ ] |
| 8 | Frontend tests pass | `cd frontend && npm test` | All suites pass | [ ] |

---

## 2. Render Backend Deployment

### 2.1 Environment Variables

Set these in the Render dashboard under your web service's **Environment** tab:

| Variable | Value | Notes |
|:---------|:------|:------|
| `MODEL_PATH` | `/opt/model/ecg_cnn.onnx` | Must match the Render disk mount path |
| `ALLOWED_ORIGINS` | `https://ecg-ai-platform.vercel.app,http://localhost:3000` | Comma-separated, no trailing slashes |
| `USE_ONNX` | `true` | Forces ONNX Runtime (not PyTorch) |
| `LOG_LEVEL` | `info` | Set to `debug` only for troubleshooting |
| `METRICS_AUTH_TOKEN` | `(generate a secure random string)` | Secures `/metrics` endpoint |
| `PYTHON_VERSION` | `3.10.0` | Render build setting |

### 2.2 Render Disk Setup

1. In the Render dashboard, add a **Disk** to the web service.
2. **Name:** `model-storage`
3. **Mount Path:** `/opt/model`
4. **Size:** 1 GB
5. After the first deploy, SSH into the Render shell and upload the ONNX model:
   ```bash
   # From the Render shell tab
   curl -L -o /opt/model/ecg_cnn.onnx <YOUR_MODEL_DOWNLOAD_URL>
   ```
6. Restart the service. Verify `/health` returns `model_loaded: true`.

### 2.3 Render Configuration (render.yaml)

The `render.yaml` at the project root defines:
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn api.main:app -k uvicorn.workers.UvicornWorker --workers 2 --bind 0.0.0.0:$PORT`
- **Health check:** `GET /health` every 30s
- Render natively supports WebSockets on web services — no additional proxy config needed.

### 2.4 Post-Deploy Verification

```bash
# Replace with your actual Render URL
BACKEND=https://ecg-ai-backend.onrender.com

curl -s $BACKEND/health | python -m json.tool
# Expect: model_loaded=true, model_type=onnx

curl -s -u admin:YOUR_TOKEN $BACKEND/metrics | head -20
# Expect: Prometheus text format with ecg_inferences_total
```

### 2.5 Known Issue: Render Cold Starts

Render free-tier services sleep after 15 minutes of inactivity. The first request after sleep takes 30–60 seconds. Mitigations:
- The frontend should display a "Backend is waking up..." message during the reconnection window.
- The `/health` endpoint is lightweight and wakes the service quickly.
- For paid tier: enable "Always On" in Render settings.

---

## 3. Vercel Frontend Deployment

### 3.1 Environment Variables

Set these in the Vercel dashboard under **Settings → Environment Variables**:

| Variable | Value | Notes |
|:---------|:------|:------|
| `NEXT_PUBLIC_API_URL` | `https://ecg-ai-backend.onrender.com` | REST API base URL |
| `NEXT_PUBLIC_WS_URL` | `wss://ecg-ai-backend.onrender.com` | WebSocket URL (must use `wss://` in production) |

### 3.2 Vercel Configuration (vercel.json)

Located at `frontend/vercel.json`. Key settings:
- **Framework:** `nextjs`
- **Build command:** `npm run build`
- **Rewrites:** `/api/*` and `/ws/*` are proxied to the Render backend
- **Security headers:** `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`

### 3.3 Deployment Steps

1. Connect the GitHub repository to Vercel.
2. Set the **Root Directory** to `frontend`.
3. Add environment variables from section 3.1.
4. Deploy. Vercel auto-builds on every push to `main`.

### 3.4 Post-Deploy Verification

1. Open the Vercel URL in an incognito browser window.
2. Verify the dashboard loads (not a blank page or build error).
3. Check the WebSocket connection status indicator shows "Connected" (green).
4. Verify the ECG waveform is actively streaming.
5. Click a beat classification row — Grad-CAM panel should open.
6. Click "Generate Report" — PDF should download.

---

## 4. Production Failure Modes & Fixes

### 4.1 CORS Failures

**Symptom:** Browser console shows `Access-Control-Allow-Origin` errors. Dashboard loads but WebSocket/REST calls fail.

**Fix:** Ensure `ALLOWED_ORIGINS` on Render includes the exact Vercel deployment URL (with `https://`, no trailing slash). Restart the Render service after changing env vars.

### 4.2 WebSocket Connection Fails in Production

**Symptom:** Dashboard shows "Disconnected". Works locally but not on Vercel/Render.

**Fix checklist:**
1. Verify `NEXT_PUBLIC_WS_URL` uses `wss://` (not `ws://`) — required for HTTPS pages.
2. Verify Render service is running (not sleeping). Hit `/health` first to wake it.
3. Check browser console for mixed-content warnings (HTTPS page trying to connect to HTTP WebSocket).

### 4.3 ONNX Model File Missing on Render

**Symptom:** `/health` returns `model_loaded: false`. Backend logs show `FileNotFoundError`.

**Fix:** The ONNX file must be manually uploaded to the Render disk at the `MODEL_PATH` location. See section 2.2.

### 4.4 Next.js SSR Hydration Errors

**Symptom:** Console shows `Hydration failed because the initial UI does not match what was rendered on the server`.

**Fix:** The dashboard component uses a `mounted` state guard (`if (!mounted) return null;`) to prevent Recharts from initializing during server-side rendering. If this error appears, verify the guard is present in `frontend/app/dashboard/page.tsx`.

### 4.5 PDF Report Returns 0 Bytes

**Symptom:** "Generate Report" downloads a file but it's empty or corrupt.

**Fix:** Verify `reportlab` is in `requirements.txt`. Check that `report_router.py` sets `Content-Type: application/pdf` and uses `StreamingResponse` correctly.

---

## 5. Production URLs

| Service | URL | Status |
|:--------|:----|:-------|
| Backend API | `https://ecg-ai-backend.onrender.com` | [ ] Verified |
| Backend Health | `https://ecg-ai-backend.onrender.com/health` | [ ] Verified |
| Frontend Dashboard | `https://ecg-ai-platform.vercel.app` | [ ] Verified |
| GitHub Repository | `https://github.com/pr6thv3/ecg-ai-platform` | [ ] Verified |

---

## 6. Deployment Sign-Off

```
Backend deployed and healthy:    [ ] YES  [ ] NO
Frontend deployed and rendering: [ ] YES  [ ] NO
WebSocket streaming in prod:     [ ] YES  [ ] NO
ONNX inference confirmed:        [ ] YES  [ ] NO
CORS configured correctly:       [ ] YES  [ ] NO
All env vars set:                [ ] YES  [ ] NO

Deployed by: ___________________  Date: ___________
```
