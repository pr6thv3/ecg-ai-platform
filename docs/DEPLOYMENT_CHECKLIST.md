# Deployment Checklist

## Local Gate

| Check | Command | Expected |
| :--- | :--- | :--- |
| Compose config | `docker compose config` | Valid config |
| Frontend install | `cd frontend && npm ci` | No install errors |
| Frontend lint | `cd frontend && npm run lint` | Zero warnings or errors |
| Frontend typecheck | `cd frontend && npm run typecheck` | Route types generate and TypeScript reports zero errors |
| Frontend tests | `cd frontend && npm test -- --runInBand` | All tests pass |
| Frontend build | `cd frontend && npm run build` | Production build succeeds |
| Backend syntax | `py -m compileall backend` | No syntax errors |
| Docker run | `docker compose up --build` | Backend and frontend become healthy |

## Render Backend

Use the root `render.yaml`.

Required environment:

| Variable | Value |
| :--- | :--- |
| `MODEL_PATH` | `models/ecg_cnn.onnx` |
| `MODEL_SHA256` | `4d796c63201114a5bfeb128fb32d78da0c13b6b92385682cddf4d7b6260c8c25` |
| `ALLOWED_ORIGINS` | Comma-separated deployed frontend origins |
| `USE_ONNX` | `true` |
| `METRICS_AUTH_TOKEN` | Secure random value |
| `REPORT_AUTH_TOKEN` | Secure random value, if report/session endpoints should be private |
| `PYTHON_VERSION` | Python 3.10 runtime |

Post-deploy checks:

```bash
curl https://ecg-ai-backend.onrender.com/health
curl -u admin:$METRICS_AUTH_TOKEN https://ecg-ai-backend.onrender.com/metrics
```

Expected health fields:

- `status` is `ok`
- `model_loaded` is `true`
- `model_type` is `onnx`
- `model_sha256` matches `MODEL_SHA256`

## Vercel Frontend

Use `frontend/` as the Vercel project root.

Required environment:

| Variable | Value |
| :--- | :--- |
| `NEXT_PUBLIC_API_URL` | Render backend HTTPS URL |
| `NEXT_PUBLIC_WS_URL` | Render backend WSS URL |

Post-deploy checks:

- Dashboard renders without a blank page.
- WebSocket status changes from connecting to live stream active.
- Waveform and beat list update.
- Beat inspector opens and either renders Grad-CAM or shows a clear unavailable state.
- Generate PDF returns a non-empty PDF or a clear authorization error if `REPORT_AUTH_TOKEN` is enabled.

## Failure Checks

- If the dashboard is disconnected, verify `NEXT_PUBLIC_WS_URL` uses `wss://`.
- If CORS or WebSocket origin checks fail, add the exact Vercel URL to `ALLOWED_ORIGINS`.
- If `/health` returns `model_unavailable`, verify `MODEL_PATH`, `MODEL_SHA256`, and the committed ONNX artifact.
- If metrics return `401`, use HTTP Basic Auth with username `admin` and password `METRICS_AUTH_TOKEN`.
- If report endpoints return `401`, send `Authorization: Bearer $REPORT_AUTH_TOKEN`.
