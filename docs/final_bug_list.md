# Final Bug List

## Resolved In Current Working Tree

| Area | Resolution |
| :--- | :--- |
| Backend model path | Default runtime now loads `models/ecg_cnn.onnx`. |
| Model hash | `MODEL_SHA256` validation and model manifest added. |
| Frontend localhost coupling | Dashboard uses `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_WS_URL`. |
| Frontend package audit | Next, ESLint, PostCSS, and lockfile updated; npm audit passes. |
| Frontend test contracts | Jest fixtures match the current stream payload. |
| Runtime schemas | `zod` schemas validate stream and explainability payloads. |
| WebSocket security | Origin, mode, pattern, and MIT-BIH record checks added. |
| Report security | Optional Bearer auth and size limits added. |
| Demo resilience | Model unavailable, backend offline, malformed payload, and explainability unavailable states added. |
| Documentation honesty | Public docs avoid unverified accuracy, latency, and live deployment claims. |
| Backend pytest | Python 3.10 Docker runner passed 39 tests with the 80% coverage gate. |

## Still Open

| Area | Required next action |
| :--- | :--- |
| MIT-BIH split | Persist record-level splits and leakage-check artifact. |
| Benchmarks | Generate reproducible accuracy, F1, latency, and throughput artifacts. |
| E2E smoke | Add Playwright or Cypress coverage for dashboard, WS, analyze, and PDF report. |
