# Final Bug List — ECG AI Platform

**Date:** 2026-05-12
**Auditor:** Release Engineering

---

## Critical (breaks startup, inference, WebSocket, dashboard, or deployment)

| # | Description | File | Severity | Status |
|---|-------------|------|----------|--------|
| 1 | `docker-compose.yml` defaults `MODEL_PATH` to `best_model.h5` (a Keras path) instead of the ONNX model path. If `.env` is missing, backend loads wrong file or crashes. | `docker-compose.yml:16` | Critical | **FIXED** |
| 2 | Verification runbook uses placeholder `<repository_url>` instead of real GitHub URL. A cold-start tester cannot follow the guide. | `docs/VERIFICATION_RUNBOOK.md:12` | Critical | **FIXED** |
| 3 | `render.yaml` ALLOWED_ORIGINS uses placeholder `your-vercel-app.vercel.app`. CORS will block all frontend requests in production. | `render.yaml:25` | Critical | **FIXED** |

## Medium (degrades experience but system still works)

| # | Description | File | Severity | Status |
|---|-------------|------|----------|--------|
| 1 | `.env.example` MODEL_PATH references `backend/models/ecg_model.onnx` but `docker-compose.yml` mounts to `/app/models/`. Inconsistent paths confuse setup. | `.env.example:3` + `docker-compose.yml:14` | Medium | **FIXED** |
| 2 | Root-level `test_ws_client.py` is a loose debugging script not part of any test suite. Clutters root directory. | `test_ws_client.py` | Medium | Noted |
| 3 | `docs/benchmarks.md` contains `[RUN: ...]` placeholders. Benchmark numbers must be populated after script execution. | `docs/benchmarks.md` | Medium | Deferred to benchmark run |

## Minor (cosmetic, non-blocking)

| # | Description | File | Severity | Status |
|---|-------------|------|----------|--------|
| 1 | Nginx service in `docker-compose.yml` is commented out but still defined. May confuse new contributors reading the compose file. | `docker-compose.yml:50-66` | Minor | Noted |
| 2 | `backend/__pycache__` directory exists in `inference/`. Should be gitignored (already in `.gitignore`, needs cleanup). | `backend/inference/__pycache__` | Minor | Noted |
