# Fix Backlog

## Done In Current Working Tree

- ONNX-first backend startup with optional sibling PyTorch checkpoint for Grad-CAM.
- `MODEL_SHA256` validation and `backend/models/model_manifest.json`.
- Graceful backend `model_unavailable` state.
- Env-driven frontend REST and WebSocket URLs.
- Frontend `zod` runtime validation for stream and explainability payloads.
- WebSocket origin validation and query allowlists.
- REST input length and finite-number validation.
- Metrics Basic Auth and report Bearer Auth hooks.
- Report payload size/event limits.
- Dashboard disconnected, malformed-payload, explainability-unavailable, and report-generation states.
- Frontend dependency upgrade, lockfile regeneration, lint/type/test/build/audit cleanup.
- Backend pytest and 80% coverage gate in a Python 3.10 Docker runner.
- CI Node upgrade for Next 16.
- Documentation cleanup to remove unverified benchmark and clinical claims.

## Remaining P1 Work

### 1. Generate ML reproducibility artifacts

Acceptance:

- Train, validation, and test split record IDs are persisted.
- Leakage check proves no `record_id` appears in more than one split.
- MIT-BIH license/provenance note is documented.
- Macro F1, per-class metrics, and confusion matrix are generated from the persisted test split.

Expected tests:

- Split manifest validation.
- Leakage check.
- ONNX/PyTorch parity check when a PyTorch checkpoint is available.

### 2. Publish measured performance artifacts

Acceptance:

- ONNX inference p50/p95/p99 and throughput are recorded with model hash and runtime versions.
- Per-stage backend timing covers preprocessing, inference, and WebSocket send.
- Browser smoke timing records first valid dashboard payload.
- Public docs are updated only from generated artifacts.

Expected tests:

- `backend/scripts/benchmark_inference.py`
- `backend/scripts/system_benchmark.py`
- Prometheus metric scrape verification.

### 3. Add end-to-end smoke automation

Acceptance:

- Playwright or Cypress starts the app stack or targets a configured local stack.
- Test verifies dashboard render, WebSocket connection, one valid stream payload, `/analyze` response shape, and non-empty PDF response.
- CI runs the smoke test on PRs or on a scheduled deployment-readiness workflow.

Expected tests:

- Desktop viewport smoke.
- Mobile viewport smoke.
- Backend-offline visible state.

## Remaining P2 Work

### 4. Decide report/session persistence strategy

Acceptance:

- If reports remain demo-only, docs explicitly state session JSON is process-local.
- If reports become durable, replace `SESSION_DB` with a real database or object store.

### 5. Add real user authentication if report data becomes private

Acceptance:

- Browser flows do not expose private bearer tokens as public environment variables.
- Report endpoints are protected by a user-auth strategy or kept admin-only.

### 6. Refresh screenshot and demo assets

Acceptance:

- Screenshots match the current UI.
- Demo script avoids unverified accuracy or latency claims.
- Public portfolio copy keeps the research-use disclaimer visible.

## Blocker Summary

| Area | Blocker |
| :--- | :--- |
| Local run | Backend and frontend verified with a Python 3.10 backend image plus local Next dev server; clean backend Docker rebuild remains slow. |
| Public deployment | Needs Render/Vercel secrets and production URLs configured by the deployer. |
| Real metrics | Needs generated benchmark artifacts. |
| Schema contracts | Runtime validation implemented; shared schema package remains optional future improvement. |
| ML reproducibility | Needs persisted split and leakage check artifacts. |
| Security hardening | Basic protections implemented; user auth remains future work for private report data. |
| Demo resilience | Implemented for main failure states; automated E2E smoke is pending. |
| Documentation honesty | Updated to avoid unverified metric and clinical claims. |
