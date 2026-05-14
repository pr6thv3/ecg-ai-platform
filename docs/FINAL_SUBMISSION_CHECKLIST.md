# Final Submission Checklist

## Required Before Public Submission

- `docker compose up --build` starts backend and frontend.
- `http://localhost:8000/health` is reachable.
- `http://localhost:3000` renders the dashboard.
- WebSocket stream connects and emits valid beat payloads.
- `/analyze` returns a valid classification response for a 360-sample input.
- `/report/generate` returns a non-empty PDF for a valid session payload.
- Frontend lint, typecheck, tests, build, and audit pass.
- Backend pytest and coverage pass in Python 3.10 or Docker.
- README keeps the research-use disclaimer visible.
- Public copy does not assert clinical use, diagnosis, or unverified benchmark numbers.

## Optional Before Portfolio Launch

- Add updated screenshots from the current UI.
- Add a short demo video using `docs/DEMO_VIDEO_SCRIPT.md`.
- Add a Playwright smoke test for dashboard, WebSocket, analyze, and report generation.
- Generate MIT-BIH evaluation, leakage-check, and benchmark artifacts.
- Add a deployed Vercel URL only after the Render backend and frontend connection are verified.

## Sign-Off

| Area | Status |
| :--- | :--- |
| Local backend/frontend run | Verified with Python 3.10 backend image and local Next dev server |
| Frontend verification | Passed locally |
| Backend syntax verification | Passed locally |
| Backend pytest in target runtime | Passed in Python 3.10 Docker runner |
| Public deployment | Requires deployer credentials and secrets |
| Benchmark artifacts | Pending |
| Clinical disclaimer | Present |
