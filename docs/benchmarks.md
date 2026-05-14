# Benchmark And Verification Notes

This document records how performance and readiness numbers should be generated. It intentionally avoids static accuracy or latency claims unless they are produced by a reproducible command and committed as an artifact.

## Targets

| Area | Target |
| :--- | :--- |
| ONNX CPU inference | p95 less than 200 ms per 360-sample beat |
| Dashboard startup | First valid stream payload within 5 seconds of page load |
| MIT-BIH evaluation | Macro F1 at least 0.85 on a persisted, non-leaking test split |
| Frontend security audit | `npm audit --audit-level=moderate` returns zero vulnerabilities |
| Local deployment | `docker compose up --build` starts backend and frontend without manual patches |

## Current Verified Checks

| Check | Result |
| :--- | :--- |
| Frontend dependency install | `npm ci` passed |
| Frontend lint | `npm run lint` passed |
| Frontend typecheck | `npm run typecheck` passed |
| Frontend unit tests | `npm test -- --runInBand` passed, 26 tests |
| Frontend production build | `npm run build` passed |
| Frontend dependency audit | `npm audit --audit-level=moderate` passed with zero vulnerabilities |
| Backend pytest and coverage | Python 3.10 Docker runner passed 39 tests with 80.05% coverage |
| Backend Python syntax | `py -m compileall backend` passed |
| Docker Compose config | `docker compose config` passed |

## Commands

Frontend:

```bash
cd frontend
npm ci
npm run lint
npm run typecheck
npm test -- --runInBand
npm run build
npm audit --audit-level=moderate
```

Backend:

```bash
cd backend
pytest --cov=. --cov-fail-under=80
python scripts/benchmark_inference.py
python scripts/system_benchmark.py
python scripts/evaluate_mitbih.py
python scripts/run_ablation.py
```

Docker:

```bash
docker compose up --build
curl http://localhost:8000/health
```

## Artifact Rules

- Benchmark output must include commit SHA, Python version, Node version, Docker version, OS, and command line.
- MIT-BIH train, validation, and test record IDs must be persisted with the output.
- Any published model result must state whether it is verified by a run artifact or inferred from code/configuration.
- Public README claims must be updated only after the benchmark artifact is generated.
