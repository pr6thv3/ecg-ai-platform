# ECG AI Platform

Production-style ECG arrhythmia ML system with a modular Python pipeline, FastAPI inference API, Streamlit UI, Docker packaging, MIT-BIH WFDB integration, leakage-safe record splitting, evaluation artifacts, robustness checks, explainability outputs, and monitoring logs.

This is a research, education, and portfolio project. It is not a regulated clinical product and must not be used for diagnosis, treatment, patient monitoring, triage, or medical decision-making.

## Current Status

- Real MIT-BIH WFDB files are loaded locally from `datasets/mit-bih`.
- The loader auto-detects complete `.dat`, `.hea`, and `.atr` triples.
- `48` complete records are usable; `102-0` is skipped because it is missing matching `.dat` and `.hea` files.
- Current reports are real MIT-BIH reports, not synthetic fallback reports.
- The model is functional but still weak. Current post-threshold test macro F1 is `0.2824`; `L` recall is still `0.0`, and `A`/`R` false negatives remain high.

## Architecture

```text
MIT-BIH WFDB / CSV / TXT
  -> signal validation
  -> preprocessing and normalization
  -> beat/window extraction
  -> model factory: baseline_cnn | resnet1d | inceptiontime | cnn_lstm
  -> PyTorch or explicit ONNX fallback inference
  -> optional per-class thresholding
  -> probabilities, confidence, timing
  -> API / CLI / Streamlit UI
  -> monitoring, evaluation, error analysis, robustness, explainability
```

## Dataset Setup

Place MIT-BIH Arrhythmia Database WFDB files here:

```text
datasets/mit-bih/
  100.dat
  100.hea
  100.atr
  ...
```

Each usable record must have matching `.dat`, `.hea`, and `.atr` files. Synthetic fallback is only allowed when explicitly enabled in config or CI environment variables, and synthetic outputs must never be cited as MIT-BIH performance.

Validate the dataset:

```bash
python -m scripts.prepare_mitbih --config configs/default.yaml
python -m scripts.validate_dataset --config configs/default.yaml
python -m scripts.leakage_check --config configs/default.yaml
```

## Training

Select architecture in `configs/default.yaml`:

```yaml
model:
  type: baseline_cnn  # baseline_cnn | resnet1d | inceptiontime | cnn_lstm
```

Train:

```bash
python -m scripts.train_model --config configs/default.yaml
```

Training outputs:

```text
artifacts/models/best_model.pt
artifacts/metrics/training_history.csv
artifacts/metrics/training_curves.png
artifacts/metrics/training_summary.json
```

Checkpoints include class order, class mapping, preprocessing config, model type, input size, sampling rate, git commit hash when available, training timestamp, and validation metrics summary.

## Evaluation

```bash
python -m scripts.evaluate_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
python -m scripts.tune_thresholds --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
```

Generated reports include:

```text
reports/evaluation/metrics_summary.json
reports/evaluation/classification_report.json
reports/evaluation/confusion_matrix.csv
reports/evaluation/confusion_matrix.png
reports/evaluation/prediction_distribution.csv
reports/evaluation/prediction_distribution.png
reports/evaluation/false_negatives.csv
reports/evaluation/low_confidence_predictions.csv
reports/evaluation/roc_curves.png
reports/evaluation/pr_curves.png
artifacts/evaluation/thresholds.json
artifacts/evaluation/dataset_summary.json
artifacts/evaluation/splits.json
```

Current verified post-threshold metrics from `artifacts/models/best_model.pt`:

```text
source: mitbih
real_mitbih: true
record_count: 48
accuracy: 0.6284
macro_f1: 0.2824
weighted_f1: 0.6057
roc_auc_ovr_macro: 0.6663
```

Threshold tuning improved validation macro F1 from `0.3025` to `0.4276`, but held-out test macro F1 remains weak. This is an engineering-complete baseline, not a reliable medical classifier.

## Inference

CLI:

```bash
python -m scripts.run_inference --input artifacts/samples/demo_ecg.csv --config configs/default.yaml
```

The inference response includes `predicted_class`, `confidence`, `probabilities`, `thresholds_used`, timing, model metadata, warnings, and the research-only disclaimer.

FastAPI:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8010
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-file`

Streamlit:

```bash
python -m streamlit run streamlit_app.py --server.headless true --server.port 8502
```

Set `ECG_API_URL=https://your-api-host` to make Streamlit call a deployed backend instead of the local Python pipeline.

## Optional Slow Workflows

```bash
python -m scripts.cross_validate --config configs/default.yaml
python -m scripts.compare_models --config configs/default.yaml
```

`cross_validate` performs grouped record-level checkpoint evaluation. `compare_models` trains selected architectures with the comparison training cap and reports the best by macro F1.

## Robustness, Explainability, Error Analysis

```bash
python -m scripts.error_analysis --config configs/default.yaml
python -m scripts.robustness_test --config configs/default.yaml
python -m scripts.explain_prediction --input artifacts/samples/demo_ecg.csv --config configs/default.yaml
python -m scripts.benchmark_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt --iterations 50
```

Saliency maps are technical model-debugging artifacts, not medical explanations.

## Docker

Build and run the API image:

```bash
docker build -t ecg-ai-system .
docker run -p 8001:8000 ecg-ai-system
```

The image excludes local `.pt` files and MIT-BIH data. It explicitly enables ONNX fallback through `ECG_ALLOW_MODEL_FALLBACK=true` and serves `backend/models/ecg_cnn.onnx`.

To serve a mounted PyTorch checkpoint, build an image that includes PyTorch dependencies or export the trained model to ONNX. Do not silently rely on a mismatched fallback model.

## Tests

```bash
python -m pytest -q
python -m scripts.smoke_test_pipeline --config configs/default.yaml
```

CI uses explicit environment flags for synthetic/no-checkpoint smoke mode and does not require MIT-BIH data or trained `.pt` files.

## Deployment

Backend start command for Render/Railway:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

Do not add public Render, Railway, Streamlit Cloud, or Vercel URLs until they are live and verified. A verified deployment means `/health` returns `status: "ok"`, the UI reaches the backend through environment variables, and no deployed page references localhost.

## Limitations

- Current model quality is not sufficient for medical use.
- The held-out split is leakage-safe by record ID, but this is still a compact academic ML engineering baseline.
- `L`, `A`, and `R` performance requires better modeling, split strategy validation, thresholding, and likely full uncapped training before performance claims.
- HIPAA, GDPR, regulatory validation, hospital deployment, and full production MLOps are out of scope.

## Documentation

- [Technical Report](docs/TECHNICAL_REPORT.md)
- [Testing And Validation](docs/TESTING_AND_VALIDATION.md)
- [Model Card](docs/MODEL_CARD.md)
