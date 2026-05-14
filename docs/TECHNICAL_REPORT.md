# Technical Report

## Scope

ECG AI Platform is a production-style academic ML systems project for real-time ECG arrhythmia classification using MIT-BIH records. It is for research, education, and portfolio review only. It is not a regulated clinical device and must not be used for diagnosis, treatment, monitoring, or medical decision-making.

Out of scope: HIPAA/GDPR compliance, medical-device validation, hospital deployment, production load testing, full retraining infrastructure, and clinical reliability claims.

## System Components

| Layer | Implementation |
| :--- | :--- |
| Configuration | YAML config in `configs/default.yaml` with environment overrides in `src.config`. |
| Data | MIT-BIH WFDB inventory/loading from `datasets/mit-bih` with `.dat`, `.hea`, and `.atr` validation. Synthetic mode is disabled by default and must be explicitly enabled. |
| Split | Record-level train/validation/test split with leakage checks by `record_id`. |
| Validation | Missing files, incomplete records, empty signals, short signals, NaN/Inf, sampling-rate mismatch, label mismatch, class imbalance, and corrupted CSV checks. |
| Preprocessing | Bandpass filtering, peak detection, fixed-window beat extraction, and normalization. |
| Models | Config-selectable `baseline_cnn`, `resnet1d`, `inceptiontime`, and `cnn_lstm` PyTorch architectures. |
| Training | Weighted loss, optional weighted sampler, ECG augmentations, macro-F1 early stopping, LR scheduling, gradient clipping, mixed precision on CUDA, deterministic seed control, and checkpoint metadata. |
| Evaluation | Accuracy, macro/weighted F1, ROC-AUC where defined, per-class sensitivity/specificity, confusion matrix, prediction distribution, false negatives, low-confidence cases, ROC curves, and PR curves. |
| Thresholds | Validation-set per-class threshold tuning for macro F1, saved to `artifacts/evaluation/thresholds.json`. Inference falls back to argmax when thresholds are absent. |
| Error analysis | Misclassified cases, worst classes, high-confidence wrong predictions, low-confidence correct predictions, false-negative summary, waveform examples, and Markdown summary. |
| Robustness | Noise, baseline wander, amplitude scaling, short signal rejection, NaN/Inf rejection, corrupted CSV rejection, wrong sampling rate, and missing annotation checks. |
| Explainability | PyTorch saliency map with research-only disclaimer and fallback metadata. |
| API | FastAPI `/health`, `/model-info`, `/predict`, and `/predict-file` with clear validation and inference logging. |
| UI | Streamlit upload UI with local inference or `ECG_API_URL` backend mode, probabilities, metrics display, and disclaimer. |
| Docker | API image on `0.0.0.0:8000`; dataset and `.pt` checkpoints are mounted/provided externally. |
| CI | GitHub Actions for dependency install, pytest, smoke test, dataset validation in explicit demo mode, leakage check, robustness, and Docker build. |

## Data Provenance

The local workspace contains real MIT-BIH WFDB records:

```text
datasets/mit-bih/*.dat
datasets/mit-bih/*.hea
datasets/mit-bih/*.atr
```

Preparation result:

```text
real_mitbih: true
valid_records: 48
loaded_samples: 100012
skipped: 102-0, missing .dat and .hea
```

The split is record-level:

```text
train_records: 34
validation_records: 7
test_records: 7
leakage_check: passed
```

Primary artifacts:

```text
reports/evaluation/mitbih_preparation.json
reports/evaluation/dataset_validation.json
reports/evaluation/leakage_check.json
artifacts/evaluation/dataset_summary.json
artifacts/evaluation/splits.json
```

## Checkpoint Contract

`configs/default.yaml` points training, evaluation, and inference to:

```text
artifacts/models/best_model.pt
```

The checkpoint metadata includes:

- `model_type`
- `class_order`
- `class_mapping`
- `input_size`
- `sampling_rate`
- preprocessing configuration
- source and `real_mitbih`
- git commit hash when available
- training timestamp
- metrics summary

Inference validates model type, class order, class mapping, input size, and preprocessing metadata before serving predictions. It does not silently fall back to another checkpoint unless `ECG_ALLOW_MODEL_FALLBACK=true` or `model.allow_fallback_checkpoint=true` is explicitly set.

## Current Metrics

Current verified held-out MIT-BIH evaluation from `artifacts/models/best_model.pt` after threshold tuning:

```text
accuracy: 0.6284
macro_f1: 0.2824
weighted_f1: 0.6057
roc_auc_ovr_macro: 0.6663
```

Per-class F1:

```text
N: 0.8975
V: 0.4527
A: 0.0014
L: 0.0000
R: 0.0606
```

Threshold tuning improved validation macro F1 from `0.3025` to `0.4276`, but held-out macro F1 remains weak. The anti-collapse check passes because all five classes are predicted and the largest prediction share is `63.10%`, but the classifier is not medically reliable. `L` recall is zero, and `A`/`R` false negatives remain high.

## Deployment Notes

Local API:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8010
```

Docker API:

```bash
docker build -t ecg-ai-system .
docker run -p 8001:8000 ecg-ai-system
```

Render/Railway start command:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

No public deployment URLs are documented until live backend/UI connectivity is verified.
