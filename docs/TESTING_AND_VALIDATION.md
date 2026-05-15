# Testing And Validation

## Verified Commands

These commands were run locally against the real MIT-BIH path unless noted otherwise:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.prepare_mitbih --config configs/default.yaml
python -m scripts.validate_dataset --config configs/default.yaml
python -m scripts.leakage_check --config configs/default.yaml
python -m scripts.smoke_test_pipeline --config configs/default.yaml
python -m scripts.train_model --config configs/default.yaml
python -m scripts.evaluate_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
python -m scripts.tune_thresholds --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
python -m scripts.diagnose_model_quality --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
python -m scripts.compare_models --config configs/default.yaml
python -m scripts.cross_validate --config configs/default.yaml
python -m scripts.error_analysis --config configs/default.yaml
python -m scripts.robustness_test --config configs/default.yaml
python -m scripts.explain_prediction --input artifacts/samples/demo_ecg.csv --config configs/default.yaml
python -m scripts.benchmark_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt --iterations 50
docker build -t ecg-ai-system .
```

Additional runtime checks:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8010
python -m streamlit run streamlit_app.py --server.headless true --server.port 8502
docker run -p 8001:8000 ecg-ai-system
```

The API `/health` endpoint returned `status: "ok"`, Streamlit returned HTTP 200, and the Docker container returned a healthy API response after startup.

## Test Coverage Areas

- Config loading and validation
- Signal preprocessing and segmentation
- MIT-BIH inventory loading and leakage report
- NaN/Inf, short signal, sampling-rate, and corrupted-file edge cases
- Model forward pass for `baseline_cnn`, `resnet1d`, `inceptiontime`, and `cnn_lstm`
- Checkpoint metadata mismatch detection
- Inference output schema and threshold use
- API health, model-info, and prediction routes
- Robustness acceptance/rejection paths

## Generated Artifacts

| Artifact | Purpose |
| :--- | :--- |
| `reports/evaluation/mitbih_preparation.json` | WFDB inventory and complete-record validation. |
| `reports/evaluation/dataset_validation.json` | Dataset source, class distribution, record count, and warnings. |
| `reports/evaluation/leakage_check.json` | Record-level leakage report. |
| `artifacts/evaluation/dataset_summary.json` | Dataset summary for GitHub/project review. |
| `artifacts/evaluation/splits.json` | Record-level split metadata. |
| `artifacts/metrics/training_history.csv` | Per-epoch training and validation metrics. |
| `artifacts/metrics/training_curves.png` | Loss and macro-F1 training curves. |
| `reports/evaluation/mitbih_evaluation_summary.json` | Current evaluation metrics and checkpoint metadata. |
| `reports/evaluation/classification_report.json` | Per-class precision, recall, F1, sensitivity, and specificity. |
| `reports/evaluation/confusion_matrix.csv` | Confusion matrix values. |
| `reports/evaluation/confusion_matrix.png` | Confusion matrix visualization. |
| `reports/evaluation/prediction_distribution.json` | Anti-collapse prediction distribution. |
| `reports/evaluation/false_negative_summary.json` | False-negative counts and rates by class. |
| `reports/error_analysis/summary.json` | Misclassification, confidence, and worst-class analysis. |
| `artifacts/evaluation/error_analysis.md` | Concise error-analysis summary. |
| `reports/robustness/robustness_summary.json` | Robustness cases and rejection behavior. |
| `reports/explainability/explanation_summary.json` | Explanation method, image, probabilities, and disclaimer. |
| `artifacts/metrics/benchmark_summary.json` | Inference latency benchmark summary. |
| `reports/benchmarks/latency_breakdown.json` | Report copy of the latency breakdown. |

## Current Validation Result

```text
source: mitbih
real_mitbih: true
record_count: 48
sample_count: 100012
leakage_check: passed
accuracy: 0.6284
macro_f1: 0.2824
weighted_f1: 0.6057
roc_auc_ovr_macro: 0.6663
```

The model is not collapsed to one class, but it is not yet a high-quality ECG classifier. `L` recall is zero, and `A`/`R` false-negative rates are high. These metrics are reported honestly and are not clinical-performance claims.

## Known Failed/Transient Checks

- A prior full uncapped training attempt exceeded the local 20-minute command timeout before producing a checkpoint. The current config uses a per-class cap for local reproducibility.
- The first Docker health probe at 8 seconds was too early and closed the connection. The same container passed `/health` after a longer startup wait, and the Dockerfile now has a startup healthcheck window.
- Direct `streamlit` executable launch did not stay up in this shell. `python -m streamlit run ...` passed with HTTP 200.
