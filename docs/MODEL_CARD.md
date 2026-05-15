# Model Card

## Model Details

| Field | Value |
| :--- | :--- |
| Project | ECG AI Platform |
| Model artifact | `artifacts/models/best_model.pt` |
| Model type | `baseline_cnn` |
| Input shape | 360-sample single-channel ECG beat/window |
| Runtime | PyTorch for local training/evaluation/inference |
| Dataset | MIT-BIH Arrhythmia Database loaded from local WFDB files |
| Classes | `N`, `V`, `A`, `L`, `R` |
| Current status | Functional academic ML baseline with weak held-out performance |

## Intended Use

This model is intended for research, education, portfolio review, and ML systems engineering demonstration. It shows a complete ECG ML workflow: WFDB data loading, leakage-safe splitting, training, evaluation, threshold tuning, explainability, API serving, Streamlit demo, Docker packaging, and CI tests.

## Not Intended Use

This model is not intended for clinical diagnosis, treatment, monitoring, triage, patient care, or medical decision-making. It is not a medical device and has not been clinically validated.

## Training Data

The project uses 48 complete local MIT-BIH WFDB records under:

```text
datasets/mit-bih/
```

Each record requires matching:

```text
<record>.dat
<record>.hea
<record>.atr
```

The extra local annotation `102-0.atr` is skipped because matching `.dat` and `.hea` files are not present.

## Class Mapping

```text
N -> 0
V -> 1
A -> 2
L -> 3
R -> 4
```

Class order is stored in checkpoint metadata and validated before inference.

## Evaluation

Current verified held-out MIT-BIH test metrics from `artifacts/models/best_model.pt` after threshold tuning:

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

Threshold tuning improved validation macro F1 from `0.3025` to `0.4276`, but held-out test macro F1 remained weak. The anti-collapse check passed because all five classes are predicted and the largest prediction share is `63.10%`, but `L` recall is still zero and `A`/`R` false negatives remain high.

## Research Checkpoint

A limited-budget model comparison produced a `resnet1d` checkpoint with macro F1 `0.3119` and `L` F1 `0.6823`. It is saved locally as:

```text
artifacts/models/best_model_research.pt
```

This checkpoint is not promoted to the main model because `A`, `V`, and `R` collapse to zero F1 in that limited run. It is useful for research follow-up only.

## Limitations

- Current model quality is weak and not clinically reliable.
- Class imbalance and record-level generalization remain difficult.
- `L`, `A`, and `R` performance need substantial improvement.
- Limited-budget architecture experiments show class-specific tradeoffs rather than a stable quality improvement.
- The local training configuration uses a per-class cap for practical runtime; stronger training should run uncapped on better hardware.
- Saliency maps are technical debugging aids, not medical explanations.
- Docker does not include MIT-BIH files or `.pt` checkpoints; those must be provided externally.

## Ethical And Medical Disclaimer

This project is for research and education only. It must not be used to diagnose, treat, monitor, or make decisions about any person. Any future clinical use would require expert clinical review, stronger validation, regulatory analysis, privacy/security work, and controlled deployment processes.

## Recommended Next Work

- Train `resnet1d` and `inceptiontime` uncapped on stronger hardware.
- Run grouped fold validation to estimate record-level generalization stability.
- Investigate `L`, `A`, and `R` false negatives with waveform-level review.
- Add calibrated probabilities and threshold review by class.
- Export a verified ONNX artifact from the best PyTorch checkpoint for lightweight API deployment.
