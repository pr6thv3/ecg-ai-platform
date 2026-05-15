# Model Quality Diagnostics

This report is technical failure analysis only. It is not clinical validation.

- Source: `mitbih`
- Real MIT-BIH: `True`
- Record count: `48`
- Sample count: `100012`
- Window size: `360`
- Normalization: `maxabs`
- Argmax macro F1: `0.2769`
- Thresholded macro F1: `0.2824`

## Findings

- A has very low held-out support, so A metrics are high-variance and sensitive to record choice.
- L is effectively not recovered on the held-out records despite non-trivial train support.
- R has severe record-level generalization failure and is mostly predicted as another morphology class.
- Current thresholds improve or match held-out macro F1 relative to argmax.
- Training loss falls while validation macro F1 plateaus, indicating overfitting and record-level domain shift.
