# High-Budget MIT-BIH Research Experiments

These experiments are grouped-record MIT-BIH research runs. They do not change the preserved production baseline checkpoint unless the explicit promotion rule passes.

- Generated: 2026-05-15T06:28:38.422132+00:00
- Budget label: larger_cpu_budget_epochs_5_cap_2500
- Baseline macro F1: 0.2824
- Promoted: True

| Experiment | Status | Macro F1 | Accuracy | N F1 | V F1 | A F1 | L F1 | R F1 | Promotion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| resnet1d_maxabs_focal_balanced | completed | 0.4567 | 0.7693 | 0.8699 | 0.7936 | 0.0000 | 0.0000 | 0.6200 | no |
| resnet1d_zscore_focal_balanced | completed | 0.3211 | 0.7003 | 0.8361 | 0.7106 | 0.0000 | 0.0000 | 0.0588 | no |
| inceptiontime_maxabs_focal_balanced | completed | 0.2949 | 0.6745 | 0.9512 | 0.5221 | 0.0010 | 0.0000 | 0.0000 | no |
| cnn_lstm_maxabs_focal_balanced | completed | 0.3295 | 0.5520 | 0.8390 | 0.7493 | 0.0153 | 0.0026 | 0.0414 | yes |

Best candidate: `cnn_lstm_maxabs_focal_balanced`.

## Promotion Notes

- Copied artifacts\models\candidates\cnn_lstm_maxabs_focal_balanced.pt to artifacts/models/best_model_research.pt after passing the promotion rule.
