# Error Analysis

This report is for technical model debugging only. It is not medical reasoning or clinical validation.

- Source: `mitbih`
- Real MIT-BIH: `True`
- Total samples: `11749`
- Misclassified samples: `5315`
- High-confidence wrong predictions: `2798`
- Low-confidence predictions: `5002`

## Worst Classes

- `L`: failure rate `1.0000` (1457/1457)
- `R`: failure rate `0.9707` (1854/1910)
- `A`: failure rate `0.8561` (119/139)
- `N`: failure rate `0.2547` (1872/7349)
- `V`: failure rate `0.0145` (13/894)

## False Negative Focus

- `N`: false negative rate `0.2547` (1872/7349)
- `V`: false negative rate `0.0145` (13/894)
- `A`: false negative rate `0.8561` (119/139)
- `L`: false negative rate `1.0000` (1457/1457)
- `R`: false negative rate `0.9707` (1854/1910)
