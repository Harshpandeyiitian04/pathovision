# PathoVision Final Results

This file is a generated-result template. It intentionally contains no fabricated metrics.

Run:

```bash
python scripts/prepare_data.py --config configs/default.yaml --validate-images
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
python scripts/generate_final_report.py
```

Then inspect:

- `results/metrics/test_metrics.json`
- `results/metrics/aggregation_comparison.csv`
- `results/metrics/by_magnification.csv`
- `results/metrics/final_report.json`
- `results/predictions/test_predictions.csv`
- `results/figures/`

Only measured values from those files should be added to a public README.
