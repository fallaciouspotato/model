# ASTRA Traffic Priority Model

This project cleans the Bengaluru traffic event dataset and trains a prediction model aligned with the ASTRA hackathon plan.

## What The Pipeline Does

- Cleans `UncleanedDataset.csv`
- Removes empty columns and normalizes categorical fields
- Parses event timestamps and creates time-based features
- Treats invalid `0,0` end coordinates as missing
- Trains a dependency-light categorical Naive Bayes model to predict `priority`
- Converts predicted high-priority probability into:
  - `congestion_risk_score`
  - `event_severity_score`
  - `impact_category`
- Exports corridor stress and junction risk rankings

## Outputs

Run:

```powershell
python train_astra_model.py
```

Generated files:

- `artifacts/cleaned_traffic_events.csv`
- `artifacts/astra_priority_model.json`
- `artifacts/model_metrics.json`
- `artifacts/corridor_stress_ranking.csv`
- `artifacts/junction_risk_ranking.csv`

## Current Model Result

Target: `priority` (`High` vs `Low`)

The latest train/test split produced:

- Accuracy: `0.9963`
- High-priority precision: `0.9980`
- High-priority recall: `0.9960`
- High-priority F1: `0.9970`
- ROC AUC: `1.0000`

The dataset has a very strong relationship between corridor/junction/location fields and priority, so these results should be validated with fresher data before operational use.
