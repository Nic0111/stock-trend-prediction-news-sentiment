# Model Performance dashboard

The presentation dashboard at `/model-performance` is read-only. It combines the
active stock-model registry, its frozen evaluation bundle and the active production
sentiment registry without changing any champion, threshold, calibration object or
runtime prediction.

## Refreshing the detailed stock diagnostics

Run the generator after intentionally changing the active stock registry or its
report bundle:

```powershell
.\.venv\Scripts\python.exe -m pipeline.generate_model_performance_artifact
```

The command writes
`results/model-performance/model-performance.json` atomically. Its sample-size
learning curves use five purged chronological development folds and the frozen
champion configuration. Calibration, confusion, ROC and precision-recall evidence
uses only the locked 2025 retrospective test.

For a fast UI/API smoke test that deliberately omits the expensive learning curves:

```powershell
.\.venv\Scripts\python.exe -m pipeline.generate_model_performance_artifact --skip-learning-curves
```

Missing sections are returned as unavailable; the API does not invent chart points.
The lightweight Stock Outlook endpoint remains separate from this full presentation
artifact.

## API

`GET /api/v1/system/model-performance` returns public presentation metadata only.
Filesystem paths, raw datasets, hashes and private artifact metadata are not exposed.

## Interpretation boundary

- Development CV covers 2015–2024 stock development evidence.
- The retrospective headline evaluation covers 2025.
- Stock shadow evidence covers 2026 where available and remains separate.
- Sentiment headline evidence is locked to the production 2015–2025 scope.
- Historical performance is not the accuracy or certainty of an individual live
  prediction.
