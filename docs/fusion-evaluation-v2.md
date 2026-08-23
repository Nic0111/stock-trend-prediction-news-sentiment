# Fusion evaluation v2

This protocol asks whether point-in-time company news adds incremental
probability information beyond a capacity-matched stock-only meta-model.  It
is a retrospective 2019–2025 study.  It does not make a prospective claim and
cannot change the active StockSense runtime by itself.

## Frozen chronology

- Development and hyperparameter selection: 2019–2023
- Calibration and decision-threshold selection: 2024
- Retrospective benchmark: 2025
- Supporting rolling-origin evaluations: 2022, 2023, 2024 and 2025

Rows whose future target crosses the next period boundary are purged.  Window
selection, elastic-net settings, news influence caps and all other choices are
made without looking at 2025 benchmark labels.

## Reproducible execution

Run from the repository root.  The scientific stock OOF step is intentionally
slow because it repeats model selection inside every outer year.

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli stock-oof `
  --stock-path data\model-input-stock-dataset `
  --mode scientific `
  --search-profile official `
  --output data\fusion\scientific-stock-oof-v2.parquet

.\.venv\Scripts\python.exe -m pipeline.fusion_cli fusion-window-candidates `
  --stock-path data\model-input-stock-dataset `
  --stock-probabilities data\fusion\scientific-stock-oof-v2.parquet `
  --news-dir data\historical-news-fusion `
  --policy config\fusion_evaluation_v2.json `
  --output data\fusion\fusion-window-candidates-v2.parquet

.\.venv\Scripts\python.exe -m pipeline.fusion_cli evaluate-v2 `
  --dataset data\fusion\fusion-window-candidates-v2.parquet `
  --policy config\fusion_evaluation_v2.json `
  --report results\fusion\fusion-evaluation-v2.json `
  --registry models\fusion\staged-fusion-v2-registry.json
```

Run the same three commands a second time in a separate output directory and
compare the scientific OOF file hash, selected parameters, predictions and
report hash.  A difference is a reproducibility failure.

## Safety and activation

The evaluator writes only a staged `fusion-registry-v4`.  Runtime refuses it
unless every 1-, 5- and 10-day gate passes, all stock/news/dataset identities
still match and a reviewed runtime-baseline state has been materialised.  Any
failure retains the stock-led aggregate outlook.

The Model Performance page can display the v2 report only when the report's
recorded stock identity is compatible with the active stock run.  It labels
the evidence as **Retrospective evidence, 2019–2025**.
