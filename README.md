# Stock Investment Analytics

Final Year Project decision-support foundation for novice Malaysian investors. The system keeps stock-direction prediction and recent financial-news sentiment as separately evaluated signals.

## Frozen analytical benchmark

- Stocks: ten selected Bursa Malaysia companies listed in `data/stock-metadata.csv`.
- Historical range: 2021-01-01 through 2025-12-31.
- Development period: feature rows through 2024 whose future target date is still before 2025.
- Final holdout: eligible feature and target dates inside 2025.
- Prediction horizons: 1, 5, and 10 trading days.
- Market context: FTSE Bursa Malaysia KLCI (`^KLSE`).

The final 2025 holdout must not be used for feature design, tuning, calibration, or threshold selection.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Add the NewsAPI key to `config/newsapi_key.txt` or the `NEWSAPI_KEY` environment variable. The key file is Git-ignored.

## Reproducible pipeline

Run individual stages when developing:

```powershell
python main.py download_raw
python main.py preprocess_raw
python main.py build_model_input
python main.py data_understanding
python main.py preprocess_financial_news_sentiment
python main.py financial_news_sentiment_understanding
python main.py train_sentiment_model --search-profile full
python main.py train_models --search-profile full
```

Or run the complete frozen benchmark:

```powershell
python main.py all --search-profile full
```

`--search-profile quick` is provided only for smoke testing. It does not satisfy the final modelling search plan.

## Runtime interfaces

Predict from the latest completed daily observation:

```powershell
python main.py predict_stock --ticker 1155.KL --horizon 5
```

Retrieve recent news context without an interactive prompt:

```powershell
python main.py download_financial_news --ticker 1155.KL
```

News is retrieved once over 21 days and summarized into 7, 14, and 21-day contexts for the 1, 5, and 10-day stock horizons. Article sentiment below 0.55 model probability is excluded from aggregation.

## Evaluation rules

Stock candidates are selected using five expanding, date-grouped, horizon-purged folds on 2021-2024. The three refinement rounds compare the original features, stationary/ticker-aware features, and Bursa KLCI context features. The chosen model is calibrated and its decision threshold is locked before the 2025 holdout is opened.

A horizon is accepted only when its 2025 balanced accuracy exceeds its strongest naive baseline by at least 0.02, MCC is positive, and macro-F1 is no worse than the baseline. Otherwise it remains available with `experimental` status.

The sentiment classifier uses a fixed stratified 80/20 development/test split and five-fold development tuning. Runtime news retrieval never trains a model implicitly.

## Tests

The offline test suite does not call Yahoo Finance or NewsAPI:

```powershell
python -m unittest discover -s tests -v
```

Tests cover target dates, missing target tails, leakage-free folds, same-date grouping, article relevance and deduplication, sentiment weighting and coverage, sentiment training artifacts, and registry-driven inference.

## Generated artifacts

- `models/model-registry.json`: champion, feature schema, threshold, benchmark status, and metrics per horizon.
- `models/financial-news-sentiment-model-metadata.json`: sentiment dataset hash, parameters, and evaluation evidence.
- `results/model-comparison-summary.csv`: final stock-model results.
- `results/stock-model-candidates.csv`: all development-search candidates.
- `results/stock-model-baselines.csv`: final holdout baselines.
- `results/stock-model-per-ticker-metrics.csv`: final champion performance by ticker.

Large datasets and `.joblib` artifacts are generated locally and remain Git-ignored.
