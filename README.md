# StockSense

Final Year Project decision-support foundation for novice Malaysian investors. The system keeps stock-direction prediction and recent financial-news sentiment as separately evaluated signals.

## Frozen analytical benchmark

- Stocks: ten selected Bursa Malaysia companies listed in `data/stock-metadata.csv`.
- Historical range: 2015-01-01 through 2025-12-31.
- Development period: feature rows from 2015 through 2024 whose future target date is still before 2025.
- Final holdout: eligible feature and target dates inside 2025.
- Prediction horizons: 1, 5, and 10 trading days.
- Market context: FTSE Bursa Malaysia KLCI (`^KLSE`).

The 2025 benchmark is retained as a fixed observed benchmark. New model choices must be selected from
2015–2024 development evidence only; because the 2025 results have already been inspected during project
review, the final report must not describe it as a newly untouched test set.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Add the NewsAPI key to `config/newsapi_key.txt` or the `NEWSAPI_KEY` environment variable. The key file is Git-ignored.

Launch the React + FastAPI decision-support MVP after installing the requirements:

```powershell
Start-StockSense.cmd
```

The app retrieves stock direction and recent news sentiment independently. Successful live results are stored
in the local Git-ignored `.cache/dashboard/` directory so a later external-service outage can show the last
result with an explicit timestamp and stale-data warning.

The sidebar is grouped into beginner-oriented pages:

- `Overview`: recommended workflow and project boundaries.
- `Stock Outlook`: ticker/horizon prediction workspace.
- `Market Snapshot`: latest available daily price, range, volume, and closing-price chart; changing the company refreshes it automatically.
- `Compare Tickers`: starts with two aligned selector columns and can expand to five unique companies. Additional slots can be removed; each comparison includes trend icons, sparklines, price/change metrics, model evidence, and normalized relative performance.
- `News & Sentiment`: filtered article and sentiment context.
- `Ask & Learn`: optional conversational tutor connected to OpenRouter when a key is configured. The Learning Center remains the deterministic offline teaching path.
- `Learning Center`: explanations of probabilities, sentiment, validation status, and freshness.

Experimental horizons are deliberately shown as research-only evidence. They retain their Up/Down
classification, probability and validation details for transparency, with a visible warning that the
direction has not passed the frozen acceptance benchmark.

The Market Snapshot page is deliberately labeled as a daily close rather than guaranteed real-time data. It
shows the observation date so a future real-time provider can be added without changing the page contract.

Ask & Learn reads `OPENROUTER_API_KEY` from the process environment or the local
ignored file `config/openrouter_api_key.txt`, then sends the conversation plus any matching
Stock Outlook evidence to OpenRouter. The key is never stored in source code or tracked by Git.

Ticker selectors and company result sections use explicit official or Wikimedia-hosted SVG/PNG logo assets
with initials and favicon fallbacks, so company identity remains recognizable even when an external logo
endpoint is unavailable.

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
python main.py benchmark_sentiment_models --profile full
python main.py train_finbert --profile full --device cuda --no-promote
python main.py train_models --search-profile full
python main.py train_models --search-profile official
python main.py audit_system --level full
```

Or run the complete frozen benchmark:

```powershell
python main.py all --search-profile full
```

`--search-profile focused` applies the full tuning budgets only to the strongest evidence-supported model and feature combinations for each horizon. `--search-profile quick` evaluates the expanded candidate set with one bounded configuration per candidate and is useful for a complete smoke benchmark. It does not satisfy the final modelling search plan.

The `official` profile is the reproducibility-controlled challenger workflow. It
uses the fixed eight-model universe, one locked feature round per horizon (1D:
round 3; 5D/10D: round 4), a seed-42 hyperparameter manifest, and five fixed
evaluation seeds (11, 42, 73, 101, 202). It is staged under
`models/staged-stock-runs/<run-id>/` by default and cannot replace the active
registry accidentally. After reviewing the observed and prospective-shadow
evidence, promote explicitly:

```powershell
python main.py promote_stock_models --run-id <run-id> --approve
python main.py verify_stock_reproducibility --run-a <run-id-a> --run-b <run-id-b>
```

The selection contract is versioned in
`config/stock_model_selection_policy.json`. Ranking uses mean balanced
accuracy, MCC, macro-F1, seed variance, model complexity, and model name as
stable tie-breakers. The run records a candidate confidence set when multiple
models are within the practical 0.5 percentage-point margin. Promotion remains
manual and requires prospective shadow evidence.

## Runtime interfaces

Predict from the latest completed daily observation:

```powershell
python main.py predict_stock --ticker 1155.KL --horizon 5
```

Stock and KLCI features are calculated only on shared valid Bursa sessions. This prevents provider-only
holiday/zero-volume rows from invalidating the 60-session market-sensitivity features required by the 5- and
10-day models. Runtime inference uses a 420-calendar-day buffer and reports the market observation date,
reference session, session lag, and freshness status. A saved or delayed result remains visible for research
transparency, but it is labelled as stale/fallback instead of being presented as live.

Retrieve recent news context without an interactive prompt:

```powershell
python main.py download_financial_news --ticker 1155.KL
```

News is retrieved across 30 calendar dates and summarized into 7-, 14-, and 30-day contexts for the 1-, 5-, and 10-day stock horizons. The NewsAPI request stays inside its rolling free-plan boundary when the stock observation is one day behind today. Recency half-lives are 3.5, 7, and 15 days, so an article at each context boundary receives 25% of the same-day recency weight before FinBERT confidence is applied. Article sentiment below 0.55 model probability is excluded from aggregation. Runtime sentiment uses the evaluated FinBERT model; the TF-IDF classifier remains available only to the offline training and benchmark pipelines.

### Conditional Marketaux coverage

NewsAPI is always queried first. Marketaux is activated automatically only when
the selected NewsAPI context is Low/Unavailable or NewsAPI fails. When it is
used, its articles are scored with the same FinBERT model, deduplicated against
NewsAPI by URL/title, and merged only when they add evidence from an independent
publisher. The dashboard labels the provider mode, cache state, duplicate count,
and source coverage so Marketaux is never mistaken for an independent article
without verification.

Add the token to `config/marketaux_api_key.txt` (Git-ignored) or set
`MARKETAUX_API_TOKEN`. A local cap of 100 Marketaux requests per Kuala Lumpur
calendar day is enforced by default; override it with
`STOCKSENSE_MARKETAUX_DAILY_CAP` when appropriate. Successful responses are
cached for six hours and may be reused as labelled stale fallback data for up to
24 hours. HTTP 402/429 responses are recorded without automatic retries.

If both providers return zero relevant articles in the current 30-day window,
StockSense makes one additional quota-aware Marketaux request for days 31–90.
Those results are stored separately as older company context, shown without a
sentiment badge, and never enter the current News Sentiment score or stock
prediction. Empty recent and historical responses are cached too, so revisiting
the same ticker within the cache period does not spend another request.

The offline provider benchmark remains available for evaluating retrieval
quality before changing the fallback policy. It stores dated snapshots and
does not alter the runtime provider automatically:

```text
python main.py collect_news_provider_benchmark --as-of 2026-07-18 --run-id day-01 --provider newsapi --provider marketaux
python main.py evaluate_news_provider_benchmark
```

The evaluator writes entity coverage, annotations, JSON metrics and Markdown
comparisons under `results/news-provider-marketaux-benchmark/`. It reports the
comparison for review and does not change the deployed fallback policy.

## Evaluation rules

Stock candidates are selected using five expanding, date-grouped, horizon-purged folds on 2015-2024. Four refinement rounds compare the original features, stationary/ticker-aware features, Bursa KLCI context features, and derived regime/relative-market features. The benchmark includes logistic regression, elastic-net logistic regression, random forest, Extra Trees, histogram gradient boosting, XGBoost, LightGBM, and a development-selected soft-voting ensemble of logistic, Extra Trees, and histogram gradient boosting learners. CatBoost is available as an exploratory candidate through `SklearnCompatibleCatBoostClassifier`, which adapts CatBoost 1.2.8 to scikit-learn 1.8's cloning/tag protocol; it is intentionally outside the official fixed eight-model universe until comparable evidence is collected.

A horizon uses the versioned `horizon-acceptance-v2` policy in `config/horizon_acceptance_policy.json`. The policy evaluates each horizon separately with locked horizon-matched baselines, date-block bootstrap uncertainty, Brier skill and directional safeguards. `accepted` requires the full gate set; `limited` identifies a positive but incomplete signal; `experimental` means the evidence is not strong enough. Company-level results can downgrade a horizon but cannot upgrade it. The UI keeps the Up/Down classification for limited and experimental results, with a clear evidence warning and failed-check explanation.

The original 2025 results remain a fixed observed benchmark. Run the forward shadow evaluation after preparing 2026 data:

```powershell
python main.py evaluate_horizon_shadow --start 2026-01-01 --end 2026-06-30
python main.py evaluate_horizon_shadow --run-id <official-run-id> --start 2026-07-23 --end 2026-12-31
```

Shadow evaluation reuses frozen artifacts and thresholds, never retrains or tunes them, and records the conservative effective status in the selected registry. Passing `--run-id` evaluates a staged official challenger; this is required before its manual promotion gate can pass.

Stock Outlook shows local supporting and opposing factors for every available signal. Each factor is estimated by replacing one feature with its 2015–2024 development median and measuring the change in calibrated Up probability. This is a model-sensitivity explanation—not a causal claim—and experimental horizons retain the same explanation with an explicit validation warning.

The sentiment classifier uses a fixed stratified 80/20 development/test split and five-fold development tuning. Runtime news retrieval never trains a model implicitly.

The approved runtime sentiment backend is the evaluated FinBERT checkpoint recorded in
`models/sentiment-runtime-config.json`; its calibrated temperature is applied before
article confidence weighting. The previous TF-IDF model remains available as an explicit
offline benchmark and training reference. The promotion is based only on the frozen
Financial PhraseBank test split, and does not claim Malaysian Bursa-news generalisation.

The optional `benchmark_sentiment_models` command evaluates the existing TF-IDF +
Logistic Regression sentiment baseline against a leakage-safe FinBERT challenger
(`yiyanghkust/finbert-pretrain`) and a calibrated probability ensemble. It writes
the comparison and a manual recommendation under `results/finbert-benchmark/`;
it never replaces the deployed sentiment artifact automatically. The public
`ProsusAI/finbert` checkpoint is not used for this benchmark because it was
already fine-tuned on Financial PhraseBank, the dataset used by this project.

## Tests

The offline test suite does not call Yahoo Finance or NewsAPI:

```powershell
python -m unittest discover -s tests -v
```

Tests cover target dates, missing target tails, leakage-free folds, same-date grouping, article relevance and deduplication, sentiment weighting and coverage, sentiment training artifacts, and registry-driven inference.
The dashboard service, market snapshot calculations, navigation startup, and per-signal fallback are also tested offline.

## Generated artifacts

- `models/model-registry.json`: champion, feature schema, threshold, benchmark status, and metrics per horizon.
- `models/financial-news-sentiment-model-metadata.json`: sentiment dataset hash, parameters, and evaluation evidence.
- `models/sentiment-runtime-config.json`: active FinBERT runtime backend and Financial PhraseBank evaluation scope.
- `results/model-comparison-summary.csv`: final stock-model results.
- `results/stock-model-candidates.csv`: all development-search candidates.
- `results/stock-model-baselines.csv`: final holdout baselines.
- `results/stock-model-per-ticker-metrics.csv`: final champion performance by ticker.
- `results/finbert-benchmark/sentiment-model-comparison.md`: TF-IDF, FinBERT and ensemble sentiment results.
- `results/finbert-benchmark/sentiment-model-recommendation.json`: manual-approval recommendation; deployment is unchanged.

The next expanded training run writes registry schema version 2 with the candidate model set, fold-level
balanced accuracy/MCC, positive-MCC fold count, stability status and the fixed-benchmark disclosure.

Large datasets and `.joblib` artifacts are generated locally and remain Git-ignored.

The official FinBERT challenger run requires a CUDA GPU. If the local environment has no CUDA device,
run `notebooks/finbert_gpu_training_colab.ipynb` in a GPU-enabled Google Colab runtime and import the
staged checkpoint/provenance bundle for review. CPU remains supported for debugging and production inference,
but the official full benchmark must use `--device cuda` and fails clearly when CUDA is unavailable.
On an NVIDIA laptop, install the CUDA wheel with `python -m pip install -r requirements-cuda.txt` after
activating `.venv`.
TF-IDF + Logistic Regression remains a frozen academic baseline; it is not used by the runtime sentiment path.

`python main.py audit_system --level full` checks data coverage, artifact hashes, registry/report consistency,
cache contracts, Python tests, backend tests, frontend tests and the production build. Provider smoke tests are
opt-in with `--live-providers` so normal audits do not spend API quota.

## React + FastAPI migration (parity phase)

The repository contains the production React + FastAPI web runtime that preserves the existing
analytical pipeline:

- `backend/app/main.py` exposes typed FastAPI endpoints under `/api/v1` for
  company metadata, market snapshots, predictions, analyses, comparisons,
  news retrieval and Ask & Learn.
- `frontend/` is a Vite-built React client. Its routes mirror the current
  StockSense navigation and keep the same evidence labels, horizons, logos,
  article metadata and educational boundary.
- The client uses session state for the selected company, horizons, comparison
  tickers and the current analysis context. TanStack Query handles request
  caching and cancellation without changing the underlying data cache policy.
- `docker-compose.yml` provides a reproducible local deployment with an
  Nginx SPA gateway and a single-worker API. It mounts the existing `data`,
  `models`, `config` and `.cache` directories rather than copying credentials
  or generated artifacts into images.

### Run the parallel runtime locally

Start the API from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

In a second terminal, install the frontend lockfile and start Vite using the
project's Node/pnpm runtime:

```powershell
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

The React client is available at `http://localhost:5173`; FastAPI's interactive
contract documentation is at `http://localhost:8000/api/docs`.

### Combined outlook rollout

The Stock Outlook page now presents a transparent Stage 1 evidence synthesis. It keeps the calibrated stock
probability unchanged, then labels recent FinBERT news as reinforcing, conflicting, balanced, or unavailable.
The synthesis is anchored to the stock observation date and the Bursa session cutoff (`17:00 Asia/Kuala_Lumpur`),
so articles published after the completed market session cannot leak into the explanation. A missing news signal
therefore falls back to the stock-led result instead of inventing a neutral probability.

Stage 2 numerical fusion is intentionally disabled until a lawful, point-in-time historical Bursa news archive is
available. Provider exports can be prepared with the additive experimental CLI:

```powershell
python -m pipeline.fusion_cli import-news --output-dir data/fusion/raw provider-export.csv
python -m pipeline.fusion_cli readiness --data-dir data/fusion/raw
python -m pipeline.fusion_cli features --stock-path data/model-input-stock-dataset/1155-kl-model-input.csv --news-dir data/fusion/raw
python -m pipeline.fusion_cli train --dataset data/fusion/fusion-features.parquet
```

The trainer refuses incomplete history and writes only experimental, horizon-specific artifacts. It does not change
`models/model-registry.json` or enable fusion at runtime. Promotion requires the predeclared 2025 benchmark and a
later untouched 2026 shadow period to pass the balanced-accuracy/MCC, calibration, class-recall, fold-stability and
per-company gates.

### Verification gates

Run the backend and legacy regression tests before cutover:

```powershell
.\.venv\Scripts\python.exe -m unittest backend.tests.test_api -q
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Run the frontend checks from `frontend`:

```powershell
pnpm typecheck
pnpm test
pnpm build
```

The Docker images are intentionally a later deployment option; Docker Desktop
must be installed before running `docker compose up --build`.

### One-click local startup

After Node.js and pnpm are installed, double-click `Start-StockSense.cmd` in
the project root. It starts FastAPI and Vite in hidden windows, waits for the
frontend, and opens `http://localhost:5173`. Use `Stop-StockSense.cmd` to stop
the processes started by that shortcut.
