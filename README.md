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
uses the fixed nine-model universe (including CatBoost), one locked feature round per horizon (1D:
round 3; 5D/10D: round 4), a bounded development-only hyperparameter search,
and a fixed-manifest panel over five evaluation seeds (11, 42, 73, 101, 202).
It is staged under
`models/staged-stock-runs/<run-id>/` by default and cannot replace the active
registry accidentally. After reviewing the observed and prospective-shadow
evidence for all three horizons, promote the registry atomically:

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

If an authorised reviewer intentionally activates an official run before the
100-matured-session shadow gate is complete, the exception must be explicit and
audited. This does not turn failed evidence gates into accepted results; the
runtime continues to show the horizon as limited/experimental:

```powershell
python main.py promote_stock_models --run-id <run-id> --approve `
  --override-gates `
  --override-reason "User-approved early activation before prospective shadow evidence"
```

Generate presentation diagnostics for an active or staged run without changing
the production registry:

```powershell
python -m pipeline.generate_model_performance_artifact
python -m pipeline.generate_model_performance_artifact `
  --registry models/staged-stock-runs/<run-id>/model-registry.json `
  --output results/model-performance/runs/<run-id>/model-performance.json
```

The controlled stock/news comparison is diagnostics-only and uses the same
matched 2025 rows for stock-only, news-only, stock+news, and a shuffled-news
placebo arm:

```powershell
python -m pipeline.fusion_cli ablation
```

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

Runtime news retrieval never trains a sentiment model implicitly. The approved runtime
backend and its component hashes, calibration, weights and evaluation scope are recorded
in `models/sentiment-runtime-config.json`. Historical fusion evidence must be rescored by
that exact frozen Bursa/PhraseBank ensemble; a feature table carrying a different or
missing article-level model identity is rejected.

The optional `benchmark_sentiment_models` command evaluates the existing TF-IDF +
Logistic Regression sentiment baseline against a leakage-safe FinBERT challenger
(`yiyanghkust/finbert-pretrain`) and a calibrated probability ensemble. It writes
the comparison and a manual recommendation under `results/finbert-benchmark/`;
it never replaces the deployed sentiment artifact automatically. The public
`ProsusAI/finbert` checkpoint is not used for this benchmark because it was
already fine-tuned on Financial PhraseBank, the dataset used by this project.

The former TF-IDF classifier and earlier PhraseBank-only/Bursa-only checkpoints remain
offline benchmark references. They are not silently selected by runtime.

Benchmark the weight grid without changing production or cached production
artifacts:

```powershell
python main.py benchmark_finbert_ensemble --bursa-run-id 20260801T040002 --weight-step 0.05
```

The benchmark selects weights on the chronological 2024 validation split and
reports one untouched 2025 test evaluation under
`results/finbert-ensemble-benchmark/latest.json`. Every scored article records
the variant, component weights and (for the ensemble) Bursa and PhraseBank class
probabilities. Existing NewsAPI/Marketaux retrieval, entity filtering, recency
decay and context aggregation are unchanged.

Fusion evaluation v3 is a separate retrospective incremental-value study. It fixes the
stock log-odds coefficient at one and learns only a capped, reliability-aware news
correction. Stock-only and fusion use one threshold, and no-news rows must retain the
exact stock probability and direction. Window, regularisation and cap choices are made
inside purged 2019–2023 development folds; 2024 selects the shared threshold and 2025 is
the retrospective benchmark. Promotion is horizon-specific and production remains
stock-led unless every gate for that horizon passes. Rebuild commands, lawful official-
disclosure import and the complete safety contract are documented in
[`docs/fusion-evaluation-v3.md`](docs/fusion-evaluation-v3.md).

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

Stage 2 numerical fusion is registry-gated. The live API uses it only when a
horizon artifact is marked `accepted` and the current news context has Medium
or High coverage. Otherwise the calibrated stock probability is preserved and
the response explains why numerical fusion was not applied.

The historical archive workflow uses GDELT only as a URL/entity index and
Common Crawl to recover article text. GDELT's own tone is never used; recovered
text passes through StockSense entity disambiguation and the promoted FinBERT
model. The `high-recall-v4` retrieval profile searches GKG organisation names
and page titles with company-specific distinctive aliases and carefully
anchored ambiguous aliases. Every candidate records its match channel, aliases
and anchors; a retrieval match is provenance only and never bypasses the strict
downstream entity-disambiguation layer.

This workflow is free-only: use BigQuery Sandbox with billing disabled,
download results locally, and stop when the free monthly query allowance is
reached. Before executing any shard, `gdelt-run` dry-runs every unfinished
shard and compares their combined cost plus the project's current calendar
month usage with the configured project-wide ceiling. Common Crawl is queried
sequentially with checkpoints and backoff.
Start with the 2023 pilot so 2024 remains validation data and 2025 remains the
frozen test period:

```powershell
python -m pipeline.fusion_cli gdelt-query `
  --pilot-year 2023 `
  --shard monthly `
  --query-profile high-recall-v4 `
  --max-candidates-per-ticker-month 500
# The default workflow is manual: run each generated query in BigQuery Sandbox with
# billing disabled, perform a dry run, and place the monthly CSV/JSONL/Parquet
# exports in one local folder.
# An authenticated local workflow is also available after installing
# requirements-gdelt.txt and running `gcloud auth application-default login`:
python -m pipeline.fusion_cli gdelt-run `
  --project-id <your-google-project-id> `
  --query-dir data/historical-news-fusion/gdelt/pilot-2023/queries `
  --export-dir data/historical-news-fusion/gdelt/pilot-2023/exports `
  --monthly-budget-gib 900 `
  --dry-run-only
# Remove --dry-run-only only after reviewing the complete projected-month total.
python -m pipeline.fusion_cli gdelt-enrich `
  --export data/historical-news-fusion/gdelt/pilot-2023/exports `
  --output-dir data/historical-news-fusion/gdelt/pilot-2023 `
  --pilot-per-ticker 150 `
  --checkpoint data/historical-news-fusion/gdelt/pilot-2023/raw/recovery-state.jsonl `
  --heartbeat data/historical-news-fusion/gdelt/pilot-2023/raw/recovery-heartbeat.json `
  --max-runtime-minutes 45 `
  --request-interval-seconds 1 `
  --require-indexed-gate
# The recovery job is sequential and resumable. Re-run the same command after a
# runtime pause; checkpointed successful/no-capture URLs are not repeated.
python -m pipeline.fusion_cli label-bursa-news-llm `
  --source data/historical-news-fusion/gdelt/pilot-2023/raw/entity-candidates.jsonl `
  --output data/news-evidence-benchmark/bursa-news-llm-pseudolabelled.jsonl `
  --checkpoint data/historical-news-fusion/gdelt/pilot-2023/raw/llm-label-checkpoint.jsonl `
  --heartbeat data/historical-news-fusion/gdelt/pilot-2023/raw/llm-label-heartbeat.json `
  --approval-queue data/historical-news-fusion/gdelt/pilot-2023/raw/llm-label-approval-queue.jsonl `
  --max-runtime-minutes 45
python -m pipeline.fusion_cli gdelt-pilot `
  --pilot-year 2023 `
  --indexed data/historical-news-fusion/gdelt/pilot-2023/raw/pilot-indexed.csv `
  --recovered data/historical-news-fusion/gdelt/pilot-2023/raw/commoncrawl-recovered.jsonl `
  --accepted data/historical-news-fusion/gdelt/pilot-2023/raw/accepted-scored.jsonl `
  --llm-labels data/news-evidence-benchmark/bursa-news-llm-pseudolabelled.jsonl
```

The historical pilot remains fail-closed and records its model outputs for
review. The active corpus is populated from labels accepted as LLM-assisted
labels;
FinBERT outputs are never included in labelling prompts. Legacy automatic
source names are retained only as audit metadata after migration.
Only abstained rows enter the approval queue. After reviewing that small queue,
rerun the label command with `--approvals <reviewed-queue.jsonl>`; confident
automatic consensus rows cannot be overwritten through this path.

After the pilot report says `ready_for_full_collection`, plan every remaining
year together. This command is dry-run only unless `--execute` is supplied:

```powershell
python -m pipeline.fusion_cli gdelt-remaining-years `
  --project-id <your-google-project-id> `
  --pilot-report data/historical-news-fusion/gdelt/pilot-2023/pilot-2023-report.json
# Add --execute only after reviewing remaining-years-plan.json.
```

The remaining-year guard blocks 2019–2022 and 2024–2025 unless the complete
aggregate dry-run plus current-month project usage fits below the configured
BigQuery ceiling. It never substitutes a paid source when a gate fails.

`--require-indexed-gate` stops before Common Crawl unless every configured
company meets its declared URL floor, two source domains, six covered months
and a supported match channel. The default URL floor is 50; Press Metal uses a
documented 30-URL floor because the 2023 high-recall pilot found materially
lower publication volume while still covering 19 domains and all 12 months.
The exception changes only candidate sufficiency—not entity relevance, text
recovery, source diversity, annotation or final model-quality requirements.

Do not run the full 2019–2025 collection until the pilot reaches at least 90%
entity precision, 75% entity recall, 60% text recovery, 95% timestamp completeness,
all-company coverage, two source domains per company, 15 accepted articles per
company and 200 accepted articles overall. Feature coverage is evaluated
separately after the archive exists. Raw recovered text remains under
Git-ignored local storage.

#### 2023 Web NGram pseudo-document usability pilot

The Common Crawl workflow above remains available. A separate, non-production
pilot tests whether URL-linked GDELT Web NGrams can safely fill some historical
full-text gaps. It uses the already exported 2023 high-recall candidates and
never changes the live NewsAPI/Marketaux service, production FinBERT artifact,
or fusion registry:

```powershell
python -m pipeline.fusion_cli gdelt-ngram-pilot `
  --year 2023 `
  --candidate-dir data/historical-news-fusion/gdelt/pilot-2023-high-recall-v3b/exports `
  --sample-per-ticker 30 `
  --paired-per-ticker 10 `
  --output-dir data/historical-news-fusion/gdelt/pilot-2023-ngram `
  --max-runtime-minutes 45 `
  --max-download-gib 5
```

Recovery order is direct publisher HTML, High-quality Web NGram
reconstruction, Common Crawl, then Wayback. Medium/Low NGram reconstructions
remain audit evidence and cannot enter sentiment labelling. Direct publisher
successes are also reconstructed for a paired full-text versus pseudo-document
test. The command is checkpointed and can be rerun after a runtime, download,
provider, or OpenRouter pause.

For a bounded network smoke test that does not call OpenRouter or archives:

```powershell
python -m pipeline.fusion_cli gdelt-ngram-pilot `
  --max-articles 5 `
  --sample-per-ticker 1 `
  --paired-per-ticker 1 `
  --output-dir data/historical-news-fusion/gdelt/pilot-2023-ngram-smoke `
  --max-runtime-minutes 5 `
  --max-download-gib 0.25 `
  --request-interval-seconds 0.2 `
  --skip-labels `
  --skip-archives
```

The principal output is
`pilot-2023-ngram-usability-report.json`. A `usable_for_weak_sentiment_labels`
decision requires every frozen recovery, reconstruction, entity, sentiment,
FinBERT-sensitivity and timestamp gate to pass. A weaker
`entity_filtering_only` decision permits NGrams only for relevance filtering.
All other results retain archive-only recovery. Only records accepted under the
LLM-assisted-label policy are eligible for the active Bursa corpus.

### Bursa FinBERT dataset (LLM-assisted labelling workflow)

The live NewsAPI lookback is not used to construct the historical training
dataset. GDELT/Common Crawl acquisition is offline across complete calendar
years. The 7/14/30-day sentiment windows are applied later when point-in-time
features are built.

After the 2023 pilot passes, collect 2019–2022 and 2024–2025 using the guarded
remaining-year command above. Each year follows the same Common Crawl recovery
and dual-LLM consensus workflow. The candidate builder locks 2019–2023 to
`training`, 2024 to `validation`, and 2025 to `test`; removes exact duplicates;
clusters near-date similar headlines into events; and keeps each event entirely
in the latest applicable split.

The models label the article's implication for the selected company, not future
stock direction. Existing FinBERT outputs must never be copied into prompts or
pseudo-labels.

Validate the completed annotations before any GPU work:

```powershell
python main.py validate_bursa_news_dataset
```

The validator no longer uses a frozen corpus-size, per-company, or
independent-double-labelling minimum as a readiness requirement. The default
policy accepts single labelled rows while retaining the locked chronological
splits, required company coverage, text/language checks, entity-quality checks,
duplicate/event isolation, and other integrity safeguards. Dataset counts and
any available agreement statistics are reported descriptively. Independent
double-labelling and Cohen's kappa can be enabled in a stricter policy when
additional annotation evidence is available; they are not required by the
default exploratory dataset policy.

Train the Bursa challenger without changing production:

```powershell
python main.py train_bursa_finbert --profile full --device cuda
```

The challenger starts from the deployed PhraseBank FinBERT, mixes 70% Bursa
training examples with 30% PhraseBank replay, uses class-weighted loss, selects
epochs only from 2024 validation evidence, calibrates on 2024, and evaluates
three fixed seeds on the frozen 2025 Bursa test plus the unchanged PhraseBank
test. Results and staged artifacts are written under ignored local directories.

Promotion is separate and fails unless every absolute, uplift, calibration,
bootstrap, PhraseBank non-inferiority and seed-stability gate passed:

```powershell
python main.py promote_bursa_finbert --run-id <reviewed-run-id> --approve
```

Optional 2015–2018 articles are added only after each earlier year passes the
same recovery, entity, timestamp, source-diversity and volume gates. Supply each
passing report explicitly with `--backfill-report <year-report.json>`; an older
year cannot enter the annotation template without that evidence.

After the archive is ready, build features using leakage-safe out-of-fold stock
probabilities and evaluate both challengers:

```powershell
python -m pipeline.fusion_cli stock-oof --stock-path <all-stock-feature-rows.csv> --output data/fusion/stock-oof-probabilities.csv
python -m pipeline.fusion_cli features --stock-path <all-stock-feature-rows.csv> --stock-probabilities <out-of-fold-stock-probabilities.csv> --news-dir data/historical-news-fusion --output data/fusion/fusion-features.parquet
python -m pipeline.fusion_cli coverage --dataset data/fusion/fusion-features.parquet --report results/fusion/fusion-coverage-report.json
python -m pipeline.fusion_cli signal-diagnostics --dataset data/fusion/fusion-features.parquet --report results/fusion/fusion-signal-diagnostics.json
python -m pipeline.fusion_cli evaluate --dataset data/fusion/fusion-features.parquet --ticker 1155.KL --ticker 1295.KL --ticker 1023.KL --ticker 5347.KL --ticker 5225.KL --ticker 6947.KL --ticker 5183.KL --ticker 8869.KL --ticker 3816.KL --ticker 4677.KL
python -m pipeline.fusion_cli promote --approve
```

Run `coverage` before evaluation. It writes the exact ticker-by-horizon-by-period
matrix of Medium/High evidence used by the frozen evaluator and returns a
non-success status while any requested company is absent from development,
2024 calibration or the 2025 benchmark. `signal-diagnostics` is explicitly
non-promotional: it reports whether signed FinBERT tone ranks later Up outcomes
above chance, including Low-coverage rows as a sensitivity check.

Supplying only a subset of tickers creates a research-only evaluation. The
training rows are restricted to that subset, the scope is recorded in the
report, and a `full_company_universe = false` gate prevents promotion even if
all numerical checks pass. This is useful for diagnosis but cannot replace the
ten-company benchmark.

The evaluator compares a transparent horizon-weighted log-odds formula with an
L2 logistic late-fusion model. It uses five expanding purged folds in
2019–2023, 2024 for calibration and threshold locking, and the frozen 2025
benchmark for promotion. The registry records accepted or experimental status
per horizon; no artifact can become live merely because training completed.
Evaluation writes a staged registry. The explicit `promote --approve` command
copies only horizons that passed every frozen gate into the active runtime
registry and leaves failed horizons inactive.
Passing balanced-accuracy/MCC, calibration, recall, bootstrap, fold-stability
and per-company gates is mandatory. Results from 2026 onward remain prospective
shadow monitoring and are not used for tuning.

The active Bursa corpus is
`data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl` and is governed
by `config/bursa_news_dataset_policy.json`. Every accepted sentiment label is
recorded as `llm_assisted_label` (displayed as **LLM-assisted label**); the former `external_single_weak` and
`external_consensus` values are migration-only audit values. Independent
double-labelling is optional and is not required by the active policy.

`stock-oof` reuses the frozen stock-model architecture but refits it separately
for every chronological output year. Its calibration and threshold are nested
inside earlier target-complete data, so a fusion row never receives a stock
probability trained or calibrated on its own future label.

```powershell
python -m pipeline.fusion_cli shadow --dataset <fusion-features-with-2026-labels.parquet> --start-date 2026-01-01
```

### Evidence governance

Stock model training is always staged. `quick`, `focused` and `full` are
research profiles; only the tracked `official` profile can be considered for
manual promotion. Official selection uses fixed candidate families, feature
rounds and seeds, expanding purged validation, cross-fitted calibration and
threshold evaluation, the frozen 2025 observed benchmark, and a later
prospective shadow period. Audit commands never repair the registry they
inspect.

The news score is an article-language context signal, not a probability that a
stock will rise. It uses the versioned
`recency_x_finbert_confidence_v1` weighting policy. Coverage additionally counts
independent events so syndicated copies cannot create false evidence
diversity. User article corrections are what-if evidence changes and are
written to a local append-only SQLite audit ledger under `data/audit/`.

FinBERT remains external-domain evidence until the benchmark described by
`config/news_evidence_policy.json` is supplied. The required dataset is a
time-split, dual-LLM consensus pseudo-labelled Bursa Malaysia news set covering
all ten companies;
the system reports this gate as not ready when the dataset is absent.

Numerical fusion additionally requires:

- an accepted stock horizon with a valid locked threshold;
- Medium or High news coverage;
- an accepted, hash-verified horizon-specific fusion artifact;
- at least 1,000 point-in-time rows per horizon and at least 50 unique
  observation dates per company/horizon in the historical fusion dataset.

If any condition is missing, StockSense keeps the transparent stock-led
synthesis and reports why fusion was not applied.

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
the processes started by that shortcut. If startup fails, the hidden launcher
shows an error and keeps diagnostic logs under `.cache/launcher-logs`.

Run the local connection audit after startup with:

```powershell
.\Test-StockSense.cmd
```

Add `-IncludeExternalCalls` to also make one NewsAPI/FinBERT request and one
OpenRouter tutor request. This optional mode can consume provider quota.
