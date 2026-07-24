# StockSense Full-System Freshness and Pipeline Audit

Audit date: 22 July 2026 (Asia/Kuala_Lumpur)

## Executive conclusion

The active React + FastAPI StockSense runtime is now aligned to freshly downloaded stock/KLCI data, freshly regenerated preprocessing/features, and newly trained horizon-specific stock models.

- Active stock model-input hash: `838558e8092340a152c135965d0d149a6e20ad8cb056c182598fdd9dda56901f`
- Active registry hash: exact match
- Active stock training run: `20260722T162145`
- Search profile: `focused`
- Active feature schema: `aligned-market-sessions-v2`
- Runtime prediction model version: `838558e80923`
- API result: ready
- Frontend result: HTTP 200

Before retraining, the active registry declared model-input hash `fb584f...`, while the freshly regenerated inputs produced `838558...`. That was a genuine stale-model condition. Retraining replaced the registry and created new timestamped model artifacts against the current hash.

All three horizons are nevertheless **Experimental in the effective runtime status** after the newly evaluated 2026 shadow period. This is a model-performance finding, not a freshness or pipeline failure.

## 1. Environment and configuration

### Implementation

- Python runs from `.venv` using Python 3.12.3.
- `pip check` reports no broken Python dependencies.
- Node.js 24.18.0 and pnpm 11.9.0 are available.
- NewsAPI, Marketaux and OpenRouter configuration files are Git-ignored.
- FastAPI reads the current registry at runtime and includes a registry signature in its in-memory prediction cache key.

### Result

- Environment: passed.
- FastAPI readiness: `ready`.
- Stock artifacts available: 1D, 5D and 10D all true.
- FinBERT runtime dependency check: true.
- React frontend: HTTP 200 at `http://localhost:5173`.

## 2. Raw stock and KLCI download

### Implementation

`pipeline/download_raw_stock_data.py` downloads unadjusted Yahoo Finance daily OHLCV, adjusted close and corporate actions for ten Bursa companies and `^KLSE`. The offline research interval is deliberately locked to 1 January 2015 through 31 December 2025.

### Fresh run result

- Ten company files downloaded successfully.
- Total company rows: 27,081.
- Each company starts on 2 January 2015 and ends on 31 December 2025.
- KLCI rows: 2,691.
- KLCI starts on 2 January 2015 and ends on 31 December 2025.
- No download failure occurred.
- Row counts and raw hashes reproduced the previous download snapshot.

## 3. Stock preprocessing

### Implementation

`pipeline/preprocess_raw_stock_data.py` validates the required schema, fills missing corporate-action values with zero, drops rows missing essential market fields, removes logically impossible OHLCV records, preserves statistical outliers, validates dates/tickers and produces preprocessed data plus audit summaries.

### Fresh run result

- Input rows: 27,081.
- Output rows: 27,076.
- Logically invalid rows removed: 5.
- Statistical outlier rows flagged but retained: 2,745.
- Missing essential-value failures: none reported.
- All ten preprocessed files were regenerated successfully.

Keeping statistical outliers is intentional because market shocks, pandemic moves and other regime events are valid information rather than automatically erroneous data.

## 4. Feature and target construction

### Implementation

`pipeline/build_model_input_stock_data.py` aligns each stock with valid KLCI sessions and builds the stationary, technical, market-relative and regime feature sets defined in `pipeline/stock_features.py`. It creates independent future targets and future target dates for 1, 5 and 10 trading days. Tail rows retain missing targets rather than being assigned false labels.

### Fresh run result

- Model-input rows across ten companies: 26,198.
- Per-company feature rows: 2,609 to 2,624.
- Columns per file: 61.
- Feature date range: 1 April 2015 through 31 December 2025.
- Current file-set hash: `838558e8092340a152c135965d0d149a6e20ad8cb056c182598fdd9dda56901f`.
- Registry hash after retraining: exact match.

The April 2015 feature start is expected because rolling features need warm-up observations from January onward.

## 5. EDA and data-understanding outputs

### Implementation

`pipeline/data_understanding.py` rebuilds descriptive statistics, variable descriptions and visual EDA for returns, prices, volume, volatility, drawdowns and KLCI-relative features.

### Result

- Completed successfully.
- Markdown summaries under `summary/data-understanding/` were regenerated.
- EDA charts were regenerated from the current 2015–2025 model inputs.

## 6. Financial PhraseBank preprocessing

### Implementation

`pipeline/preprocess_financial_news_sentiment_data.py` cleans labels/text, rejects invalid rows and removes duplicate sentences from the fixed Financial PhraseBank training dataset.

### Fresh run result

- Raw records: 4,846.
- Final unique valid records: 4,838.
- Development split: 3,870.
- Frozen test split: 968.
- The deterministic split manifest was restored and now contains all 4,838 current sentence hashes.

## 7. TF-IDF + logistic-regression sentiment benchmark

### Implementation

`pipeline/train_financial_news_sentiment_model.py` performs a fixed stratified 80/20 split, five-fold development tuning, class-balanced logistic regression and full-data artifact fitting. This model is an offline baseline; it is not the active runtime sentiment backend.

### Fresh full-search result

- Status: Accepted.
- Test balanced accuracy: 0.7175.
- Test macro-F1: 0.7232.
- Majority macro-F1: 0.2484.
- Negative recall: 0.6777.
- Neutral recall: 0.8313.
- Positive recall: 0.6434.

## 8. FinBERT runtime sentiment model

### Implementation

The runtime configuration selects `yiyanghkust/finbert-pretrain`, fine-tuned on the frozen PhraseBank development split and calibrated with temperature 1.52. Retrieved articles below 0.55 maximum class probability are labelled uncertain and excluded from the aggregated tone.

### Audit result

- Runtime backend: FinBERT.
- Stored test accuracy: 0.8440.
- Stored test balanced accuracy: 0.8303.
- Stored test macro-F1: 0.8248.
- Runtime configuration, checkpoint directory, class order, temperature and stored comparison metrics agree.
- Direct live NewsAPI inference successfully produced article class probabilities.

### Limitation encountered

A complete CPU-only FinBERT retraining was attempted but exceeded the one-hour execution ceiling. No checkpoint or benchmark file was partially replaced; the previous verified checkpoint remained intact. Therefore this audit validates the deployed artifact by metadata, hash, manifest, loading and live inference, but does **not** claim that the FinBERT weights were freshly retrained on 22 July.

The current validation scope remains Financial PhraseBank only. Malaysian Bursa-news generalisation has not been established.

## 9. Horizon-specific stock-model training

### Implementation

`pipeline/train_stock_models.py` uses:

- 2015–2024 development data.
- Five expanding, date-grouped, horizon-purged validation folds.
- A fixed observed 2025 benchmark.
- Horizon-specific feature/model candidate sets.
- Out-of-fold probability calibration and a locked threshold.
- Versioned Horizon Acceptance V2 evaluation.
- Timestamped model files and an atomic registry update.

The focused profile applies the full tuning budget to the strongest horizon-specific candidates rather than repeating the broad exploratory full search.

### Fresh observed-benchmark results

| Horizon | Fresh champion | 2025 balanced accuracy | MCC | ROC-AUC | Observed status | Main failed gate |
|---|---|---:|---:|---:|---|---|
| 1 day | Elastic-net logistic, round 3 | 0.5192 | 0.0401 | 0.5342 | Experimental | Bootstrap interval crosses zero |
| 5 days | Random forest, round 4 | 0.5448 | 0.0896 | 0.5612 | Accepted | None globally |
| 10 days | Soft-voting ensemble, round 4 | 0.5685 | 0.1379 | 0.5981 | Experimental | Development stability and negative Brier skill |

Company-level observed acceptance is more conservative:

- 1 day: 10 Experimental.
- 5 days: 3 Accepted and 7 Experimental.
- 10 days: 10 Experimental.

## 10. Horizon Acceptance V2 reports

### Implementation

`pipeline/horizon_acceptance.py` evaluates balanced-accuracy gain over a locked baseline, MCC, macro-F1, both directional recalls, predicted class share, Brier skill, development-fold stability and a date-block bootstrap confidence interval. Company-level status may downgrade but never upgrade a horizon.

### Result

The pre-audit acceptance CSV still contained the former quick-profile champions. It was stale reporting data, although runtime inference used the registry rather than this CSV. `main.py recompute_horizon_acceptance` regenerated the report against the new artifacts:

| Horizon | Observed gain | 95% block-bootstrap interval | Brier skill | Observed status |
|---|---:|---:|---:|---|
| 1 day | +0.0271 | -0.0085 to +0.0626 | +0.00186 | Experimental |
| 5 days | +0.0633 | +0.0081 to +0.1182 | +0.00399 | Accepted |
| 10 days | +0.0876 | +0.0237 to +0.1619 | -0.00049 | Experimental |

## 11. Frozen 2026 shadow evaluation

### Implementation

`pipeline/evaluate_horizon_shadow.py` downloads the forward 2026 period, applies the frozen model artifacts and thresholds without retraining, and evaluates 115 unique observation dates / 1,150 pooled rows per horizon. The effective runtime status is the weaker of observed and shadow evidence.

### Fresh result

| Horizon | Shadow balanced accuracy | Gain over locked baseline | 95% interval | p-value | Shadow status |
|---|---:|---:|---:|---:|---|
| 1 day | 0.5273 | +0.0197 | -0.0465 to +0.0824 | 0.262 | Experimental |
| 5 days | 0.5173 | +0.0128 | -0.0500 to +0.0749 | 0.342 | Experimental |
| 10 days | 0.4950 | -0.0196 | -0.0853 to +0.0448 | 0.713 | Experimental |

Consequently, every current runtime prediction is labelled Experimental even though the 5-day model passed the observed 2025 global benchmark. This is the correct conservative behaviour.

## 12. Runtime stock inference

### Implementation

`pipeline/predict_stock_trend.py` loads only the artifact path recorded for the requested horizon in `models/model-registry.json`, downloads a 420-calendar-day runtime buffer, aligns stock and KLCI sessions, builds the current feature row, applies the stored calibrated threshold and combines global, ticker and shadow evidence conservatively.

### Fresh-process API matrix result

- 30/30 company/horizon combinations returned HTTP 200.
- All 30 used model version `838558e80923`.
- All 30 used market observation 21 July 2026.
- All 30 reported freshness `fresh`.
- All 30 reported effective status `experimental`.
- No artifact-not-found, schema mismatch or stale-observation error occurred.

Old timestamped `.joblib` files remain on disk for reproducibility, but no active code path references the old `stock-trend-champion-*.joblib` files. Runtime selection is registry-driven.

## 13. Market snapshot and ticker comparison

### Implementation

Market snapshots fetch the latest Yahoo Finance daily close independently from the prediction feature row. Comparison reuses the same snapshot/prediction contracts for two to five unique tickers.

### Result

- 10/10 market snapshots returned HTTP 200.
- All snapshot dates were 22 July 2026.
- No snapshot was labelled stale.
- A three-ticker, 5-day comparison returned three items with no item-level errors.

The one-session difference between prediction observation (21 July) and snapshot (22 July) is expected during the current Bursa session: model inference requires a completed, feature-aligned session, while the market snapshot can display the provider's current latest daily row.

## 14. NewsAPI and conditional Marketaux pipeline

### Implementation

NewsAPI is queried first over 30 calendar days. FinBERT scores relevant articles and builds 7-, 14- and 30-day sentiment contexts for the 1-, 5- and 10-day horizons. Marketaux is queried only when NewsAPI coverage is Low/Unavailable or NewsAPI fails. Marketaux output is deduplicated and only incorporated if it adds relevant independent evidence. A separate 31–90-day Marketaux request is display-only historical context and cannot enter the current tone.

### Fresh Maybank result

- NewsAPI request source: live.
- Raw NewsAPI articles: 4.
- Relevant articles: 3.
- 7-day context: Unavailable.
- 14-day context: Unavailable.
- 30-day context: Low, one usable neutral article, weighted score 0.0.
- Marketaux fallback trigger: NewsAPI low coverage.
- Marketaux fresh result: no relevant recent-window article, so it was not merged.
- Current Marketaux ledger after testing: 24/100 requests used; quota ledger preserved.

This confirms both the primary provider and conditional fallback decision path execute. It does not mean Marketaux improves every company query.

## 15. Evidence synthesis

### Implementation

`pipeline/evidence_synthesis.py` keeps the calibrated stock probability unchanged and describes recent news as reinforcing, conflicting, balanced or unavailable. It is anchored to the completed stock observation and Bursa 17:00 cutoff. Numerical stock/news fusion remains disabled until lawful point-in-time historical news is available.

### Result

For the fresh 10-day Maybank analysis:

- Stock source: live.
- Stock observation: 21 July 2026.
- Stock model version: `838558e80923`.
- News source: live NewsAPI.
- News tone: balanced.
- Relationship: balanced.
- Combined mode: `stock_led_synthesis`.
- No fabricated fused probability was produced.

## 16. Ask & Learn / OpenRouter

### Implementation

The FastAPI chat endpoint grounds the prompt in the selected company, horizon and optional analysis context. It applies the educational/no-personal-advice system prompt, uses the configured primary OpenRouter model and then a configured fallback only if necessary.

### Live result

- HTTP 200.
- Primary model: `nvidia/nemotron-3-super-120b-a12b:free`.
- Fallback used: false.
- Response time: approximately 3.5 seconds.
- The answer correctly described 55% as a slight probability advantage rather than certainty.

## 17. FastAPI verification

### Result

- Backend API unit tests: 6 passed.
- Pipeline/regression tests: 80 passed.
- `/api/v1/health/live`: HTTP 200.
- `/api/v1/health/ready`: ready.
- Registry, metadata, schema and all three model files: available.
- System-status response exposed configuration booleans but no secret values.

Non-blocking warning: Starlette reports that the current `httpx` TestClient integration is deprecated and recommends `httpx2`. This affects test-maintenance, not the running API.

## 18. React/Vite verification

### Result

- TypeScript typecheck: passed.
- Vitest: 2 test files / 2 tests passed.
- Production Vite build: passed (2,236 modules transformed).
- Local frontend: HTTP 200.
- FastAPI + Vite restarted successfully through `start_stocksense.ps1`.

## 19. Cache and obsolete-artifact controls

### Implementation

- FastAPI prediction memory cache keys include the model-registry signature.
- FastAPI process-local prediction cache TTL: 15 minutes.
- Dashboard disk fallback cache schema: version 7.
- Dashboard disk fallback TTL: 15 minutes.
- Marketaux success-cache TTL and stale-fallback rules remain provider-specific and explicitly labelled.

### Audit action and result

- Archived 32 dashboard cache files.
- Archived 27 Marketaux response-cache files.
- Preserved Marketaux daily quota ledgers.
- Recreated empty runtime cache directories.
- Ran a fresh Maybank analysis to repopulate only current-version cache entries.
- Restarted FastAPI to clear all process-local memory caches.

The archived cache is stored under this audit directory and is no longer in a runtime lookup path.

## 20. Experimental numerical fusion

### Implementation and result

The fusion modules and CLI exist, but `/api/v1/system/status` correctly reports:

- Fusion data ready: false.
- Fusion model available: false for all horizons.
- Validation status: unavailable.

This is intentional. The active combined outlook is the transparent stock-led evidence synthesis, not an unvalidated numerical stock/news model.

## Defects found and corrected during this audit

1. **Stale active stock registry** — the regenerated model-input hash did not match the active registry. Corrected by fresh focused stock training.
2. **Stale acceptance summary** — the CSV described the previous quick champions. Corrected by recomputing Horizon Acceptance V2 reports against the new registry.
3. **FinBERT test contamination** — `test_split_manifest_is_stratified_and_disjoint` wrote its synthetic 60-row fixture into the real benchmark manifest. Corrected by redirecting the test write to a temporary directory and restoring the deterministic 4,838-row manifest.
4. **Pre-training runtime caches** — old dashboard and Marketaux response files were archived outside runtime paths; quota ledgers were preserved.

## Remaining limitations before acceptance-criteria work

1. The effective forward evidence is Experimental for all horizons; this is the next modelling problem, not a freshness problem.
2. Full CPU-only FinBERT retraining exceeded one hour. The artifact works and its evidence is internally consistent, but release-grade retraining should use a longer controlled job or suitable GPU.
3. FinBERT still lacks a Malaysian Bursa-news labelled evaluation set.
4. The frontend currently has only two component tests; API contracts and pipeline logic are well covered, but browser-level end-to-end interaction coverage should be expanded later.
5. The Starlette TestClient deprecation warning should be addressed during dependency maintenance.

## Final audit verdict

The active stock models, registry, feature schema and runtime outputs are current and mutually consistent. The old runtime caches are no longer active, and the API was restarted after cache archival. NewsAPI, conditional Marketaux, FinBERT inference, evidence synthesis, OpenRouter, FastAPI, React and Vite all executed successfully.

The system is therefore ready for the next separate task: improving and defending horizon-specific acceptance and modelling performance without confusing stale-artifact problems with genuine forecast weakness.
