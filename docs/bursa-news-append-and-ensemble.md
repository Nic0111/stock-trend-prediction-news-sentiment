# Append-only Bursa news and sentiment ensemble workflow

This workflow expands the historical Bursa-news corpus without replacing the
existing validated rows. It does not change the live NewsAPI/Marketaux
retrieval path, the production FinBERT artifact, or the stock/news fusion
registry.

## 1. Prepare a resumable, deduplicated recovery cohort

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli bursa-news-append prepare `
  --work-dir data/historical-news-fusion/gdelt/bursa-news-append-v1 `
  --target-relevant-per-ticker 100 `
  --recovery-multiplier 5
```

The command reads candidate exports only. It removes candidates already found
in the canonical corpus by ticker + normalised URL, article ID, and matching
same-event headline. It balances selection over month, source domain and the
2019–2025 train/validation/test periods. Re-running `prepare` in the same
workspace safely returns the original manifest instead of replacing a recovery
checkpoint. Use a new `--work-dir` to deliberately make a new sample.

### Class-balance cycle (recommended for the current corpus)

The balance-v2 planner fills sparse ticker x chronological-split x sentiment
cells rather than blindly adding a fixed number per company. The target is 40
rows per class/ticker: 28 training, 6 validation and 6 test. Sentiment is
unknown during retrieval, so the recovery cohort is oversampled eight-fold and
later prioritised for review.

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli bursa-news-append prepare `
  --dataset data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl `
  --candidate-root data/historical-news-fusion/gdelt/external-corpus-2015-2025 `
  --work-dir data/historical-news-fusion/gdelt/bursa-news-balance-v2 `
  --target-per-class-per-ticker 40 `
  --training-target-per-class 28 `
  --validation-target-per-class 6 `
  --test-target-per-class 6 `
  --recovery-multiplier 8 `
  --max-candidates-per-cycle 1000
```

The manifest records the exact deficit matrix and canonical dataset hash. A
second run against the same workspace resumes the checkpoint and does not
replace the selected cohort.

After a recovery/label/import cycle is complete, pass `--new-cycle` to select
the next bounded cohort in the same workspace. Prior-cycle candidate identities
are excluded and the recovery checkpoint is written under a new cycle folder.

## 2. Recover text within one safety budget

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli bursa-news-append recover `
  --work-dir data/historical-news-fusion/gdelt/bursa-news-balance-v2 `
  --max-runtime-minutes 45 `
  --max-download-gib 5
```

Recovery uses the existing free-only order: direct publisher text, GDELT Web
NGram reconstruction, Common Crawl, then Wayback. The 45 minutes applies to
the whole command, not to every calendar year. Each stage has checkpoints, so
the same command resumes unfinished work without reissuing completed requests.

## 3. Export a fresh Codex-review package

For the balance-v2 cycle, run the local triage step first. It uses the frozen
PhraseBank FinBERT only as a sampling hint (likely negative/neutral/positive,
uncertain and a 10% random-control lane). Predictions are not labels and are
not included in the Codex-facing package.

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli bursa-news-append triage `
  --work-dir data/historical-news-fusion/gdelt/bursa-news-balance-v2 `
  --random-control-share 0.10
```

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli bursa-news-append export-labels `
  --work-dir data/historical-news-fusion/gdelt/bursa-news-balance-v2
```

Only recovered, eligible, non-duplicate text enters the package. A returned
label file is validated into staging first:

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli bursa-news-append import-labels `
  --work-dir data/historical-news-fusion/gdelt/bursa-news-balance-v2 `
  --returned-labels C:\path\to\returned-labels.jsonl
```

Imported rows use the active project policy of `llm_assisted_label`, while
retaining `label_source_original=codex_single_review` and
`external_labelled_by=codex` for honest auditability. The canonical dataset is
still untouched. Only labelled rows with confidence >= 0.70 and
`label_status=labelled` enter append staging; lower-confidence or
`needs_review` rows are written to `staging/unresolved-confidence.jsonl`.

## 4. Commit only after review

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli bursa-news-append commit `
  --work-dir data/historical-news-fusion/gdelt/bursa-news-balance-v2
```

Commit rechecks the original dataset hash, deduplicates again, checks the
chronological split, validates the combined dataset and makes a timestamped
backup before atomically appending records. It changes the dataset only; it
does not retrain or deploy any model. In balance-v2 mode, only records that
fill a currently deficient ticker/split/class cell are appended. Valid surplus
labels and entity-negative examples are preserved under the cycle's
`reserves/` folder for later use instead of changing the class-balanced corpus.

## 5. Legacy workflow (superseded — do not use for promotion)

After the append is committed, run a fixed, recorded recipe. The recommended
first run below uses a 60% Bursa / 40% PhraseBank replay mix during
fine-tuning; this is not a runtime ensemble weight.

```powershell
& .\.venv\Scripts\python.exe -m pipeline.benchmark_bursa_finbert train `
  --profile full --device cuda `
  --bursa-replay-share 0.60 `
  --learning-rate 0.00002 `
  --seed 42 --seed 52 --seed 62 --primary-seed 42
```

The run records seeds, replay share, learning rate, epoch cap, best validation
epoch, validation metrics and one frozen 2025 confirmation result. Training
settings must be selected using the 2024 validation period—not by repeatedly
trying settings against the 2025 report.

## 6. Benchmark the 60/40 starting hypothesis fairly

```powershell
& .\.venv\Scripts\python.exe -m pipeline.benchmark_finbert_ensemble `
  --variant-id bursa_phrasebank_soft_vote_v2 `
  --weight-step 0.05
```

The weight grid is selected using 2024 validation macro-F1, balanced accuracy,
calibration and log loss. If choices tie, 60% Bursa is the deterministic
tie-breaker. The 2025 score is retrospective because it is already known; it
cannot promote a model on its own. Promotion also requires a separate,
untouched labelled holdout meeting all configured coverage, per-company,
calibration, class-F1, bootstrap, phrasebank non-inferiority and seed-stability
gates.

## 7. Promote only with explicit approval

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli sentiment-ensemble promote `
  --variant bursa_phrasebank_soft_vote_v2 `
  --benchmark-report results/finbert-ensemble-benchmark/latest.json `
  --approve
```

Promotion copies the immutable component paths, temperatures, weights and
artifact hashes from the passed benchmark report into runtime configuration.
It then switches `active_variant`. If the checks fail, the command stops with
no runtime change. Rollback is explicit too:

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli sentiment-ensemble rollback --approve
```

## Current guarded ensemble workflow

Use the commands below for all new Bursa + PhraseBank ensemble work. They
preserve the active PhraseBank model until an independently labelled 2026
holdout passes every acceptance gate.

### 1. Select the retrospective candidate on development evidence only

The runner compares the predeclared recipes and replay shares using expanding
2015--2024 folds. It uses the fixed Bursa seeds (42, 52 and 62) and records
all settings, hashes and safe-resume checkpoints. The known 2025 result is
reported only after selection and is not allowed to promote a model.

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli defensible-sentiment-ensemble `
  --run-id bursa-phrasebank-production-candidate-v1 `
  --profile full --device cuda `
  --training-recipe-grid config\finbert_ensemble_training_recipe_grid.json `
  --max-runtime-minutes 45
```

The grid contains the baseline, target-aware text, ticker/class-balanced
sampling and balanced focal-loss challengers. Re-run the same command after a
safe pause. If the recipe grid changes, use a new run ID rather than mixing
evidence into the old folder.

### 2. Prepare and seal a prospective 2026 acceptance cohort

The 2026 cohort remains outside the canonical 2015--2025 training corpus. Its
deterministic final/shadow assignment is made before LLM-assisted labels are imported.

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli prospective-sentiment-prepare `
  --candidates data\news-evidence-benchmark\bursa-news-2026-candidates.csv `
  --output-dir data\news-evidence-benchmark\prospective-ensemble-2026

& .\.venv\Scripts\python.exe -m pipeline.fusion_cli prospective-sentiment-import-labels `
  --labels data\news-evidence-benchmark\prospective-ensemble-2026\returned-labels.csv `
  --output-dir data\news-evidence-benchmark\prospective-ensemble-2026

& .\.venv\Scripts\python.exe -m pipeline.fusion_cli prospective-sentiment-bind `
  --benchmark-report results\finbert-ensemble-defensible\bursa-phrasebank-production-candidate-v1\report.json `
  --output-dir data\news-evidence-benchmark\prospective-ensemble-2026

& .\.venv\Scripts\python.exe -m pipeline.fusion_cli prospective-sentiment-evaluate `
  --output-dir data\news-evidence-benchmark\prospective-ensemble-2026
```

Only LLM-assisted labels with `llm_assisted_entity_relevant`,
`llm_assisted_sentiment_label`, and a confirmed/reviewed/validated `label_status`
are accepted. The one-time evaluation gates coverage, per-class F1, balanced
accuracy, calibration, clustered bootstrap non-inferiority, component weights
and PhraseBank retention without changing runtime configuration.

### 3. Promote only after the independent gates pass

```powershell
& .\.venv\Scripts\python.exe -m pipeline.fusion_cli sentiment-ensemble promote `
  --variant bursa_phrasebank_soft_vote_v2 `
  --benchmark-report data\news-evidence-benchmark\prospective-ensemble-2026\prospective-report.json `
  --approve
```

Promotion verifies the sealed protocol, exact three-seed Bursa average,
PhraseBank component, blend weights and artifact hashes before it changes the
runtime configuration. A failed gate leaves production unchanged.
