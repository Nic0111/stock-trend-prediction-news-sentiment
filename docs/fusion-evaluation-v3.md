# Fusion evaluation v3

Fusion v3 asks one narrow question: **does point-in-time company news add
incremental out-of-sample probability information beyond the scientific stock
OOF probability?** It is a retrospective 2019–2025 study and does not change
production merely because a point estimate improves.

## Scientific contract

- Development and nested selection: 2019–2023.
- Shared decision-threshold selection: 2024.
- Retrospective benchmark: 2025.
- Supporting rolling evaluations: 2022, 2023, 2024 and 2025.
- The active Bursa/PhraseBank sentiment ensemble is frozen.
- No-news rows use exactly the stock probability, threshold and direction.
- Promotion is horizon-specific.

The primary challenger is an anchored correction:

```text
stock probability + point-in-time NewsEvidence
    -> IncrementalNewsFusion
    -> fused probability + explicit news adjustment + provenance
```

The stock log-odds coefficient is fixed at one. Only the regularised news
correction is learned, and its magnitude is capped using development data.
An explicit news-present intercept represents the occurrence/attention effect;
it is never applied to rows without eligible news.
The previous broad 17-news-feature stacker is still reported as a repaired
diagnostic (shared threshold and exact no-news fallback), but it is explicitly
ineligible for promotion.

## Rebuild commands

Run from the repository root. The scientific stock OOF file is reusable unless
the active stock model, stock dataset or feature schema changes.

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli score-news `
  --source data\fusion\historical-news-production-finbert.parquet `
  --output data\fusion\historical-news-production-ensemble-v3.parquet

.\.venv\Scripts\python.exe -m pipeline.fusion_cli import-news `
  data\fusion\historical-news-production-ensemble-v3.parquet `
  --output-dir data\historical-news-fusion\normalized-production-v3

.\.venv\Scripts\python.exe -m pipeline.fusion_cli fusion-window-candidates `
  --stock-path data\model-input-stock-dataset `
  --stock-probabilities data\fusion\scientific-stock-oof-v2.parquet `
  --news-dir data\historical-news-fusion\normalized-production-v3 `
  --policy config\fusion_evaluation_v3.json `
  --output data\fusion\fusion-window-candidates-v3.parquet

.\.venv\Scripts\python.exe -m pipeline.fusion_cli evaluate-v3 `
  --dataset data\fusion\fusion-window-candidates-v3.parquet `
  --policy config\fusion_evaluation_v3.json `
  --report results\fusion\fusion-evaluation-v3.json `
  --registry models\fusion\staged-fusion-v3-registry.json
```

## Promotion-grade route

Do not begin the expensive permutation/bootstrap evaluation until the selected
development configuration passes the frozen evidence-readiness policy:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli preflight-v3 `
  --dataset data\fusion\fusion-window-candidates-v3.parquet `
  --policy config\fusion_evaluation_v3.json `
  --report results\fusion\fusion-v3-readiness.json
```

The command deliberately returns a non-zero status when evidence is not ready.
That is a scientific stop signal, not a program failure. Add lawful Bursa or
issuer announcement exports, rescore them with the frozen sentiment model,
rebuild the window-candidate table, and repeat the preflight. Never relax the
readiness gates merely to obtain a positive fusion result.

Once every intended horizon passes, run the complete protocol twice. Each run
has a horizon-level atomic checkpoint and resumes completed horizons without
repeating them:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli evaluate-v3 `
  --dataset data\fusion\fusion-window-candidates-v3.parquet `
  --policy config\fusion_evaluation_v3.json `
  --report results\fusion\fusion-evaluation-v3-run-a.json `
  --registry models\fusion\staged-fusion-v3-registry-run-a.json `
  --checkpoint results\fusion\fusion-evaluation-v3-run-a.checkpoint.json

.\.venv\Scripts\python.exe -m pipeline.fusion_cli evaluate-v3 `
  --dataset data\fusion\fusion-window-candidates-v3.parquet `
  --policy config\fusion_evaluation_v3.json `
  --report results\fusion\fusion-evaluation-v3-run-b.json `
  --registry models\fusion\staged-fusion-v3-registry-run-b.json `
  --checkpoint results\fusion\fusion-evaluation-v3-run-b.checkpoint.json

.\.venv\Scripts\python.exe -m pipeline.fusion_cli compare-v3-runs `
  results\fusion\fusion-evaluation-v3-run-a.json `
  results\fusion\fusion-evaluation-v3-run-b.json `
  --output results\fusion\fusion-evaluation-v3-reproducibility.json
```

The comparison excludes timestamps and file locations, but includes scientific
inputs, selected configurations, predictions, statistics and acceptance
outcomes. Its canonical hash therefore checks reproducibility without treating
different output filenames as scientific differences.

## Runtime staging and promotion

An accepted offline horizon is still not automatically safe for runtime use.
Materialize the trailing 252-session surprise baselines required by the frozen
features:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli materialize-v3-runtime-state `
  --dataset data\fusion\fusion-window-candidates-v3.parquet `
  --registry models\fusion\staged-fusion-v3-registry-run-a.json `
  --output models\fusion\fusion-v3-runtime-state.json
```

The generated state remains explicitly not runtime-ready until an online
NewsEvidence adapter replays frozen historical cases and produces a passing
row-level parity report. Run the parity replay before promotion. It checks the
complete v3 feature schema, numerical parity for a deterministic sample from
every accepted horizon, and the exact stock-only input when news is absent:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli runtime-parity-v3 `
  --dataset data\fusion\fusion-window-candidates-v3.parquet `
  --registry models\fusion\staged-fusion-v3-registry-run-a.json `
  --runtime-state models\fusion\fusion-v3-runtime-state.json `
  --output results\fusion\fusion-v3-runtime-parity.json `
  --sample-per-ticker 10
```

Only after this command returns a passing report may the staged registry be
promoted:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli promote-v3 `
  --registry models\fusion\staged-fusion-v3-registry-run-a.json `
  --reproducibility-report results\fusion\fusion-evaluation-v3-reproducibility.json `
  --runtime-state models\fusion\fusion-v3-runtime-state.json `
  --runtime-parity-report results\fusion\fusion-v3-runtime-parity.json
```

Promotion fails closed if there is no accepted horizon, the runs differ, the
runtime state belongs to another dataset, or online/offline feature parity has
not passed. Rejected horizons remain stock-led and no-news inputs must preserve
the stock probability, threshold and direction exactly.

For a quick integrity run, add `--horizon 1 --permutations 10 --bootstraps 50
--skip-rolling --allow-unready`. Such a run can never provide promotion
evidence; normal `evaluate-v3` runs stop before resampling when preflight fails.

## Official disclosures

StockSense does not scrape or bypass access controls. Lawfully exported Bursa
or issuer metadata can be normalized into the frozen material-event taxonomy:

The source register and empty import template define the ten-company scope and
the provenance fields required before a row can enter the v3 evidence corpus:

- `data/fusion/official-disclosures-v3-source-register.json`
- `data/fusion/official-disclosures-v3-import-template.csv`

The register is a collection checklist, not a dataset. It deliberately contains
no generated article rows and does not claim that a source has been collected.
Exports must retain the original publication timestamp, URL and source kind.
Do not bypass Bursa controls or infer missing article text.

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli prepare-official-disclosures `
  <export.csv> --source-kind bursa_announcement `
  --output data\fusion\official-disclosures.parquet
```

Score and import that output before rebuilding window candidates. Routine
administrative notices are excluded; unclassified records remain visible for
review rather than being silently assigned a material category.

## Safety

The evaluator writes `fusion-registry-v5` to a **staged** path. Runtime keeps
the stock-led result unless a horizon passes every readiness/statistical gate,
all artifact identities still match, and the corresponding runtime surprise
baseline state has been materialized. The UI identifies the outcome as
`Fusion-supported` or `Stock-led fallback`.
