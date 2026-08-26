# Fusion v4 evaluation workflow

Fusion v4 is an isolated research candidate. It is designed to test whether
point-in-time news adds incremental probability quality to the frozen stock
model; it does not alter the active runtime registry.

## Frozen protocol

- Development and nested selection: 2019–2023
- Calibration and shared decision threshold: 2024
- Retrospective benchmark: 2025
- Rolling-origin checks: 2022, 2023, 2024 and 2025
- Horizons: 1, 5 and 10 trading days

The protocol, feature contract, identity hashes and acceptance gates are locked
in `config/fusion_evaluation_v4.json`.

The current protocol uses `automated_structured_provenance` as its evidence
audit mode because this project does not require manual labels.  The audit
checks event IDs, point-in-time sentiment identity and aggregation-policy
identity.  It is not an independent semantic relevance review and is never
described as human ground truth.

## Safe data preparation

The input must be a point-in-time feature table generated from scientific
nested stock OOF probabilities and the active sentiment ensemble. It must carry
the v4 schema and matching stock, sentiment, corpus and aggregation-policy
identities. Never use the runtime-compatibility OOF table as the scientific
benchmark input.

First run the cheap preflight:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli preflight-v4 `
  --dataset data/fusion/fusion-window-candidates-v3-current-config.parquet `
  --report results/fusion/fusion-evaluation-v4-preflight.json
```

Then create an immutable, isolated copy for the evaluation:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli seal-v4-dataset `
  --source data/fusion/fusion-window-candidates-v3-current-config.parquet `
  --output data/fusion/fusion-features-v4-sealed.parquet `
  --manifest data/fusion/fusion-features-v4-sealed.manifest.json
```

Sealing is a copy-and-contract operation. It does not retrain, rescore or
modify production data. A stale identity fails closed instead of being
silently relabelled.

## Full benchmark

Only run the expensive command after preflight evidence gates pass:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli evaluate-v4 `
  --dataset data/fusion/fusion-features-v4-sealed.parquet `
  --report results/fusion/fusion-evaluation-v4.json `
  --registry models/fusion/staged-fusion-v4-registry.json `
  --permutations 1000 `
  --bootstraps 2000
```

The report contains compact occurrence/tone/reliability/event-impact arms,
content and complete-event placebos, moving-block bootstrap intervals,
Clark–West support, rolling origins, leave-one-company-out diagnostics and
Holm-adjusted acceptance checks. If any horizon fails, the staged registry
records `stock_led_fallback`; the active production registry is unchanged.

## Current sealed-candidate status

The current sealed table has strict artifact identity for all rows and covers
all ten companies and all three horizons. Window selection is constrained by
calibration-year evidence sufficiency before the development performance rule
chooses among eligible windows; this prevents a current-session-only variant
from winning a negligible score difference and then failing the locked sample
gate. The v12 corpus already contains the required point-in-time evidence for
the selected windows, so no manual relabelling is required.

## Independent blind-audit queue

When a feature table has no independent entity/materiality review, export a
deterministic queue from the matching normalized article corpus:

```powershell
.\.venv\Scripts\python.exe -m pipeline.fusion_cli export-v4-blind-audit `
  --dataset data/fusion/fusion-features-v4-expanded-v12-sealed.parquet `
  --article-root data/historical-news-fusion/normalized-production-v3-official-prepared-expanded-v12/normalized `
  --output results/fusion/fusion-v4-blind-audit-queue.csv `
  --manifest results/fusion/fusion-v4-blind-audit-queue.manifest.json `
  --sample-size 200
```

The CSV contains only the company, article text/URL and event identity. This
optional queue is retained for future academic audit work, but it is not used
by the current protocol and no manual labels are required. Do not replace a
future independent review with the model's existing relevance score or
sentiment label; automated provenance checks and semantic annotation are
different claims.
