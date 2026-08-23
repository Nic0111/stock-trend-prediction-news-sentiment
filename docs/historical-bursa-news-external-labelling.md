# Historical Bursa news corpus: 2015–2025

This workflow collects GDELT candidate URLs for StockSense's ten configured
companies, recovers article text using free sources, and creates JSONL batches
for review or labelling. Candidate collection and recovery never call an LLM.
The separate automated weak-label runner uses only explicitly configured free
models. Neither path promotes FinBERT or changes the production fusion
registry.

## Run or resume

From PowerShell at the repository root:

```powershell
.\Collect-HistoricalBursaNews.cmd -ExecuteCandidates -Recover
```

If the complete range exceeds the remaining monthly BigQuery allowance, run a
safe subset with `-Years`:

```powershell
.\Collect-HistoricalBursaNews.cmd -Years "2015,2018,2019,2020,2021,2022,2023,2024,2025" -ExecuteCandidates -Recover
```

Run the deferred years after the UTC BigQuery month resets:

```powershell
.\Collect-HistoricalBursaNews.cmd -Years "2016,2017" -ExecuteCandidates -Recover
```

BigQuery monthly exports use `query-state.jsonl`; publisher, Web-NGram, Common
Crawl and Wayback recovery each keep their own append-only checkpoint. After
all 12 monthly exports exist, resume recovery without issuing another BigQuery
planning request:

```powershell
.\Collect-HistoricalBursaNews.cmd -RecoveryOnly -Recover
```

For one bounded recovery pass per year:

```powershell
.\Collect-HistoricalBursaNews.cmd -RecoveryOnly -Recover -SingleRecoveryPass
```

## Safety rules

- BigQuery first dry-runs every pending 2015–2025 shard.
- The completed 2023 export is reused.
- No candidate query executes if current-month use plus all pending dry-runs
  exceeds the configured free-only ceiling.
- `-RecoveryOnly` makes no BigQuery request and is the required resume mode
  when candidate exports are already complete or the monthly allowance is
  nearly exhausted.
- Each recovery pass pauses after 45 minutes by default.
- A year stops after two bounded passes without checkpoint progress.
- Web NGrams are attempted only for 2020 onward.
- Only publisher/Common Crawl/Wayback full text and High-quality Web-NGram
  pseudo-documents enter the labelling package.

## Labelling package

When recovery checkpoints are available, the runner regenerates:

```text
data/historical-news-fusion/gdelt/external-corpus-2015-2025/external-labelling/
```

The automated wrapper explicitly passes `--recommended-start-year 2015`.
This is intentional for the approved historical backfill; the reusable Python
API keeps its safer 2019 default unless an earlier start year is requested.

Do not begin full labelling until `package-summary.json` lists all intended
years and acceptable company coverage. Upload one file from `batches/` plus
`external-label-output.schema.json` and use `CHATGPT_LABELLING_PROMPT.md`.

For the repository's free-only automated weak-label path, run:

```powershell
.\scripts\Run-AutomatedBursaNewsLabelling.ps1
```

The runner uses two distinct `:free` OpenRouter models, resumes from its
append-only checkpoint, stops after repeated no-progress passes, and writes
disagreements or low-confidence cases to
`automated-weak-labels-approval-queue.jsonl`. Only rows subsequently accepted
by the LLM-assisted review workflow enter the active corpus; the active
provenance value is `llm_assisted_label` (displayed as **LLM-assisted label**).

After all batches are labelled, concatenate the raw JSONL outputs without a
header and return the combined file to Codex for schema, coverage, entity,
class-balance, leakage, chronology and training-viability checks.

## Codex or CSV labelling workflow

The repository cannot upload local files into a Codex conversation by itself.
It can, however, create deterministic, bounded batches that can be attached to
Codex (or another review platform), and it can validate the returned labels
before they become an experimental dataset. This is the supported workflow:

```powershell
python -m pipeline.fusion_cli gdelt-external-label-export `
  --recovery-root data/historical-news-fusion/gdelt/external-corpus-2015-2025 `
  --output-dir data/historical-news-fusion/gdelt/external-corpus-2015-2025/external-labelling-codex-v2 `
  --batch-size 25
```

For the canonical repository root, the exporter also includes the two saved
2023 pilot recovery checkpoints and deduplicates them with the year folders.
The versioned output directory keeps the older automated-label package intact.

Before labelling, inspect `package-summary.json` and
`external-label-snapshot-manifest.json`. Attach one file from `batches/`, its
matching `.manifest.json`, and `external-label-output.schema.json` to Codex.
Ask Codex to return raw JSONL only, one object per input line. The batch file
contains the bounded `label_text`, identity fields, and recovery provenance;
the full `article_text` is deliberately kept in the frozen snapshot and
`review-pack.csv`, not duplicated in the chat upload. Every returned object
must copy `article_id`, `content_sha256`, `label_text_sha256`, `batch_id`, and
`batch_sha256` exactly.

For manual or platform-based labelling, use `review-pack.csv` instead. Keep
the immutable identity columns unchanged and fill only the annotation columns.
The importer accepts the CSV directly, including `label_a`/`label_b` or
`annotator_1`/`annotator_2` columns when independent review is available.

Import a completed response into the experimental namespace:

```powershell
python -m pipeline.fusion_cli gdelt-external-label-import `
  --input-snapshot data/historical-news-fusion/gdelt/external-corpus-2015-2025/external-labelling/external-label-input.jsonl `
  --returned-labels path\to\returned-labels.jsonl `
  --output-dataset data/news-evidence-benchmark/external-news-experimental.jsonl `
  --review-queue data/news-evidence-benchmark/external-news-review-queue.jsonl `
  --report results/external-news-import/import-report.json
```

The import command rejects missing, duplicate, stale, or hash-mismatched
rows, malformed sentiment values, sentiment assigned to an irrelevant entity,
and incomplete JSONL batches. Accepted labels are written as
`llm_assisted_label`; the former source is retained in `label_source_original` for
audit only. Disagreements and `needs_review` rows are written to the review
queue.

The report records coverage by ticker/year/sentiment/source, confidence
summary, and review reasons. It does not call an LLM, train FinBERT, or alter
the runtime news retrieval pipeline. Run the dataset-readiness and FinBERT
benchmark checks after import before training or promoting a model.

## Balanced historical recovery

The GDELT exports are deliberately high-recall candidate lists. They are kept
unchanged, but recovery runs in a balanced per-ticker batch so a large number
of passing mentions for one bank cannot consume the whole free recovery
budget. The default is 60 candidates per company per year; re-running the
command resumes checkpointed requests without repeating completed work.

```powershell
.\scripts\Collect-HistoricalBursaNews.ps1 -RecoveryOnly -Recover -SingleRecoveryPass `
  -MaxRuntimeMinutes 5 -RecoveryCandidatesPerTicker 60
```

Use `-RecoverAllCandidates` only for a deliberate long-running research job.
It does not improve relevance by itself, because all raw discovery candidates
still require downstream entity disambiguation.
