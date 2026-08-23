"""Command-line entry points for the provider-ready fusion experiment.

These commands only prepare and evaluate historical data. They do not promote
an artifact into the runtime Stock Outlook path; promotion remains a separate,
reviewed action after both benchmark gates pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.fusion_model import (
    DEFAULT_FUSION_DATASET,
    DEFAULT_FUSION_REGISTRY,
    DEFAULT_STAGED_FUSION_REGISTRY,
    FusionDatasetNotReadyError,
    train_fusion_models,
)
from pipeline.fusion_training import (
    DEFAULT_FUSION_COVERAGE_REPORT,
    DEFAULT_FUSION_REPORT,
    DEFAULT_FUSION_SIGNAL_REPORT,
    DEFAULT_FUSION_SHADOW_REPORT,
    evaluate_fusion_dataset,
    evaluate_fusion_shadow,
    promote_fusion_registry,
    write_fusion_coverage_report,
    write_fusion_signal_diagnostics,
)
from pipeline.fusion_ablation import DEFAULT_REPORT as DEFAULT_FUSION_ABLATION_REPORT, run_fusion_ablation
from pipeline.official_disclosure_news import run_prepare_official_disclosures
from pipeline.official_disclosure_export import (
    DEFAULT_CACHE as DEFAULT_OFFICIAL_EXPORT_CACHE,
    DEFAULT_OUTPUT as DEFAULT_OFFICIAL_EXPORT_OUTPUT,
    DEFAULT_REPORT as DEFAULT_OFFICIAL_EXPORT_REPORT,
    export_official_disclosures_v3,
)
from pipeline.fusion_stock_oof import run_build_stock_oof_probabilities
from pipeline.fusion_evaluation_v2 import (
    DEFAULT_POLICY as DEFAULT_FUSION_V2_POLICY,
    DEFAULT_REGISTRY as DEFAULT_FUSION_V2_REGISTRY,
    DEFAULT_REPORT as DEFAULT_FUSION_V2_REPORT,
    run_fusion_evaluation_v2,
)
from pipeline.fusion_evaluation_v3 import (
    DEFAULT_POLICY as DEFAULT_FUSION_V3_POLICY,
    DEFAULT_REGISTRY as DEFAULT_FUSION_V3_REGISTRY,
    DEFAULT_REPORT as DEFAULT_FUSION_V3_REPORT,
    FusionV3ReadinessError,
    compare_fusion_v3_reports,
    run_fusion_evaluation_v3,
    run_fusion_v3_preflight,
)
from pipeline.fusion_v3_promotion import (
    materialize_runtime_surprise_state,
    prepare_fusion_v3_acceptance_override,
    promote_fusion_v3_registry,
    run_runtime_feature_parity,
)
from pipeline.gdelt_historical_news import (
    DEFAULT_GDELT_DIR,
    DEFAULT_GDELT_QUERY_PROFILE,
    DEFAULT_CANDIDATES_PER_TICKER_MONTH,
    DEFAULT_CAPTURE_LAG_DAYS,
    DEFAULT_EXTERNAL_COLLECTION_YEARS,
    DEFAULT_PILOT_YEAR,
    DEFAULT_REQUEST_INTERVAL_SECONDS,
    BigQueryRunnerError,
    build_gdelt_pilot_report,
    collect_external_label_candidate_years,
    collect_remaining_gdelt_years,
    load_gdelt_export,
    run_gdelt_enrichment,
    run_authenticated_gdelt_queries,
    write_gdelt_query_bundle,
)
from pipeline.gdelt_ngram_pilot import (
    DEFAULT_CANDIDATE_DIR as DEFAULT_NGRAM_CANDIDATE_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_NGRAM_OUTPUT_DIR,
    run_gdelt_ngram_pilot,
)
from pipeline.external_news_label_export import (
    DEFAULT_OUTPUT_DIR as DEFAULT_EXTERNAL_LABEL_OUTPUT_DIR,
    DEFAULT_RECOVERY_ROOT as DEFAULT_EXTERNAL_RECOVERY_ROOT,
    export_external_label_package,
)
from pipeline.external_news_label_import import (
    ExternalNewsImportError,
    import_external_news_labels,
)
from pipeline.bursa_news_append import (
    DEFAULT_BALANCE_WORK_DIR,
    DEFAULT_MAX_CANDIDATES_PER_CYCLE,
    DEFAULT_RECOVERY_MULTIPLIER,
    DEFAULT_TARGET_RELEVANT_PER_TICKER,
    DEFAULT_TEST_CLASS_TARGET,
    DEFAULT_TRAINING_CLASS_TARGET,
    DEFAULT_VALIDATION_CLASS_TARGET,
    DEFAULT_WORK_DIR as DEFAULT_BURSA_APPEND_WORK_DIR,
    BursaNewsAppendError,
    append_status,
    commit_bursa_news_append,
    export_bursa_news_append_labels,
    import_bursa_news_append_labels,
    prepare_bursa_news_append,
    recover_bursa_news_append,
    triage_bursa_news_append,
)
from pipeline.sentiment_ensemble_promotion import (
    SentimentEnsemblePromotionError,
    manually_activate_sentiment_ensemble,
    promote_sentiment_ensemble,
    rollback_sentiment_ensemble,
)
from pipeline.chronological_finbert_ensemble import (
    CALIBRATION_MODE_SCALAR_TEMPERATURE,
    SUPPORTED_CALIBRATION_MODES,
    DEFAULT_RUN_ID as DEFAULT_CHRONOLOGICAL_ENSEMBLE_RUN_ID,
    ChronologicalEnsembleError,
    run_chronological_finbert_ensemble,
)
from pipeline.defensible_finbert_ensemble import (
    DEFAULT_RUN_ID as DEFAULT_DEFENSIBLE_ENSEMBLE_RUN_ID,
    DefensibleEnsembleError,
    load_training_recipe_grid,
    run_defensible_finbert_ensemble,
)
from pipeline.prospective_finbert_holdout import (
    ProspectiveHoldoutError,
    bind_prospective_candidate,
    evaluate_prospective_holdout,
    import_prospective_labels,
    prepare_prospective_holdout,
    prospective_holdout_status,
)
from pipeline.llm_bursa_news_labeler import (
    DEFAULT_APPROVAL_QUEUE,
    DEFAULT_HEARTBEAT_PATH,
    DEFAULT_LABEL_CHECKPOINT,
    DEFAULT_LABEL_OUTPUT,
    DEFAULT_LABEL_REPORT,
    run_llm_consensus_labelling,
)
from pipeline.bursa_news_remediation import (
    DEFAULT_MASTER_DATASET as DEFAULT_BURSA_REMEDIATION_MASTER,
    DEFAULT_OUTPUT_DIR as DEFAULT_BURSA_REMEDIATION_OUTPUT,
    DEFAULT_REVIEW_PACKS as DEFAULT_BURSA_REMEDIATION_PACKS,
    audit_bursa_news_labels,
    export_label_expansion_queue,
)
from pipeline.sentiment_remediation import (
    DEFAULT_BENCHMARK as DEFAULT_SENTIMENT_REMEDIATION_BENCHMARK,
    DEFAULT_CANDIDATE_ROOT as DEFAULT_SENTIMENT_REMEDIATION_CANDIDATES,
    DEFAULT_DATASET as DEFAULT_SENTIMENT_REMEDIATION_DATASET,
    DEFAULT_RUN_ROOT as DEFAULT_SENTIMENT_REMEDIATION_ROOT,
    SentimentRemediationError,
    remediation_status,
    resume_sentiment_remediation,
    run_sentiment_remediation,
)
from pipeline.historical_news_fusion_data import (
    run_build_fusion_window_candidates,
    run_build_point_in_time_features,
    run_fusion_data_readiness,
    run_import_historical_news_pipeline,
    run_score_historical_news_for_fusion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and train the experimental StockSense stock/news fusion challenger.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-news", help="Normalize provider exports into ticker/year Parquet partitions.")
    import_parser.add_argument("sources", nargs="+", help="CSV, JSON, JSONL or Parquet provider export paths.")
    import_parser.add_argument("--output-dir", default=None)
    import_parser.add_argument("--provider", default=None)

    disclosure_parser = subparsers.add_parser(
        "prepare-official-disclosures",
        help=(
            "Classify lawfully exported Bursa or issuer announcement metadata "
            "into the frozen material-event taxonomy; no scraping is performed."
        ),
    )
    disclosure_parser.add_argument("sources", nargs="+")
    disclosure_parser.add_argument("--output", required=True)
    disclosure_parser.add_argument(
        "--source-kind",
        choices=("bursa_announcement", "issuer_ir", "press_release"),
        required=True,
    )

    official_export_parser = subparsers.add_parser(
        "export-official-disclosures-v3",
        help=(
            "Fetch public issuer/Bursa disclosure metadata for fusion v3. "
            "Responses are cached and blocked sources are reported; no access "
            "control is bypassed."
        ),
    )
    official_export_parser.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    official_export_parser.add_argument("--tickers", nargs="*", default=None)
    official_export_parser.add_argument("--output", default=str(DEFAULT_OFFICIAL_EXPORT_OUTPUT))
    official_export_parser.add_argument("--report", default=str(DEFAULT_OFFICIAL_EXPORT_REPORT))
    official_export_parser.add_argument("--cache-dir", default=str(DEFAULT_OFFICIAL_EXPORT_CACHE))
    official_export_parser.add_argument("--request-interval-seconds", type=float, default=0.25)

    score_parser = subparsers.add_parser(
        "score-news",
        help=(
            "Score reviewed historical news with the unchanged production "
            "FinBERT artifact for runtime-matched fusion features."
        ),
    )
    score_parser.add_argument("--source", required=True)
    score_parser.add_argument("--output", required=True)

    readiness_parser = subparsers.add_parser("readiness", help="Report historical-news coverage and feature readiness.")
    readiness_parser.add_argument("--data-dir", default=None)

    feature_parser = subparsers.add_parser("features", help="Build point-in-time fusion features from stock rows and normalized news.")
    feature_parser.add_argument("--stock-path", required=True)
    feature_parser.add_argument("--news-dir", default=None)
    feature_parser.add_argument("--output", default=None)
    feature_parser.add_argument("--stock-probabilities", default=None)

    window_parser = subparsers.add_parser(
        "fusion-window-candidates",
        help="Build every frozen v2 news-window/half-life candidate from scientific stock OOF probabilities.",
    )
    window_parser.add_argument("--stock-path", required=True)
    window_parser.add_argument("--stock-probabilities", required=True)
    window_parser.add_argument("--news-dir", default=None)
    window_parser.add_argument("--policy", default=str(DEFAULT_FUSION_V2_POLICY))
    window_parser.add_argument("--output", required=True)
    window_parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Optional horizon shard(s); omit to build all policy horizons.",
    )

    oof_parser = subparsers.add_parser(
        "stock-oof",
        help="Generate nested out-of-fold stock probabilities for fusion features.",
    )
    oof_parser.add_argument("--stock-path", required=True)
    oof_parser.add_argument("--registry", default=None)
    oof_parser.add_argument("--output", required=True)
    oof_parser.add_argument(
        "--mode",
        choices=("scientific", "runtime-compatibility"),
        default="runtime-compatibility",
    )
    oof_parser.add_argument(
        "--search-profile",
        choices=("quick", "focused", "full", "official"),
        default="focused",
    )

    v2_parser = subparsers.add_parser(
        "evaluate-v2",
        help="Run the frozen, all-row, capacity-matched retrospective fusion protocol.",
    )
    v2_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    v2_parser.add_argument("--policy", default=str(DEFAULT_FUSION_V2_POLICY))
    v2_parser.add_argument("--report", default=str(DEFAULT_FUSION_V2_REPORT))
    v2_parser.add_argument("--registry", default=str(DEFAULT_FUSION_V2_REGISTRY))
    v2_parser.add_argument("--permutations", type=int, default=None)
    v2_parser.add_argument("--bootstraps", type=int, default=None)
    v2_parser.add_argument(
        "--horizon", action="append", type=int, dest="v2_horizons", default=None
    )
    v2_parser.add_argument(
        "--skip-rolling",
        action="store_true",
        help="Smoke-test only: omit rolling origins and force the acceptance gate to fail.",
    )

    v3_parser = subparsers.add_parser(
        "evaluate-v3",
        help=(
            "Run horizon-specific anchored residual fusion with exact no-news "
            "fallback, ablations, placebos and strict article provenance."
        ),
    )
    v3_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    v3_parser.add_argument("--policy", default=str(DEFAULT_FUSION_V3_POLICY))
    v3_parser.add_argument("--report", default=str(DEFAULT_FUSION_V3_REPORT))
    v3_parser.add_argument("--registry", default=str(DEFAULT_FUSION_V3_REGISTRY))
    v3_parser.add_argument("--permutations", type=int, default=None)
    v3_parser.add_argument("--bootstraps", type=int, default=None)
    v3_parser.add_argument(
        "--horizon", action="append", type=int, dest="v3_horizons", default=None
    )
    v3_parser.add_argument(
        "--skip-rolling",
        action="store_true",
        help="Smoke-test only: omit rolling-origin evidence so acceptance fails.",
    )
    v3_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Resumable horizon checkpoint (defaults beside the report).",
    )
    v3_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and start the requested run again.",
    )
    v3_parser.add_argument(
        "--allow-unready",
        action="store_true",
        help=(
            "Diagnostic-only override for smoke tests. It bypasses the evidence "
            "readiness stop and can never produce a promotion-ready result."
        ),
    )

    preflight_parser = subparsers.add_parser(
        "preflight-v3",
        help="Run selection-aware v3 readiness checks without the final resampling.",
    )
    preflight_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    preflight_parser.add_argument("--policy", default=str(DEFAULT_FUSION_V3_POLICY))
    preflight_parser.add_argument("--report", required=True)
    preflight_parser.add_argument(
        "--horizon",
        action="append",
        type=int,
        dest="preflight_horizons",
        default=None,
    )

    reproduce_parser = subparsers.add_parser(
        "compare-v3-runs",
        help="Compare two v3 reports by canonical scientific content.",
    )
    reproduce_parser.add_argument("first_report")
    reproduce_parser.add_argument("second_report")
    reproduce_parser.add_argument("--output", required=True)

    baseline_parser = subparsers.add_parser(
        "materialize-v3-runtime-state",
        help="Stage trailing news histories for accepted v3 horizons.",
    )
    baseline_parser.add_argument("--dataset", required=True)
    baseline_parser.add_argument("--registry", required=True)
    baseline_parser.add_argument("--output", required=True)

    parity_parser = subparsers.add_parser(
        "runtime-parity-v3",
        help=(
            "Replay frozen v3 rows through the live NewsEvidence adapter and "
            "verify feature and no-news parity before promotion."
        ),
    )
    parity_parser.add_argument("--dataset", required=True)
    parity_parser.add_argument("--registry", required=True)
    parity_parser.add_argument("--runtime-state", required=True)
    parity_parser.add_argument("--output", required=True)
    parity_parser.add_argument("--sample-per-ticker", type=int, default=10)
    parity_parser.add_argument("--tolerance", type=float, default=1e-6)

    promote_v3_parser = subparsers.add_parser(
        "promote-v3",
        help="Fail-closed promotion after reproducibility and runtime parity pass.",
    )
    promote_v3_parser.add_argument("--registry", required=True)
    promote_v3_parser.add_argument("--reproducibility-report", required=True)
    promote_v3_parser.add_argument("--runtime-state", required=True)
    promote_v3_parser.add_argument("--runtime-parity-report", required=True)
    promote_v3_parser.add_argument("--active-registry", default=None)
    promote_v3_parser.add_argument(
        "--allow-acceptance-override",
        action="store_true",
        help="Allow promotion only when the registry carries an explicit audited override.",
    )

    override_v3_parser = subparsers.add_parser(
        "prepare-v3-acceptance-override",
        help="Create an explicit audit-preserving v3 runtime override registry.",
    )
    override_v3_parser.add_argument("--registry", required=True)
    override_v3_parser.add_argument("--evaluation-report", required=True)
    override_v3_parser.add_argument("--output", required=True)
    override_v3_parser.add_argument("--reason", required=True)

    train_parser = subparsers.add_parser("train", help="Train experimental horizon-specific fusion artifacts when readiness passes.")
    train_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    train_parser.add_argument("--registry", default=str(DEFAULT_STAGED_FUSION_REGISTRY))
    train_parser.add_argument("--ticker", action="append", dest="tickers", default=None)
    train_parser.add_argument("--min-rows", type=int, default=100)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run the frozen 2019-2025 challenger evaluation and write a guarded registry.",
    )
    evaluate_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    evaluate_parser.add_argument("--registry", default=str(DEFAULT_STAGED_FUSION_REGISTRY))
    evaluate_parser.add_argument("--report", default=str(DEFAULT_FUSION_REPORT))
    evaluate_parser.add_argument("--ticker", action="append", dest="tickers", default=None)
    evaluate_parser.add_argument("--min-rows", type=int, default=100)

    ablation_parser = subparsers.add_parser(
        "ablation",
        help="Compare stock-only, news-only, stock+news and shuffled-news placebo on matched 2025 rows.",
    )
    ablation_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    ablation_parser.add_argument("--report", default=str(DEFAULT_FUSION_ABLATION_REPORT))

    coverage_parser = subparsers.add_parser(
        "coverage",
        help=(
            "Write the ticker-by-horizon-by-period Medium/High evidence "
            "matrix used by the guarded evaluator."
        ),
    )
    coverage_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    coverage_parser.add_argument(
        "--report", default=str(DEFAULT_FUSION_COVERAGE_REPORT)
    )
    coverage_parser.add_argument(
        "--ticker", action="append", dest="tickers", default=None
    )

    signal_parser = subparsers.add_parser(
        "signal-diagnostics",
        help=(
            "Measure the historical association between signed news tone and "
            "future returns without training or promoting a model."
        ),
    )
    signal_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    signal_parser.add_argument(
        "--report", default=str(DEFAULT_FUSION_SIGNAL_REPORT)
    )

    shadow_parser = subparsers.add_parser(
        "shadow",
        help="Evaluate frozen fusion artifacts on later labelled rows without tuning.",
    )
    shadow_parser.add_argument("--dataset", required=True)
    shadow_parser.add_argument("--registry", default=str(DEFAULT_FUSION_REGISTRY))
    shadow_parser.add_argument("--report", default=str(DEFAULT_FUSION_SHADOW_REPORT))
    shadow_parser.add_argument("--start-date", default="2026-01-01")
    shadow_parser.add_argument("--end-date", default=None)

    promote_parser = subparsers.add_parser(
        "promote",
        help="Promote accepted staged fusion horizons into the active runtime registry.",
    )
    promote_parser.add_argument(
        "--staged-registry", default=str(DEFAULT_STAGED_FUSION_REGISTRY)
    )
    promote_parser.add_argument(
        "--active-registry", default=str(DEFAULT_FUSION_REGISTRY)
    )
    promote_parser.add_argument("--approve", action="store_true")

    gdelt_query = subparsers.add_parser(
        "gdelt-query",
        help="Write the GDELT BigQuery SQL and reproducibility manifest.",
    )
    gdelt_query.add_argument("--output-dir", default=str(DEFAULT_GDELT_DIR))
    gdelt_query.add_argument("--start-date", default="2023-01-01")
    gdelt_query.add_argument("--end-date", default="2023-12-31")
    gdelt_query.add_argument("--pilot-year", type=int, default=None)
    gdelt_query.add_argument("--shard", choices=["single", "monthly"], default="monthly")
    gdelt_query.add_argument("--max-shard-gib", type=float, default=75.0)
    gdelt_query.add_argument("--monthly-budget-gib", type=float, default=900.0)
    gdelt_query.add_argument("--ticker", action="append", dest="tickers", default=None)
    gdelt_query.add_argument(
        "--query-profile", default=DEFAULT_GDELT_QUERY_PROFILE
    )
    gdelt_query.add_argument(
        "--max-candidates-per-ticker-month",
        type=int,
        default=DEFAULT_CANDIDATES_PER_TICKER_MONTH,
    )

    gdelt_run = subparsers.add_parser(
        "gdelt-run",
        help="Run generated GDELT SQL shards through authenticated BigQuery and export local JSONL.",
    )
    gdelt_run.add_argument("--project-id", required=True)
    gdelt_run.add_argument("--query-dir", default=str(DEFAULT_GDELT_DIR / "queries"))
    gdelt_run.add_argument("--export-dir", default=str(DEFAULT_GDELT_DIR / "exports"))
    gdelt_run.add_argument("--state", default=None)
    gdelt_run.add_argument("--maximum-bytes-billed-gib", type=float, default=75.0)
    gdelt_run.add_argument("--monthly-budget-gib", type=float, default=900.0)
    gdelt_run.add_argument("--location", default="US")
    gdelt_run.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Validate all shards and project-wide quota without executing or exporting them.",
    )

    gdelt_enrich = subparsers.add_parser(
        "gdelt-enrich",
        help="Recover exported GDELT URLs through Common Crawl and score accepted text.",
    )
    gdelt_enrich.add_argument("--export", required=True)
    gdelt_enrich.add_argument("--output-dir", default=str(DEFAULT_GDELT_DIR))
    gdelt_enrich.add_argument("--max-articles", type=int, default=None)
    gdelt_enrich.add_argument("--pilot-per-ticker", type=int, default=None)
    gdelt_enrich.add_argument("--checkpoint", default=None)
    gdelt_enrich.add_argument(
        "--request-interval-seconds",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL_SECONDS,
    )
    gdelt_enrich.add_argument(
        "--capture-lag-days", type=int, default=DEFAULT_CAPTURE_LAG_DAYS
    )
    gdelt_enrich.add_argument(
        "--score-sentiment",
        action="store_true",
        help="Optionally run the local FinBERT scorer after entity filtering.",
    )
    gdelt_enrich.add_argument(
        "--approvals",
        default=None,
        help="Optional CSV/JSONL decisions exported from entity-approval-queue.jsonl.",
    )
    gdelt_enrich.add_argument(
        "--require-indexed-gate",
        action="store_true",
        help="Stop before Common Crawl unless every configured company passes the indexed candidate gate.",
    )
    gdelt_enrich.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=45.0,
        help="Pause safely after this many minutes; rerun with the same checkpoint to resume.",
    )
    gdelt_enrich.add_argument("--heartbeat", default=None)

    llm_labels = subparsers.add_parser(
        "label-bursa-news-llm",
        help="Create dual-model consensus pseudo-labels from recovered entity candidates.",
    )
    llm_labels.add_argument("--source", required=True)
    llm_labels.add_argument("--output", default=str(DEFAULT_LABEL_OUTPUT))
    llm_labels.add_argument("--report", default=str(DEFAULT_LABEL_REPORT))
    llm_labels.add_argument("--checkpoint", default=str(DEFAULT_LABEL_CHECKPOINT))
    llm_labels.add_argument("--heartbeat", default=str(DEFAULT_HEARTBEAT_PATH))
    llm_labels.add_argument(
        "--approval-queue", default=str(DEFAULT_APPROVAL_QUEUE)
    )
    llm_labels.add_argument(
        "--approvals",
        default=None,
        help="Optional reviewed queue; only approved prior abstentions are applied.",
    )
    llm_labels.add_argument("--max-runtime-minutes", type=float, default=45.0)
    llm_labels.add_argument("--max-requests", type=int, default=None)
    llm_labels.add_argument("--request-interval-seconds", type=float, default=1.0)
    llm_labels.add_argument(
        "--recommended-start-year",
        type=int,
        default=2019,
        choices=range(2015, 2026),
        help=(
            "Earliest publication year to label. Use 2015 only when the "
            "historical backfill has been explicitly approved."
        ),
    )

    gdelt_pilot = subparsers.add_parser(
        "gdelt-pilot",
        help="Evaluate the mandatory pilot archive from prepared tabular files.",
    )
    gdelt_pilot.add_argument("--indexed", required=True)
    gdelt_pilot.add_argument("--recovered", required=True)
    gdelt_pilot.add_argument("--accepted", required=True)
    gdelt_pilot.add_argument("--manual-review", default=None)
    gdelt_pilot.add_argument(
        "--llm-labels",
        default=None,
        help="Dual-model consensus pseudo-label artifact used as the pilot reference.",
    )
    gdelt_pilot.add_argument("--features", default=None)
    gdelt_pilot.add_argument("--pilot-year", type=int, default=DEFAULT_PILOT_YEAR)
    gdelt_pilot.add_argument("--output-dir", default=str(DEFAULT_GDELT_DIR))

    ngram_pilot = subparsers.add_parser(
        "gdelt-ngram-pilot",
        help=(
            "Run or resume free-only historical publisher/Web-NGram/archive "
            "recovery (Web NGrams are used from 2020 onward)."
        ),
    )
    ngram_pilot.add_argument("--year", type=int, default=DEFAULT_PILOT_YEAR)
    ngram_pilot.add_argument(
        "--candidate-dir", default=str(DEFAULT_NGRAM_CANDIDATE_DIR)
    )
    ngram_pilot.add_argument("--sample-per-ticker", type=int, default=30)
    ngram_pilot.add_argument("--paired-per-ticker", type=int, default=10)
    ngram_pilot.add_argument("--output-dir", default=str(DEFAULT_NGRAM_OUTPUT_DIR))
    ngram_pilot.add_argument("--max-runtime-minutes", type=float, default=45.0)
    ngram_pilot.add_argument("--max-download-gib", type=float, default=5.0)
    ngram_pilot.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Bounded smoke-test override; the full pilot leaves this unset.",
    )
    ngram_pilot.add_argument(
        "--request-interval-seconds",
        type=float,
        default=1.0,
    )
    ngram_pilot.add_argument(
        "--network-timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request timeout for publisher and archive recovery.",
    )
    ngram_pilot.add_argument(
        "--max-stage-attempts",
        type=int,
        default=3,
        help=(
            "Maximum attempts for clearly transient publisher, NGram or "
            "archive failures. Recovered and deterministic failures are never "
            "repeated."
        ),
    )
    ngram_pilot.add_argument(
        "--max-label-requests",
        type=int,
        default=None,
        help="Pause paired LLM labelling after this many requests per representation.",
    )
    ngram_pilot.add_argument(
        "--skip-labels",
        action="store_true",
        help="Recover and score FinBERT pairs without calling OpenRouter.",
    )
    ngram_pilot.add_argument(
        "--skip-archives",
        action="store_true",
        help="Smoke-test option that stops after publisher and Web NGrams.",
    )
    ngram_pilot.add_argument(
        "--skip-ngrams",
        action="store_true",
        help=(
            "Skip Web NGram reconstruction and continue unresolved publisher "
            "rows directly to the archive fallback."
        ),
    )
    ngram_pilot.add_argument(
        "--candidate-only",
        action="store_true",
        help=(
            "Export every deduplicated candidate for the selected year. "
            "No publisher, Web-NGram, archive, FinBERT, LLM, or acceptance-gate "
            "stage is run."
        ),
    )

    remaining = subparsers.add_parser(
        "gdelt-remaining-years",
        help="Dry-run all remaining years together and optionally execute only after the 2023 pilot passes.",
    )
    remaining.add_argument("--project-id", required=True)
    remaining.add_argument("--pilot-report", required=True)
    remaining.add_argument("--output-dir", default=str(DEFAULT_GDELT_DIR))
    remaining.add_argument(
        "--year", action="append", dest="years", type=int, default=None
    )
    remaining.add_argument("--maximum-bytes-billed-gib", type=float, default=75.0)
    remaining.add_argument("--monthly-budget-gib", type=float, default=900.0)
    remaining.add_argument("--location", default="US")
    remaining.add_argument(
        "--execute",
        action="store_true",
        help="Execute only after the aggregate dry-run fits the monthly safety ceiling.",
    )

    external_corpus = subparsers.add_parser(
        "gdelt-external-corpus",
        help=(
            "Dry-run or collect free-only 2015-2025 GDELT candidates for "
            "external labelling without calling an LLM."
        ),
    )
    external_corpus.add_argument("--project-id", required=True)
    external_corpus.add_argument(
        "--output-dir",
        default=str(DEFAULT_GDELT_DIR / "external-corpus-2015-2025"),
    )
    external_corpus.add_argument(
        "--year", action="append", dest="years", type=int, default=None
    )
    external_corpus.add_argument(
        "--reuse-2023-export",
        default=str(
            DEFAULT_GDELT_DIR / "pilot-2023-high-recall-v3b" / "exports"
        ),
    )
    external_corpus.add_argument(
        "--maximum-bytes-billed-gib", type=float, default=20.0
    )
    external_corpus.add_argument(
        "--monthly-budget-gib", type=float, default=900.0
    )
    external_corpus.add_argument("--location", default="US")
    external_corpus.add_argument(
        "--query-profile", default=DEFAULT_GDELT_QUERY_PROFILE
    )
    external_corpus.add_argument(
        "--max-candidates-per-ticker-month",
        type=int,
        default=DEFAULT_CANDIDATES_PER_TICKER_MONTH,
    )
    external_corpus.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute only when all pending years plus current-month usage fit "
            "under the free-only project ceiling."
        ),
    )
    label_export = subparsers.add_parser(
        "gdelt-external-label-export",
        help=(
            "Package recovered 2015-2025 article text for external labelling "
            "without calling an LLM or changing production."
        ),
    )
    label_export.add_argument(
        "--recovery-root", default=str(DEFAULT_EXTERNAL_RECOVERY_ROOT)
    )
    label_export.add_argument(
        "--output-dir", default=str(DEFAULT_EXTERNAL_LABEL_OUTPUT_DIR)
    )
    label_export.add_argument(
        "--year", action="append", dest="years", type=int, default=None
    )
    label_export.add_argument(
        "--extra-file", action="append", dest="extra_files", default=None
    )
    label_export.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Requested rows per batch; exporter hard-caps this at 25 and 60,000 characters.",
    )

    label_import = subparsers.add_parser(
        "gdelt-external-label-import",
        help=(
            "Validate returned Codex/reviewer labels against a frozen external "
            "snapshot and write an experimental-only dataset plus review queue."
        ),
    )
    label_import.add_argument(
        "--input-snapshot",
        required=True,
        help="Frozen external-label-input.jsonl produced by gdelt-external-label-export.",
    )
    label_import.add_argument(
        "--returned-labels",
        required=True,
        help="Returned JSONL or reviewer CSV; every eligible article ID must be present.",
    )
    label_import.add_argument(
        "--output-dataset",
        default=None,
        help="Experimental JSONL destination (never a production dataset).",
    )
    label_import.add_argument("--review-queue", default=None)
    label_import.add_argument("--report", dest="report_path", default=None)
    label_import.add_argument(
        "--allow-backfill",
        action="store_true",
        help="Opt in to including 2015–2018 rows in the experimental training split.",
    )

    bursa_append = subparsers.add_parser(
        "bursa-news-append",
        help=(
            "Append-only, deduplicated Bursa historical-news expansion. "
            "The canonical dataset is changed only by the explicit commit action."
        ),
    )
    bursa_append_subcommands = bursa_append.add_subparsers(
        dest="bursa_append_action", required=True
    )
    append_prepare = bursa_append_subcommands.add_parser(
        "prepare", help="Select unseen, balanced candidates without network calls."
    )
    append_prepare.add_argument(
        "--dataset",
        default="data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl",
    )
    append_prepare.add_argument(
        "--candidate-root", default=str(DEFAULT_EXTERNAL_RECOVERY_ROOT)
    )
    append_prepare.add_argument("--work-dir", default=str(DEFAULT_BURSA_APPEND_WORK_DIR))
    append_prepare.add_argument("--year", action="append", dest="years", type=int)
    append_prepare.add_argument(
        "--ticker",
        action="append",
        dest="bursa_append_tickers",
        help="Repeat to restrict a remediation cycle to selected Bursa tickers.",
    )
    append_prepare.add_argument(
        "--target-relevant-per-ticker",
        type=int,
        default=DEFAULT_TARGET_RELEVANT_PER_TICKER,
    )
    append_prepare.add_argument(
        "--recovery-multiplier", type=int, default=DEFAULT_RECOVERY_MULTIPLIER
    )
    append_prepare.add_argument(
        "--target-per-class-per-ticker",
        type=int,
        default=None,
        help=(
            "Activate class/split balancing; target this many relevant rows "
            "per sentiment class and ticker."
        ),
    )
    append_prepare.add_argument(
        "--training-target-per-class",
        type=int,
        default=DEFAULT_TRAINING_CLASS_TARGET,
    )
    append_prepare.add_argument(
        "--validation-target-per-class",
        type=int,
        default=DEFAULT_VALIDATION_CLASS_TARGET,
    )
    append_prepare.add_argument(
        "--test-target-per-class",
        type=int,
        default=DEFAULT_TEST_CLASS_TARGET,
    )
    append_prepare.add_argument(
        "--max-candidates-per-cycle",
        type=int,
        default=DEFAULT_MAX_CANDIDATES_PER_CYCLE,
        help="Bound the raw recovery cohort; rerun the same workspace to continue.",
    )
    append_prepare.add_argument(
        "--new-cycle",
        action="store_true",
        help=(
            "Explicitly select a next bounded cohort in the same balance workspace; "
            "prior-cycle candidate identities are excluded."
        ),
    )
    append_recover = bursa_append_subcommands.add_parser(
        "recover", help="Run one bounded, resumable free-only recovery pass."
    )
    append_recover.add_argument("--work-dir", default=str(DEFAULT_BURSA_APPEND_WORK_DIR))
    append_recover.add_argument("--max-runtime-minutes", type=float, default=45.0)
    append_recover.add_argument("--max-download-gib", type=float, default=5.0)
    append_recover.add_argument("--request-interval-seconds", type=float, default=1.0)
    append_recover.add_argument(
        "--skip-archives",
        action="store_true",
        help="Skip Common Crawl/Wayback and finish direct/NGram recovery when archive services are unavailable.",
    )
    append_triage = bursa_append_subcommands.add_parser(
        "triage",
        help="Prioritise recovered rows for Codex review using a frozen local hint model.",
    )
    append_triage.add_argument(
        "--work-dir", default=str(DEFAULT_BALANCE_WORK_DIR)
    )
    append_triage.add_argument("--random-control-share", type=float, default=0.10)
    append_export = bursa_append_subcommands.add_parser(
        "export-labels", help="Create a fresh Codex batch package from recovered rows."
    )
    append_export.add_argument("--work-dir", default=str(DEFAULT_BURSA_APPEND_WORK_DIR))
    append_export.add_argument("--batch-size", type=int, default=25)
    append_import = bursa_append_subcommands.add_parser(
        "import-labels", help="Validate Codex labels into append-only staging."
    )
    append_import.add_argument("--work-dir", default=str(DEFAULT_BURSA_APPEND_WORK_DIR))
    append_import.add_argument("--returned-labels", required=True)
    append_commit = bursa_append_subcommands.add_parser(
        "commit", help="Atomically append validated staging rows to the canonical corpus."
    )
    append_commit.add_argument("--work-dir", default=str(DEFAULT_BURSA_APPEND_WORK_DIR))
    append_commit.add_argument("--dataset", default=None)
    append_status_parser = bursa_append_subcommands.add_parser(
        "status", help="Show append workspace progress without changing data."
    )
    append_status_parser.add_argument("--work-dir", default=str(DEFAULT_BURSA_APPEND_WORK_DIR))

    sentiment_ensemble = subparsers.add_parser(
        "sentiment-ensemble",
        help="Explicitly promote or roll back an independently benchmarked FinBERT ensemble.",
    )
    sentiment_ensemble_actions = sentiment_ensemble.add_subparsers(
        dest="sentiment_ensemble_action", required=True
    )
    ensemble_promote = sentiment_ensemble_actions.add_parser(
        "promote", help="Activate a passed candidate only with --approve."
    )
    ensemble_promote.add_argument("--variant", required=True)
    ensemble_promote.add_argument(
        "--benchmark-report",
        default="results/finbert-ensemble-benchmark/latest.json",
    )
    ensemble_promote.add_argument(
        "--runtime-config", default="models/sentiment-runtime-config.json"
    )
    ensemble_promote.add_argument("--policy", default="config/bursa_news_dataset_policy.json")
    ensemble_promote.add_argument("--approve", action="store_true")
    ensemble_manual = sentiment_ensemble_actions.add_parser(
        "manual-activate",
        help="Activate a candidate as an explicitly marked experimental override.",
    )
    ensemble_manual.add_argument("--variant", required=True)
    ensemble_manual.add_argument(
        "--benchmark-report",
        default="results/finbert-ensemble-defensible/production-ready-20260807-v1/report.json",
    )
    ensemble_manual.add_argument(
        "--runtime-config", default="models/sentiment-runtime-config.json"
    )
    ensemble_manual.add_argument("--reason", required=True)
    ensemble_manual.add_argument("--approve", action="store_true")
    ensemble_rollback = sentiment_ensemble_actions.add_parser(
        "rollback", help="Restore the prior sentiment variant with --approve."
    )
    ensemble_rollback.add_argument("--variant", default=None)
    ensemble_rollback.add_argument(
        "--runtime-config", default="models/sentiment-runtime-config.json"
    )
    ensemble_rollback.add_argument("--approve", action="store_true")

    chronological_ensemble = subparsers.add_parser(
        "chronological-sentiment-ensemble",
        help=(
            "Run/resume the 2015-2025 retrospective Bursa/PhraseBank FinBERT "
            "ensemble benchmark without changing runtime sentiment."
        ),
    )
    chronological_ensemble.add_argument(
        "--dataset", default="data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl"
    )
    chronological_ensemble.add_argument("--policy", default="config/bursa_news_dataset_policy.json")
    chronological_ensemble.add_argument(
        "--phrasebank-model-directory", default="models/experimental/finbert-sentiment"
    )
    chronological_ensemble.add_argument("--run-id", default=DEFAULT_CHRONOLOGICAL_ENSEMBLE_RUN_ID)
    chronological_ensemble.add_argument("--profile", choices=["full", "quick"], default="full")
    chronological_ensemble.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    chronological_ensemble.add_argument("--seed", dest="chronological_seeds", type=int, action="append")
    chronological_ensemble.add_argument("--bursa-replay-share", type=float, default=0.70)
    chronological_ensemble.add_argument("--learning-rate", type=float, default=2e-5)
    chronological_ensemble.add_argument("--max-epochs", type=int, default=None)
    chronological_ensemble.add_argument("--embargo-days", type=int, default=7)
    chronological_ensemble.add_argument("--bootstrap-iterations", type=int, default=2000)
    chronological_ensemble.add_argument("--max-runtime-minutes", type=float, default=45.0)
    chronological_ensemble.add_argument("--fold", dest="chronological_folds", action="append")
    chronological_ensemble.add_argument("--dry-run", action="store_true")
    chronological_ensemble.add_argument(
        "--text-mode", choices=["headline_lead", "target_aware"], default="headline_lead"
    )
    chronological_ensemble.add_argument(
        "--sampling-mode",
        choices=["shuffle", "ticker_class_balanced", "ticker_class_neutral_balanced"],
        default="shuffle",
    )
    chronological_ensemble.add_argument(
        "--loss-mode", choices=["inverse_frequency_ce", "effective_number_focal"], default="inverse_frequency_ce"
    )
    chronological_ensemble.add_argument(
        "--calibration-mode", choices=SUPPORTED_CALIBRATION_MODES,
        default=CALIBRATION_MODE_SCALAR_TEMPERATURE,
    )

    defensible_ensemble = subparsers.add_parser(
        "defensible-sentiment-ensemble",
        help=(
            "Run/resume the PhraseBank-anchored Bursa ensemble protocol: select replay "
            "mixtures on 2015-2024, then evaluate once on the locked 2025 test."
        ),
    )
    defensible_ensemble.add_argument(
        "--dataset", default="data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl"
    )
    defensible_ensemble.add_argument("--policy", default="config/bursa_news_dataset_policy.json")
    defensible_ensemble.add_argument(
        "--phrasebank-model-directory", default="models/experimental/finbert-sentiment"
    )
    defensible_ensemble.add_argument("--run-id", default=DEFAULT_DEFENSIBLE_ENSEMBLE_RUN_ID)
    defensible_ensemble.add_argument(
        "--phrasebank-replay-share",
        dest="defensible_replay_shares",
        type=float,
        action="append",
        help="Repeat for candidate replay shares; defaults to 0, 0.10, 0.20 and 0.30.",
    )
    defensible_ensemble.add_argument(
        "--seed", dest="defensible_seeds", type=int, action="append"
    )
    defensible_ensemble.add_argument("--profile", choices=["full", "quick"], default="full")
    defensible_ensemble.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    defensible_ensemble.add_argument("--learning-rate", type=float, default=2e-5)
    defensible_ensemble.add_argument("--max-epochs", type=int, default=None)
    defensible_ensemble.add_argument("--embargo-days", type=int, default=7)
    defensible_ensemble.add_argument("--bootstrap-iterations", type=int, default=2000)
    defensible_ensemble.add_argument("--max-runtime-minutes", type=float, default=45.0)
    defensible_ensemble.add_argument(
        "--no-time-limit",
        action="store_true",
        help="Opt into an unbounded wall-clock run; phase checkpoints remain enabled.",
    )
    defensible_ensemble.add_argument("--dry-run", action="store_true")
    defensible_ensemble.add_argument(
        "--text-mode", choices=["headline_lead", "target_aware"], default="headline_lead"
    )
    defensible_ensemble.add_argument(
        "--sampling-mode",
        choices=["shuffle", "ticker_class_balanced", "ticker_class_neutral_balanced"],
        default="shuffle",
    )
    defensible_ensemble.add_argument(
        "--loss-mode", choices=["inverse_frequency_ce", "effective_number_focal"], default="inverse_frequency_ce"
    )
    defensible_ensemble.add_argument(
        "--calibration-mode", choices=SUPPORTED_CALIBRATION_MODES,
        default=CALIBRATION_MODE_SCALAR_TEMPERATURE,
    )
    defensible_ensemble.add_argument(
        "--training-recipe-grid",
        default=None,
        help="Optional JSON array of development-only text/sampling/loss recipes.",
    )
    defensible_ensemble.add_argument(
        "--storage-mode",
        choices=["all", "sequential"],
        default="all",
        help=(
            "Keep all model artifacts (all) or hash and remove each completed "
            "development child before continuing (sequential)."
        ),
    )
    defensible_ensemble.add_argument(
        "--min-free-gib",
        type=float,
        default=10.0,
        help="Pause sequential runs when filesystem free space falls below this GiB.",
    )

    bursa_remediation_audit = subparsers.add_parser(
        "bursa-news-remediation-audit",
        help="Audit Bursa label quality and company/class coverage without changing training data.",
    )
    bursa_remediation_audit.add_argument("--master", default=str(DEFAULT_BURSA_REMEDIATION_MASTER))
    bursa_remediation_audit.add_argument("--review-pack", action="append", default=None)
    bursa_remediation_audit.add_argument("--output-dir", default=str(DEFAULT_BURSA_REMEDIATION_OUTPUT))
    bursa_remediation_queue = subparsers.add_parser(
        "bursa-news-remediation-export",
        help="Export unlabeled recovered Bursa articles for LLM-assisted labelling.",
    )
    bursa_remediation_queue.add_argument("--master", default=str(DEFAULT_BURSA_REMEDIATION_MASTER))
    bursa_remediation_queue.add_argument("--review-pack", action="append", default=None)
    bursa_remediation_queue.add_argument(
        "--output", default=str(DEFAULT_BURSA_REMEDIATION_OUTPUT / "label-expansion-queue.csv")
    )

    sentiment_remediation = subparsers.add_parser(
        "sentiment-remediation",
        help="Run or inspect resumable low-coverage Bursa news recovery.",
    )
    sentiment_remediation_actions = sentiment_remediation.add_subparsers(
        dest="sentiment_remediation_action", required=True
    )
    for action in ("run", "resume"):
        remediation_action = sentiment_remediation_actions.add_parser(
            action,
            help=(
                "Prepare and recover a new remediation workspace."
                if action == "run"
                else "Resume an existing remediation workspace from checkpoints."
            ),
        )
        remediation_action.add_argument("--run-id", required=True)
        remediation_action.add_argument(
            "--run-root", default=str(DEFAULT_SENTIMENT_REMEDIATION_ROOT)
        )
        remediation_action.add_argument(
            "--dataset", default=str(DEFAULT_SENTIMENT_REMEDIATION_DATASET)
        )
        remediation_action.add_argument(
            "--candidate-root", default=str(DEFAULT_SENTIMENT_REMEDIATION_CANDIDATES)
        )
        remediation_action.add_argument(
            "--benchmark-report", default=str(DEFAULT_SENTIMENT_REMEDIATION_BENCHMARK)
        )
        remediation_action.add_argument("--max-runtime-minutes", type=float, default=45.0)
        remediation_action.add_argument("--max-download-gib", type=float, default=5.0)
        remediation_action.add_argument("--request-interval-seconds", type=float, default=1.0)
        remediation_action.add_argument("--min-free-gib", type=float, default=20.0)
        remediation_action.add_argument("--max-passes", type=int, default=50)
        remediation_action.add_argument("--skip-provenance-repair", action="store_true")
        remediation_action.add_argument("--skip-archives", action="store_true")
    remediation_status_parser = sentiment_remediation_actions.add_parser(
        "status", help="Show recovery progress without changing any files."
    )
    remediation_status_parser.add_argument("--run-id", required=True)
    remediation_status_parser.add_argument(
        "--run-root", default=str(DEFAULT_SENTIMENT_REMEDIATION_ROOT)
    )

    prospective_prepare = subparsers.add_parser(
        "prospective-sentiment-prepare",
        help="Create a sealed 2026 LLM-assisted-label cohort without changing the training corpus.",
    )
    prospective_prepare.add_argument("--candidates", required=True)
    prospective_prepare.add_argument("--output-dir", required=True)
    prospective_prepare.add_argument("--policy", default="config/bursa_news_dataset_policy.json")
    prospective_prepare.add_argument(
        "--master-dataset", default="data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl"
    )

    prospective_import = subparsers.add_parser(
        "prospective-sentiment-import-labels",
        help="Seal LLM-assisted labels for a prepared prospective cohort.",
    )
    prospective_import.add_argument("--labels", required=True)
    prospective_import.add_argument("--output-dir", required=True)

    prospective_bind = subparsers.add_parser(
        "prospective-sentiment-bind",
        help="Bind one immutable three-seed Bursa/PhraseBank recipe before final evaluation.",
    )
    prospective_bind.add_argument("--benchmark-report", required=True)
    prospective_bind.add_argument("--output-dir", required=True)

    prospective_evaluate = subparsers.add_parser(
        "prospective-sentiment-evaluate",
        help="Evaluate the sealed 2026 final holdout once; it never promotes a runtime model.",
    )
    prospective_evaluate.add_argument("--output-dir", required=True)

    prospective_status = subparsers.add_parser(
        "prospective-sentiment-status",
        help="Show the current state and coverage of a prospective 2026 holdout.",
    )
    prospective_status.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import-news":
        run_import_historical_news_pipeline(args.sources, args.output_dir, args.provider)
        return 0
    if args.command == "prepare-official-disclosures":
        run_prepare_official_disclosures(
            args.sources,
            output_path=args.output,
            source_kind=args.source_kind,
        )
        return 0
    if args.command == "export-official-disclosures-v3":
        payload = export_official_disclosures_v3(
            years=args.years,
            tickers=args.tickers,
            output_path=args.output,
            report_path=args.report,
            cache_dir=args.cache_dir,
            request_interval_seconds=args.request_interval_seconds,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.command == "score-news":
        run_score_historical_news_for_fusion(
            args.source, output_path=args.output
        )
        return 0
    if args.command == "readiness":
        report = run_fusion_data_readiness(args.data_dir)
        return 0 if report.get("ready") else 2
    if args.command == "features":
        run_build_point_in_time_features(
            args.stock_path,
            args.news_dir,
            args.output,
            args.stock_probabilities,
        )
        return 0
    if args.command == "fusion-window-candidates":
        run_build_fusion_window_candidates(
            args.stock_path,
            stock_probability_path=args.stock_probabilities,
            news_dir=args.news_dir,
            policy_path=args.policy,
            output_path=args.output,
            horizons=args.horizons,
        )
        return 0
    if args.command == "stock-oof":
        run_build_stock_oof_probabilities(
            args.stock_path,
            registry_path=args.registry,
            output_path=args.output,
            mode=args.mode,
            search_profile=args.search_profile,
        )
        return 0
    if args.command == "evaluate-v2":
        report = run_fusion_evaluation_v2(
            Path(args.dataset),
            policy_path=Path(args.policy),
            report_path=Path(args.report),
            registry_path=Path(args.registry),
            permutation_iterations=args.permutations,
            bootstrap_iterations=args.bootstraps,
            horizons=args.v2_horizons,
            include_rolling_origins=not args.skip_rolling,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("overall_all_horizon_accepted") else 2
    if args.command == "evaluate-v3":
        try:
            report = run_fusion_evaluation_v3(
                Path(args.dataset),
                policy_path=Path(args.policy),
                report_path=Path(args.report),
                registry_path=Path(args.registry),
                permutation_iterations=args.permutations,
                bootstrap_iterations=args.bootstraps,
                horizons=args.v3_horizons,
                include_rolling_origins=not args.skip_rolling,
                checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
                resume=not args.no_resume,
                enforce_readiness=not args.allow_unready,
            )
        except FusionV3ReadinessError as exc:
            print(json.dumps({"completed": False, "readiness_blocked": True, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("accepted_horizons") else 2
    if args.command == "preflight-v3":
        report = run_fusion_v3_preflight(
            Path(args.dataset),
            policy_path=Path(args.policy),
            report_path=Path(args.report),
            horizons=args.preflight_horizons,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("overall_ready") else 2
    if args.command == "compare-v3-runs":
        report = compare_fusion_v3_reports(
            Path(args.first_report),
            Path(args.second_report),
            output_path=Path(args.output),
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("passed") else 2
    if args.command == "materialize-v3-runtime-state":
        report = materialize_runtime_surprise_state(
            Path(args.dataset),
            Path(args.registry),
            output_path=Path(args.output),
        )
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "runtime-parity-v3":
        report = run_runtime_feature_parity(
            Path(args.dataset),
            Path(args.registry),
            Path(args.runtime_state),
            output_path=Path(args.output),
            sample_per_ticker=args.sample_per_ticker,
            tolerance=args.tolerance,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("passed") else 2
    if args.command == "promote-v3":
        kwargs = {}
        if args.active_registry:
            kwargs["active_registry_path"] = Path(args.active_registry)
        kwargs["allow_acceptance_override"] = bool(args.allow_acceptance_override)
        report = promote_fusion_v3_registry(
            Path(args.registry),
            Path(args.reproducibility_report),
            Path(args.runtime_state),
            Path(args.runtime_parity_report),
            **kwargs,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "prepare-v3-acceptance-override":
        report = prepare_fusion_v3_acceptance_override(
            Path(args.registry),
            Path(args.evaluation_report),
            output_path=Path(args.output),
            reason=args.reason,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "train":
        try:
            artifacts = train_fusion_models(
                Path(args.dataset),
                registry_path=Path(args.registry),
                supported_tickers=args.tickers,
                min_rows_per_horizon=args.min_rows,
            )
        except FusionDatasetNotReadyError as exc:
            print(json.dumps({"ready": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps({str(horizon): artifact.to_dict() for horizon, artifact in artifacts.items()}, indent=2, default=str))
        return 0
    if args.command == "evaluate":
        try:
            artifacts = evaluate_fusion_dataset(
                Path(args.dataset),
                registry_path=Path(args.registry),
                report_path=Path(args.report),
                supported_tickers=args.tickers,
                min_rows_per_horizon=args.min_rows,
            )
        except FusionDatasetNotReadyError as exc:
            print(json.dumps({"ready": False, "reason": str(exc)}, indent=2))
            return 2
        print(
            json.dumps(
                {str(horizon): artifact.to_dict() for horizon, artifact in artifacts.items()},
                indent=2,
                default=str,
            )
        )
        return 0
    if args.command == "ablation":
        report = run_fusion_ablation(
            Path(args.dataset),
            output_path=Path(args.report),
        )
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "coverage":
        report = write_fusion_coverage_report(
            Path(args.dataset),
            output_path=Path(args.report),
            supported_tickers=args.tickers,
        )
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("ready_for_requested_universe") else 2
    if args.command == "signal-diagnostics":
        report = write_fusion_signal_diagnostics(
            Path(args.dataset), output_path=Path(args.report)
        )
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "gdelt-query":
        start_date = args.start_date
        end_date = args.end_date
        if args.pilot_year is not None:
            start_date = f"{args.pilot_year:04d}-01-01"
            end_date = f"{args.pilot_year:04d}-12-31"
        print(
            json.dumps(
                write_gdelt_query_bundle(
                    Path(args.output_dir),
                    start_date=start_date,
                    end_date=end_date,
                    tickers=args.tickers,
                    shard=args.shard,
                    max_shard_gib=args.max_shard_gib,
                    monthly_budget_gib=args.monthly_budget_gib,
                    pilot_year=args.pilot_year,
                    query_profile=args.query_profile,
                    max_candidates_per_ticker_month=args.max_candidates_per_ticker_month,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "gdelt-run":
        try:
            result = run_authenticated_gdelt_queries(
                args.project_id,
                query_dir=args.query_dir,
                export_dir=args.export_dir,
                state_path=args.state,
                maximum_bytes_billed_gib=args.maximum_bytes_billed_gib,
                monthly_budget_gib=args.monthly_budget_gib,
                location=args.location,
                dry_run_only=args.dry_run_only,
            )
        except BigQueryRunnerError as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "shadow":
        print(
            json.dumps(
                evaluate_fusion_shadow(
                    Path(args.dataset),
                    registry_path=Path(args.registry),
                    report_path=Path(args.report),
                    start_date=args.start_date,
                    end_date=args.end_date,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "promote":
        try:
            registry = promote_fusion_registry(
                Path(args.staged_registry),
                active_registry_path=Path(args.active_registry),
                approve=args.approve,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(json.dumps({"promoted": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps({"promoted": True, "registry": registry.to_dict()}, indent=2))
        return 0
    if args.command == "gdelt-enrich":
        result = run_gdelt_enrichment(
            args.export,
            output_dir=args.output_dir,
            max_articles=args.max_articles,
            pilot_per_ticker=args.pilot_per_ticker,
            checkpoint_path=args.checkpoint,
            request_interval_seconds=args.request_interval_seconds,
            capture_lag_days=args.capture_lag_days,
            score_sentiment=args.score_sentiment,
            approvals_path=args.approvals,
            enforce_indexed_gate=args.require_indexed_gate,
            max_runtime_minutes=args.max_runtime_minutes,
            heartbeat_path=args.heartbeat,
        )
        return 2 if result.get("blocked") else 0
    if args.command == "label-bursa-news-llm":
        result = run_llm_consensus_labelling(
            args.source,
            output_path=args.output,
            report_path=args.report,
            checkpoint_path=args.checkpoint,
            heartbeat_path=args.heartbeat,
            approval_queue_path=args.approval_queue,
            approvals_path=args.approvals,
            max_runtime_minutes=args.max_runtime_minutes,
            max_requests=args.max_requests,
            request_interval_seconds=args.request_interval_seconds,
            recommended_start_year=args.recommended_start_year,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("stop_reason") is None else 2
    if args.command == "gdelt-pilot":
        def read_table(value: str):
            path = Path(value)
            if path.suffix.casefold() in {".parquet", ".pq"}:
                return __import__("pandas").read_parquet(path)
            if path.suffix.casefold() in {".json", ".jsonl"}:
                return __import__("pandas").read_json(
                    path, lines=path.suffix.casefold() == ".jsonl"
                )
            return __import__("pandas").read_csv(path)

        result = build_gdelt_pilot_report(
            read_table(args.indexed),
            read_table(args.recovered),
            read_table(args.accepted),
            output_dir=Path(args.output_dir),
            manually_reviewed=(
                read_table(args.manual_review) if args.manual_review else None
            ),
            llm_labelled=(
                read_table(args.llm_labels) if args.llm_labels else None
            ),
            feature_frame=read_table(args.features) if args.features else None,
            pilot_year=args.pilot_year,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ready_for_full_collection else 2
    if args.command == "gdelt-ngram-pilot":
        result = run_gdelt_ngram_pilot(
            year=args.year,
            candidate_dir=args.candidate_dir,
            sample_per_ticker=args.sample_per_ticker,
            paired_per_ticker=args.paired_per_ticker,
            output_dir=args.output_dir,
            max_runtime_minutes=args.max_runtime_minutes,
            max_download_gib=args.max_download_gib,
            max_articles=args.max_articles,
            request_interval_seconds=args.request_interval_seconds,
            run_labels=not args.skip_labels,
            max_label_requests=args.max_label_requests,
            skip_archives=args.skip_archives,
            skip_ngrams=args.skip_ngrams,
            candidate_only=args.candidate_only,
            max_stage_attempts=args.max_stage_attempts,
            network_timeout_seconds=args.network_timeout_seconds,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("completed") else 2
    if args.command == "gdelt-remaining-years":
        try:
            result = collect_remaining_gdelt_years(
                args.project_id,
                pilot_report_path=args.pilot_report,
                output_dir=args.output_dir,
                years=args.years or (2019, 2020, 2021, 2022, 2024, 2025),
                maximum_bytes_billed_gib=args.maximum_bytes_billed_gib,
                monthly_budget_gib=args.monthly_budget_gib,
                location=args.location,
                execute=args.execute,
            )
        except BigQueryRunnerError as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "gdelt-external-corpus":
        try:
            result = collect_external_label_candidate_years(
                args.project_id,
                output_dir=args.output_dir,
                years=args.years or DEFAULT_EXTERNAL_COLLECTION_YEARS,
                reusable_2023_export=args.reuse_2023_export,
                maximum_bytes_billed_gib=args.maximum_bytes_billed_gib,
                monthly_budget_gib=args.monthly_budget_gib,
                location=args.location,
                execute=args.execute,
                query_profile=args.query_profile,
                max_candidates_per_ticker_month=(
                    args.max_candidates_per_ticker_month
                ),
            )
        except BigQueryRunnerError as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ready_for_execution") else 2
    if args.command == "gdelt-external-label-export":
        result = export_external_label_package(
            args.recovery_root,
            output_dir=args.output_dir,
            years=args.years or DEFAULT_EXTERNAL_COLLECTION_YEARS,
            extra_files=args.extra_files,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "gdelt-external-label-import":
        # Keep the default destination under the importer’s experimental
        # namespace.  No production registry, FinBERT artifact, or fusion
        # dataset is touched by this command.
        kwargs = {
            "input_snapshot": args.input_snapshot,
            "returned_labels": args.returned_labels,
            "review_queue": args.review_queue,
            "report_path": args.report_path,
            "allow_backfill": args.allow_backfill,
        }
        if args.output_dataset:
            kwargs["output_dataset"] = args.output_dataset
        try:
            result = import_external_news_labels(**kwargs)
        except (ExternalNewsImportError, FileNotFoundError) as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "bursa-news-append":
        try:
            if args.bursa_append_action == "prepare":
                result = prepare_bursa_news_append(
                    dataset_path=args.dataset,
                    candidate_root=args.candidate_root,
                    work_dir=args.work_dir,
                    years=args.years or tuple(range(2015, 2026)),
                    target_relevant_per_ticker=args.target_relevant_per_ticker,
                    recovery_multiplier=args.recovery_multiplier,
                    target_per_class_per_ticker=args.target_per_class_per_ticker,
                    training_target_per_class=args.training_target_per_class,
                    validation_target_per_class=args.validation_target_per_class,
                    test_target_per_class=args.test_target_per_class,
                    max_candidates_per_cycle=args.max_candidates_per_cycle,
                    new_cycle=args.new_cycle,
                    tickers=args.bursa_append_tickers,
                )
            elif args.bursa_append_action == "recover":
                result = recover_bursa_news_append(
                    work_dir=args.work_dir,
                    max_runtime_minutes=args.max_runtime_minutes,
                    max_download_gib=args.max_download_gib,
                    request_interval_seconds=args.request_interval_seconds,
                    skip_archives=args.skip_archives,
                )
            elif args.bursa_append_action == "triage":
                result = triage_bursa_news_append(
                    work_dir=args.work_dir,
                    random_control_share=args.random_control_share,
                )
            elif args.bursa_append_action == "export-labels":
                result = export_bursa_news_append_labels(
                    work_dir=args.work_dir, batch_size=args.batch_size
                )
            elif args.bursa_append_action == "import-labels":
                result = import_bursa_news_append_labels(
                    args.returned_labels, work_dir=args.work_dir
                )
            elif args.bursa_append_action == "commit":
                result = commit_bursa_news_append(
                    work_dir=args.work_dir, dataset_path=args.dataset
                )
            elif args.bursa_append_action == "status":
                result = append_status(work_dir=args.work_dir)
            else:
                raise BursaNewsAppendError(
                    f"Unknown Bursa append action: {args.bursa_append_action}"
                )
        except (BursaNewsAppendError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "sentiment-ensemble":
        try:
            if args.sentiment_ensemble_action == "promote":
                result = promote_sentiment_ensemble(
                    variant=args.variant,
                    benchmark_report=args.benchmark_report,
                    runtime_config=args.runtime_config,
                    policy_path=args.policy,
                    approve=args.approve,
                )
            elif args.sentiment_ensemble_action == "manual-activate":
                result = manually_activate_sentiment_ensemble(
                    variant=args.variant,
                    benchmark_report=args.benchmark_report,
                    runtime_config=args.runtime_config,
                    reason=args.reason,
                    approve=args.approve,
                )
            elif args.sentiment_ensemble_action == "rollback":
                result = rollback_sentiment_ensemble(
                    variant=args.variant,
                    runtime_config=args.runtime_config,
                    approve=args.approve,
                )
            else:
                raise SentimentEnsemblePromotionError(
                    f"Unknown sentiment ensemble action: {args.sentiment_ensemble_action}"
                )
        except SentimentEnsemblePromotionError as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "chronological-sentiment-ensemble":
        try:
            result = run_chronological_finbert_ensemble(
                Path(args.dataset),
                policy_path=Path(args.policy),
                phrasebank_model_directory=Path(args.phrasebank_model_directory),
                run_id=args.run_id,
                profile=args.profile,
                device=args.device,
                seeds=tuple(args.chronological_seeds) if args.chronological_seeds else (42, 52, 62),
                bursa_replay_share=args.bursa_replay_share,
                learning_rate=args.learning_rate,
                max_epochs=args.max_epochs,
                embargo_days=args.embargo_days,
                bootstrap_iterations=args.bootstrap_iterations,
                max_runtime_minutes=args.max_runtime_minutes,
                selected_folds=args.chronological_folds,
                dry_run=args.dry_run,
                text_mode=args.text_mode,
                sampling_mode=args.sampling_mode,
                loss_mode=args.loss_mode,
                calibration_mode=args.calibration_mode,
            )
        except (ChronologicalEnsembleError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "bursa-news-remediation-audit":
        packs = tuple(Path(value) for value in args.review_pack) if args.review_pack else DEFAULT_BURSA_REMEDIATION_PACKS
        result = audit_bursa_news_labels(
            Path(args.master), review_packs=packs, output_dir=Path(args.output_dir)
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "bursa-news-remediation-export":
        packs = tuple(Path(value) for value in args.review_pack) if args.review_pack else DEFAULT_BURSA_REMEDIATION_PACKS
        result = export_label_expansion_queue(
            packs, master_path=Path(args.master), output_path=Path(args.output)
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "sentiment-remediation":
        try:
            if args.sentiment_remediation_action == "status":
                result = remediation_status(
                    run_id=args.run_id, run_root=args.run_root
                )
            else:
                runner = (
                    resume_sentiment_remediation
                    if args.sentiment_remediation_action == "resume"
                    else run_sentiment_remediation
                )
                result = runner(
                    run_id=args.run_id,
                    run_root=args.run_root,
                    dataset_path=args.dataset,
                    candidate_root=args.candidate_root,
                    benchmark_report=args.benchmark_report,
                    max_runtime_minutes=args.max_runtime_minutes,
                    max_download_gib=args.max_download_gib,
                    request_interval_seconds=args.request_interval_seconds,
                    minimum_free_gib=args.min_free_gib,
                    max_passes=args.max_passes,
                    repair_provenance=not args.skip_provenance_repair,
                    skip_archives=args.skip_archives,
                )
        except (SentimentRemediationError, BursaNewsAppendError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("status") not in {"failed", "paused_storage"} else 2
    if args.command == "defensible-sentiment-ensemble":
        try:
            result = run_defensible_finbert_ensemble(
                Path(args.dataset),
                policy_path=Path(args.policy),
                phrasebank_model_directory=Path(args.phrasebank_model_directory),
                run_id=args.run_id,
                phrasebank_replay_shares=args.defensible_replay_shares,
                seeds=tuple(args.defensible_seeds)
                if args.defensible_seeds
                else (42, 52, 62),
                profile=args.profile,
                device=args.device,
                learning_rate=args.learning_rate,
                max_epochs=args.max_epochs,
                embargo_days=args.embargo_days,
                bootstrap_iterations=args.bootstrap_iterations,
                max_runtime_minutes=(
                    None if args.no_time_limit else args.max_runtime_minutes
                ),
                dry_run=args.dry_run,
                text_mode=args.text_mode,
                sampling_mode=args.sampling_mode,
                loss_mode=args.loss_mode,
                calibration_mode=args.calibration_mode,
                training_recipe_grid=(
                    load_training_recipe_grid(args.training_recipe_grid)
                    if args.training_recipe_grid
                    else None
                ),
                storage_mode=args.storage_mode,
                min_free_gib=args.min_free_gib,
            )
        except (DefensibleEnsembleError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("status") in {"complete", "dry_run"} else 2
    if args.command == "prospective-sentiment-prepare":
        try:
            result = prepare_prospective_holdout(
                args.candidates,
                output_dir=args.output_dir,
                policy_path=args.policy,
                master_dataset_path=args.master_dataset,
            )
        except (ProspectiveHoldoutError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"prepared": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "prospective-sentiment-import-labels":
        try:
            result = import_prospective_labels(args.labels, output_dir=args.output_dir)
        except (ProspectiveHoldoutError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"imported": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "prospective-sentiment-bind":
        try:
            result = bind_prospective_candidate(args.benchmark_report, output_dir=args.output_dir)
        except (ProspectiveHoldoutError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"bound": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "prospective-sentiment-evaluate":
        try:
            result = evaluate_prospective_holdout(output_dir=args.output_dir)
        except (ProspectiveHoldoutError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("status") == "complete" else 2
    if args.command == "prospective-sentiment-status":
        try:
            result = prospective_holdout_status(output_dir=args.output_dir)
        except (ProspectiveHoldoutError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"available": False, "reason": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
