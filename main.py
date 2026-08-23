from __future__ import annotations

import argparse
import json


def run_download_raw() -> None:
    from pipeline.download_raw_stock_data import run_download_pipeline

    run_download_pipeline()


def run_download_financial_news(ticker: str | None = None, as_of: str | None = None) -> None:
    from pipeline.download_financial_news_data import run_financial_news_download_pipeline

    run_financial_news_download_pipeline(ticker=ticker, as_of_date=as_of)


def run_preprocess_raw() -> None:
    from pipeline.preprocess_raw_stock_data import run_preprocess_pipeline

    run_preprocess_pipeline()


def run_build_model_input() -> None:
    from pipeline.build_model_input_stock_data import run_model_input_pipeline

    run_model_input_pipeline()


def run_data_understanding() -> None:
    from pipeline.data_understanding import run_data_understanding_pipeline

    run_data_understanding_pipeline()


def run_preprocess_financial_news_sentiment() -> None:
    from pipeline.preprocess_financial_news_sentiment_data import (
        run_financial_news_sentiment_preprocess_pipeline,
    )

    run_financial_news_sentiment_preprocess_pipeline()


def run_financial_news_sentiment_understanding() -> None:
    from pipeline.financial_news_sentiment_understanding import (
        run_financial_news_sentiment_understanding_pipeline,
    )

    run_financial_news_sentiment_understanding_pipeline()


def run_train_sentiment_model(search_profile: str = "full") -> None:
    from pipeline.train_financial_news_sentiment_model import run_sentiment_training_pipeline

    run_sentiment_training_pipeline(search_profile=search_profile)


def run_benchmark_sentiment_models(profile: str = "full") -> None:
    from pipeline.benchmark_finbert_sentiment import run_finbert_benchmark

    run_finbert_benchmark(profile=profile)


def run_train_finbert(profile: str = "full", device: str = "cuda", promote: bool = True) -> None:
    from pipeline.benchmark_finbert_sentiment import run_finbert_benchmark

    report = run_finbert_benchmark(profile=profile, device=device, promote=promote)
    print(json.dumps(report, indent=2, default=str))


def run_prepare_bursa_news_dataset(
    source: str,
    output: str,
    backfill_reports: list[str] | None = None,
) -> None:
    from pathlib import Path

    from pipeline.bursa_news_dataset import (
        build_annotation_template,
        read_table,
    )

    approved_backfill_years: list[int] = []
    for report_value in backfill_reports or []:
        report = json.loads(Path(report_value).read_text(encoding="utf-8"))
        year = int(report.get("year", 0))
        if not report.get("ready") or year not in {2015, 2016, 2017, 2018}:
            raise ValueError(
                f"Backfill report is not an approved 2015-2018 year: {report_value}"
            )
        approved_backfill_years.append(year)
    frame = build_annotation_template(
        read_table(source), approved_backfill_years=approved_backfill_years
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.casefold() == ".csv":
        frame.to_csv(destination, index=False)
    elif destination.suffix.casefold() in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    elif destination.suffix.casefold() == ".jsonl":
        frame.to_json(
            destination, orient="records", lines=True, date_format="iso"
        )
    else:
        raise ValueError("Bursa news candidate output must be CSV, JSONL, or Parquet")
    print(json.dumps({"rows": len(frame), "output": str(destination)}, indent=2))


def run_validate_bursa_news_dataset(
    dataset: str, report: str, policy: str
) -> None:
    from pathlib import Path

    from pipeline.bursa_news_dataset import (
        load_policy,
        read_table,
        validate_bursa_news_dataset,
    )

    result = validate_bursa_news_dataset(
        read_table(dataset),
        policy=load_policy(Path(policy)),
        dataset_path=dataset,
    )
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.ready:
        raise SystemExit(2)


def run_report_bursa_news_year(
    year: int,
    indexed: str,
    recovered: str,
    accepted: str,
    reviewed: str,
    output: str,
) -> None:
    from pathlib import Path

    from pipeline.bursa_news_dataset import (
        build_year_quality_report,
        read_table,
    )

    result = build_year_quality_report(
        year,
        read_table(indexed),
        read_table(recovered),
        read_table(accepted),
        read_table(reviewed),
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(2)


def run_train_bursa_finbert(
    dataset: str,
    profile: str = "full",
    device: str = "cuda",
    policy: str = "config/bursa_news_dataset_policy.json",
) -> None:
    from pathlib import Path

    from pipeline.benchmark_bursa_finbert import run_bursa_finbert_benchmark

    report = run_bursa_finbert_benchmark(
        Path(dataset),
        profile=profile,
        device=device,
        policy_path=Path(policy),
    )
    print(json.dumps(report, indent=2, default=str))


def run_promote_bursa_finbert(run_id: str, approve: bool = False) -> None:
    from pipeline.benchmark_bursa_finbert import promote_bursa_finbert

    print(
        json.dumps(
            promote_bursa_finbert(run_id, approve=approve),
            indent=2,
            default=str,
        )
    )


def run_benchmark_finbert_ensemble(
    dataset: str = "data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl",
    policy: str = "config/bursa_news_dataset_policy.json",
    bursa_model_directory: str | None = None,
    bursa_run_id: str | None = None,
    phrasebank_model_directory: str | None = None,
    bursa_temperature: float = 1.27,
    phrasebank_temperature: float = 1.52,
    weight_step: float = 0.05,
) -> None:
    from pathlib import Path

    from pipeline.benchmark_finbert_ensemble import run_finbert_ensemble_benchmark

    kwargs: dict[str, object] = {
        "policy_path": Path(policy),
        "bursa_temperature": bursa_temperature,
        "phrasebank_temperature": phrasebank_temperature,
        "weight_step": weight_step,
    }
    if bursa_model_directory:
        kwargs["bursa_model_directory"] = Path(bursa_model_directory)
    elif bursa_run_id:
        kwargs["bursa_model_directory"] = (
            Path("models/experimental")
            / f"bursa-finbert-staging-{bursa_run_id}"
            / "seed-42"
        )
    if phrasebank_model_directory:
        kwargs["phrasebank_model_directory"] = Path(phrasebank_model_directory)
    report = run_finbert_ensemble_benchmark(Path(dataset), **kwargs)
    print(json.dumps(report, indent=2, default=str))


def run_audit_system(level: str = "full", live_providers: bool = False) -> None:
    from pipeline.system_audit import run_system_audit

    report = run_system_audit(level=level, live_providers=live_providers)
    print(json.dumps(report, indent=2, default=str))


def run_collect_news_provider_benchmark(
    as_of: str | None = None,
    tickers: list[str] | None = None,
    providers: list[str] | None = None,
    run_id: str | None = None,
    cutoff_lag_days: int = 1,
) -> None:
    from pipeline.news_provider_benchmark import collect_news_provider_benchmark

    records = collect_news_provider_benchmark(
        as_of_date=as_of,
        tickers=tickers,
        providers=providers or ["newsapi", "marketaux"],
        run_id=run_id,
        cutoff_lag_days=cutoff_lag_days,
    )
    print(json.dumps([record.__dict__ for record in records], indent=2))


def run_evaluate_news_provider_benchmark(
    annotations: str | None = None,
    apply_decision_rules: bool = False,
) -> None:
    from pathlib import Path

    from pipeline.news_provider_benchmark import evaluate_news_provider_benchmark

    report = evaluate_news_provider_benchmark(
        annotation_path=Path(annotations) if annotations else None,
        apply_decision_rules=apply_decision_rules,
    )
    print(json.dumps(report, indent=2))


def run_train_models(search_profile: str | None = None) -> None:
    from pipeline.train_stock_models import run_training_pipeline

    run_training_pipeline(search_profile=search_profile, promote=False)


def run_promote_stock_models(
    run_id: str,
    approve: bool = False,
    override_gates: bool = False,
    override_reason: str | None = None,
) -> None:
    from pipeline.train_stock_models import promote_stock_models

    print(
        promote_stock_models(
            run_id,
            approve=approve,
            override_gates=override_gates,
            override_reason=override_reason,
        )
    )


def run_verify_stock_reproducibility(run_a: str, run_b: str) -> None:
    from pipeline.train_stock_models import verify_stock_reproducibility

    print(json.dumps(verify_stock_reproducibility(run_a, run_b), indent=2))


def run_predict_stock(ticker: str, horizon: int, as_of: str | None = None) -> None:
    from pipeline.predict_stock_trend import predict_stock_trend

    result = predict_stock_trend(ticker, horizon, as_of)
    print(json.dumps(result.to_dict(), indent=2))


def run_evaluate_horizon_shadow(
    start: str = "2026-01-01",
    end: str = "2026-06-30",
    run_id: str | None = None,
) -> None:
    from pipeline.evaluate_horizon_shadow import run_shadow_evaluation
    from pathlib import Path

    registry_path = None
    if run_id:
        registry_path = Path(__file__).resolve().parent / "models" / "staged-stock-runs" / run_id / "model-registry.json"
    kwargs = {"registry_path": registry_path} if registry_path is not None else {}
    print(json.dumps(run_shadow_evaluation(start, end, **kwargs), indent=2))


def run_recompute_horizon_acceptance() -> None:
    from pipeline.recompute_horizon_acceptance import recompute_acceptance

    print(json.dumps(recompute_acceptance(), indent=2))


def run_all(search_profile: str = "full") -> None:
    print("Step 1/8: Downloading 2015-2025 stock and market-context data...")
    run_download_raw()
    print("\nStep 2/8: Preprocessing stock data...")
    run_preprocess_raw()
    print("\nStep 3/8: Building corrected model-input data...")
    run_build_model_input()
    print("\nStep 4/8: Generating stock data summaries...")
    run_data_understanding()
    print("\nStep 5/8: Preprocessing the financial sentiment dataset...")
    run_preprocess_financial_news_sentiment()
    print("\nStep 6/8: Generating sentiment data summaries...")
    run_financial_news_sentiment_understanding()
    print("\nStep 7/8: Training and evaluating the sentiment model...")
    run_train_sentiment_model(search_profile=search_profile)
    print("\nStep 8/8: Selecting and evaluating stock models...")
    run_train_models(search_profile=search_profile)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FYP analytics pipeline controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "download_raw",
        "preprocess_raw",
        "build_model_input",
        "data_understanding",
        "preprocess_financial_news_sentiment",
        "financial_news_sentiment_understanding",
    ):
        subparsers.add_parser(command)

    news_parser = subparsers.add_parser("download_financial_news")
    news_parser.add_argument("--ticker")
    news_parser.add_argument("--as-of")

    sentiment_parser = subparsers.add_parser("train_sentiment_model")
    sentiment_parser.add_argument("--search-profile", choices=["full", "quick"], default="full")

    benchmark_parser = subparsers.add_parser(
        "benchmark_sentiment_models",
        help="Compare TF-IDF sentiment with a leakage-safe FinBERT challenger",
    )
    benchmark_parser.add_argument("--profile", choices=["full", "quick"], default="full")

    finbert_parser = subparsers.add_parser(
        "train_finbert",
        help="Run the full FinBERT challenger training with an explicit device and guarded promotion",
    )
    finbert_parser.add_argument("--profile", choices=["full", "quick"], default="full")
    finbert_parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    finbert_parser.add_argument("--no-promote", action="store_true")

    bursa_prepare_parser = subparsers.add_parser(
        "prepare_bursa_news_dataset",
        help="Create the chronological candidate table used by the offline dual-LLM labelling workflow",
    )
    bursa_prepare_parser.add_argument("--source", required=True)
    bursa_prepare_parser.add_argument(
        "--output",
        default="data/news-evidence-benchmark/bursa-news-candidates.jsonl",
    )
    bursa_prepare_parser.add_argument(
        "--backfill-report",
        action="append",
        default=[],
        help="Passing year-quality report for an optional 2015-2018 year",
    )

    bursa_validate_parser = subparsers.add_parser(
        "validate_bursa_news_dataset",
        help="Validate Bursa news integrity and the quality gates enabled by policy",
    )
    bursa_validate_parser.add_argument(
        "--dataset",
        default="data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl",
    )
    bursa_validate_parser.add_argument(
        "--report",
        default="results/bursa-news-dataset/readiness-report.json",
    )
    bursa_validate_parser.add_argument(
        "--policy",
        default="config/bursa_news_dataset_policy.json",
    )

    bursa_year_parser = subparsers.add_parser(
        "report_bursa_news_year",
        help="Evaluate one archive year before it enters the labelled dataset",
    )
    bursa_year_parser.add_argument("--year", type=int, required=True)
    bursa_year_parser.add_argument("--indexed", required=True)
    bursa_year_parser.add_argument("--recovered", required=True)
    bursa_year_parser.add_argument("--accepted", required=True)
    bursa_year_parser.add_argument("--reviewed", required=True)
    bursa_year_parser.add_argument("--output", required=True)

    bursa_train_parser = subparsers.add_parser(
        "train_bursa_finbert",
        help="Train the staged three-seed Bursa-adapted FinBERT challenger",
    )
    bursa_train_parser.add_argument(
        "--dataset",
        default="data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl",
    )
    bursa_train_parser.add_argument(
        "--profile", choices=["full", "quick"], default="full"
    )
    bursa_train_parser.add_argument(
        "--device", choices=["cuda", "cpu", "auto"], default="cuda"
    )
    bursa_train_parser.add_argument(
        "--policy",
        default="config/bursa_news_dataset_policy.json",
    )

    bursa_promote_parser = subparsers.add_parser(
        "promote_bursa_finbert",
        help="Promote a reviewed, gate-passing Bursa FinBERT staged run",
    )
    bursa_promote_parser.add_argument("--run-id", required=True)
    bursa_promote_parser.add_argument("--approve", action="store_true")

    ensemble_parser = subparsers.add_parser(
        "benchmark_finbert_ensemble",
        help="Select and benchmark the opt-in Bursa/PhraseBank FinBERT probability ensemble",
    )
    ensemble_parser.add_argument(
        "--dataset",
        default="data/news-evidence-benchmark/bursa-news-llm-assisted-label.jsonl",
    )
    ensemble_parser.add_argument(
        "--policy",
        default="config/bursa_news_dataset_policy.json",
    )
    ensemble_parser.add_argument("--bursa-model-directory")
    ensemble_parser.add_argument("--bursa-run-id")
    ensemble_parser.add_argument("--phrasebank-model-directory")
    ensemble_parser.add_argument("--bursa-temperature", type=float, default=1.27)
    ensemble_parser.add_argument("--phrasebank-temperature", type=float, default=1.52)
    ensemble_parser.add_argument("--weight-step", type=float, default=0.05)

    audit_parser = subparsers.add_parser("audit_system", help="Run the end-to-end StockSense integrity audit")
    audit_parser.add_argument("--level", choices=["quick", "full"], default="full")
    audit_parser.add_argument("--live-providers", action="store_true")

    provider_collect_parser = subparsers.add_parser(
        "collect_news_provider_benchmark",
        help="Collect a dated NewsAPI/Marketaux challenger snapshot",
    )
    provider_collect_parser.add_argument("--as-of")
    provider_collect_parser.add_argument("--ticker", action="append")
    provider_collect_parser.add_argument("--provider", action="append", choices=["newsapi", "marketaux"])
    provider_collect_parser.add_argument("--run-id")
    provider_collect_parser.add_argument("--cutoff-lag-days", type=int, default=1)

    provider_evaluate_parser = subparsers.add_parser(
        "evaluate_news_provider_benchmark",
        help="Evaluate saved NewsAPI/Marketaux challenger snapshots",
    )
    provider_evaluate_parser.add_argument("--annotations")
    provider_evaluate_parser.add_argument(
        "--apply-decision-rules",
        action="store_true",
        help="Apply promotion gates only after explicit approval",
    )

    stock_model_parser = subparsers.add_parser("train_models")
    stock_model_parser.add_argument(
        "--search-profile", choices=["full", "focused", "quick", "official"], default="full"
    )

    promote_parser = subparsers.add_parser(
        "promote_stock_models",
        help="Manually promote a reviewed staged official stock-model run",
    )
    promote_parser.add_argument("--run-id", required=True)
    promote_parser.add_argument("--approve", action="store_true")
    promote_parser.add_argument(
        "--override-gates",
        action="store_true",
        help="Explicitly bypass prospective/acceptance gates while preserving their failed status",
    )
    promote_parser.add_argument(
        "--override-reason",
        help="Required audit reason when --override-gates is used",
    )

    reproducibility_parser = subparsers.add_parser(
        "verify_stock_reproducibility",
        help="Compare two staged official stock-model runs",
    )
    reproducibility_parser.add_argument("--run-a", required=True)
    reproducibility_parser.add_argument("--run-b", required=True)

    prediction_parser = subparsers.add_parser("predict_stock")
    prediction_parser.add_argument("--ticker", required=True)
    prediction_parser.add_argument("--horizon", type=int, choices=[1, 5, 10], required=True)
    prediction_parser.add_argument("--as-of")

    shadow_parser = subparsers.add_parser(
        "evaluate_horizon_shadow",
        help="Evaluate frozen stock champions on the forward 2026 shadow period",
    )
    shadow_parser.add_argument("--start", default="2026-01-01")
    shadow_parser.add_argument("--end", default="2026-06-30")
    shadow_parser.add_argument("--run-id", help="Evaluate a staged official run")

    subparsers.add_parser(
        "recompute_horizon_acceptance",
        help="Reevaluate existing frozen stock artifacts under Horizon Acceptance V2",
    )

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--search-profile", choices=["full", "quick"], default="full")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command
    if command == "download_raw":
        run_download_raw()
    elif command == "download_financial_news":
        run_download_financial_news(args.ticker, args.as_of)
    elif command == "preprocess_raw":
        run_preprocess_raw()
    elif command == "build_model_input":
        run_build_model_input()
    elif command == "data_understanding":
        run_data_understanding()
    elif command == "preprocess_financial_news_sentiment":
        run_preprocess_financial_news_sentiment()
    elif command == "financial_news_sentiment_understanding":
        run_financial_news_sentiment_understanding()
    elif command == "train_sentiment_model":
        run_train_sentiment_model(args.search_profile)
    elif command == "benchmark_sentiment_models":
        run_benchmark_sentiment_models(args.profile)
    elif command == "train_finbert":
        run_train_finbert(args.profile, args.device, not args.no_promote)
    elif command == "prepare_bursa_news_dataset":
        run_prepare_bursa_news_dataset(
            args.source, args.output, args.backfill_report
        )
    elif command == "validate_bursa_news_dataset":
        run_validate_bursa_news_dataset(args.dataset, args.report, args.policy)
    elif command == "report_bursa_news_year":
        run_report_bursa_news_year(
            args.year,
            args.indexed,
            args.recovered,
            args.accepted,
            args.reviewed,
            args.output,
        )
    elif command == "train_bursa_finbert":
        run_train_bursa_finbert(
            args.dataset, args.profile, args.device, args.policy
        )
    elif command == "promote_bursa_finbert":
        run_promote_bursa_finbert(args.run_id, args.approve)
    elif command == "benchmark_finbert_ensemble":
        run_benchmark_finbert_ensemble(
            args.dataset,
            args.policy,
            args.bursa_model_directory,
            args.bursa_run_id,
            args.phrasebank_model_directory,
            args.bursa_temperature,
            args.phrasebank_temperature,
            args.weight_step,
        )
    elif command == "audit_system":
        run_audit_system(args.level, args.live_providers)
    elif command == "collect_news_provider_benchmark":
        run_collect_news_provider_benchmark(
            args.as_of,
            args.ticker,
            args.provider,
            args.run_id,
            args.cutoff_lag_days,
        )
    elif command == "evaluate_news_provider_benchmark":
        run_evaluate_news_provider_benchmark(args.annotations, args.apply_decision_rules)
    elif command == "train_models":
        run_train_models(args.search_profile)
    elif command == "promote_stock_models":
        run_promote_stock_models(
            args.run_id,
            args.approve,
            args.override_gates,
            args.override_reason,
        )
    elif command == "verify_stock_reproducibility":
        run_verify_stock_reproducibility(args.run_a, args.run_b)
    elif command == "predict_stock":
        run_predict_stock(args.ticker, args.horizon, args.as_of)
    elif command == "evaluate_horizon_shadow":
        run_evaluate_horizon_shadow(args.start, args.end, args.run_id)
    elif command == "recompute_horizon_acceptance":
        run_recompute_horizon_acceptance()
    elif command == "all":
        run_all(args.search_profile)


if __name__ == "__main__":
    main()
