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

    run_training_pipeline(search_profile=search_profile)


def run_predict_stock(ticker: str, horizon: int, as_of: str | None = None) -> None:
    from pipeline.predict_stock_trend import predict_stock_trend

    result = predict_stock_trend(ticker, horizon, as_of)
    print(json.dumps(result.to_dict(), indent=2))


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
        "--search-profile", choices=["full", "focused", "quick"], default="full"
    )

    prediction_parser = subparsers.add_parser("predict_stock")
    prediction_parser.add_argument("--ticker", required=True)
    prediction_parser.add_argument("--horizon", type=int, choices=[1, 5, 10], required=True)
    prediction_parser.add_argument("--as-of")

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
    elif command == "predict_stock":
        run_predict_stock(args.ticker, args.horizon, args.as_of)
    elif command == "all":
        run_all(args.search_profile)


if __name__ == "__main__":
    main()
