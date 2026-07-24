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
    FusionDatasetNotReadyError,
    train_fusion_models,
)
from pipeline.historical_news_fusion_data import (
    run_build_point_in_time_features,
    run_fusion_data_readiness,
    run_import_historical_news_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and train the experimental StockSense stock/news fusion challenger.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-news", help="Normalize provider exports into ticker/year Parquet partitions.")
    import_parser.add_argument("sources", nargs="+", help="CSV, JSON, JSONL or Parquet provider export paths.")
    import_parser.add_argument("--output-dir", default=None)
    import_parser.add_argument("--provider", default=None)

    readiness_parser = subparsers.add_parser("readiness", help="Report historical-news coverage and feature readiness.")
    readiness_parser.add_argument("--data-dir", default=None)

    feature_parser = subparsers.add_parser("features", help="Build point-in-time fusion features from stock rows and normalized news.")
    feature_parser.add_argument("--stock-path", required=True)
    feature_parser.add_argument("--news-dir", default=None)
    feature_parser.add_argument("--output", default=None)

    train_parser = subparsers.add_parser("train", help="Train experimental horizon-specific fusion artifacts when readiness passes.")
    train_parser.add_argument("--dataset", default=str(DEFAULT_FUSION_DATASET))
    train_parser.add_argument("--registry", default=str(DEFAULT_FUSION_REGISTRY))
    train_parser.add_argument("--ticker", action="append", dest="tickers", default=None)
    train_parser.add_argument("--min-rows", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "import-news":
        run_import_historical_news_pipeline(args.sources, args.output_dir, args.provider)
        return 0
    if args.command == "readiness":
        report = run_fusion_data_readiness(args.data_dir)
        return 0 if report.get("ready") else 2
    if args.command == "features":
        run_build_point_in_time_features(args.stock_path, args.news_dir, args.output)
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
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
