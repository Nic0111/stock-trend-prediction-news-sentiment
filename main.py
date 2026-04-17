from __future__ import annotations

import argparse
import sys


def run_download_raw() -> None:
    from pipeline.download_raw_stock_data import run_download_pipeline
    run_download_pipeline()


def run_download_financial_news() -> None:
    from pipeline.download_financial_news_data import run_financial_news_download_pipeline
    run_financial_news_download_pipeline()


def run_preprocess_raw() -> None:
    try:
        from pipeline.preprocess_raw_stock_data import run_preprocess_pipeline
    except ImportError:
        print(
            "preprocess_raw_stock_data.py is not ready yet.\n"
            "Create pipeline/preprocess_raw_stock_data.py with a function named "
            "`run_preprocess_pipeline()` first."
        )
        return

    run_preprocess_pipeline()


def run_build_model_input() -> None:
    try:
        from pipeline.build_model_input_stock_data import run_model_input_pipeline
    except ImportError:
        print(
            "build_model_input_stock_data.py is not ready yet.\n"
            "Create pipeline/build_model_input_stock_data.py with a function named "
            "`run_model_input_pipeline()` first."
        )
        return

    run_model_input_pipeline()


def run_data_understanding() -> None:
    try:
        from pipeline.data_understanding import run_data_understanding_pipeline
    except ImportError:
        print(
            "data_understanding.py is not ready yet.\n"
            "Create pipeline/data_understanding.py with a function named "
            "`run_data_understanding_pipeline()` first."
        )
        return

    run_data_understanding_pipeline()


def run_train_models() -> None:
    try:
        from pipeline.train_stock_models import run_training_pipeline
    except ImportError:
        print(
            "train_stock_models.py is not ready yet.\n"
            "Create pipeline/train_stock_models.py with a function named "
            "`run_training_pipeline()` first."
        )
        return

    run_training_pipeline()


def run_all() -> None:
    print("Step 1/5: Downloading raw stock data...")
    run_download_raw()

    print("\nStep 2/5: Preprocessing stock data...")
    run_preprocess_raw()

    print("\nStep 3/5: Building model-input stock data...")
    run_build_model_input()

    print("\nStep 4/5: Generating data understanding summaries...")
    run_data_understanding()

    print("\nStep 5/5: Training stock models...")
    run_train_models()

    print("\nPipeline finished.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FYP data pipeline controller"
    )

    parser.add_argument(
        "command",
        choices=[
            "download_raw",
            "download_financial_news",
            "preprocess_raw",
            "build_model_input",
            "data_understanding",
            "train_models",
            "all",
        ],
        help=(
            "Pipeline command to run: "
            "download_raw, download_financial_news, preprocess_raw, "
            "build_model_input, data_understanding, train_models, or all"
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download_raw":
        run_download_raw()
    elif args.command == "download_financial_news":
        run_download_financial_news()
    elif args.command == "preprocess_raw":
        run_preprocess_raw()
    elif args.command == "build_model_input":
        run_build_model_input()
    elif args.command == "data_understanding":
        run_data_understanding()
    elif args.command == "train_models":
        run_train_models()
    elif args.command == "all":
        run_all()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
