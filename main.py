from __future__ import annotations

import argparse
import sys


def run_download_raw() -> None:
    from pipeline.download_raw_stock_data import run_download_pipeline
    run_download_pipeline()


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
    print("Step 1/4: Downloading raw stock data...")
    run_download_raw()

    print("\nStep 2/4: Preprocessing stock data...")
    run_preprocess_raw()

    print("\nStep 3/4: Building model-input stock data...")
    run_build_model_input()

    print("\nStep 4/4: Training stock models...")
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
            "preprocess_raw",
            "build_model_input",
            "train_models",
            "all",
        ],
        help=(
            "Pipeline command to run: "
            "download_raw, preprocess_raw, build_model_input, train_models, or all"
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download_raw":
        run_download_raw()
    elif args.command == "preprocess_raw":
        run_preprocess_raw()
    elif args.command == "build_model_input":
        run_build_model_input()
    elif args.command == "train_models":
        run_train_models()
    elif args.command == "all":
        run_all()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()