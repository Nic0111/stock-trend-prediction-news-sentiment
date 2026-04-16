from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# Project configuration
# ============================================================
# This file is inside: project-folder/pipeline/
# So project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw-stock-dataset"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed-stock-dataset"
MODEL_INPUT_DATA_DIR = DATA_DIR / "model-input-stock-dataset"

SUMMARY_DIR = PROJECT_ROOT / "summary"
DATA_PREPROCESSING_SUMMARY_DIR = SUMMARY_DIR / "data-preprocessing"

SUMMARY_FILE = DATA_DIR / "model-input-summary.csv"

REQUIRED_COLUMNS = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]

# Locked target horizons
TARGET_HORIZONS = [1, 5, 10]

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Return_1D",
    "Return_5D",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_5",
    "MA_5",
    "MA_20",
    "MA_60",
    "Volatility_5D",
    "Volatility_20D",
    "High_Low_Spread",
    "Open_Close_Change",
    "Volume_Change_1D",
    "Volume_MA_5",
]

TARGET_COLUMNS = [
    "Target_1D",
    "Target_5D",
    "Target_10D",
]

MODEL_INPUT_COLUMNS = [
    "Date",
    "Ticker",
    *FEATURE_COLUMNS,
    *TARGET_COLUMNS,
]


# ============================================================
# Helper functions
# ============================================================
def ensure_data_structure() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PREPROCESSING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def format_number(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    if not rows:
        return "_No rows to report._"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_number(value) for value in row) + " |")
    return "\n".join(lines)


def get_preprocessed_csv_files() -> List[Path]:
    return sorted(PREPROCESSED_DATA_DIR.glob("*.csv"))


def build_output_filename(preprocessed_file: Path) -> str:
    """
    Convert:
    malayan-banking-berhad-preprocessed.csv
    -> malayan-banking-berhad-model-input.csv
    """
    name = preprocessed_file.name
    if name.endswith("-preprocessed.csv"):
        return name.replace("-preprocessed.csv", "-model-input.csv")
    return preprocessed_file.stem + "-model-input.csv"


def validate_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.copy()


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the locked baseline derived columns using Adj Close as the main price series.
    """
    df = df.copy()

    # Ensure correct order
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------
    df["Return_1D"] = df["Adj Close"].pct_change(1)
    df["Return_5D"] = df["Adj Close"].pct_change(5)

    # --------------------------------------------------------
    # Lags
    # --------------------------------------------------------
    df["Lag_1"] = df["Adj Close"].shift(1)
    df["Lag_2"] = df["Adj Close"].shift(2)
    df["Lag_3"] = df["Adj Close"].shift(3)
    df["Lag_5"] = df["Adj Close"].shift(5)

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------
    df["MA_5"] = df["Adj Close"].rolling(window=5).mean()
    df["MA_20"] = df["Adj Close"].rolling(window=20).mean()
    df["MA_60"] = df["Adj Close"].rolling(window=60).mean()

    # --------------------------------------------------------
    # Volatility (rolling std of daily returns)
    # --------------------------------------------------------
    daily_returns = df["Adj Close"].pct_change(1)
    df["Volatility_5D"] = daily_returns.rolling(window=5).std()
    df["Volatility_20D"] = daily_returns.rolling(window=20).std()

    # --------------------------------------------------------
    # Price spread / intraday movement
    # --------------------------------------------------------
    df["High_Low_Spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Change"] = (df["Close"] - df["Open"]) / df["Open"]

    # --------------------------------------------------------
    # Volume features
    # --------------------------------------------------------
    # Replace 0 with NaN before pct_change-style calculation to avoid divide-by-zero problems
    volume_safe = df["Volume"].replace(0, np.nan)
    df["Volume_Change_1D"] = volume_safe / volume_safe.shift(1) - 1
    df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all locked target columns:
    - Target_1D
    - Target_5D
    - Target_10D

    Definition:
    Target_h = 1 if AdjClose_(t+h) > AdjClose_t else 0
    """
    df = df.copy()

    for horizon in TARGET_HORIZONS:
        future_price = df["Adj Close"].shift(-horizon)
        target = (future_price > df["Adj Close"]).astype("float")
        target[future_price.isna()] = np.nan
        df[f"Target_{horizon}D"] = target

    return df


def finalize_model_input(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Drop rows with NaN caused by lag / rolling / future target generation.
    Also clean up any inf values.
    """
    df = df.copy()
    original_rows = len(df)

    # Replace inf with NaN before final dropping
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df[MODEL_INPUT_COLUMNS].copy()

    rows_with_nan_or_inf = int(df.isna().any(axis=1).sum())
    df = df.dropna().reset_index(drop=True)

    # Convert targets from float to int after dropping NaN rows
    for target_column in TARGET_COLUMNS:
        df[target_column] = df[target_column].astype(int)

    # Standardize date string format for saved CSV
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    summary = {
        "original_rows": original_rows,
        "rows_dropped_due_to_feature_nan_or_inf": rows_with_nan_or_inf,
        "final_rows": len(df),
    }

    return df, summary


def process_single_file(file_path: Path) -> tuple[pd.DataFrame, Dict[str, int | str]]:
    df = pd.read_csv(file_path)
    df = validate_required_columns(df)
    df = add_derived_features(df)
    df = add_targets(df)
    df, summary = finalize_model_input(df)

    result_summary = {
        "file_name": file_path.name,
        **summary,
    }
    return df, result_summary


def save_model_input_csv(df: pd.DataFrame, preprocessed_file: Path) -> Path:
    output_name = build_output_filename(preprocessed_file)
    output_path = MODEL_INPUT_DATA_DIR / output_name
    temp_path = output_path.with_suffix(".tmp")
    try:
        df.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
    except PermissionError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise PermissionError(
            f"Could not replace {output_path}. Close the file if it is open "
            "in Excel or another application, then rerun build_model_input."
        ) from exc
    return output_path


def save_summary(summary_rows: List[Dict[str, int | str]]) -> None:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)


def write_model_input_ready_summary(summary_rows: List[Dict[str, int | str]]) -> None:
    total_original = int(sum(int(row["original_rows"]) for row in summary_rows))
    total_dropped = int(
        sum(int(row["rows_dropped_due_to_feature_nan_or_inf"]) for row in summary_rows)
    )
    total_final = int(sum(int(row["final_rows"]) for row in summary_rows))

    table_rows = [
        [
            row["file_name"],
            row["original_rows"],
            row["rows_dropped_due_to_feature_nan_or_inf"],
            row["final_rows"],
        ]
        for row in summary_rows
    ]

    feature_list = "\n".join(f"- `{column}`" for column in FEATURE_COLUMNS)
    target_list = "\n".join(f"- `{column}`" for column in TARGET_COLUMNS)

    content = f"""# 3.4.1.6 Model Input-Ready Dataset Summary

## Purpose
This section documents the dataset construction step that converts the preprocessed stock dataset into model input-ready rows. This is feature engineering and target construction, not standardization or normalization.

## Dataset Location
`data/model-input-stock-dataset/`

## Method Applied
- Daily and multi-day return features are created from `Adj Close`.
- Lag, moving-average, volatility, price movement, and volume features are created.
- Classification targets are created for 1-day, 5-day, and 10-day future price direction.
- Rows with missing or infinite values from lag, rolling-window, volume-change, or future-target construction are removed.

## Overall Result
- Files processed: {len(summary_rows)}
- Preprocessed input rows: {total_original:,}
- Rows dropped during model input preparation: {total_dropped:,}
- Model input-ready rows: {total_final:,}

## File-Level Statistics
{markdown_table(
        [
            "Preprocessed File",
            "Input Rows",
            "Rows Dropped",
            "Final Rows",
        ],
        table_rows,
    )}

## Feature Columns
{feature_list}

## Target Columns
{target_list}

## Relationship To Standardization/Normalization
This stage prepares the columns required by the prediction models. Scaling remains model-dependent and is handled during training when needed, such as the `StandardScaler` used inside the Logistic Regression training pipeline.
"""
    output_path = DATA_PREPROCESSING_SUMMARY_DIR / "model-input-ready-dataset-summary.md"
    output_path.write_text(content, encoding="utf-8")


# ============================================================
# Main pipeline
# ============================================================
def run_model_input_pipeline() -> None:
    ensure_data_structure()

    preprocessed_files = get_preprocessed_csv_files()
    if not preprocessed_files:
        print(f"No preprocessed CSV files found in: {PREPROCESSED_DATA_DIR}")
        return

    summary_rows: List[Dict[str, int | str]] = []

    print("[Starting model-input dataset pipeline]\n")

    for preprocessed_file in preprocessed_files:
        try:
            print(f"Processing: {preprocessed_file.name}")
            model_df, summary = process_single_file(preprocessed_file)
            output_path = save_model_input_csv(model_df, preprocessed_file)

            summary_rows.append(summary)

            print(f"Saved: {output_path}")
            print(f"Original rows: {summary['original_rows']}")
            print(f"Final rows: {summary['final_rows']}")
            print(
                "Rows dropped due to lag/rolling/target NaN or inf: "
                f"{summary['rows_dropped_due_to_feature_nan_or_inf']}"
            )
            print()

        except Exception as exc:
            print(f"Failed to build model input for {preprocessed_file.name}: {exc}\n")

    save_summary(summary_rows)
    write_model_input_ready_summary(summary_rows)
    print(f"Model-input summary saved to: {SUMMARY_FILE}")
    print(
        "Model input-ready Markdown summary saved to: "
        f"{DATA_PREPROCESSING_SUMMARY_DIR / 'model-input-ready-dataset-summary.md'}"
    )
    print("\n[Model-input dataset pipeline finished]")


def main() -> None:
    run_model_input_pipeline()


if __name__ == "__main__":
    main()
