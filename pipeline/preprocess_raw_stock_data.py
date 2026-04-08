from __future__ import annotations

from pathlib import Path
from typing import Dict, List

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

SUMMARY_FILE = DATA_DIR / "preprocessing-summary.csv"

EXPECTED_COLUMNS = [
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

ESSENTIAL_NUMERIC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

OPTIONAL_ACTION_COLUMNS = [
    "Dividends",
    "Stock Splits",
]


# ============================================================
# Helper functions
# ============================================================
def ensure_data_structure() -> None:
    """
    Ensure the project data folders exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_csv_files() -> List[Path]:
    """
    Return all raw CSV files from the raw-stock-dataset folder.
    """
    return sorted(RAW_DATA_DIR.glob("*.csv"))


def build_output_filename(raw_file: Path) -> str:
    """
    Convert:
    malayan-banking-berhad-raw.csv
    -> malayan-banking-berhad-preprocessed.csv
    """
    name = raw_file.name
    if name.endswith("-raw.csv"):
        return name.replace("-raw.csv", "-preprocessed.csv")
    return raw_file.stem + "-preprocessed.csv"


def ensure_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expected columns exist.
    If Dividends / Stock Splits are missing, create them with 0.
    """
    df = df.copy()

    for col in OPTIONAL_ACTION_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    missing_required = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    return df


def preprocess_single_file(file_path: Path) -> tuple[pd.DataFrame, Dict[str, int | str]]:
    """
    Preprocess one raw stock dataset according to the locked rules.

    Rules applied:
    - Standardize columns
    - Convert Date to datetime
    - Convert numeric columns to numeric
    - Sort by Date ascending
    - Remove duplicate Date rows
    - Fill missing Dividends and Stock Splits with 0
    - Drop rows with missing essential OHLC / Adj Close / Volume values
    - Drop logically invalid rows:
        * non-positive prices
        * negative volume
        * High < Low
        * High < Open or High < Close
        * Low > Open or Low > Close
    """
    df = pd.read_csv(file_path)
    original_rows = len(df)

    df = ensure_expected_columns(df)

    # Keep only expected columns in the locked order
    df = df[EXPECTED_COLUMNS].copy()

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_date_rows = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"])

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # Fill optional action columns
    # --------------------------------------------------------
    missing_dividends_before_fill = int(df["Dividends"].isna().sum())
    missing_splits_before_fill = int(df["Stock Splits"].isna().sum())

    df["Dividends"] = df["Dividends"].fillna(0.0)
    df["Stock Splits"] = df["Stock Splits"].fillna(0.0)

    # --------------------------------------------------------
    # Sort by date ascending
    # --------------------------------------------------------
    df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    # --------------------------------------------------------
    # Remove duplicate dates within the same stock file
    # --------------------------------------------------------
    duplicate_date_rows = int(df.duplicated(subset=["Date"]).sum())
    df = df.drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)

    # --------------------------------------------------------
    # Drop rows with missing essential numeric values
    # --------------------------------------------------------
    missing_essential_rows = int(df[ESSENTIAL_NUMERIC_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=ESSENTIAL_NUMERIC_COLUMNS).reset_index(drop=True)

    # --------------------------------------------------------
    # Logical validity checks
    # --------------------------------------------------------
    invalid_mask = (
        (df["Open"] <= 0) |
        (df["High"] <= 0) |
        (df["Low"] <= 0) |
        (df["Close"] <= 0) |
        (df["Adj Close"] <= 0) |
        (df["Volume"] < 0) |
        (df["High"] < df["Low"]) |
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"]) |
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["Close"])
    )

    logically_invalid_rows = int(invalid_mask.sum())
    df = df.loc[~invalid_mask].reset_index(drop=True)

    # --------------------------------------------------------
    # Standardize Date format for CSV output
    # --------------------------------------------------------
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    final_rows = len(df)

    summary = {
        "file_name": file_path.name,
        "original_rows": original_rows,
        "invalid_date_rows_dropped": invalid_date_rows,
        "duplicate_date_rows_removed": duplicate_date_rows,
        "missing_dividends_filled_with_zero": missing_dividends_before_fill,
        "missing_stock_splits_filled_with_zero": missing_splits_before_fill,
        "rows_with_missing_essential_values_dropped": missing_essential_rows,
        "logically_invalid_rows_dropped": logically_invalid_rows,
        "final_rows": final_rows,
    }

    return df, summary


def save_preprocessed_csv(df: pd.DataFrame, raw_file: Path) -> Path:
    """
    Save preprocessed stock data into data/preprocessed-stock-dataset/
    """
    output_name = build_output_filename(raw_file)
    output_path = PREPROCESSED_DATA_DIR / output_name
    df.to_csv(output_path, index=False)
    return output_path


def save_summary(summary_rows: List[Dict[str, int | str]]) -> None:
    """
    Save preprocessing summary log as CSV.
    """
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)


# ============================================================
# Main pipeline
# ============================================================
def run_preprocess_pipeline() -> None:
    ensure_data_structure()

    raw_files = get_raw_csv_files()
    if not raw_files:
        print(f"No raw CSV files found in: {RAW_DATA_DIR}")
        return

    summary_rows: List[Dict[str, int | str]] = []

    print("[Starting preprocessing pipeline]\n")

    for raw_file in raw_files:
        try:
            print(f"Processing: {raw_file.name}")
            cleaned_df, summary = preprocess_single_file(raw_file)
            output_path = save_preprocessed_csv(cleaned_df, raw_file)

            summary_rows.append(summary)

            print(f"Saved: {output_path}")
            print(f"Original rows: {summary['original_rows']}")
            print(f"Final rows: {summary['final_rows']}")
            print(f"Duplicate dates removed: {summary['duplicate_date_rows_removed']}")
            print(
                "Rows dropped due to missing essential values: "
                f"{summary['rows_with_missing_essential_values_dropped']}"
            )
            print(
                "Rows dropped due to logical invalidity: "
                f"{summary['logically_invalid_rows_dropped']}"
            )
            print()

        except Exception as exc:
            print(f"Failed to preprocess {raw_file.name}: {exc}\n")

    save_summary(summary_rows)
    print(f"Preprocessing summary saved to: {SUMMARY_FILE}")
    print("\n[Preprocessing pipeline finished]")


def main() -> None:
    run_preprocess_pipeline()


if __name__ == "__main__":
    main()