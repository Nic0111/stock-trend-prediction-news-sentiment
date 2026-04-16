from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ============================================================
# Project configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw-stock-dataset"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed-stock-dataset"
MODEL_INPUT_DATA_DIR = DATA_DIR / "model-input-stock-dataset"
SAMPLE_ONLY_DIR = DATA_DIR / "sample-only"
STANDARDIZED_SAMPLE_DATA_DIR = SAMPLE_ONLY_DIR / "standardized-stock-dataset"

SUMMARY_DIR = PROJECT_ROOT / "summary"
DATA_PREPROCESSING_SUMMARY_DIR = SUMMARY_DIR / "data-preprocessing"

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

ESSENTIAL_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

ESSENTIAL_NUMERIC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

NUMERIC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]

MARKET_NUMERIC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

STANDARDIZATION_COLUMNS = MARKET_NUMERIC_COLUMNS

OPTIONAL_ACTION_COLUMNS = [
    "Dividends",
    "Stock Splits",
]


# ============================================================
# Generic helpers
# ============================================================
def ensure_data_structure() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    STANDARDIZED_SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PREPROCESSING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_csv_files() -> List[Path]:
    return sorted(RAW_DATA_DIR.glob("*.csv"))


def build_output_filename(raw_file: Path) -> str:
    name = raw_file.name
    if name.endswith("-raw.csv"):
        return name.replace("-raw.csv", "-preprocessed.csv")
    return raw_file.stem + "-preprocessed.csv"


def build_standardized_sample_filename(raw_file: Path) -> str:
    name = raw_file.name
    if name.endswith("-raw.csv"):
        return name.replace("-raw.csv", "-standardized-sample.csv")
    return raw_file.stem + "-standardized-sample.csv"


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def format_number(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_No rows to report._"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(format_number(value) for value in row) + " |")

    return "\n".join(lines)


def summarize_total(summary_rows: List[Dict[str, Any]], column: str) -> int:
    return int(sum(int(row.get(column, 0)) for row in summary_rows))


# ============================================================
# 3.4.1.1 Missing Values
# ============================================================
def ensure_expected_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    added_optional_columns: List[str] = []

    for col in OPTIONAL_ACTION_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            added_optional_columns.append(col)

    missing_required = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    return df[EXPECTED_COLUMNS].copy(), added_optional_columns


def handle_missing_values(
    df: pd.DataFrame,
    file_name: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    original_rows = len(df)
    df, added_optional_columns = ensure_expected_columns(df)

    missing_before = {
        col: int(df[col].isna().sum())
        for col in EXPECTED_COLUMNS
    }

    missing_dividends_before_fill = int(df["Dividends"].isna().sum())
    missing_splits_before_fill = int(df["Stock Splits"].isna().sum())

    df["Dividends"] = df["Dividends"].fillna(0.0)
    df["Stock Splits"] = df["Stock Splits"].fillna(0.0)

    rows_with_missing_essential = int(df[ESSENTIAL_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=ESSENTIAL_COLUMNS).reset_index(drop=True)

    summary = {
        "file_name": file_name,
        "original_rows": original_rows,
        "added_optional_columns": ", ".join(added_optional_columns) if added_optional_columns else "None",
        "missing_date_before": missing_before["Date"],
        "missing_open_before": missing_before["Open"],
        "missing_high_before": missing_before["High"],
        "missing_low_before": missing_before["Low"],
        "missing_close_before": missing_before["Close"],
        "missing_adj_close_before": missing_before["Adj Close"],
        "missing_volume_before": missing_before["Volume"],
        "missing_dividends_filled_with_zero": missing_dividends_before_fill,
        "missing_stock_splits_filled_with_zero": missing_splits_before_fill,
        "rows_with_missing_essential_values_dropped": rows_with_missing_essential,
        "final_rows_after_missing_values": len(df),
    }

    return df, summary


def write_missing_values_summary(summary_rows: List[Dict[str, Any]]) -> None:
    total_original = summarize_total(summary_rows, "original_rows")
    total_final = summarize_total(summary_rows, "final_rows_after_missing_values")
    total_dropped = summarize_total(summary_rows, "rows_with_missing_essential_values_dropped")
    total_dividends = summarize_total(summary_rows, "missing_dividends_filled_with_zero")
    total_splits = summarize_total(summary_rows, "missing_stock_splits_filled_with_zero")

    table_rows = [
        [
            row["file_name"],
            row["original_rows"],
            row["rows_with_missing_essential_values_dropped"],
            row["missing_dividends_filled_with_zero"],
            row["missing_stock_splits_filled_with_zero"],
            row["final_rows_after_missing_values"],
        ]
        for row in summary_rows
    ]

    content = f"""# 3.4.1.1 Missing Values Summary

## Purpose
This step validates the raw stock dataset columns, fills optional corporate action fields, and removes records that cannot be used because essential market values are missing.

## Method Applied
- Required stock columns are checked before processing.
- `Dividends` and `Stock Splits` are filled with `0.0` because missing values in these fields mean no recorded action for that trading day.
- Rows missing `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, or `Volume` are removed because these fields are required for stock prediction features.

## Overall Result
- Files processed: {len(summary_rows)}
- Original rows: {total_original:,}
- Rows dropped due to missing essential values: {total_dropped:,}
- Missing dividend values filled with zero: {total_dividends:,}
- Missing stock split values filled with zero: {total_splits:,}
- Rows after missing-value handling: {total_final:,}

## File-Level Statistics
{markdown_table(
        [
            "File",
            "Original Rows",
            "Rows Dropped",
            "Dividends Filled",
            "Stock Splits Filled",
            "Final Rows",
        ],
        table_rows,
    )}
"""
    write_markdown(DATA_PREPROCESSING_SUMMARY_DIR / "missing-values-summary.md", content)


# ============================================================
# 3.4.1.2 Outlier
# ============================================================
def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_logically_invalid_mask(df: pd.DataFrame) -> pd.Series:
    return (
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


def detect_iqr_outliers(df: pd.DataFrame) -> tuple[Dict[str, Dict[str, float | int]], int]:
    details: Dict[str, Dict[str, float | int]] = {}
    any_outlier_mask = pd.Series(False, index=df.index)

    for col in MARKET_NUMERIC_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            details[col] = {
                "q1": 0.0,
                "q3": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "outlier_count": 0,
            }
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        any_outlier_mask = any_outlier_mask | col_mask.fillna(False)

        details[col] = {
            "q1": q1,
            "q3": q3,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(col_mask.sum()),
        }

    return details, int(any_outlier_mask.sum())


def handle_outliers(
    df: pd.DataFrame,
    file_name: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df = convert_numeric_columns(df)
    original_rows = len(df)

    missing_numeric_after_conversion = int(df[ESSENTIAL_NUMERIC_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=ESSENTIAL_NUMERIC_COLUMNS).reset_index(drop=True)

    invalid_mask = build_logically_invalid_mask(df)
    logically_invalid_rows = int(invalid_mask.sum())
    df = df.loc[~invalid_mask].reset_index(drop=True)

    iqr_details, rows_with_any_statistical_outlier = detect_iqr_outliers(df)

    summary: Dict[str, Any] = {
        "file_name": file_name,
        "original_rows_before_outlier_step": original_rows,
        "rows_with_numeric_conversion_missing_dropped": missing_numeric_after_conversion,
        "logically_invalid_rows_dropped": logically_invalid_rows,
        "statistical_outlier_rows_flagged_not_removed": rows_with_any_statistical_outlier,
        "final_rows_after_outlier_step": len(df),
    }

    for col, detail in iqr_details.items():
        safe_col = col.lower().replace(" ", "_")
        summary[f"{safe_col}_iqr_lower_bound"] = detail["lower_bound"]
        summary[f"{safe_col}_iqr_upper_bound"] = detail["upper_bound"]
        summary[f"{safe_col}_statistical_outlier_count"] = detail["outlier_count"]

    return df, summary


def write_outlier_summary(summary_rows: List[Dict[str, Any]]) -> None:
    total_logical = summarize_total(summary_rows, "logically_invalid_rows_dropped")
    total_flagged = summarize_total(summary_rows, "statistical_outlier_rows_flagged_not_removed")
    total_final = summarize_total(summary_rows, "final_rows_after_outlier_step")

    file_rows = [
        [
            row["file_name"],
            row["logically_invalid_rows_dropped"],
            row["statistical_outlier_rows_flagged_not_removed"],
            row["final_rows_after_outlier_step"],
        ]
        for row in summary_rows
    ]

    outlier_rows: List[List[Any]] = []
    for row in summary_rows:
        for col in MARKET_NUMERIC_COLUMNS:
            safe_col = col.lower().replace(" ", "_")
            outlier_rows.append([
                row["file_name"],
                col,
                row[f"{safe_col}_iqr_lower_bound"],
                row[f"{safe_col}_iqr_upper_bound"],
                row[f"{safe_col}_statistical_outlier_count"],
            ])

    content = f"""# 3.4.1.2 Outlier Summary

## Purpose
This step removes records that are logically impossible for OHLCV stock data and documents statistical outliers for interpretation.

## Method Applied
- Logical invalid rows are removed because they break stock price rules.
- IQR outliers are detected for `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.
- Statistical outliers are kept in the dataset because sharp market movement and unusual volume may carry predictive information.

## Overall Result
- Files processed: {len(summary_rows)}
- Logically invalid rows removed: {total_logical:,}
- Rows flagged as statistical outliers but kept: {total_flagged:,}
- Rows after outlier handling: {total_final:,}

## File-Level Result
{markdown_table(
        [
            "File",
            "Logical Rows Removed",
            "Statistical Rows Flagged",
            "Final Rows",
        ],
        file_rows,
    )}

## IQR Outlier Details
{markdown_table(
        [
            "File",
            "Column",
            "IQR Lower Bound",
            "IQR Upper Bound",
            "Outlier Count",
        ],
        outlier_rows,
    )}
"""
    write_markdown(DATA_PREPROCESSING_SUMMARY_DIR / "outlier-summary.md", content)


# ============================================================
# 3.4.1.3 Data Transformation
# ============================================================
def transform_stock_data(
    df: pd.DataFrame,
    file_name: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    original_rows = len(df)
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_date_rows = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"])

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    duplicate_date_rows = int(df.duplicated(subset=["Date"]).sum())
    df = df.drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)

    df = df[EXPECTED_COLUMNS].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    summary = {
        "file_name": file_name,
        "original_rows_before_transformation": original_rows,
        "invalid_date_rows_dropped": invalid_date_rows,
        "duplicate_date_rows_removed": duplicate_date_rows,
        "final_rows_after_transformation": len(df),
        "columns_after_transformation": ", ".join(df.columns),
        "first_date": df["Date"].min() if not df.empty else "N/A",
        "last_date": df["Date"].max() if not df.empty else "N/A",
    }

    return df, summary


def write_data_transformation_summary(summary_rows: List[Dict[str, Any]]) -> None:
    total_invalid_dates = summarize_total(summary_rows, "invalid_date_rows_dropped")
    total_duplicates = summarize_total(summary_rows, "duplicate_date_rows_removed")
    total_final = summarize_total(summary_rows, "final_rows_after_transformation")

    table_rows = [
        [
            row["file_name"],
            row["invalid_date_rows_dropped"],
            row["duplicate_date_rows_removed"],
            row["first_date"],
            row["last_date"],
            row["final_rows_after_transformation"],
        ]
        for row in summary_rows
    ]

    content = f"""# 3.4.1.3 Data Transformation Summary

## Purpose
This step converts the cleaned data into a consistent dataset structure before saving the preprocessed stock dataset.

## Method Applied
- `Date` is converted to datetime and saved as `YYYY-MM-DD`.
- Numeric fields are converted to numeric values.
- Records are sorted in ascending date order.
- Duplicate dates inside each stock file are removed.
- The final column order is kept consistent across all stock datasets.

## Overall Result
- Files processed: {len(summary_rows)}
- Invalid date rows removed: {total_invalid_dates:,}
- Duplicate date rows removed: {total_duplicates:,}
- Rows after transformation: {total_final:,}

## File-Level Statistics
{markdown_table(
        [
            "File",
            "Invalid Dates Removed",
            "Duplicate Dates Removed",
            "First Date",
            "Last Date",
            "Final Rows",
        ],
        table_rows,
    )}

## Final Column Order
`{", ".join(EXPECTED_COLUMNS)}`
"""
    write_markdown(DATA_PREPROCESSING_SUMMARY_DIR / "data-transformation-summary.md", content)


# ============================================================
# 3.4.1.4 Preprocessed Dataset
# ============================================================
def save_preprocessed_csv(df: pd.DataFrame, raw_file: Path) -> Path:
    output_name = build_output_filename(raw_file)
    output_path = PREPROCESSED_DATA_DIR / output_name
    df.to_csv(output_path, index=False)
    return output_path


def build_preprocessed_dataset_summary_row(
    df: pd.DataFrame,
    raw_file: Path,
    output_path: Path,
    original_rows: int,
) -> Dict[str, Any]:
    if df.empty:
        first_date = "N/A"
        last_date = "N/A"
        ticker = "N/A"
    else:
        first_date = str(df["Date"].min())
        last_date = str(df["Date"].max())
        ticker = str(df["Ticker"].iloc[0])

    return {
        "file_name": raw_file.name,
        "ticker": ticker,
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "raw_rows": original_rows,
        "preprocessed_rows": len(df),
        "rows_removed_total": original_rows - len(df),
        "first_date": first_date,
        "last_date": last_date,
        "column_count": len(df.columns),
    }


def write_preprocessed_dataset_summary(summary_rows: List[Dict[str, Any]]) -> None:
    total_raw_rows = summarize_total(summary_rows, "raw_rows")
    total_preprocessed_rows = summarize_total(summary_rows, "preprocessed_rows")
    total_removed = summarize_total(summary_rows, "rows_removed_total")

    table_rows = [
        [
            row["file_name"],
            row["ticker"],
            row["raw_rows"],
            row["preprocessed_rows"],
            row["rows_removed_total"],
            row["first_date"],
            row["last_date"],
            row["output_file"],
        ]
        for row in summary_rows
    ]

    content = f"""# 3.4.1.4 Preprocessed Dataset Summary

## Purpose
This section documents the dataset that has undergone preprocessing. The output is the cleaned stock dataset used as the source for model input-ready feature construction.

## Dataset Location
`data/preprocessed-stock-dataset/`

## Difference From Raw Dataset
- Raw data is downloaded directly from Yahoo Finance.
- The preprocessed dataset has validated columns, handled missing values, removed logically invalid rows, converted data types, removed duplicate dates, and standardized date formatting.
- The preprocessed dataset is not scaled. Scaling is discussed separately because it depends on the model training scenario.

## Overall Result
- Files processed: {len(summary_rows)}
- Raw rows: {total_raw_rows:,}
- Preprocessed rows: {total_preprocessed_rows:,}
- Total rows removed through preprocessing: {total_removed:,}

## File-Level Dataset Output
{markdown_table(
        [
            "Raw File",
            "Ticker",
            "Raw Rows",
            "Preprocessed Rows",
            "Rows Removed",
            "First Date",
            "Last Date",
            "Output File",
        ],
        table_rows,
    )}

## Dataset Columns
`{", ".join(EXPECTED_COLUMNS)}`
"""
    write_markdown(DATA_PREPROCESSING_SUMMARY_DIR / "preprocessed-dataset-summary.md", content)


# ============================================================
# 3.4.1.5 Data Standardization/Normalization
# ============================================================
def create_standardized_sample_dataset(
    df: pd.DataFrame,
    raw_file: Path,
) -> tuple[Path, List[Dict[str, Any]]]:
    standardized_df = df.copy()
    output_name = build_standardized_sample_filename(raw_file)
    output_path = STANDARDIZED_SAMPLE_DATA_DIR / output_name
    summary_rows: List[Dict[str, Any]] = []

    for col in STANDARDIZATION_COLUMNS:
        standardized_df[col] = pd.to_numeric(standardized_df[col], errors="coerce")
        original_mean = float(standardized_df[col].mean())
        original_std = float(standardized_df[col].std(ddof=1))

        if pd.isna(original_std) or original_std == 0:
            standardized_df[col] = 0.0
            status = "std_missing_or_zero"
        else:
            standardized_df[col] = (standardized_df[col] - original_mean) / original_std
            status = "standardized"

        standardized_mean = float(standardized_df[col].mean())
        standardized_std = float(standardized_df[col].std(ddof=1))

        summary_rows.append({
            "file_name": raw_file.name,
            "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            "column": col,
            "row_count": len(standardized_df),
            "original_mean": original_mean,
            "original_std": original_std,
            "standardized_mean": standardized_mean,
            "standardized_std": standardized_std,
            "status": status,
        })

    standardized_df = standardized_df[EXPECTED_COLUMNS].copy()
    standardized_df.to_csv(output_path, index=False)
    return output_path, summary_rows


def write_standardization_normalization_summary(
    summary_rows: List[Dict[str, Any]],
) -> None:
    output_files = sorted({row["output_file"] for row in summary_rows})
    table_rows = [
        [
            row["file_name"],
            row["column"],
            row["row_count"],
            row["original_mean"],
            row["original_std"],
            row["standardized_mean"],
            row["standardized_std"],
            row["status"],
        ]
        for row in summary_rows
    ]

    output_file_rows = [[output_file] for output_file in output_files]

    content = f"""# 3.4.1.5 Data Standardization/Normalization Summary

## Purpose
This section explains standardization and normalization for the stock prediction system. The main preprocessed dataset is not scaled at this stage because scaling requirements depend on the model training scenario.

## Sample-Only Dataset Location
`data/sample-only/standardized-stock-dataset/`

These standardized CSV files are generated only to show what a standardized dataset would look like for documentation. They are not used as the production preprocessing output and are not used by the model input-ready dataset pipeline.

## Understanding Of Scaling
- Standardization converts a feature to a distribution centered around mean `0` with standard deviation `1`.
- Normalization commonly rescales values into a fixed range such as `0` to `1`.
- These methods change feature magnitude, but they do not create new stock market information.

## Sample-Only Method Applied
- Z-score standardization is applied to `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.
- `Date`, `Ticker`, `Dividends`, and `Stock Splits` are preserved without scaling.
- The full row count and original column order are preserved for each sample-only standardized file.
- If a column has missing or zero standard deviation, the standardized sample column is set to `0.0` and marked in the status field.

## Why Scaling Is Not Applied To The Main Preprocessed Dataset
- The preprocessed dataset is intended to preserve interpretable stock prices, volume, dividends, and split values.
- Scaling before train/test splitting can leak future information because scaling parameters would be calculated using future dates.
- Scaling should be fitted on the training or development period only, then applied to validation/test data.

## When Scaling Is Needed
- Logistic Regression is sensitive to feature scale because it optimizes coefficients across features with different magnitudes.
- The current training pipeline handles this correctly by placing `StandardScaler` inside the Logistic Regression `Pipeline`.
- Random Forest and XGBoost are tree-based models and generally do not require standardization in the same way because they split based on feature thresholds.

## Project Decision
For this project stage, scaling is documented as model-dependent preprocessing knowledge. The main preprocessed dataset remains unscaled, and model-specific scaling is handled during training where required.

## Sample-Only Output Files
{markdown_table(["Output File"], output_file_rows)}

## Standardization Statistics
{markdown_table(
        [
            "Raw File",
            "Column",
            "Rows",
            "Original Mean",
            "Original Std",
            "Standardized Mean",
            "Standardized Std",
            "Status",
        ],
        table_rows,
    )}
"""
    write_markdown(
        DATA_PREPROCESSING_SUMMARY_DIR / "data-standardization-normalization-summary.md",
        content,
    )


# ============================================================
# Main pipeline
# ============================================================
def preprocess_single_file(raw_file: Path) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    raw_df = pd.read_csv(raw_file)

    missing_df, missing_summary = handle_missing_values(raw_df, raw_file.name)
    outlier_df, outlier_summary = handle_outliers(missing_df, raw_file.name)
    transformed_df, transformation_summary = transform_stock_data(outlier_df, raw_file.name)

    return transformed_df, {
        "missing_values": missing_summary,
        "outlier": outlier_summary,
        "data_transformation": transformation_summary,
    }


def save_csv_summary(summary_rows: List[Dict[str, Any]]) -> None:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)


def run_preprocess_pipeline() -> None:
    ensure_data_structure()

    raw_files = get_raw_csv_files()
    if not raw_files:
        print(f"No raw CSV files found in: {RAW_DATA_DIR}")
        return

    missing_summary_rows: List[Dict[str, Any]] = []
    outlier_summary_rows: List[Dict[str, Any]] = []
    transformation_summary_rows: List[Dict[str, Any]] = []
    preprocessed_summary_rows: List[Dict[str, Any]] = []
    standardization_summary_rows: List[Dict[str, Any]] = []
    csv_summary_rows: List[Dict[str, Any]] = []

    print("[Starting preprocessing pipeline]\n")

    for raw_file in raw_files:
        try:
            print(f"Processing: {raw_file.name}")

            cleaned_df, summaries = preprocess_single_file(raw_file)
            output_path = save_preprocessed_csv(cleaned_df, raw_file)
            standardized_sample_path, standardization_summaries = (
                create_standardized_sample_dataset(cleaned_df, raw_file)
            )

            missing_summary = summaries["missing_values"]
            outlier_summary = summaries["outlier"]
            transformation_summary = summaries["data_transformation"]
            preprocessed_summary = build_preprocessed_dataset_summary_row(
                cleaned_df,
                raw_file,
                output_path,
                int(missing_summary["original_rows"]),
            )

            missing_summary_rows.append(missing_summary)
            outlier_summary_rows.append(outlier_summary)
            transformation_summary_rows.append(transformation_summary)
            preprocessed_summary_rows.append(preprocessed_summary)
            standardization_summary_rows.extend(standardization_summaries)

            csv_summary_rows.append({
                "file_name": raw_file.name,
                "original_rows": missing_summary["original_rows"],
                "rows_with_missing_essential_values_dropped": missing_summary[
                    "rows_with_missing_essential_values_dropped"
                ],
                "rows_with_numeric_conversion_missing_dropped": outlier_summary[
                    "rows_with_numeric_conversion_missing_dropped"
                ],
                "logically_invalid_rows_dropped": outlier_summary[
                    "logically_invalid_rows_dropped"
                ],
                "statistical_outlier_rows_flagged_not_removed": outlier_summary[
                    "statistical_outlier_rows_flagged_not_removed"
                ],
                "invalid_date_rows_dropped": transformation_summary[
                    "invalid_date_rows_dropped"
                ],
                "duplicate_date_rows_removed": transformation_summary[
                    "duplicate_date_rows_removed"
                ],
                "final_rows": len(cleaned_df),
            })

            print(f"Saved: {output_path}")
            print(f"Saved sample-only standardized dataset: {standardized_sample_path}")
            print(f"Original rows: {missing_summary['original_rows']}")
            print(f"Final rows: {len(cleaned_df)}")
            print(
                "Rows dropped due to missing essential values: "
                f"{missing_summary['rows_with_missing_essential_values_dropped']}"
            )
            print(
                "Rows dropped due to logical invalidity: "
                f"{outlier_summary['logically_invalid_rows_dropped']}"
            )
            print(
                "Statistical outlier rows flagged and kept: "
                f"{outlier_summary['statistical_outlier_rows_flagged_not_removed']}"
            )
            print()

        except Exception as exc:
            print(f"Failed to preprocess {raw_file.name}: {exc}\n")

    save_csv_summary(csv_summary_rows)
    write_missing_values_summary(missing_summary_rows)
    write_outlier_summary(outlier_summary_rows)
    write_data_transformation_summary(transformation_summary_rows)
    write_preprocessed_dataset_summary(preprocessed_summary_rows)
    write_standardization_normalization_summary(standardization_summary_rows)

    print(f"Preprocessing CSV summary saved to: {SUMMARY_FILE}")
    print(f"Markdown summaries saved to: {DATA_PREPROCESSING_SUMMARY_DIR}")
    print("\n[Preprocessing pipeline finished]")


def main() -> None:
    run_preprocess_pipeline()


if __name__ == "__main__":
    main()
