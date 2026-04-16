from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ============================================================
# Project configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed-stock-dataset"
MODEL_INPUT_DATA_DIR = DATA_DIR / "model-input-stock-dataset"
MODEL_INPUT_SUMMARY_FILE = DATA_DIR / "model-input-summary.csv"

SUMMARY_DIR = PROJECT_ROOT / "summary"
DATA_UNDERSTANDING_SUMMARY_DIR = SUMMARY_DIR / "data-understanding"

COLUMN_MEANINGS = {
    "Date": "Trading date for the daily stock record.",
    "Ticker": "Stock ticker symbol from Yahoo Finance.",
    "Open": "Opening price for the trading day.",
    "High": "Highest traded price for the trading day.",
    "Low": "Lowest traded price for the trading day.",
    "Close": "Closing price for the trading day.",
    "Adj Close": "Adjusted closing price after dividends and splits.",
    "Volume": "Number of shares traded for the day.",
    "Dividends": "Dividend amount recorded for the trading day.",
    "Stock Splits": "Stock split ratio recorded for the trading day.",
}

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


# ============================================================
# Helpers
# ============================================================
def ensure_summary_structure() -> None:
    DATA_UNDERSTANDING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def get_preprocessed_csv_files() -> List[Path]:
    return sorted(PREPROCESSED_DATA_DIR.glob("*.csv"))


def get_model_input_csv_files() -> List[Path]:
    return sorted(MODEL_INPUT_DATA_DIR.glob("*.csv"))


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


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_stock_files(files: List[Path]) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError(f"No preprocessed CSV files found in: {PREPROCESSED_DATA_DIR}")

    frames: List[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df["Source_File"] = file_path.name
        frames.append(df)

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    return combined_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ============================================================
# 3.4.2.1 Variables
# ============================================================
def write_variables_summary(df: pd.DataFrame) -> None:
    rows: List[List[Any]] = []
    for col in df.columns:
        if col == "Source_File":
            continue

        series = df[col]
        min_value: Any = "N/A"
        max_value: Any = "N/A"
        if pd.api.types.is_numeric_dtype(series):
            min_value = series.min()
            max_value = series.max()
        elif pd.api.types.is_datetime64_any_dtype(series):
            min_value = series.min().date() if not series.dropna().empty else "N/A"
            max_value = series.max().date() if not series.dropna().empty else "N/A"

        rows.append([
            col,
            str(series.dtype),
            COLUMN_MEANINGS.get(col, "Generated or supporting dataset field."),
            int(series.isna().sum()),
            int(series.nunique(dropna=True)),
            min_value,
            max_value,
        ])

    content = f"""# 3.4.2.1 Variables Summary

## Purpose
This section documents the variables available in the preprocessed stock dataset and describes their basic data characteristics.

## Variable Dictionary
{markdown_table(
        [
            "Variable",
            "Data Type",
            "Meaning",
            "Missing Count",
            "Unique Count",
            "Minimum",
            "Maximum",
        ],
        rows,
    )}
"""
    write_markdown(DATA_UNDERSTANDING_SUMMARY_DIR / "variables-summary.md", content)


# ============================================================
# 3.4.2.2 Observations
# ============================================================
def write_observations_summary(df: pd.DataFrame) -> None:
    rows: List[List[Any]] = []
    for ticker, ticker_df in df.groupby("Ticker", dropna=False):
        valid_dates = ticker_df["Date"].dropna()
        duplicate_dates = int(ticker_df.duplicated(subset=["Ticker", "Date"]).sum())

        if valid_dates.empty:
            first_date: Any = "N/A"
            last_date: Any = "N/A"
            business_days = 0
            coverage_rate: Any = "N/A"
        else:
            first_date = valid_dates.min().date()
            last_date = valid_dates.max().date()
            business_days = len(pd.bdate_range(valid_dates.min(), valid_dates.max()))
            coverage_rate = (
                len(valid_dates) / business_days
                if business_days > 0
                else "N/A"
            )

        rows.append([
            ticker,
            int(len(ticker_df)),
            first_date,
            last_date,
            duplicate_dates,
            business_days,
            coverage_rate,
        ])

    content = f"""# 3.4.2.2 Observations Summary

## Purpose
This section documents the observation coverage of the preprocessed dataset by stock ticker.

## Notes
Trading-day coverage is compared against weekday business days only. Public holidays and market closures are not separately removed from that denominator, so this value is an approximate coverage indicator.

## Observation Coverage
{markdown_table(
        [
            "Ticker",
            "Rows",
            "First Date",
            "Last Date",
            "Duplicate Dates",
            "Weekday Business Days",
            "Approx Coverage Rate",
        ],
        rows,
    )}
"""
    write_markdown(DATA_UNDERSTANDING_SUMMARY_DIR / "observations-summary.md", content)


# ============================================================
# 3.4.2.3 Exploratory Data Analysis
# ============================================================
def build_descriptive_rows(df: pd.DataFrame) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        rows.append([
            col,
            int(series.count()),
            series.mean(),
            series.std(),
            series.min(),
            series.quantile(0.25),
            series.median(),
            series.quantile(0.75),
            series.max(),
        ])
    return rows


def build_return_rows(df: pd.DataFrame) -> List[List[Any]]:
    working_df = df.sort_values(["Ticker", "Date"]).copy()
    working_df["Daily_Return"] = working_df.groupby("Ticker")["Adj Close"].pct_change()

    rows: List[List[Any]] = []
    for ticker, ticker_df in working_df.groupby("Ticker", dropna=False):
        returns = ticker_df["Daily_Return"].dropna()
        if returns.empty:
            rows.append([ticker, 0, "N/A", "N/A", "N/A", "N/A"])
            continue
        rows.append([
            ticker,
            int(returns.count()),
            returns.mean(),
            returns.std(),
            returns.min(),
            returns.max(),
        ])
    return rows


def build_correlation_rows(df: pd.DataFrame) -> List[List[Any]]:
    available_columns = [col for col in NUMERIC_COLUMNS if col in df.columns]
    corr_df = df[available_columns].corr(numeric_only=True)
    rows: List[List[Any]] = []

    for row_index, row_name in enumerate(available_columns):
        for col_name in available_columns[row_index + 1:]:
            correlation = corr_df.loc[row_name, col_name]
            if pd.isna(correlation):
                continue
            rows.append([row_name, col_name, float(correlation)])

    rows = sorted(rows, key=lambda item: abs(float(item[2])), reverse=True)
    return rows[:10]


def write_eda_summary(df: pd.DataFrame) -> None:
    descriptive_rows = build_descriptive_rows(df)
    return_rows = build_return_rows(df)
    correlation_rows = build_correlation_rows(df)

    content = f"""# 3.4.2.3 Exploratory Data Analysis (EDA) Summary

## Purpose
This section summarizes the numerical behavior of the preprocessed stock dataset.

## Descriptive Statistics
{markdown_table(
        [
            "Column",
            "Count",
            "Mean",
            "Std",
            "Min",
            "Q1",
            "Median",
            "Q3",
            "Max",
        ],
        descriptive_rows,
    )}

## Daily Return Summary By Ticker
{markdown_table(
        [
            "Ticker",
            "Return Count",
            "Mean Return",
            "Return Std",
            "Minimum Return",
            "Maximum Return",
        ],
        return_rows,
    )}

## Highest Absolute Correlations
{markdown_table(
        [
            "Column A",
            "Column B",
            "Correlation",
        ],
        correlation_rows,
    )}
"""
    write_markdown(DATA_UNDERSTANDING_SUMMARY_DIR / "eda-summary.md", content)


# ============================================================
# 3.4.2.4 Dataset Insights
# ============================================================
def write_dataset_insights_summary(df: pd.DataFrame) -> None:
    row_count = len(df)
    ticker_count = int(df["Ticker"].nunique(dropna=True)) if "Ticker" in df.columns else 0
    first_date = df["Date"].min().date() if not df["Date"].dropna().empty else "N/A"
    last_date = df["Date"].max().date() if not df["Date"].dropna().empty else "N/A"
    missing_total = int(df.drop(columns=["Source_File"], errors="ignore").isna().sum().sum())

    if MODEL_INPUT_SUMMARY_FILE.exists():
        model_input_summary_df = pd.read_csv(MODEL_INPUT_SUMMARY_FILE)
        model_input_rows = int(model_input_summary_df["final_rows"].sum())
        model_input_files = int(model_input_summary_df["file_name"].nunique())
        model_input_note = (
            f"The model input-ready dataset currently contains {model_input_rows:,} "
            f"rows across {model_input_files} files according to "
            "`data/model-input-summary.csv`."
        )
    elif get_model_input_csv_files():
        model_input_files = get_model_input_csv_files()
        model_input_note = (
            f"The model input-ready dataset exists across {len(model_input_files)} "
            "files, but `data/model-input-summary.csv` was not found."
        )
    else:
        model_input_note = (
            "The model input-ready dataset has not been generated yet. Run "
            "`python main.py build_model_input` after preprocessing to create it."
        )

    volume = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else pd.Series(dtype=float)
    close = pd.to_numeric(df["Close"], errors="coerce") if "Close" in df.columns else pd.Series(dtype=float)

    volume_note = "Volume data is unavailable."
    if not volume.dropna().empty:
        volume_note = (
            f"Trading volume ranges from {volume.min():,.0f} to {volume.max():,.0f}, "
            "which means volume-related features can be much larger in scale than price features."
        )

    close_note = "Closing price data is unavailable."
    if not close.dropna().empty:
        close_note = (
            f"Closing prices range from {close.min():,.4f} to {close.max():,.4f} "
            "in the preprocessed dataset."
        )

    content = f"""# 3.4.2.4 Dataset Insights Summary

## Purpose
This section interprets the preprocessed dataset quality and explains the modeling implications of the observed data.

## Dataset Quality
- The preprocessed dataset contains {row_count:,} rows across {ticker_count} tickers.
- The date range is {first_date} to {last_date}.
- Total missing values remaining in the preprocessed dataset: {missing_total:,}.
- The preprocessing stage keeps statistical market outliers because unusual price or volume movements may contain useful prediction signals.

## Market Data Insights
- {close_note}
- {volume_note}
- Because price and volume features have very different magnitudes, scaling is useful for scale-sensitive models such as Logistic Regression.
- Tree-based models such as Random Forest and XGBoost can usually work without standardization because their splits are threshold-based.

## Modeling Implications
- `Adj Close` is important because it accounts for corporate actions and is used for return, lag, moving-average, volatility, and target construction.
- Time order should be preserved during modeling to avoid future-data leakage.
- Scaling should be fitted only on the training or development period, then applied to later validation or test periods.
- {model_input_note}
"""
    write_markdown(DATA_UNDERSTANDING_SUMMARY_DIR / "dataset-insights-summary.md", content)


# ============================================================
# Main pipeline
# ============================================================
def run_data_understanding_pipeline() -> None:
    ensure_summary_structure()

    preprocessed_files = get_preprocessed_csv_files()
    if not preprocessed_files:
        print(f"No preprocessed CSV files found in: {PREPROCESSED_DATA_DIR}")
        return

    print("[Starting data understanding summary pipeline]\n")

    df = load_stock_files(preprocessed_files)

    write_variables_summary(df)
    write_observations_summary(df)
    write_eda_summary(df)
    write_dataset_insights_summary(df)

    print(f"Data understanding summaries saved to: {DATA_UNDERSTANDING_SUMMARY_DIR}")
    print("\n[Data understanding summary pipeline finished]")


def main() -> None:
    run_data_understanding_pipeline()


if __name__ == "__main__":
    main()
