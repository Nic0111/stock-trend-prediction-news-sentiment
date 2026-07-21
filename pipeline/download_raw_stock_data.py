from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Dict, List

import pandas as pd
import yfinance as yf


# ============================================================
# Project configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw-stock-dataset"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed-stock-dataset"
MODEL_INPUT_DATA_DIR = DATA_DIR / "model-input-stock-dataset"
RAW_MARKET_CONTEXT_DIR = DATA_DIR / "raw-market-context-dataset"
MARKET_CONTEXT_FILE = RAW_MARKET_CONTEXT_DIR / "ftse-bursa-malaysia-klci-raw.csv"

METADATA_FILE = DATA_DIR / "stock-metadata.csv"

# Locked project date range.  The longer history deliberately covers multiple
# market regimes (pre-pandemic, COVID shock, recovery, and recent conditions)
# while the model trainer keeps 2025 as the fixed benchmark holdout.
START_DATE = "2015-01-01"
END_DATE_EXCLUSIVE = "2026-01-01"
MARKET_CONTEXT_TICKER = "^KLSE"

# Top 10 Malaysian stocks chosen earlier for the project
STOCKS: Dict[str, str] = {
    "1155.KL": "Malayan Banking Berhad",
    "1295.KL": "Public Bank Berhad",
    "1023.KL": "CIMB Group Holdings Berhad",
    "5347.KL": "Tenaga Nasional Berhad",
    "5225.KL": "IHH Healthcare Berhad",
    "6947.KL": "CelcomDigi Berhad",
    "5183.KL": "PETRONAS Chemicals Group Berhad",
    "8869.KL": "Press Metal Aluminium Holdings Berhad",
    "3816.KL": "MISC Berhad",
    "4677.KL": "YTL Corporation Berhad",
}


# ============================================================
# Helper functions
# ============================================================
def make_safe_filename(company_name: str) -> str:
    """
    Convert company name into a filesystem-safe filename in the format:
    company_name-raw.csv

    Example:
    'Malayan Banking Berhad' -> 'malayan-banking-berhad-raw.csv'
    """
    safe = company_name.strip().lower()
    safe = re.sub(r"[^\w\s-]", "", safe)
    safe = re.sub(r"[\s-]+", "-", safe)
    return f"{safe}-raw.csv"


def ensure_data_structure() -> None:
    """
    Ensure the following structure exists:

    project-folder/
    └─ data/
       ├─ raw-stock-dataset/
       ├─ preprocessed-stock-dataset/
       ├─ model-input-stock-dataset/
       └─ stock-metadata.csv

    If the folders do not exist, create them.
    If stock-metadata.csv does not exist, create it with headers.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_MARKET_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_FILE.exists():
        empty_metadata = pd.DataFrame(columns=["Ticker", "Company_Name", "Sector"])
        empty_metadata.to_csv(METADATA_FILE, index=False)


def download_stock_data(ticker: str) -> pd.DataFrame:
    """
    Download raw daily stock data from yfinance for the locked 2015-2025 range.
    """
    df = yf.download(
        tickers=ticker,
        start=START_DATE,
        end=END_DATE_EXCLUSIVE,
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    df = df.reset_index()

    if "Date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Date"})

    df["Ticker"] = ticker

    desired_order = [
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
    existing_cols = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_cols]
    df = df[existing_cols + remaining_cols]

    return df


def fetch_sector(ticker: str) -> str:
    """
    Fetch sector information from yfinance metadata.
    Falls back to 'Unknown' if unavailable.
    """
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        return sector if sector else "Unknown"
    except Exception:
        return "Unknown"


def save_raw_csv(df: pd.DataFrame, company_name: str) -> Path:
    """
    Save raw stock data into data/raw-stock-dataset/<company_name>-raw.csv
    """
    filename = make_safe_filename(company_name)
    output_path = RAW_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def create_stock_metadata(metadata_rows: List[dict]) -> None:
    """
    Overwrite stock-metadata.csv with the current metadata rows.
    """
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df = metadata_df[["Ticker", "Company_Name", "Sector"]]
    metadata_df.to_csv(METADATA_FILE, index=False)


def download_market_context() -> Path:
    """Download the fixed-period FTSE Bursa Malaysia KLCI context series."""
    market_df = download_stock_data(MARKET_CONTEXT_TICKER)
    market_df.to_csv(MARKET_CONTEXT_FILE, index=False)
    return MARKET_CONTEXT_FILE


# ============================================================
# Main execution
# ============================================================
def run_download_pipeline() -> None:
    ensure_data_structure()

    metadata_rows: List[dict] = []

    print("[Starting raw stock price download]\n")

    for ticker, company_name in STOCKS.items():
        try:
            print(f"Downloading {ticker} - {company_name}")
            df = download_stock_data(ticker)
            output_path = save_raw_csv(df, company_name)

            sector = fetch_sector(ticker)

            metadata_rows.append(
                {
                    "Ticker": ticker,
                    "Company_Name": company_name,
                    "Sector": sector,
                }
            )

            print(f"Saved: {output_path}")
            print(f"Rows downloaded: {len(df)}")
            print(f"Sector: {sector}\n")

        except Exception as exc:
            print(f"Failed for {ticker} - {company_name}: {exc}\n", file=sys.stderr)

    create_stock_metadata(metadata_rows)
    print(f"Saved metadata file: {METADATA_FILE}")

    try:
        market_path = download_market_context()
        print(f"Saved market context: {market_path}")
    except Exception as exc:
        print(
            f"Failed to download {MARKET_CONTEXT_TICKER} market context: {exc}",
            file=sys.stderr,
        )

    print("\n[Done]")


def main() -> None:
    run_download_pipeline()


if __name__ == "__main__":
    main()
