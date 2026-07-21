from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

from pipeline.download_financial_news_data import load_stock_metadata


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "data" / "stock-metadata.csv"


@dataclass
class MarketSnapshot:
    ticker: str
    company_name: str
    price: float
    previous_close: float | None
    change: float | None
    change_pct: float | None
    day_high: float | None
    day_low: float | None
    volume: int | None
    as_of_date: str
    data_source: str
    stale_data_warning: bool
    history_dates: List[str]
    history_prices: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(column[0]) for column in frame.columns]
    return frame


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def fetch_market_snapshot(
    ticker: str,
    *,
    stock_metadata: Dict[str, str] | None = None,
) -> MarketSnapshot:
    normalized_ticker = ticker.strip().upper()
    metadata = stock_metadata or load_stock_metadata()
    if normalized_ticker not in metadata:
        raise ValueError(f"Unsupported ticker: {normalized_ticker}")

    data = yf.download(
        normalized_ticker,
        period="3mo",
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if data is None or data.empty:
        raise ValueError(f"No market data returned for {normalized_ticker}")

    frame = _flatten_columns(data).reset_index()
    date_column = "Date" if "Date" in frame.columns else str(frame.columns[0])
    close_column = "Close" if "Close" in frame.columns else "Adj Close"
    required_columns = {close_column, date_column}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Market data for {normalized_ticker} is missing price columns")

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[close_column] = pd.to_numeric(frame[close_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, close_column]).sort_values(date_column)
    if frame.empty:
        raise ValueError(f"No usable market rows returned for {normalized_ticker}")

    latest = frame.iloc[-1]
    previous_close = _number(frame[close_column].iloc[-2]) if len(frame) > 1 else None
    price = float(latest[close_column])
    change = price - previous_close if previous_close is not None else None
    change_pct = change / previous_close if change is not None and previous_close else None
    latest_date = pd.Timestamp(latest[date_column]).date()

    history = frame.tail(30)
    history_dates = [pd.Timestamp(value).date().isoformat() for value in history[date_column]]
    history_prices = [float(value) for value in history[close_column]]
    age_days = max(0, (date.today() - latest_date).days)

    return MarketSnapshot(
        ticker=normalized_ticker,
        company_name=metadata[normalized_ticker],
        price=price,
        previous_close=previous_close,
        change=change,
        change_pct=change_pct,
        day_high=_number(latest.get("High")),
        day_low=_number(latest.get("Low")),
        volume=_integer(latest.get("Volume")),
        as_of_date=latest_date.isoformat(),
        data_source="Yahoo Finance daily close",
        stale_data_warning=age_days > 5,
        history_dates=history_dates,
        history_prices=history_prices,
    )

