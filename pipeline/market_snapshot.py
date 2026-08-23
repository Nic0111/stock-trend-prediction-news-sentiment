from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from pipeline.download_financial_news_data import load_stock_metadata
from pipeline.market_calendar import completed_sessions_between, latest_completed_session
from pipeline.stock_features import build_stock_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "data" / "stock-metadata.csv"
KLCI_TICKER = "^KLSE"


@dataclass
class MarketHistoryPoint:
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adjusted_close: float | None
    volume: int | None
    dividend: float | None
    split: float | None
    ma20: float | None
    ma60: float | None
    rsi14: float | None
    macd: float | None
    macd_signal: float | None


@dataclass
class BenchmarkPoint:
    date: str
    close: float | None
    adjusted_close: float | None


@dataclass
class MarketPerformance:
    one_day: float | None
    five_day: float | None
    one_month: float | None
    three_month: float | None
    ytd: float | None
    one_year: float | None


@dataclass
class MarketRange:
    low: float | None
    high: float | None
    position_pct: float | None


@dataclass
class MarketActivity:
    average_volume_20d: float | None
    volume_ratio: float | None


@dataclass
class MarketRisk:
    annualized_volatility_20d: float | None
    atr14_pct: float | None
    daily_range_pct: float | None


@dataclass
class MarketContext:
    history: List[BenchmarkPoint] = field(default_factory=list)
    excess_return_one_month: float | None = None
    excess_return_three_month: float | None = None
    beta20: float | None = None
    beta60: float | None = None
    correlation20: float | None = None
    correlation60: float | None = None
    klci_return_one_month: float | None = None
    klci_trend: str = "Unavailable"
    klci_volatility_20d: float | None = None


@dataclass
class MarketSnapshot:
    ticker: str
    company_name: str
    sector: str | None
    price: float
    previous_close: float | None
    change: float | None
    change_pct: float | None
    open: float | None
    adjusted_close: float | None
    day_high: float | None
    day_low: float | None
    volume: int | None
    as_of_date: str
    data_source: str
    stale_data_warning: bool
    freshness_status: str
    freshness_reason: str
    observation_lag_sessions: int | None
    history_dates: List[str]
    history_prices: List[float]
    history: List[MarketHistoryPoint]
    performance: MarketPerformance
    range_52_week: MarketRange
    activity: MarketActivity
    risk: MarketRisk
    market_context: MarketContext

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
    value_float = float(value)
    return value_float if np.isfinite(value_float) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _sector_metadata() -> Dict[str, str]:
    if not METADATA_FILE.exists():
        return {}
    metadata = pd.read_csv(METADATA_FILE)
    return {
        str(row["Ticker"]).strip().upper(): str(row.get("Sector", "")).strip() or None
        for _, row in metadata.iterrows()
        if str(row.get("Ticker", "")).strip()
    }


def _clean_provider_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = _flatten_columns(data).reset_index()
    date_column = "Date" if "Date" in frame.columns else str(frame.columns[0])
    if date_column not in frame.columns or not {"Close", "Adj Close"}.intersection(frame.columns):
        raise ValueError(f"Market data for {ticker} is missing price columns")

    frame = frame.rename(columns={date_column: "Date"})
    if "Close" not in frame.columns:
        frame["Close"] = frame["Adj Close"]
    if "Adj Close" not in frame.columns:
        frame["Adj Close"] = frame["Close"]
    for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Some provider/test responses contain only a close plus a daily range.
    # A missing OHLC component must not erase an otherwise valid close.
    frame["Open"] = frame["Open"].fillna(frame["Close"])
    frame["High"] = frame["High"].fillna(frame["Close"])
    frame["Low"] = frame["Low"].fillna(frame["Close"])
    frame["Volume"] = frame["Volume"].fillna(0)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    frame["Ticker"] = ticker
    frame = frame.dropna(subset=["Date", "Close", "Adj Close"]).sort_values("Date")
    return frame.drop_duplicates("Date", keep="last").reset_index(drop=True)


def _fetch_daily(ticker: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period="18mo",
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if data is None or data.empty:
        raise ValueError(f"No market data returned for {ticker}")
    return _clean_provider_frame(data, ticker)


def fetch_daily_market_data(ticker: str) -> pd.DataFrame:
    """Fetch one reusable daily frame for snapshots and model inference."""

    return _fetch_daily(ticker.strip().upper())


def _safe_return(series: pd.Series, periods: int) -> float | None:
    if len(series) <= periods:
        return None
    latest = _number(series.iloc[-1])
    previous = _number(series.iloc[-periods - 1])
    if latest is None or previous in (None, 0):
        return None
    return latest / previous - 1


def _year_to_date(series: pd.Series, dates: pd.Series) -> float | None:
    if series.empty:
        return None
    latest = _number(series.iloc[-1])
    current_year = pd.Timestamp(dates.iloc[-1]).year
    start_indexes = dates[dates.dt.year == current_year].index
    if latest is None or start_indexes.empty:
        return None
    start = _number(series.loc[start_indexes[0]])
    return None if start in (None, 0) else latest / start - 1


def _feature_value(features: pd.DataFrame, column: str) -> float | None:
    if column not in features.columns or features.empty:
        return None
    return _number(features[column].iloc[-1])


def _point(row: pd.Series, features_row: pd.Series | None) -> MarketHistoryPoint:
    value = lambda column: _number(row.get(column))
    feature = lambda column: _number(features_row.get(column)) if features_row is not None else None
    return MarketHistoryPoint(
        date=pd.Timestamp(row["Date"]).date().isoformat(),
        open=value("Open"), high=value("High"), low=value("Low"), close=value("Close"),
        adjusted_close=value("Adj Close"), volume=_integer(row.get("Volume")),
        dividend=value("Dividends"), split=value("Stock Splits"),
        ma20=feature("MA_20"), ma60=feature("MA_60"), rsi14=feature("RSI_14"),
        macd=feature("MACD"), macd_signal=feature("MACD_Signal"),
    )


def _benchmark_points(frame: pd.DataFrame) -> List[BenchmarkPoint]:
    return [
        BenchmarkPoint(
            date=pd.Timestamp(row["Date"]).date().isoformat(),
            close=_number(row.get("Close")),
            adjusted_close=_number(row.get("Adj Close")),
        )
        for _, row in frame.tail(252).iterrows()
    ]


def _market_context(stock_features: pd.DataFrame, market: pd.DataFrame) -> MarketContext:
    stock_returns = stock_features["Return_1D"]
    market_returns = stock_features["Market_Return_1D"]
    beta20 = _feature_value(stock_features, "Rolling_Beta_20D")
    beta60 = _feature_value(stock_features, "Rolling_Beta_60D")
    corr20 = _feature_value(stock_features, "Rolling_Correlation_20D")
    corr60 = _feature_value(stock_features, "Rolling_Correlation_60D")
    market_features = market.copy()
    market_features["return"] = market_features["Adj Close"].pct_change(fill_method=None)
    market_features["ma20"] = market_features["Adj Close"].rolling(20, min_periods=20).mean()
    market_features["vol20"] = market_features["return"].rolling(20, min_periods=20).std()
    latest_market = market_features.iloc[-1]
    trend = "Above 20-day average" if latest_market["Adj Close"] > latest_market["ma20"] else "Below 20-day average"
    aligned_market_return = _safe_return(market_features["Adj Close"], 20)
    stock_return_one_month = _safe_return(stock_features["Adj Close"], 20)
    stock_return_three_month = _safe_return(stock_features["Adj Close"], 60)
    market_return_three_month = _safe_return(market_features["Adj Close"], 60)
    return MarketContext(
        history=_benchmark_points(market),
        excess_return_one_month=(
            stock_return_one_month - aligned_market_return
            if stock_return_one_month is not None and aligned_market_return is not None
            else None
        ),
        excess_return_three_month=(
            stock_return_three_month - market_return_three_month
            if stock_return_three_month is not None and market_return_three_month is not None
            else None
        ),
        beta20=beta20,
        beta60=beta60,
        correlation20=corr20,
        correlation60=corr60,
        klci_return_one_month=aligned_market_return,
        klci_trend=trend if _number(latest_market.get("ma20")) is not None else "Unavailable",
        klci_volatility_20d=_number(latest_market.get("vol20")),
    )


def fetch_market_snapshot(
    ticker: str,
    *,
    stock_metadata: Dict[str, str] | None = None,
    stock_df: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
) -> MarketSnapshot:
    normalized_ticker = ticker.strip().upper()
    metadata = stock_metadata or load_stock_metadata()
    if normalized_ticker not in metadata:
        raise ValueError(f"Unsupported ticker: {normalized_ticker}")

    stock = stock_df.copy() if stock_df is not None else _fetch_daily(normalized_ticker)
    market = market_df.copy() if market_df is not None else _fetch_daily(KLCI_TICKER)
    paired_market = market[["Date", "Adj Close"]].copy()
    features = build_stock_features(stock, paired_market)
    features = features.sort_values("Date").reset_index(drop=True)
    if features.empty:
        raise ValueError(f"No aligned market rows returned for {normalized_ticker}")

    latest = stock.iloc[-1]
    previous_close = _number(stock["Close"].iloc[-2]) if len(stock) > 1 else None
    price = _number(latest["Close"])
    if price is None:
        raise ValueError(f"Latest market price is unavailable for {normalized_ticker}")
    change = price - previous_close if previous_close is not None else None
    change_pct = change / previous_close if change is not None and previous_close else None
    latest_date = pd.Timestamp(latest["Date"]).date()
    expected_session = latest_completed_session()
    lag_sessions = completed_sessions_between(latest_date, expected_session)
    freshness_status = "current" if lag_sessions == 0 else "delayed" if lag_sessions <= 2 else "stale"
    freshness_reason = (
        "Stock data matches the latest completed Bursa Malaysia session."
        if freshness_status == "current"
        else f"Stock is {lag_sessions} completed Bursa session(s) behind the expected market session."
    )

    adjusted = stock["Adj Close"]
    return_history = MarketPerformance(
        one_day=_safe_return(adjusted, 1), five_day=_safe_return(adjusted, 5),
        one_month=_safe_return(adjusted, 20), three_month=_safe_return(adjusted, 60),
        ytd=_year_to_date(adjusted, pd.to_datetime(stock["Date"])), one_year=_safe_return(adjusted, 252),
    )
    trailing = stock.tail(252)
    range_low = _number(trailing["Low"].min())
    range_high = _number(trailing["High"].max())
    range_position = None if range_low is None or range_high in (None, range_low) else (price - range_low) / (range_high - range_low)
    previous_volumes = pd.to_numeric(stock["Volume"].tail(21).iloc[:-1], errors="coerce")
    average_volume = _number(previous_volumes.mean()) if not previous_volumes.empty else None
    volume = _integer(latest.get("Volume"))
    volume_ratio = None if average_volume in (None, 0) or volume is None else volume / average_volume
    feature_latest = features.iloc[-1]
    risk = MarketRisk(
        annualized_volatility_20d=(
            _feature_value(features, "Volatility_20D") * sqrt(252)
            if _feature_value(features, "Volatility_20D") is not None else None
        ),
        atr14_pct=_feature_value(features, "ATR_14_pct"),
        daily_range_pct=(
            (float(latest["High"]) - float(latest["Low"])) / price
            if _number(latest.get("High")) is not None and _number(latest.get("Low")) is not None else None
        ),
    )
    feature_rows = features.set_index("Date")
    history = [
        _point(row, feature_rows.loc[row["Date"]] if row["Date"] in feature_rows.index else None)
        for _, row in stock.tail(252).iterrows()
    ]
    history_dates = [point.date for point in history[-30:]]
    history_prices = [point.close for point in history[-30:] if point.close is not None]
    sector = _sector_metadata().get(normalized_ticker)
    return MarketSnapshot(
        ticker=normalized_ticker,
        company_name=metadata[normalized_ticker], sector=sector,
        price=price, previous_close=previous_close, change=change, change_pct=change_pct,
        open=_number(latest.get("Open")), adjusted_close=_number(latest.get("Adj Close")),
        day_high=_number(latest.get("High")), day_low=_number(latest.get("Low")), volume=volume,
        as_of_date=latest_date.isoformat(), data_source="Yahoo Finance daily close",
        stale_data_warning=freshness_status == "stale", freshness_status=freshness_status,
        freshness_reason=freshness_reason, observation_lag_sessions=lag_sessions,
        history_dates=history_dates, history_prices=history_prices, history=history,
        performance=return_history,
        range_52_week=MarketRange(low=range_low, high=range_high, position_pct=range_position),
        activity=MarketActivity(average_volume_20d=average_volume, volume_ratio=volume_ratio),
        risk=risk, market_context=_market_context(features, market),
    )
