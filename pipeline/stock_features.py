from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TARGET_HORIZONS = (1, 5, 10)

STOCK_REQUIRED_COLUMNS = (
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)

BASE_FEATURE_COLUMNS = (
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
)

STATIONARY_FEATURE_COLUMNS = (
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Lag_Return_1D_1",
    "Lag_Return_1D_2",
    "Lag_Return_1D_3",
    "Lag_Return_1D_5",
    "AdjClose_to_MA_5",
    "AdjClose_to_MA_20",
    "AdjClose_to_MA_60",
    "MA_5_to_MA_20",
    "MA_20_to_MA_60",
    "Volatility_5D",
    "Volatility_20D",
    "High_Low_Spread",
    "Open_Close_Change",
    "Volume_Change_1D",
    "Log_Volume",
    "Volume_to_MA_5",
    "Volume_to_MA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "ATR_14_pct",
)

MARKET_FEATURE_COLUMNS = (
    "Market_Return_1D",
    "Market_Return_5D",
    "Market_Return_10D",
    "Market_AdjClose_to_MA_20",
    "Market_Volatility_20D",
    "Excess_Return_1D",
    "Excess_Return_5D",
)

ALL_NUMERIC_FEATURE_COLUMNS = tuple(
    dict.fromkeys((*BASE_FEATURE_COLUMNS, *STATIONARY_FEATURE_COLUMNS, *MARKET_FEATURE_COLUMNS))
)

TARGET_COLUMNS = tuple(f"Target_{horizon}D" for horizon in TARGET_HORIZONS)
TARGET_DATE_COLUMNS = tuple(f"Target_Date_{horizon}D" for horizon in TARGET_HORIZONS)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _validate_stock_columns(df: pd.DataFrame) -> None:
    missing = [column for column in STOCK_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required stock columns: {missing}")


def _prepare_stock_frame(stock_df: pd.DataFrame) -> pd.DataFrame:
    _validate_stock_columns(stock_df)
    df = stock_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype("string").str.strip().str.upper()

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", "Ticker", *numeric_columns])
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(
        subset=["Ticker", "Date"], keep="last"
    )
    return df.reset_index(drop=True)


def _build_single_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.sort_values("Date").copy()
    adjusted_close = frame["Adj Close"]
    close = frame["Close"]
    volume = frame["Volume"]
    daily_return = adjusted_close.pct_change(1, fill_method=None)

    frame["Return_1D"] = daily_return
    frame["Return_5D"] = adjusted_close.pct_change(5, fill_method=None)
    frame["Return_10D"] = adjusted_close.pct_change(10, fill_method=None)

    for lag in (1, 2, 3, 5):
        frame[f"Lag_{lag}"] = adjusted_close.shift(lag)
        frame[f"Lag_Return_1D_{lag}"] = daily_return.shift(lag)

    frame["MA_5"] = adjusted_close.rolling(5, min_periods=5).mean()
    frame["MA_20"] = adjusted_close.rolling(20, min_periods=20).mean()
    frame["MA_60"] = adjusted_close.rolling(60, min_periods=60).mean()
    frame["Volatility_5D"] = daily_return.rolling(5, min_periods=5).std()
    frame["Volatility_20D"] = daily_return.rolling(20, min_periods=20).std()

    frame["High_Low_Spread"] = _safe_divide(frame["High"] - frame["Low"], close)
    frame["Open_Close_Change"] = _safe_divide(close - frame["Open"], frame["Open"])

    volume_safe = volume.replace(0, np.nan)
    frame["Volume_Change_1D"] = volume_safe.pct_change(1, fill_method=None)
    frame["Volume_MA_5"] = volume.rolling(5, min_periods=5).mean()
    volume_ma_20 = volume.rolling(20, min_periods=20).mean()
    frame["Log_Volume"] = np.log1p(volume.clip(lower=0))
    frame["Volume_to_MA_5"] = _safe_divide(volume, frame["Volume_MA_5"])
    frame["Volume_to_MA_20"] = _safe_divide(volume, volume_ma_20)

    frame["AdjClose_to_MA_5"] = _safe_divide(adjusted_close, frame["MA_5"]) - 1
    frame["AdjClose_to_MA_20"] = _safe_divide(adjusted_close, frame["MA_20"]) - 1
    frame["AdjClose_to_MA_60"] = _safe_divide(adjusted_close, frame["MA_60"]) - 1
    frame["MA_5_to_MA_20"] = _safe_divide(frame["MA_5"], frame["MA_20"]) - 1
    frame["MA_20_to_MA_60"] = _safe_divide(frame["MA_20"], frame["MA_60"]) - 1

    delta = adjusted_close.diff()
    average_gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    relative_strength = _safe_divide(average_gain, average_loss)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    frame["RSI_14"] = rsi

    ema_12 = adjusted_close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = adjusted_close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_absolute = ema_12 - ema_26
    frame["MACD"] = _safe_divide(macd_absolute, adjusted_close)
    frame["MACD_Signal"] = _safe_divide(
        macd_absolute.ewm(span=9, adjust=False, min_periods=9).mean(),
        adjusted_close,
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["ATR_14_pct"] = _safe_divide(
        true_range.rolling(14, min_periods=14).mean(), close
    )
    return frame


def _build_market_features(market_df: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "Adj Close"]
    missing = [column for column in required if column not in market_df.columns]
    if missing:
        raise ValueError(f"Missing required market columns: {missing}")

    market = market_df[required].copy()
    market["Date"] = pd.to_datetime(market["Date"], errors="coerce")
    market["Adj Close"] = pd.to_numeric(market["Adj Close"], errors="coerce")
    market = market.dropna().sort_values("Date").drop_duplicates("Date", keep="last")

    market_return_1d = market["Adj Close"].pct_change(1, fill_method=None)
    market["Market_Return_1D"] = market_return_1d
    market["Market_Return_5D"] = market["Adj Close"].pct_change(5, fill_method=None)
    market["Market_Return_10D"] = market["Adj Close"].pct_change(10, fill_method=None)
    market_ma_20 = market["Adj Close"].rolling(20, min_periods=20).mean()
    market["Market_AdjClose_to_MA_20"] = (
        _safe_divide(market["Adj Close"], market_ma_20) - 1
    )
    market["Market_Volatility_20D"] = market_return_1d.rolling(
        20, min_periods=20
    ).std()
    return market[["Date", *MARKET_FEATURE_COLUMNS[:5]]]


def build_stock_features(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all reusable stock features without requiring future target values."""
    prepared = _prepare_stock_frame(stock_df)
    stock_frames = [
        _build_single_stock_features(group)
        for _, group in prepared.groupby("Ticker", sort=True)
    ]
    if not stock_frames:
        return prepared

    features = pd.concat(stock_frames, ignore_index=True)
    if market_df is not None:
        market_features = _build_market_features(market_df)
        features = features.merge(market_features, on="Date", how="left", validate="many_to_one")
        features["Excess_Return_1D"] = (
            features["Return_1D"] - features["Market_Return_1D"]
        )
        features["Excess_Return_5D"] = (
            features["Return_5D"] - features["Market_Return_5D"]
        )
    else:
        for column in MARKET_FEATURE_COLUMNS:
            features[column] = np.nan

    return features.replace([np.inf, -np.inf], np.nan).sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)


def add_stock_targets(
    feature_df: pd.DataFrame,
    horizons: Sequence[int] = TARGET_HORIZONS,
) -> pd.DataFrame:
    """Add direction targets and their actual future trading dates per ticker."""
    df = feature_df.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)
    grouped = df.groupby("Ticker", sort=False)

    for horizon in horizons:
        future_price = grouped["Adj Close"].shift(-horizon)
        future_date = grouped["Date"].shift(-horizon)
        target = (future_price > df["Adj Close"]).astype("float")
        target = target.mask(future_price.isna())
        df[f"Target_{horizon}D"] = target
        df[f"Target_Date_{horizon}D"] = pd.to_datetime(future_date)

    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def feature_columns_for_round(round_number: int) -> tuple[str, ...]:
    if round_number == 1:
        return BASE_FEATURE_COLUMNS
    if round_number == 2:
        return STATIONARY_FEATURE_COLUMNS
    if round_number == 3:
        return (*STATIONARY_FEATURE_COLUMNS, *MARKET_FEATURE_COLUMNS)
    raise ValueError(f"Unsupported refinement round: {round_number}")


def require_finite_features(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected = list(columns)
    missing = [column for column in selected if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=selected)
