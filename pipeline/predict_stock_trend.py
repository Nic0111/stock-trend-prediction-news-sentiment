from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict

import joblib
import pandas as pd
import yfinance as yf

from pipeline.model_artifacts import CalibratedStockModel
from pipeline.stock_features import build_stock_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "data" / "stock-metadata.csv"
REGISTRY_FILE = PROJECT_ROOT / "models" / "model-registry.json"
MARKET_TICKER = "^KLSE"
SUPPORTED_HORIZONS = (1, 5, 10)
MINIMUM_HISTORY_ROWS = 60
DOWNLOAD_LOOKBACK_DAYS = 260


@dataclass
class PredictionResult:
    ticker: str
    company_name: str
    horizon_days: int
    as_of_date: str
    prediction_label: str
    probability_up: float
    decision_threshold: float
    model_name: str
    model_version: str
    refinement_round: int
    benchmark_passed: bool
    status: str
    stale_data_warning: bool
    latest_data_age_days: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_metadata() -> Dict[str, str]:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Stock metadata not found: {METADATA_FILE}")
    metadata = pd.read_csv(METADATA_FILE)
    return {
        str(row["Ticker"]).strip().upper(): str(row["Company_Name"]).strip()
        for _, row in metadata.iterrows()
    }


def _download_daily_data(ticker: str, start: date, end_exclusive: date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if data is None or data.empty:
        raise ValueError(f"No daily data returned for {ticker}")
    data = data.reset_index()
    if "Date" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Date"})
    data["Ticker"] = ticker
    return data


def _resolve_download_end(as_of: date) -> date:
    today = date.today()
    return today if as_of >= today else as_of + timedelta(days=1)


def predict_stock_trend(
    ticker: str,
    horizon_days: int,
    as_of_date: date | str | None = None,
    *,
    stock_df: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
    registry_path: Path = REGISTRY_FILE,
    model_loader: Callable[[Path], Any] = joblib.load,
) -> PredictionResult:
    normalized_ticker = ticker.strip().upper()
    metadata = _load_metadata()
    if normalized_ticker not in metadata:
        raise ValueError(f"Unsupported ticker: {normalized_ticker}")
    if horizon_days not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon_days}; expected 1, 5, or 10")
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found: {registry_path}. Run `python main.py train_models`."
        )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    horizon_registry = registry.get("horizons", {}).get(str(horizon_days))
    if not horizon_registry:
        raise KeyError(f"Registry has no model for {horizon_days}D horizon")

    effective_date = pd.Timestamp(as_of_date).date() if as_of_date is not None else date.today()
    start_date = effective_date - timedelta(days=DOWNLOAD_LOOKBACK_DAYS)
    end_exclusive = _resolve_download_end(effective_date)

    stock_data = stock_df
    if stock_data is None:
        stock_data = _download_daily_data(normalized_ticker, start_date, end_exclusive)
    else:
        stock_data = stock_data.copy()
        if "Ticker" not in stock_data.columns:
            stock_data["Ticker"] = normalized_ticker

    if len(stock_data) < MINIMUM_HISTORY_ROWS:
        raise ValueError(
            f"Insufficient history for {normalized_ticker}: {len(stock_data)} rows; "
            f"at least {MINIMUM_HISTORY_ROWS} are required"
        )

    refinement_round = int(horizon_registry["refinement_round"])
    market_data = market_df
    if refinement_round >= 3 and market_data is None:
        try:
            market_data = _download_daily_data(MARKET_TICKER, start_date, end_exclusive)
        except Exception as exc:
            raise RuntimeError(
                "The selected model requires Bursa KLCI market context, but it is unavailable"
            ) from exc

    features = build_stock_features(stock_data, market_df=market_data if refinement_round >= 3 else None)
    features = features[features["Date"] <= pd.Timestamp(effective_date)].copy()
    required_features = list(horizon_registry["numeric_feature_columns"])
    feature_complete = features.dropna(subset=required_features)
    if feature_complete.empty:
        raise ValueError("No feature-complete inference row is available")
    latest = feature_complete.sort_values("Date").tail(1)

    artifact_path = PROJECT_ROOT / horizon_registry["artifact_path"]
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    model = model_loader(artifact_path)
    if not isinstance(model, CalibratedStockModel):
        raise TypeError("Model artifact is not a calibrated stock model")
    if list(model.input_columns) != list(horizon_registry["input_columns"]):
        raise ValueError("Model artifact and registry feature schemas do not match")

    probability_up = float(model.predict_proba(latest[list(model.input_columns)])[:, 1][0])
    label = "Up" if probability_up >= float(model.threshold) else "Down"
    latest_date = pd.Timestamp(latest["Date"].iloc[0]).date()
    age_days = max(0, (effective_date - latest_date).days)
    return PredictionResult(
        ticker=normalized_ticker,
        company_name=metadata[normalized_ticker],
        horizon_days=horizon_days,
        as_of_date=latest_date.isoformat(),
        prediction_label=label,
        probability_up=probability_up,
        decision_threshold=float(model.threshold),
        model_name=str(horizon_registry["model_name"]),
        model_version=str(registry.get("dataset_sha256", "unknown"))[:12],
        refinement_round=refinement_round,
        benchmark_passed=bool(horizon_registry["threshold_passed"]),
        status=str(horizon_registry["status"]),
        stale_data_warning=age_days > 5,
        latest_data_age_days=age_days,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict stock direction from the latest completed day")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--horizon", type=int, choices=SUPPORTED_HORIZONS, required=True)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    result = predict_stock_trend(args.ticker, args.horizon, args.as_of)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
