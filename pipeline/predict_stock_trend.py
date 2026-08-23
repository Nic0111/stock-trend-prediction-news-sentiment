from __future__ import annotations

import argparse
import json
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict

import joblib
import pandas as pd
import yfinance as yf

from pipeline.model_artifacts import CalibratedStockModel
from pipeline.market_calendar import completed_sessions_between, latest_completed_session
from pipeline.runtime_validation import validate_stock_artifact
from pipeline.runtime_provenance import sha256_file
from pipeline.horizon_acceptance import weaker_status
from pipeline.stock_features import FEATURE_SCHEMA_VERSION, build_stock_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "data" / "stock-metadata.csv"
REGISTRY_FILE = PROJECT_ROOT / "models" / "model-registry.json"
MARKET_TICKER = "^KLSE"
SUPPORTED_HORIZONS = (1, 5, 10)
MINIMUM_HISTORY_ROWS = 61
# 420 calendar days provides a buffer for 60-session features, market holidays
# and provider gaps while keeping runtime requests bounded.
DOWNLOAD_LOOKBACK_DAYS = 420

FEATURE_LABELS = {
    "Return_1D": "1-day momentum",
    "Return_5D": "5-day momentum",
    "Return_10D": "10-day momentum",
    "Lag_Return_1D_1": "previous trading day's return",
    "Lag_Return_1D_2": "return two trading days ago",
    "Lag_Return_1D_3": "return three trading days ago",
    "Lag_Return_1D_5": "return five trading days ago",
    "AdjClose_to_MA_5": "price versus 5-day trend",
    "AdjClose_to_MA_20": "price versus 20-day trend",
    "AdjClose_to_MA_60": "price versus 60-day trend",
    "MA_5_to_MA_20": "short-term versus medium-term trend",
    "MA_20_to_MA_60": "medium-term versus long-term trend",
    "RSI_14": "RSI momentum",
    "MACD": "MACD momentum",
    "MACD_Signal": "MACD signal line",
    "Volatility_20D": "20-day volatility",
    "High_Low_Spread": "daily high-low price range",
    "Open_Close_Change": "open-to-close movement",
    "Volume_to_MA_20": "volume versus 20-day average",
    "Market_Return_1D": "KLCI 1-day direction",
    "Market_Return_5D": "KLCI 5-day direction",
    "Market_Return_10D": "KLCI 10-day direction",
    "Market_AdjClose_to_MA_20": "KLCI versus 20-day trend",
    "Excess_Return_1D": "1-day performance versus KLCI",
    "Excess_Return_5D": "5-day performance versus KLCI",
    "Excess_Return_10D": "10-day performance versus KLCI",
    "Rolling_Beta_20D": "20-day market sensitivity",
    "Rolling_Beta_60D": "60-day market sensitivity",
    "Rolling_Correlation_20D": "20-day market correlation",
    "Rolling_Correlation_60D": "60-day market correlation",
    "Volatility_Ratio_5D_20D": "short-term versus long-term volatility",
    "KLCI_Trend_Regime": "KLCI trend regime",
    "KLCI_Volatility_Regime": "KLCI volatility regime",
}


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
    threshold_constraints_passed: bool = True
    evidence_strength: str = "Unavailable"
    supporting_factors: list[Dict[str, Any]] = field(default_factory=list)
    opposing_factors: list[Dict[str, Any]] = field(default_factory=list)
    market_reference_date: str | None = None
    observation_lag_sessions: int | None = None
    freshness_status: str = "unknown"
    freshness_reason: str | None = None
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    global_evidence_status: str = "experimental"
    ticker_evidence_status: str = "experimental"
    shadow_evidence_status: str | None = None
    acceptance_policy_version: str = "legacy"
    failed_acceptance_checks: list[str] = field(default_factory=list)
    forecast_issued_at: str | None = None
    target_window_start_date: str | None = None
    target_window_end_date: str | None = None
    probability_interval: dict[str, float] | None = None
    directional_margin: float = 0.0
    abstention_reason: str | None = None

    @property
    def evidence_status(self) -> str:
        """Human-facing evidence tier used by the dashboard and tutor."""
        return str(self.status).lower()

    @property
    def directional_signal_available(self) -> bool:
        """Whether the frozen validation benchmark accepts this direction as evidence."""
        return self.evidence_status == "accepted" and self.threshold_constraints_passed

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
    return latest_completed_session(as_of) + timedelta(days=1)


def _minimum_history_rows(required_features: list[str]) -> int:
    """Derive a safe row floor from the largest rolling feature window."""
    largest_window = 10
    for feature in required_features:
        if "60" in feature:
            largest_window = max(largest_window, 60)
        elif "20" in feature:
            largest_window = max(largest_window, 20)
        elif "14" in feature:
            largest_window = max(largest_window, 14)
    # Returns consume one earlier observation; a small buffer also tolerates
    # one provider row that is removed during stock/KLCI alignment.
    return max(MINIMUM_HISTORY_ROWS, largest_window + 5)


@lru_cache(maxsize=6)
def _load_model_cached(path_text: str, size: int, modified_ns: int) -> Any:
    """Load a deployed model once, invalidating it when the artifact changes."""

    del size, modified_ns
    return joblib.load(Path(path_text))


def _production_model_loader(path: Path) -> Any:
    stat = path.stat()
    return _load_model_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def predict_stock_trend(
    ticker: str,
    horizon_days: int,
    as_of_date: date | str | None = None,
    *,
    stock_df: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
    registry_path: Path = REGISTRY_FILE,
    model_loader: Callable[[Path], Any] | None = None,
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
    registry_schema_version = registry.get("feature_schema_version")
    if registry_schema_version != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            "Model registry is missing or uses an incompatible stock/KLCI feature schema. "
            "Rebuild model-input data and retrain the stock models."
        )
    horizon_registry = registry.get("horizons", {}).get(str(horizon_days))
    if not horizon_registry:
        raise KeyError(f"Registry has no model for {horizon_days}D horizon")

    global_status = str(
        horizon_registry.get(
            "global_status",
            horizon_registry.get("status", "experimental"),
        )
    ).lower()
    shadow_acceptance = horizon_registry.get("shadow_acceptance") or {}
    shadow_status = str(shadow_acceptance.get("status", "")).lower() or None
    if shadow_status not in {"accepted", "limited", "experimental"}:
        shadow_status = None
    effective_global_status = weaker_status(global_status, shadow_status) if shadow_status else global_status
    ticker_entry = (horizon_registry.get("ticker_acceptance") or {}).get(normalized_ticker, {})
    ticker_evidence_available = bool(
        ticker_entry.get("development_evidence_available", bool(ticker_entry))
    )
    if ticker_entry and not ticker_evidence_available:
        ticker_status = "unavailable"
        # A missing ticker-specific development sample is disclosed but does
        # not silently override a properly accepted pooled panel model.
        effective_status = effective_global_status
    else:
        ticker_status = str(ticker_entry.get("status", "accepted")).lower()
        if ticker_status not in {"accepted", "limited", "experimental"}:
            ticker_status = "experimental"
        effective_status = weaker_status(effective_global_status, ticker_status)
    observed_acceptance = horizon_registry.get("observed_acceptance") or {}
    failed_checks = list(observed_acceptance.get("failed_checks", []))
    registry_is_official = (
        str(registry.get("search_profile", "")).casefold() == "official"
        and str(registry.get("publication_status", "")).casefold() == "validated"
    )
    if not registry_is_official:
        effective_status = "experimental"
        failed_checks.append("registry_not_officially_validated")
    if shadow_status and shadow_status != "accepted":
        failed_checks.extend(
            f"shadow:{name}" for name in (shadow_acceptance.get("failed_checks") or [])
        )
    policy_version = str(
        registry.get("benchmark_context", {}).get("acceptance_policy_version", "legacy")
    )

    effective_date = pd.Timestamp(as_of_date).date() if as_of_date is not None else date.today()
    completed_session = latest_completed_session(effective_date)
    start_date = completed_session - timedelta(days=DOWNLOAD_LOOKBACK_DAYS)
    end_exclusive = _resolve_download_end(effective_date)
    required_features = list(horizon_registry["numeric_feature_columns"])

    stock_data = stock_df
    if stock_data is None:
        stock_data = _download_daily_data(normalized_ticker, start_date, end_exclusive)
    else:
        stock_data = stock_data.copy()
        if "Ticker" not in stock_data.columns:
            stock_data["Ticker"] = normalized_ticker
    stock_data["Date"] = pd.to_datetime(stock_data["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    stock_data = stock_data[stock_data["Date"] <= pd.Timestamp(completed_session)].copy()

    minimum_history = _minimum_history_rows(required_features)
    if len(stock_data) < minimum_history:
        raise ValueError(
            f"Insufficient history for {normalized_ticker}: {len(stock_data)} rows; "
            f"at least {minimum_history} are required by the selected feature schema"
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
    features = features[features["Date"] <= pd.Timestamp(completed_session)].copy()
    feature_complete = features.dropna(subset=required_features)
    if feature_complete.empty:
        raise ValueError("No feature-complete inference row is available")
    latest = feature_complete.sort_values("Date").tail(1)

    artifact_path = PROJECT_ROOT / horizon_registry["artifact_path"]
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    expected_artifact_hash = str(horizon_registry.get("artifact_sha256") or "").strip()
    if not expected_artifact_hash:
        raise RuntimeError(
            "The stock artifact has no integrity hash in the model registry; "
            "run an official staged training and promotion."
        )
    if sha256_file(artifact_path) != expected_artifact_hash:
        raise RuntimeError(
            "The stock artifact integrity check failed; inference was stopped."
        )
    model = (model_loader or _production_model_loader)(artifact_path)
    artifact_errors = validate_stock_artifact(
        model, horizon_registry, horizon_days=horizon_days
    )
    if artifact_errors:
        raise ValueError(
            "Model artifact and registry contract do not match: "
            + "; ".join(artifact_errors)
        )

    probability_up = float(model.predict_proba(latest[list(model.input_columns)])[:, 1][0])
    label = "Up" if probability_up >= float(model.threshold) else "Down"
    direction_sign = 1.0 if label == "Up" else -1.0
    contributions = model.local_feature_contributions(latest[list(model.input_columns)])

    def present_factor(item: Dict[str, Any], relationship: str) -> Dict[str, Any]:
        feature = str(item["feature"])
        return {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
            "impact_probability_points": round(abs(float(item["contribution"])) * 100, 2),
            "relationship": relationship,
        }

    supporting_factors = [
        present_factor(item, "supports")
        for item in contributions
        if float(item["contribution"]) * direction_sign > 0
    ][:3]
    opposing_factors = [
        present_factor(item, "opposes")
        for item in contributions
        if float(item["contribution"]) * direction_sign < 0
    ][:2]
    threshold_margin = abs(probability_up - float(model.threshold))
    margin_tier = (
        "Strong" if threshold_margin >= 0.05 else "Moderate" if threshold_margin >= 0.02 else "Limited"
    )
    latest_date = pd.Timestamp(latest["Date"].iloc[0]).date()
    aligned_dates = (
        pd.to_datetime(features["Date"], errors="coerce")
        .dropna()
        .dt.date
        .drop_duplicates()
        .sort_values()
    )
    reference_date = aligned_dates.iloc[-1] if len(aligned_dates) else latest_date
    observation_lag_sessions = completed_sessions_between(latest_date, completed_session)
    age_days = max(0, (effective_date - latest_date).days)
    if observation_lag_sessions == 0 and age_days <= 7:
        freshness_status = "fresh"
        freshness_reason = "Latest feature-complete row matches the latest valid aligned market session."
    elif observation_lag_sessions <= 1 and age_days <= 7:
        freshness_status = "delayed"
        freshness_reason = "One valid aligned market session is newer than the observation used."
    else:
        freshness_status = "stale"
        freshness_reason = (
            f"Observation is {observation_lag_sessions} aligned session(s) and {age_days} calendar day(s) "
            "behind the latest valid market data."
        )
    abstention_reasons: list[str] = []
    if effective_status != "accepted":
        abstention_reasons.append(
            f"Validation status is {effective_status}; the directional result is research-only."
        )
    if freshness_status == "stale":
        abstention_reasons.append(freshness_reason)
    if not bool(model.threshold_constraints_passed):
        abstention_reasons.append("The frozen threshold did not pass directional-balance constraints.")
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
        benchmark_passed=effective_status == "accepted",
        status=effective_status,
        stale_data_warning=freshness_status == "stale",
        latest_data_age_days=age_days,
        threshold_constraints_passed=bool(
            horizon_registry.get(
                "threshold_constraints_passed",
                horizon_registry.get("threshold_passed", False),
            )
        ),
        evidence_strength=margin_tier,
        supporting_factors=supporting_factors,
        opposing_factors=opposing_factors,
        market_reference_date=reference_date.isoformat(),
        observation_lag_sessions=observation_lag_sessions,
        freshness_status=freshness_status,
        freshness_reason=freshness_reason,
        feature_schema_version=str(registry.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
        global_evidence_status=effective_global_status,
        ticker_evidence_status=ticker_status,
        shadow_evidence_status=shadow_status,
        acceptance_policy_version=policy_version,
        failed_acceptance_checks=sorted(set(failed_checks)),
        forecast_issued_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        # Future Bursa sessions are not fabricated from weekday arithmetic;
        # the exact target window remains unavailable until an exchange
        # calendar service is part of the runtime contract.
        target_window_start_date=None,
        target_window_end_date=None,
        probability_interval=None,
        directional_margin=threshold_margin,
        abstention_reason=" ".join(reason for reason in abstention_reasons if reason) or None,
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
