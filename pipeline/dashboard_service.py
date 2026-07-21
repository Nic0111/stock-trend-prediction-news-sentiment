from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal

from pipeline.finbert_sentiment_model import FinBERTRuntimeError
from pipeline.download_financial_news_data import (
    NEWS_CONTEXT_CONFIG_VERSION,
    NewsContextResult,
    load_stock_metadata,
)
from pipeline.marketaux_news_provider import retrieve_news_with_marketaux_fallback
from pipeline.predict_stock_trend import (
    FEATURE_SCHEMA_VERSION,
    PredictionResult,
    SUPPORTED_HORIZONS,
    predict_stock_trend,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_CACHE_DIR = PROJECT_ROOT / ".cache" / "dashboard"
STOCK_CACHE_DIR = DASHBOARD_CACHE_DIR / "stock"
NEWS_CACHE_DIR = DASHBOARD_CACHE_DIR / "news"
CACHE_SCHEMA_VERSION = 6
CACHE_TTL_SECONDS = 900
logger = logging.getLogger(__name__)

SignalSource = Literal["live", "fallback", "unavailable"]


@dataclass
class DashboardAnalysisResult:
    ticker: str
    company_name: str
    horizon_days: int
    stock: PredictionResult | None
    stock_source: SignalSource
    stock_captured_at: str | None
    stock_warning: str | None
    news: NewsContextResult | None
    news_source: SignalSource
    news_captured_at: str | None
    news_warning: str | None
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _stock_cache_path(ticker: str, horizon_days: int) -> Path:
    return STOCK_CACHE_DIR / f"{_slugify(ticker)}-{horizon_days}d.json"


def _news_cache_path(ticker: str) -> Path:
    return NEWS_CACHE_DIR / f"{_slugify(ticker)}.json"


def _write_cache(path: Path, payload: Dict[str, Any], captured_at: str) -> None:
    """Persist a cache entry without leaving a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "captured_at": captured_at,
                    "payload": payload,
                },
                temporary,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary_path, path)
    except OSError:
        # A cache write must never turn a successful live request into a failure.
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_cache(path: Path) -> tuple[Dict[str, Any], str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    payload = data.get("payload")
    captured_at = data.get("captured_at")
    if not isinstance(payload, dict) or not isinstance(captured_at, str) or not captured_at:
        return None
    try:
        captured_time = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if captured_time.tzinfo is None:
            captured_time = captured_time.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - captured_time).total_seconds()
    except ValueError:
        return None
    if age_seconds < 0 or age_seconds > CACHE_TTL_SECONDS:
        return None
    return payload, captured_at


def _generic_warning(signal_name: str, source: SignalSource, captured_at: str | None) -> str | None:
    if source == "live":
        return None
    if source == "fallback":
        timestamp = captured_at or "an unknown time"
        return f"Live {signal_name} data was unavailable; showing the saved result captured at {timestamp}."
    return f"Live {signal_name} data was unavailable and no saved result could be loaded."


def _load_stock_fallback(
    ticker: str,
    horizon_days: int,
) -> tuple[PredictionResult | None, str | None]:
    cached = _read_cache(_stock_cache_path(ticker, horizon_days))
    if cached is None:
        return None, None
    payload, captured_at = cached
    # Old caches do not contain the corrected stock/KLCI alignment contract.
    required_fields = {
        "feature_schema_version",
        "market_reference_date",
        "observation_lag_sessions",
        "freshness_status",
    }
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or not required_fields.issubset(payload):
        return None, None
    try:
        return PredictionResult(**payload), captured_at
    except (TypeError, ValueError):
        return None, None


def _load_news_fallback(ticker: str) -> tuple[NewsContextResult | None, str | None]:
    cached = _read_cache(_news_cache_path(ticker))
    if cached is None:
        return None, None
    payload, captured_at = cached
    try:
        result = NewsContextResult(**payload)
    except (TypeError, ValueError):
        return None, None
    if int(
        (result.provider_metadata or {}).get("news_context_config_version", 0) or 0
    ) != NEWS_CONTEXT_CONFIG_VERSION:
        return None, None
    return result, captured_at


def _stock_freshness_warning(stock: PredictionResult | None) -> str | None:
    if stock is None or (
        stock.freshness_status in {"fresh", "unknown"} and not stock.stale_data_warning
    ):
        return None
    status = stock.freshness_status.title()
    reason = stock.freshness_reason or "The observation is not the newest valid market session."
    return f"{status} market data: {reason}"


def _combine_warnings(*warnings: str | None) -> str | None:
    messages = [message for message in warnings if message]
    return " ".join(messages) if messages else None


def run_dashboard_analysis(ticker: str, horizon_days: int) -> DashboardAnalysisResult:
    """Run both independent runtime signals, falling back to saved results per signal."""
    normalized_ticker = ticker.strip().upper()
    if horizon_days not in SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon_days}; expected 1, 5, or 10")

    metadata = load_stock_metadata()
    if normalized_ticker not in metadata:
        raise ValueError(f"Unsupported ticker: {normalized_ticker}")

    stock: PredictionResult | None = None
    stock_source: SignalSource = "unavailable"
    stock_captured_at: str | None = None
    try:
        stock = predict_stock_trend(normalized_ticker, horizon_days)
        stock_source = "live"
        stock_captured_at = _now_iso()
        _write_cache(_stock_cache_path(normalized_ticker, horizon_days), stock.to_dict(), stock_captured_at)
    except Exception as exc:
        logger.warning("Stock signal failed for %s at %s: %s", normalized_ticker, horizon_days, exc)
        stock, stock_captured_at = _load_stock_fallback(normalized_ticker, horizon_days)
        stock_source = "fallback" if stock is not None else "unavailable"

    news: NewsContextResult | None = None
    news_source: SignalSource = "unavailable"
    news_captured_at: str | None = None
    news_failure_detail: str | None = None
    try:
        news = retrieve_news_with_marketaux_fallback(normalized_ticker, horizon_days)
        retrieval_source = str(news.provider_metadata.get("retrieval_source", "live"))
        news_source = "fallback" if retrieval_source == "fallback" else "live"
        news_captured_at = _now_iso()
        _write_cache(_news_cache_path(normalized_ticker), news.to_dict(), news_captured_at)
    except FinBERTRuntimeError as exc:
        # A shared classifier failure is distinct from a provider outage. Keep
        # the stock signal available and show an actionable dashboard message.
        news_failure_detail = str(exc)
        logger.warning("News sentiment runtime failed for %s: %s", normalized_ticker, exc)
        news, news_captured_at = _load_news_fallback(normalized_ticker)
        news_source = "fallback" if news is not None else "unavailable"
    except Exception as exc:
        logger.warning("News signal failed for %s: %s", normalized_ticker, exc)
        news, news_captured_at = _load_news_fallback(normalized_ticker)
        news_source = "fallback" if news is not None else "unavailable"

    return DashboardAnalysisResult(
        ticker=normalized_ticker,
        company_name=(stock.company_name if stock else news.company_name if news else metadata[normalized_ticker]),
        horizon_days=horizon_days,
        stock=stock,
        stock_source=stock_source,
        stock_captured_at=stock_captured_at,
        stock_warning=_combine_warnings(
            _generic_warning("stock", stock_source, stock_captured_at),
            _stock_freshness_warning(stock),
        ),
        news=news,
        news_source=news_source,
        news_captured_at=news_captured_at,
        news_warning=(
            (
                "Sentiment engine unavailable: "
                f"{news_failure_detail} "
                + (
                    f"Showing the saved result captured at {news_captured_at}."
                    if news_source == "fallback"
                    else "Install the pinned FinBERT dependencies in the active environment."
                )
            )
            if news_failure_detail
            else _generic_warning("news", news_source, news_captured_at)
        ),
        generated_at=_now_iso(),
    )
