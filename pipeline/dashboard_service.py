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
from pipeline.finbert_sentiment_model import FINBERT_MODEL_DIR
from pipeline.fusion_runtime import build_aggregate_outlook
from pipeline.download_financial_news_data import (
    NEWS_CONTEXT_CONFIG_VERSION,
    NewsContextResult,
    load_stock_metadata,
    sentiment_runtime_variant,
)
from pipeline.marketaux_news_provider import retrieve_news_with_marketaux_fallback
from pipeline.runtime_provenance import sha256_file
from pipeline.predict_stock_trend import (
    FEATURE_SCHEMA_VERSION,
    PredictionResult,
    REGISTRY_FILE,
    SUPPORTED_HORIZONS,
    predict_stock_trend,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_CACHE_DIR = PROJECT_ROOT / ".cache" / "dashboard"
STOCK_CACHE_DIR = DASHBOARD_CACHE_DIR / "stock"
NEWS_CACHE_DIR = DASHBOARD_CACHE_DIR / "news"
# Bump after correcting provider provenance so old payloads that labelled a
# Marketaux-only fallback as NewsAPI-successful cannot be reused.
CACHE_SCHEMA_VERSION = 10
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
    combined_outlook: Any | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _stock_cache_path(ticker: str, horizon_days: int) -> Path:
    return STOCK_CACHE_DIR / f"{_slugify(ticker)}-{horizon_days}d.json"


def _news_cache_path(ticker: str, as_of_date: str | None) -> Path:
    date_key = _slugify(as_of_date or "runtime")
    return NEWS_CACHE_DIR / f"{_slugify(ticker)}-{date_key}.json"


SENTIMENT_RUNTIME_CONFIG_FILE = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"


def _signature_from_parts(parts: Dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stock_cache_signature(ticker: str, horizon_days: int) -> str:
    """Identify the exact stock artifact and feature contract that produced a cache."""
    try:
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        entry = registry["horizons"][str(horizon_days)]
        artifact = PROJECT_ROOT / str(entry["artifact_path"])
        artifact_hash = sha256_file(artifact)
        registry_hash = sha256_file(REGISTRY_FILE)
    except (OSError, ValueError, KeyError, TypeError):
        return "missing"
    return _signature_from_parts(
        {
            "kind": "stock",
            "ticker": ticker.strip().upper(),
            "horizon_days": int(horizon_days),
            "registry_hash": registry_hash,
            "run_id": registry.get("run_id"),
            "dataset_sha256": registry.get("dataset_sha256"),
            "feature_schema_version": registry.get("feature_schema_version"),
            "artifact_path": str(entry["artifact_path"]),
            "artifact_sha256": artifact_hash,
        }
    )


def sentiment_cache_signature() -> str:
    """Identify the deployed sentiment checkpoint and runtime configuration."""
    files = [SENTIMENT_RUNTIME_CONFIG_FILE]
    for name in ("config.json", "metadata.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"):
        files.append(FINBERT_MODEL_DIR / name)
    try:
        config = json.loads(SENTIMENT_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
        variant = sentiment_runtime_variant()
        challenger = (config.get("challengers") or {}).get(variant) or {}
        for component in (challenger.get("components") or {}).values():
            if not isinstance(component, dict):
                continue
            configured_models = component.get("seed_models")
            if configured_models is None:
                configured_models = [component]
            if not isinstance(configured_models, list):
                continue
            for configured_model in configured_models:
                if not isinstance(configured_model, dict) or not configured_model.get("model_directory"):
                    continue
                directory = Path(str(configured_model["model_directory"]))
                if not directory.is_absolute():
                    directory = PROJECT_ROOT / directory
                files.extend(directory / name for name in ("config.json", "metadata.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"))
        hashes = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files if path.exists()}
    except (OSError, ValueError, TypeError):
        return "missing"
    return _signature_from_parts(
        {
            "kind": "news-sentiment",
            "news_context_config_version": NEWS_CONTEXT_CONFIG_VERSION,
            "sentiment_model_variant": sentiment_runtime_variant(),
            "runtime_config": config,
            "artifact_hashes": hashes,
        }
    )


def _write_cache(
    path: Path,
    payload: Dict[str, Any],
    captured_at: str,
    producer_signature: str | None = None,
) -> None:
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
                    "producer_signature": producer_signature or "unspecified",
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


def _read_cache(path: Path, expected_signature: str | None = None) -> tuple[Dict[str, Any], str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    if expected_signature is not None and data.get("producer_signature") != expected_signature:
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
    cached = _read_cache(
        _stock_cache_path(ticker, horizon_days),
        expected_signature=stock_cache_signature(ticker, horizon_days),
    )
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


def _load_news_fallback(
    ticker: str,
    expected_as_of_date: str | None = None,
) -> tuple[NewsContextResult | None, str | None]:
    if expected_as_of_date is not None:
        candidates = [_news_cache_path(ticker, expected_as_of_date)]
    else:
        prefix = f"{_slugify(ticker)}-"
        candidates = sorted(
            NEWS_CACHE_DIR.glob(f"{prefix}*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        # Read the legacy location only to reject an old schema cleanly.
        candidates.append(NEWS_CACHE_DIR / f"{_slugify(ticker)}.json")
    for path in candidates:
        cached = _read_cache(path, expected_signature=sentiment_cache_signature())
        if cached is None:
            continue
        payload, captured_at = cached
        try:
            result = NewsContextResult(**payload)
        except (TypeError, ValueError):
            continue
        if int(
            (result.provider_metadata or {}).get("news_context_config_version", 0) or 0
        ) != NEWS_CONTEXT_CONFIG_VERSION:
            continue
        if expected_as_of_date is not None and result.as_of_date != expected_as_of_date:
            continue
        return result, captured_at
    return None, None


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


def run_dashboard_analysis(
    ticker: str,
    horizon_days: int,
    *,
    stock_result: PredictionResult | None = None,
) -> DashboardAnalysisResult:
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
        stock = stock_result or predict_stock_trend(normalized_ticker, horizon_days)
        stock_source = "live"
        stock_captured_at = _now_iso()
        _write_cache(
            _stock_cache_path(normalized_ticker, horizon_days),
            stock.to_dict(),
            stock_captured_at,
            stock_cache_signature(normalized_ticker, horizon_days),
        )
    except Exception as exc:
        logger.warning("Stock signal failed for %s at %s: %s", normalized_ticker, horizon_days, exc)
        stock, stock_captured_at = _load_stock_fallback(normalized_ticker, horizon_days)
        stock_source = "fallback" if stock is not None else "unavailable"

    news: NewsContextResult | None = None
    news_source: SignalSource = "unavailable"
    news_captured_at: str | None = None
    news_failure_detail: str | None = None
    try:
        # Anchor news to the same completed market observation used by the
        # stock signal.  Without this explicit date, a stale stock row could
        # be paired with articles published after that observation.
        news_as_of = stock.as_of_date if stock is not None else None
        # A fresh, artifact-matched result is reusable evidence, not merely an
        # outage fallback. Reading it first avoids unnecessary provider quota
        # and repeated FinBERT inference during normal navigation.
        news, news_captured_at = _load_news_fallback(normalized_ticker, news_as_of)
        if news is None:
            news = retrieve_news_with_marketaux_fallback(
                normalized_ticker,
                horizon_days,
                as_of_date=news_as_of,
            )
            news_captured_at = _now_iso()
        if news_as_of is not None and news.as_of_date != news_as_of:
            raise ValueError(
                "News context date does not match the stock observation date "
                f"({news.as_of_date} != {news_as_of})."
            )
        retrieval_source = str(news.provider_metadata.get("retrieval_source", "live"))
        news_source = "fallback" if retrieval_source == "fallback" else "live"
        if not _news_cache_path(normalized_ticker, news_as_of or news.as_of_date).exists() or news_captured_at is None:
            news_captured_at = news_captured_at or _now_iso()
            _write_cache(
                _news_cache_path(normalized_ticker, news_as_of or news.as_of_date),
                news.to_dict(),
                news_captured_at,
                sentiment_cache_signature(),
            )
    except FinBERTRuntimeError as exc:
        # A shared classifier failure is distinct from a provider outage. Keep
        # the stock signal available and show an actionable dashboard message.
        news_failure_detail = str(exc)
        logger.warning("News sentiment runtime failed for %s: %s", normalized_ticker, exc)
        expected_as_of = stock.as_of_date if stock is not None else None
        news, news_captured_at = _load_news_fallback(normalized_ticker, expected_as_of)
        news_source = "fallback" if news is not None else "unavailable"
    except Exception as exc:
        logger.warning("News signal failed for %s: %s", normalized_ticker, exc)
        expected_as_of = stock.as_of_date if stock is not None else None
        news, news_captured_at = _load_news_fallback(normalized_ticker, expected_as_of)
        news_source = "fallback" if news is not None else "unavailable"

    company_name = stock.company_name if stock else news.company_name if news else metadata[normalized_ticker]
    combined_outlook = build_aggregate_outlook(
        stock,
        news,
        ticker=normalized_ticker,
        company_name=company_name,
        horizon_days=horizon_days,
    )

    return DashboardAnalysisResult(
        ticker=normalized_ticker,
        company_name=company_name,
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
        combined_outlook=combined_outlook,
    )
