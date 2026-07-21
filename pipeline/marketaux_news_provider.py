"""Marketaux adapter and conditional runtime fallback.

NewsAPI remains the primary provider.  Marketaux is queried only when the
selected NewsAPI context is low/unavailable or NewsAPI fails.  The adapter
shares StockSense's relevance filter, FinBERT scorer and context builder so
provider differences do not silently become model differences.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline.finbert_sentiment_model import FinBERTRuntimeError
from pipeline.download_financial_news_data import (
    ARTICLE_COLUMNS,
    CONTEXT_WINDOWS,
    DASHBOARD_ARTICLES,
    NEWS_CONTEXT_CONFIG_VERSION,
    NEWS_HISTORICAL_CONTEXT_END_DAYS,
    NEWS_HISTORICAL_CONTEXT_START_DAYS,
    NEWS_RETRIEVAL_LOOKBACK_DAYS,
    NewsContextResult,
    add_sentiment_scores,
    article_age_days,
    build_news_context_metadata,
    build_company_query_variants,
    build_news_contexts,
    build_newsapi_query,
    clean_text,
    ensure_data_structure,
    load_sentiment_model,
    load_stock_metadata,
    match_relevant_alias,
    normalize_title,
    normalize_url,
    slugify,
)


MARKETAUX_ENDPOINT = "https://api.marketaux.com/v1/news/all"
MARKETAUX_ENTITY_SEARCH_ENDPOINT = "https://api.marketaux.com/v1/entity/search"
MARKETAUX_API_KEY_FILE = Path(__file__).resolve().parent.parent / "config" / "marketaux_api_key.txt"
MARKETAUX_API_TOKEN_ENV_VAR = "MARKETAUX_API_TOKEN"
MARKETAUX_TIMEOUT_SECONDS = float(os.getenv("STOCKSENSE_MARKETAUX_TIMEOUT_SECONDS", "20"))
MARKETAUX_ARTICLE_LIMIT = 3
MARKETAUX_PROVIDER = "marketaux"
MARKETAUX_DAILY_CAP = max(1, int(os.getenv("STOCKSENSE_MARKETAUX_DAILY_CAP", "100")))
MARKETAUX_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "marketaux-news"
MARKETAUX_CACHE_TTL_SECONDS = 6 * 60 * 60
MARKETAUX_STALE_TTL_SECONDS = 24 * 60 * 60
MARKETAUX_HISTORICAL_SCOPE = "historical"
MARKETAUX_RECENT_SCOPE = "recent"
MARKETAUX_TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")


class MarketauxRequestError(RuntimeError):
    """A Marketaux request failure with a safe status/code description."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        code: str | None = None,
        headers: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.headers = {
            str(key).casefold(): clean_text(value)
            for key, value in (headers or {}).items()
            if str(key).casefold() in {
                "x-usagelimit-limit",
                "x-usagelimit-remaining",
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "retry-after",
            }
        }


class MarketauxConfigurationError(MarketauxRequestError):
    """Raised when the candidate provider has not been configured."""


class MarketauxQuotaError(MarketauxRequestError):
    """Raised before a request when the local quota guard is exhausted."""


@dataclass(frozen=True)
class MarketauxEntityMatch:
    """The one-time provider-specific entity mapping for a local ticker."""

    local_ticker: str
    company_name: str
    symbol: str | None
    exchange: str | None
    country: str | None
    matched_name: str | None
    status: str
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_ticker": self.local_ticker,
            "company_name": self.company_name,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "country": self.country,
            "matched_name": self.matched_name,
            "status": self.status,
            "reason": self.reason,
        }


def get_marketaux_api_token() -> str:
    """Read the token from the ignored config file, then environment."""
    if MARKETAUX_API_KEY_FILE.exists():
        token = MARKETAUX_API_KEY_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = os.getenv(MARKETAUX_API_TOKEN_ENV_VAR, "").strip()
    if token:
        return token
    raise MarketauxConfigurationError(
        "Marketaux token is not configured. Add config/marketaux_api_key.txt "
        f"or set {MARKETAUX_API_TOKEN_ENV_VAR}."
    )


def _marketaux_local_date() -> date:
    return datetime.now(timezone.utc).astimezone(MARKETAUX_TIMEZONE).date()


def _marketaux_cache_path(
    ticker: str,
    as_of_date: date,
    *,
    scope: str = MARKETAUX_RECENT_SCOPE,
) -> Path:
    return MARKETAUX_CACHE_DIR / "responses" / f"{slugify(ticker)}-{as_of_date.isoformat()}-{scope}.json"


def _marketaux_quota_path() -> Path:
    return MARKETAUX_CACHE_DIR / "quota" / f"{_marketaux_local_date().isoformat()}.json"


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_quota_state() -> Dict[str, Any]:
    path = _marketaux_quota_path()
    state = _read_json(path) or {}
    if state.get("date") != _marketaux_local_date().isoformat():
        state = {}
    return {
        "date": _marketaux_local_date().isoformat(),
        "requests_used": max(0, int(state.get("requests_used", 0) or 0)),
        "daily_cap": MARKETAUX_DAILY_CAP,
        "last_remaining": state.get("last_remaining"),
        "last_status": clean_text(state.get("last_status")),
        "last_request_at": clean_text(state.get("last_request_at")),
    }


def _reserve_marketaux_request() -> Dict[str, Any]:
    state = _read_quota_state()
    if state["requests_used"] >= MARKETAUX_DAILY_CAP:
        raise MarketauxQuotaError(
            f"Marketaux local daily cap ({MARKETAUX_DAILY_CAP}) has been reached."
        )
    state["requests_used"] += 1
    state["last_request_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _atomic_json_write(_marketaux_quota_path(), state)
    return state


def _record_marketaux_response(headers: Mapping[str, Any] | None, status: str) -> Dict[str, Any]:
    state = _read_quota_state()
    safe = _safe_headers(headers)
    remaining = safe.get("x-usagelimit-remaining") or safe.get("x-ratelimit-remaining")
    state["last_remaining"] = remaining
    state["last_status"] = status
    _atomic_json_write(_marketaux_quota_path(), state)
    return {
        "requests_used": state["requests_used"],
        "daily_cap": MARKETAUX_DAILY_CAP,
        "last_remaining": remaining,
        "headers": safe,
    }


def _read_marketaux_cache(
    ticker: str,
    as_of_date: date,
    *,
    scope: str = MARKETAUX_RECENT_SCOPE,
    allow_stale: bool = False,
) -> tuple[NewsContextResult, Dict[str, Any]] | None:
    path = _marketaux_cache_path(ticker, as_of_date, scope=scope)
    payload = _read_json(path)
    if not payload:
        return None
    if clean_text(payload.get("cache_scope")) != scope:
        return None
    if int(payload.get("news_context_config_version", 0) or 0) != NEWS_CONTEXT_CONFIG_VERSION:
        return None
    captured_at = clean_text(payload.get("captured_at"))
    result_payload = payload.get("result")
    if not captured_at or not isinstance(result_payload, dict):
        return None
    try:
        captured_time = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if captured_time.tzinfo is None:
            captured_time = captured_time.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - captured_time).total_seconds()
    except ValueError:
        return None
    max_age = MARKETAUX_STALE_TTL_SECONDS if allow_stale else MARKETAUX_CACHE_TTL_SECONDS
    if age_seconds < 0 or age_seconds > max_age:
        return None
    try:
        result = NewsContextResult(**result_payload)
    except (TypeError, ValueError):
        return None
    if int(
        (result.provider_metadata or {}).get("news_context_config_version", 0) or 0
    ) != NEWS_CONTEXT_CONFIG_VERSION:
        return None
    return result, {
        "cache_state": "stale" if age_seconds > MARKETAUX_CACHE_TTL_SECONDS else "fresh",
        "captured_at": captured_at,
        "age_seconds": max(0, int(age_seconds)),
    }


def _write_marketaux_cache(result: NewsContextResult, captured_at: str) -> None:
    try:
        as_of_date = pd.Timestamp(result.as_of_date).date()
    except (TypeError, ValueError):
        return
    scope = clean_text((result.provider_metadata or {}).get("cache_scope")) or MARKETAUX_RECENT_SCOPE
    _atomic_json_write(
        _marketaux_cache_path(result.ticker, as_of_date, scope=scope),
        {
            "captured_at": captured_at,
            "news_context_config_version": NEWS_CONTEXT_CONFIG_VERSION,
            "cache_scope": scope,
            "result": result.to_dict(),
        },
    )


def _date_parameter(value: date) -> str:
    return value.isoformat()


def build_marketaux_request_url(
    token: str,
    *,
    symbols: Sequence[str] | None = None,
    search: str | None = None,
    as_of_date: date,
    lookback_days: int = NEWS_RETRIEVAL_LOOKBACK_DAYS,
    published_after: date | None = None,
    published_before: date | None = None,
    limit: int = MARKETAUX_ARTICLE_LIMIT,
    endpoint: str = MARKETAUX_ENDPOINT,
) -> str:
    """Build a bounded, deterministic Marketaux news request."""
    if not symbols and not search:
        raise ValueError("Marketaux requests require symbols or search")
    params: Dict[str, str] = {
        "api_token": token,
        "published_after": _date_parameter(
            published_after if published_after is not None else as_of_date - timedelta(days=lookback_days)
        ),
        "published_before": _date_parameter(
            published_before if published_before is not None else as_of_date + timedelta(days=1)
        ),
        "language": "en",
        "filter_entities": "true",
        "must_have_entities": "true",
        "group_similar": "true",
        "sort": "published_at",
        "limit": str(limit),
        "page": "1",
    }
    if symbols:
        params["symbols"] = ",".join(clean_text(item) for item in symbols if clean_text(item))
    if search:
        params["search"] = search
    return f"{endpoint}?{urlencode(params)}"


def build_marketaux_entity_search_url(
    token: str,
    search: str,
    *,
    endpoint: str = MARKETAUX_ENTITY_SEARCH_ENDPOINT,
) -> str:
    return f"{endpoint}?{urlencode({'api_token': token, 'search': search})}"


def _safe_headers(headers: Mapping[str, Any] | None) -> Dict[str, str]:
    allowed = {
        "x-usagelimit-limit",
        "x-usagelimit-remaining",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "retry-after",
    }
    return {
        str(key).casefold(): clean_text(value)
        for key, value in (headers or {}).items()
        if str(key).casefold() in allowed
    }


def _error_details(payload: Any) -> tuple[str | None, str]:
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            return clean_text(error.get("code")) or None, clean_text(error.get("message")) or "Marketaux request failed"
        return clean_text(payload.get("code")) or None, clean_text(payload.get("message")) or "Marketaux request failed"
    return None, "Marketaux request failed"


def _fetch_json(
    url: str,
    *,
    timeout: float = MARKETAUX_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = urlopen,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    request = Request(url, headers={"User-Agent": "stock-investment-analytics/2.0"})
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            headers = _safe_headers(getattr(response, "headers", {}))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        code, message = _error_details(payload)
        safe_message = f"Marketaux HTTP {exc.code}: {message}"
        raise MarketauxRequestError(safe_message, exc.code, code, _safe_headers(exc.headers)) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise MarketauxRequestError(f"Marketaux request failed: {exc}") from exc
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MarketauxRequestError("Marketaux returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MarketauxRequestError("Marketaux returned a non-object JSON response")
    if isinstance(payload.get("error"), Mapping):
        code, message = _error_details(payload)
        raise MarketauxRequestError(message, code=code, headers=headers)
    return payload, headers


def fetch_marketaux_response(
    token: str,
    *,
    symbols: Sequence[str] | None = None,
    search: str | None = None,
    as_of_date: date,
    lookback_days: int = NEWS_RETRIEVAL_LOOKBACK_DAYS,
    published_after: date | None = None,
    published_before: date | None = None,
    limit: int = MARKETAUX_ARTICLE_LIMIT,
    timeout: float = MARKETAUX_TIMEOUT_SECONDS,
    opener: Callable[..., Any] = urlopen,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    return _fetch_json(
        build_marketaux_request_url(
            token,
            symbols=symbols,
            search=search,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            published_after=published_after,
            published_before=published_before,
            limit=limit,
        ),
        timeout=timeout,
        opener=opener,
    )


def _entity_candidates(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    data = payload.get("data")
    return [item for item in data if isinstance(item, Mapping)] if isinstance(data, list) else []


def _entity_is_exact(candidate: Mapping[str, Any], ticker: str, company_name: str, aliases: Sequence[str]) -> bool:
    symbol = clean_text(candidate.get("symbol")).casefold()
    name = clean_text(candidate.get("name")).casefold()
    local_symbol = ticker.split(".", 1)[0].casefold()
    name_variants = {clean_text(item).casefold() for item in (company_name, *aliases) if clean_text(item)}
    country = clean_text(candidate.get("country")).casefold()
    exchange = clean_text(candidate.get("exchange") or candidate.get("exchange_long")).casefold()
    malaysia_listing = (
        country in {"my", "malaysia"}
        or "malaysia" in exchange
        or "bursa" in exchange
        or "myx" in exchange
        or "klse" in exchange
    )
    # A same-name foreign listing is not evidence for a Bursa company.  The
    # provider's entity search can return OTC/foreign symbols for Malaysian
    # issuers, so exact mappings require explicit Malaysian listing evidence.
    return malaysia_listing and (symbol in {ticker.casefold(), local_symbol} or name in name_variants)


def probe_marketaux_entity(
    ticker: str,
    company_name: str,
    aliases: Sequence[str],
    token: str,
    *,
    opener: Callable[..., Any] = urlopen,
) -> MarketauxEntityMatch:
    """Probe entity search once and never claim unsupported MYX coverage."""
    query = company_name or ticker
    payload, _ = _fetch_json(build_marketaux_entity_search_url(token, query), opener=opener)
    candidates = _entity_candidates(payload)
    exact = next((item for item in candidates if _entity_is_exact(item, ticker, company_name, aliases)), None)
    if exact is None:
        return MarketauxEntityMatch(
            ticker,
            company_name,
            None,
            None,
            None,
            None,
            "unsupported",
            "No exact Marketaux entity match",
        )
    return MarketauxEntityMatch(
        ticker,
        company_name,
        clean_text(exact.get("symbol")) or None,
        clean_text(exact.get("exchange")) or None,
        clean_text(exact.get("country")) or None,
        clean_text(exact.get("name")) or None,
        "supported",
    )


def probe_marketaux_entities(
    tickers: Sequence[str],
    *,
    token: str | None = None,
    opener: Callable[..., Any] = urlopen,
) -> Dict[str, Dict[str, Any]]:
    metadata = load_stock_metadata()
    api_token = token or get_marketaux_api_token()
    result: Dict[str, Dict[str, Any]] = {}
    for raw_ticker in tickers:
        ticker = clean_text(raw_ticker).upper()
        if ticker not in metadata:
            raise ValueError(f"Unsupported ticker: {ticker}")
        aliases = build_company_query_variants(ticker, metadata[ticker])
        try:
            match = probe_marketaux_entity(ticker, metadata[ticker], aliases, api_token, opener=opener)
            result[ticker] = match.to_dict()
        except MarketauxRequestError as exc:
            result[ticker] = MarketauxEntityMatch(
                ticker,
                metadata[ticker],
                None,
                None,
                None,
                None,
                "error",
                str(exc),
            ).to_dict()
    return result


def _marketaux_articles(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    data = payload.get("data")
    return [item for item in data if isinstance(item, Mapping)] if isinstance(data, list) else []


def _matching_entity(article: Mapping[str, Any], mapping: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    entities = article.get("entities")
    if not isinstance(entities, list):
        return None
    expected = clean_text((mapping or {}).get("symbol")).casefold()
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        if expected and clean_text(entity.get("symbol")).casefold() == expected:
            return entity
    return entities[0] if entities and isinstance(entities[0], Mapping) else None


def normalize_marketaux_rows(
    payload: Mapping[str, Any],
    query_variants: Sequence[str],
    *,
    entity_mapping: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for article in _marketaux_articles(payload):
        title = clean_text(article.get("title"))
        description = clean_text(article.get("description"))
        snippet = clean_text(article.get("snippet"))
        url = clean_text(article.get("url"))
        published_at = clean_text(article.get("published_at"))
        entity = _matching_entity(article, entity_mapping)
        matched_alias = match_relevant_alias(title, f"{description} {snippet}", query_variants)
        exact_symbol = bool(entity_mapping and clean_text(entity_mapping.get("symbol")) and entity)
        if not title or not url or not published_at or not (description or snippet):
            continue
        if not matched_alias and not exact_symbol:
            continue
        normalized_url = normalize_url(url)
        normalized_title = normalize_title(title)
        if normalized_url in seen_urls or normalized_title in seen_titles:
            continue
        seen_urls.add(normalized_url)
        seen_titles.add(normalized_title)
        text_completeness = (
            "title_description_snippet" if description and snippet else
            "title_description" if description else
            "title_snippet"
        )
        rows.append(
            {
                "matched_alias": matched_alias or clean_text((entity or {}).get("name")),
                "source_id": clean_text(article.get("source")),
                "source_name": clean_text(article.get("source")),
                "source_domain": clean_text(article.get("source")),
                "author": "",
                "title": title,
                "description": description or snippet,
                "url": url,
                "url_to_image": clean_text(article.get("image_url")),
                "published_at": published_at,
                "content_snippet": snippet,
                "provider": MARKETAUX_PROVIDER,
                "text_completeness": text_completeness,
                "provider_entity_symbol": clean_text((entity or {}).get("symbol")),
                "provider_entity_match_score": (entity or {}).get("match_score"),
                "provider_entity_sentiment_score": (entity or {}).get("sentiment_score"),
            }
        )
    return rows


def _save_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys([*ARTICLE_COLUMNS, "provider", "text_completeness", "source_domain", "provider_entity_symbol", "provider_entity_match_score", "provider_entity_sentiment_score"]))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _output_paths(ticker: str) -> Dict[str, Path]:
    root = Path(__file__).resolve().parent.parent / "data" / "financial-news-dataset"
    slug = slugify(ticker)
    return {
        "raw_json": root / "json" / f"marketaux-{slug}-raw.json",
        "filtered_articles": root / "csv" / f"marketaux-{slug}-filtered-articles.csv",
        "context_summary": root / "csv" / f"marketaux-{slug}-context-summary.csv",
        "dashboard_top5": root / "csv" / f"marketaux-{slug}-dashboard-top5.csv",
    }


def retrieve_marketaux_financial_news(
    ticker: str,
    as_of_date: date | str | None = None,
    *,
    token: str | None = None,
    entity_mapping: Mapping[str, Any] | None = None,
    response_data: Mapping[str, Any] | None = None,
    response_headers: Mapping[str, Any] | None = None,
    model: object | None = None,
    opener: Callable[..., Any] = urlopen,
    persist_outputs: bool = True,
    lookback_days: int = NEWS_RETRIEVAL_LOOKBACK_DAYS,
    published_after: date | None = None,
    published_before: date | None = None,
    historical_only: bool = False,
) -> NewsContextResult:
    ensure_data_structure()
    metadata = load_stock_metadata()
    normalized_ticker = clean_text(ticker).upper()
    if normalized_ticker not in metadata:
        raise ValueError(f"Unsupported ticker: {normalized_ticker}")
    company_name = metadata[normalized_ticker]
    effective_date = pd.Timestamp(as_of_date).date() if as_of_date is not None else date.today()
    variants = build_company_query_variants(normalized_ticker, company_name)
    api_token = token or get_marketaux_api_token()
    symbol = clean_text((entity_mapping or {}).get("symbol"))
    query_text = ",".join([symbol]) if symbol else build_newsapi_query(variants)
    if response_data is None:
        payload, headers = fetch_marketaux_response(
            api_token,
            symbols=[symbol] if symbol else None,
            search=None if symbol else query_text,
            as_of_date=effective_date,
            lookback_days=lookback_days,
            published_after=published_after,
            published_before=published_before,
            opener=opener,
        )
    else:
        payload = dict(response_data)
        headers = _safe_headers(response_headers)
    rows = normalize_marketaux_rows(payload, variants, entity_mapping=entity_mapping if symbol else None)
    if historical_only:
        # Keep the range invariant local to the adapter as API date filters are
        # inclusive/format-dependent across providers.
        rows = [
            row
            for row in rows
            if NEWS_HISTORICAL_CONTEXT_START_DAYS - 1 < article_age_days(
                str(row.get("published_at", "")), effective_date
            ) <= NEWS_HISTORICAL_CONTEXT_END_DAYS
        ]
        contexts = build_news_contexts([], effective_date)
    else:
        add_sentiment_scores(rows, model or load_sentiment_model())
        contexts = build_news_contexts(rows, effective_date)
    enriched_rows = [
        {
            "query_ticker": normalized_ticker,
            "query_company": company_name,
            "query_text": query_text,
            "display_rank": index,
            **row,
        }
        for index, row in enumerate(rows, start=1)
    ]
    paths = _output_paths(normalized_ticker) if persist_outputs else {}
    if persist_outputs:
        paths["raw_json"].parent.mkdir(parents=True, exist_ok=True)
        paths["raw_json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _save_rows(enriched_rows, paths["filtered_articles"])
        _save_rows(enriched_rows[:DASHBOARD_ARTICLES], paths["dashboard_top5"])
        pd.DataFrame(contexts.values()).to_csv(paths["context_summary"], index=False)
    returned = payload.get("meta", {}).get("returned", len(_marketaux_articles(payload))) if isinstance(payload.get("meta"), Mapping) else len(_marketaux_articles(payload))
    metadata_out = {
        **build_news_context_metadata(),
        "coverage_mode": "entity" if symbol else "alias_search",
        "entity_symbol": symbol or None,
        "entity_exchange": clean_text((entity_mapping or {}).get("exchange")) or None,
        "entity_country": clean_text((entity_mapping or {}).get("country")) or None,
        "marketaux_returned": int(returned or 0),
        "marketaux_headers": _safe_headers(headers),
        "cache_scope": MARKETAUX_HISTORICAL_SCOPE if historical_only else MARKETAUX_RECENT_SCOPE,
        "historical_only": historical_only,
    }
    return NewsContextResult(
        ticker=normalized_ticker,
        company_name=company_name,
        query_text=query_text,
        as_of_date=effective_date.isoformat(),
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        raw_article_count=int(payload.get("meta", {}).get("found", returned) or 0) if isinstance(payload.get("meta"), Mapping) else len(_marketaux_articles(payload)),
        relevant_article_count=0 if historical_only else len(enriched_rows),
        results_truncated=int(returned or 0) >= MARKETAUX_ARTICLE_LIMIT,
        contexts=contexts,
        articles=[] if historical_only else enriched_rows,
        output_paths={key: str(value) for key, value in paths.items()},
        provider_mode=MARKETAUX_PROVIDER,
        provider_article_counts={
            MARKETAUX_PROVIDER: 0 if historical_only else len(enriched_rows),
            "marketaux_historical": len(enriched_rows) if historical_only else 0,
        },
        provider_status={MARKETAUX_PROVIDER: "ok"},
        provider_metadata={
            **metadata_out,
            "retrieval_source": "live",
            "marketaux_cache_state": "miss",
        },
        historical_articles=enriched_rows if historical_only else [],
    )


def _article_domain(article: Mapping[str, Any]) -> str:
    domain = clean_text(article.get("source_domain"))
    if domain:
        return domain.casefold().removeprefix("www.")
    return clean_text(urlparse(clean_text(article.get("url"))).netloc).casefold().removeprefix("www.")


def _article_identity(article: Mapping[str, Any]) -> str:
    normalized_url = normalize_url(clean_text(article.get("url")))
    if normalized_url:
        return f"url:{normalized_url}"
    normalized_title = normalize_title(clean_text(article.get("title")))
    return f"title:{normalized_title}" if normalized_title else ""


def _article_dates_are_close(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    try:
        first_date = pd.Timestamp(first.get("published_at"), tz="UTC")
        second_date = pd.Timestamp(second.get("published_at"), tz="UTC")
        return abs((first_date - second_date).total_seconds()) <= 48 * 60 * 60
    except (TypeError, ValueError):
        return False


def _find_duplicate_article(
    article: Mapping[str, Any],
    merged: Sequence[Mapping[str, Any]],
) -> int | None:
    identity = _article_identity(article)
    if identity:
        for index, existing in enumerate(merged):
            if identity == _article_identity(existing):
                return index
    title = normalize_title(clean_text(article.get("title")))
    if not title:
        return None
    for index, existing in enumerate(merged):
        existing_title = normalize_title(clean_text(existing.get("title")))
        if existing_title == title and _article_dates_are_close(article, existing):
            return index
        if (
            existing_title
            and _article_domain(article) == _article_domain(existing)
            and _article_dates_are_close(article, existing)
            and SequenceMatcher(None, title, existing_title).ratio() >= 0.93
        ):
            return index
    return None


def _merge_news_results(
    newsapi_result: NewsContextResult,
    marketaux_result: NewsContextResult,
    *,
    trigger: str,
    marketaux_cache_state: str,
    retrieval_source: str = "live",
) -> NewsContextResult:
    merged: List[Dict[str, Any]] = []
    duplicate_count = 0
    for raw_article in [*newsapi_result.articles, *marketaux_result.articles]:
        article = dict(raw_article)
        provider = clean_text(article.get("provider")) or "newsapi"
        article["provider"] = provider
        article["source_domain"] = _article_domain(article)
        article["provider_sources"] = [provider]
        duplicate_index = _find_duplicate_article(article, merged)
        if duplicate_index is None:
            merged.append(article)
            continue
        duplicate_count += 1
        existing = merged[duplicate_index]
        sources = list(existing.get("provider_sources") or [])
        if provider not in sources:
            sources.append(provider)
        existing["provider_sources"] = sources
        if len(clean_text(article.get("description"))) > len(clean_text(existing.get("description"))):
            for key in ("description", "content_snippet", "author", "url_to_image"):
                if clean_text(article.get(key)):
                    existing[key] = article[key]

    for index, article in enumerate(merged, start=1):
        article["display_rank"] = index

    counts: Dict[str, int] = {}
    for article in merged:
        providers = article.get("provider_sources") or [clean_text(article.get("provider")) or "newsapi"]
        for provider in providers:
            provider_name = clean_text(provider) or "newsapi"
            counts[provider_name] = counts.get(provider_name, 0) + 1
    metadata = {
        **dict(newsapi_result.provider_metadata or {}),
        **build_news_context_metadata(),
        "retrieval_source": retrieval_source,
        "marketaux_used": True,
        "marketaux_trigger": trigger,
        "marketaux_cache_state": marketaux_cache_state,
        "deduplicated_article_count": duplicate_count,
        "provider_article_counts_before_merge": {
            "newsapi": newsapi_result.relevant_article_count,
            "marketaux": marketaux_result.relevant_article_count,
        },
        "marketaux_headers": dict((marketaux_result.provider_metadata or {}).get("marketaux_headers") or {}),
        "marketaux_usage": dict((marketaux_result.provider_metadata or {}).get("marketaux_usage") or {}),
    }
    return NewsContextResult(
        ticker=newsapi_result.ticker,
        company_name=newsapi_result.company_name,
        query_text=newsapi_result.query_text,
        as_of_date=newsapi_result.as_of_date,
        retrieved_at=max(newsapi_result.retrieved_at, marketaux_result.retrieved_at),
        raw_article_count=newsapi_result.raw_article_count + marketaux_result.raw_article_count,
        relevant_article_count=len(merged),
        results_truncated=newsapi_result.results_truncated or marketaux_result.results_truncated,
        contexts=build_news_contexts(merged, pd.Timestamp(newsapi_result.as_of_date).date()),
        articles=merged,
        output_paths=newsapi_result.output_paths,
        provider_mode="newsapi_marketaux_complement",
        provider_article_counts=counts,
        provider_status={"newsapi": "ok", "marketaux": "ok"},
        provider_metadata=metadata,
        historical_articles=[
            *list(newsapi_result.historical_articles or []),
            *list(marketaux_result.historical_articles or []),
        ],
    )


def _with_provider_metadata(
    result: NewsContextResult,
    *,
    provider_status: str,
    trigger: str | None = None,
    marketaux_used: bool = False,
    retrieval_source: str = "live",
    cache_state: str = "not_used",
    error: str | None = None,
) -> NewsContextResult:
    metadata = {
        **dict(result.provider_metadata or {}),
        **build_news_context_metadata(),
        "retrieval_source": retrieval_source,
        "marketaux_used": marketaux_used,
        "marketaux_cache_state": cache_state,
    }
    if trigger:
        metadata["marketaux_trigger"] = trigger
    if error:
        metadata["marketaux_error"] = error
    return NewsContextResult(
        ticker=result.ticker,
        company_name=result.company_name,
        query_text=result.query_text,
        as_of_date=result.as_of_date,
        retrieved_at=result.retrieved_at,
        raw_article_count=result.raw_article_count,
        relevant_article_count=result.relevant_article_count,
        results_truncated=result.results_truncated,
        contexts=result.contexts,
        articles=result.articles,
        output_paths=result.output_paths,
        provider_mode=result.provider_mode,
        provider_article_counts=result.provider_article_counts or {"newsapi": result.relevant_article_count},
        provider_status={"newsapi": "ok", "marketaux": provider_status},
        provider_metadata=metadata,
        historical_articles=list(result.historical_articles or []),
    )


def _with_historical_context(
    base_result: NewsContextResult | None,
    historical_result: NewsContextResult,
    *,
    cache_state: str,
    status: str = "ok",
    error: str | None = None,
) -> NewsContextResult:
    """Attach background articles without changing the current signal."""
    historical_articles = list(historical_result.historical_articles or [])
    if base_result is None:
        metadata = {
            **dict(historical_result.provider_metadata or {}),
            "retrieval_source": "historical_context",
            "marketaux_used": False,
            "historical_search_attempted": True,
            "historical_article_count": len(historical_articles),
            "historical_cache_state": cache_state,
            "historical_provider_status": status,
        }
        if error:
            metadata["historical_error"] = error
        return NewsContextResult(
            ticker=historical_result.ticker,
            company_name=historical_result.company_name,
            query_text=historical_result.query_text,
            as_of_date=historical_result.as_of_date,
            retrieved_at=historical_result.retrieved_at,
            raw_article_count=0,
            relevant_article_count=0,
            results_truncated=False,
            contexts=build_news_contexts([], pd.Timestamp(historical_result.as_of_date).date()),
            articles=[],
            output_paths={},
            provider_mode="marketaux_historical_context",
            provider_article_counts={"marketaux_historical": len(historical_articles)},
            provider_status={"newsapi": "unavailable", "marketaux": status},
            provider_metadata=metadata,
            historical_articles=historical_articles,
        )

    metadata = {
        **dict(base_result.provider_metadata or {}),
        "historical_search_attempted": True,
        "historical_article_count": len(historical_articles),
        "historical_cache_state": cache_state,
        "historical_provider_status": status,
    }
    if error:
        metadata["historical_error"] = error
    return NewsContextResult(
        ticker=base_result.ticker,
        company_name=base_result.company_name,
        query_text=base_result.query_text,
        as_of_date=base_result.as_of_date,
        retrieved_at=max(base_result.retrieved_at, historical_result.retrieved_at),
        raw_article_count=base_result.raw_article_count,
        relevant_article_count=base_result.relevant_article_count,
        results_truncated=base_result.results_truncated,
        contexts=base_result.contexts,
        articles=base_result.articles,
        output_paths=base_result.output_paths,
        provider_mode=base_result.provider_mode,
        provider_article_counts=base_result.provider_article_counts,
        provider_status={
            **dict(base_result.provider_status or {}),
            "marketaux_historical": status,
        },
        provider_metadata=metadata,
        historical_articles=historical_articles,
    )


def _retrieve_historical_context(
    base_result: NewsContextResult | None,
    ticker: str,
    effective_date: date,
    *,
    marketaux_token: str | None,
    marketaux_opener: Callable[..., Any],
) -> NewsContextResult | None:
    """Try the 31–90-day background search after a successful empty recent search."""
    cached = _read_marketaux_cache(
        ticker,
        effective_date,
        scope=MARKETAUX_HISTORICAL_SCOPE,
        allow_stale=False,
    )
    if cached is not None:
        historical_result, cache_info = cached
        return _with_historical_context(
            base_result,
            historical_result,
            cache_state=cache_info["cache_state"],
            status="cached",
        )

    try:
        token = marketaux_token or get_marketaux_api_token()
        _reserve_marketaux_request()
        historical_result = retrieve_marketaux_financial_news(
            ticker,
            as_of_date=effective_date,
            token=token,
            opener=marketaux_opener,
            persist_outputs=False,
            lookback_days=NEWS_HISTORICAL_CONTEXT_END_DAYS,
            published_after=effective_date - timedelta(days=NEWS_HISTORICAL_CONTEXT_END_DAYS),
            # Include the boundary calendar day in the provider query; the
            # local age filter below is the final authority for 31–90 days.
            published_before=effective_date - timedelta(days=NEWS_RETRIEVAL_LOOKBACK_DAYS - 1),
            historical_only=True,
        )
        usage = _record_marketaux_response(
            historical_result.provider_metadata.get("marketaux_headers"),
            "ok",
        )
        historical_result.provider_metadata.update(
            {
                "marketaux_usage": usage,
                "retrieval_source": "historical_context",
                "marketaux_cache_state": "fresh",
                "cache_scope": MARKETAUX_HISTORICAL_SCOPE,
            }
        )
        captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_marketaux_cache(historical_result, captured_at)
        return _with_historical_context(
            base_result,
            historical_result,
            cache_state="fresh",
            status="ok",
        )
    except Exception as exc:
        if isinstance(exc, FinBERTRuntimeError):
            raise
        if isinstance(exc, MarketauxRequestError) and exc.status is not None:
            _record_marketaux_response(exc.headers, f"error_{exc.status or 'request'}")
        if base_result is not None:
            return _with_historical_context(
                base_result,
                NewsContextResult(
                    ticker=base_result.ticker,
                    company_name=base_result.company_name,
                    query_text=base_result.query_text,
                    as_of_date=base_result.as_of_date,
                    retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    raw_article_count=0,
                    relevant_article_count=0,
                    results_truncated=False,
                    contexts={},
                    articles=[],
                    output_paths={},
                    provider_metadata=build_news_context_metadata(),
                    historical_articles=[],
                ),
                cache_state="unavailable",
                status="unavailable",
                error=str(exc),
            )
        return None


def retrieve_news_with_marketaux_fallback(
    ticker: str,
    horizon_days: int,
    as_of_date: date | str | None = None,
    *,
    newsapi_result: NewsContextResult | None = None,
    marketaux_token: str | None = None,
    marketaux_opener: Callable[..., Any] = urlopen,
) -> NewsContextResult:
    """Retrieve NewsAPI evidence, conditionally enriching it with Marketaux.

    Marketaux is not queried for medium/high NewsAPI coverage. A cached
    Marketaux response is preferred to a new request, and all network calls
    pass through the local daily quota guard.
    """
    if horizon_days not in CONTEXT_WINDOWS:
        raise ValueError(f"Unsupported news horizon: {horizon_days}")
    effective_date = pd.Timestamp(as_of_date).date() if as_of_date is not None else date.today()
    newsapi_error: Exception | None = None
    news = newsapi_result
    if news is None:
        try:
            from pipeline.download_financial_news_data import retrieve_financial_news

            news = retrieve_financial_news(ticker, as_of_date=effective_date)
        except FinBERTRuntimeError:
            # This is a shared classifier/runtime failure, not a NewsAPI
            # retrieval failure. Trying Marketaux would waste quota and fail
            # through the same FinBERT dependency path.
            raise
        except Exception as exc:
            newsapi_error = exc

    trigger: str | None = None
    marketaux_recent_empty = False
    if newsapi_error is not None:
        trigger = "newsapi_failure"
    elif news is not None:
        context = news.contexts.get(str(horizon_days), {})
        coverage = str(context.get("coverage_confidence", "Unavailable"))
        if coverage in {"Low", "Unavailable"}:
            trigger = f"newsapi_{coverage.casefold()}_coverage"
        else:
            return _with_provider_metadata(news, provider_status="skipped")

    cached_fresh = _read_marketaux_cache(ticker, effective_date, allow_stale=False)
    if cached_fresh is not None:
        marketaux_result, cache_info = cached_fresh
        if marketaux_result.relevant_article_count > 0:
            if news is not None:
                return _merge_news_results(
                    news,
                    marketaux_result,
                    trigger=trigger or "newsapi_low_coverage",
                    marketaux_cache_state="fresh",
                )
            return _with_provider_metadata(
                marketaux_result,
                provider_status="cached",
                marketaux_used=True,
                retrieval_source="fallback",
                cache_state=cache_info["cache_state"],
            )
        marketaux_recent_empty = True
        if news is None or news.relevant_article_count <= 0:
            historical = _retrieve_historical_context(
                news,
                ticker,
                effective_date,
                marketaux_token=marketaux_token,
                marketaux_opener=marketaux_opener,
            )
            if historical is not None:
                return historical
        if news is not None:
            return _with_provider_metadata(
                news,
                provider_status="cached_empty",
                trigger=trigger,
                error="Marketaux returned no relevant articles in the 30-day window",
            )
        # Preserve the original provider error path when NewsAPI also failed.

    marketaux_error: Exception | None = None
    marketaux_result: NewsContextResult | None = None
    try:
        token = marketaux_token or get_marketaux_api_token()
        _reserve_marketaux_request()
        marketaux_result = retrieve_marketaux_financial_news(
            ticker,
            as_of_date=effective_date,
            token=token,
            opener=marketaux_opener,
            persist_outputs=False,
        )
        usage = _record_marketaux_response(
            marketaux_result.provider_metadata.get("marketaux_headers"),
            "ok",
        )
        marketaux_result.provider_metadata.update(
            {
                "marketaux_usage": usage,
                "cache_scope": MARKETAUX_RECENT_SCOPE,
                "retrieval_source": "live",
                "marketaux_cache_state": "fresh",
            }
        )
        # Persist empty responses as well as positive responses. This prevents
        # repeated visits to a no-news ticker from consuming another quota unit.
        captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_marketaux_cache(marketaux_result, captured_at)
        if marketaux_result.relevant_article_count <= 0:
            marketaux_recent_empty = True
            marketaux_error = MarketauxRequestError("Marketaux returned no relevant articles", status=200)
            marketaux_result = None
    except Exception as exc:
        if isinstance(exc, FinBERTRuntimeError):
            raise
        marketaux_error = exc
        if isinstance(exc, MarketauxRequestError) and exc.status is not None:
            _record_marketaux_response(exc.headers, f"error_{exc.status or 'request'}")

    if marketaux_result is not None:
        if news is not None:
            return _merge_news_results(
                news,
                marketaux_result,
                trigger=trigger or "newsapi_low_coverage",
                marketaux_cache_state="fresh",
            )
        return _with_provider_metadata(
            marketaux_result,
            provider_status="ok",
            trigger=trigger,
            marketaux_used=True,
        )

    if marketaux_recent_empty and (news is None or news.relevant_article_count <= 0):
        historical = _retrieve_historical_context(
            news,
            ticker,
            effective_date,
            marketaux_token=marketaux_token,
            marketaux_opener=marketaux_opener,
        )
        if historical is not None:
            return historical

    if marketaux_recent_empty and news is not None and news.relevant_article_count > 0:
        return _with_provider_metadata(
            news,
            provider_status="empty_recent_window",
            trigger=trigger,
            error="Marketaux returned no relevant articles in the 30-day window",
        )

    cached_stale = _read_marketaux_cache(ticker, effective_date, allow_stale=True)
    if cached_stale is not None:
        stale_result, cache_info = cached_stale
        if news is not None:
            return _merge_news_results(
                news,
                stale_result,
                trigger=trigger or "newsapi_low_coverage",
                marketaux_cache_state="stale",
            )
        return _with_provider_metadata(
            stale_result,
            provider_status="stale_cache",
            trigger=trigger,
            marketaux_used=True,
            retrieval_source="fallback",
            cache_state=cache_info["cache_state"],
            error=str(marketaux_error) if marketaux_error else None,
        )

    if news is not None:
        return _with_provider_metadata(
            news,
            provider_status=("quota_exhausted" if isinstance(marketaux_error, MarketauxQuotaError) else "unavailable"),
            trigger=trigger,
            error=str(marketaux_error) if marketaux_error else None,
        )
    if newsapi_error is not None:
        detail = f"NewsAPI failed: {newsapi_error}"
        if marketaux_error is not None:
            detail += f"; Marketaux fallback failed: {marketaux_error}"
        raise RuntimeError(detail) from marketaux_error or newsapi_error
    raise RuntimeError(f"No news evidence available for {ticker}")
