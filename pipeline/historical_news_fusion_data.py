"""Provider-neutral historical-news data contract for fusion experiments.

This module is deliberately an offline, additive scaffold.  It does not make
provider requests, train a stacker, or change the runtime news service.  Once a
lawful point-in-time historical news export is available, callers can use the
three public operations below to:

* normalize and partition the export as Parquet;
* report whether the data is sufficient for leakage-safe fusion experiments;
* build one point-in-time feature row per stock observation and horizon.

The resulting feature table contains news features only from articles that
were published by the stock observation cutoff.  Missing news remains an
explicit ``news_available`` flag rather than being converted into neutral
sentiment.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd

from pipeline.download_financial_news_data import (
    CONTEXT_HALF_LIVES,
    CONTEXT_WINDOWS,
    SENTIMENT_SCORE_MAP,
    add_sentiment_scores,
    calculate_recency_weight,
    clean_text,
    load_sentiment_model,
    load_stock_metadata,
    stable_event_id,
)
from pipeline.fusion_model import EVIDENCE_STRENGTH_SCORES
from pipeline.official_disclosure_news import classify_material_event
from pipeline.stock_features import TARGET_HORIZONS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FUSION_DATA_DIR = DATA_DIR / "historical-news-fusion"
FUSION_SCHEMA_VERSION = "historical-news-fusion-v3"
FUSION_FEATURE_SCHEMA_VERSION = "point-in-time-news-features-v4"
DEFAULT_MINIMUM_DATE = pd.Timestamp("2019-01-01")
DEFAULT_MAXIMUM_DATE = pd.Timestamp("2026-12-31")
DEFAULT_MIN_NEWS_PRESENT_OBSERVATIONS = 100
DEFAULT_MIN_SOURCE_DOMAINS = 2
LOCAL_TIMEZONE = "Asia/Kuala_Lumpur"
MARKET_CUTOFF_HOUR = 17
MATERIAL_EVENT_CATEGORIES = (
    "financial_results_guidance",
    "dividends_distributions",
    "contracts_order_wins",
    "acquisitions_disposals_investments",
    "financing_capital_changes",
    "regulatory_legal_governance",
    "operational_product_strategy",
    "management_changes",
)

_LEGACY_EVENT_CATEGORY_MAP = {
    "earnings": "financial_results_guidance",
    "guidance": "financial_results_guidance",
    "contract": "contracts_order_wins",
    "regulatory_legal": "regulatory_legal_governance",
    "operations": "operational_product_strategy",
    "product_service": "operational_product_strategy",
    "esg": "operational_product_strategy",
    "management": "management_changes",
    "financing": "financing_capital_changes",
}


def canonical_material_event_category(
    category: Any, title: Any = "", description: Any = ""
) -> str:
    """Translate reviewed legacy labels into the frozen v3 taxonomy.

    ``corporate_action`` is intentionally resolved from its text because it
    can mean a dividend, acquisition or capital change.  Broad ``other`` and
    macro-only labels remain unclassified rather than being forced into a
    material category.
    """

    value = clean_text(category).casefold()
    if value in MATERIAL_EVENT_CATEGORIES:
        return value
    mapped = _LEGACY_EVENT_CATEGORY_MAP.get(value)
    if mapped:
        return mapped
    classified = classify_material_event(title, description)
    if classified in MATERIAL_EVENT_CATEGORIES:
        return classified
    return "unclassified"


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _sentiment_runtime_identity(config_path: Path) -> dict[str, str | None]:
    """Return stable identities for the active runtime sentiment variant."""

    if not config_path.is_file():
        return {
            "config_sha256": None,
            "variant": None,
            "variant_sha256": None,
            "component_artifacts_sha256": None,
        }
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = {}
    variant = str(payload.get("active_variant") or "").strip() or None
    challenger = dict((payload.get("challengers") or {}).get(variant) or {})
    component_identity = {
        "top_level_artifact_hashes": payload.get("artifact_hashes") or {},
        "components": challenger.get("components") or {},
        "weights": challenger.get("weights") or {},
        "ensemble_temperature": challenger.get("ensemble_temperature"),
        "text_mode": challenger.get("text_mode"),
    }
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "variant": variant,
        "variant_sha256": _canonical_json_hash(challenger) if challenger else None,
        "component_artifacts_sha256": _canonical_json_hash(component_identity),
    }


def _canonical_label_source(value: Any) -> str:
    source = clean_text(value).casefold().replace("-", "_").replace(" ", "_")
    if source in {
        "external_single_weak",
        "external_consensus",
        "user_or_codex_single_label",
        "user_adjudicated_after_llm_abstention",
        "single",
        "single_label",
    }:
        return "llm_assisted_label"
    return source

NORMALIZED_COLUMNS = (
    "article_id",
    "ticker",
    "company_name",
    "provider",
    "source_name",
    "source_domain",
    "published_at",
    "retrieved_at",
    "title",
    "description",
    "content_snippet",
    "url",
    "sentiment_label",
    "sentiment_score",
    "sentiment_probability",
    "probability_negative",
    "probability_neutral",
    "probability_positive",
    "sentiment_uncertain",
    "gdelt_organizations",
    "commoncrawl_index",
    "commoncrawl_capture_at",
    "commoncrawl_capture_lag_days",
    "content_sha256",
    "text_source",
    "is_pseudo_document",
    "recovery_status",
    "recovery_failure_reason",
    "recovered_at",
    "ngram_file_timestamps",
    "ngram_record_count",
    "deciles_present",
    "unique_token_count",
    "overlap_attachment_rate",
    "reconstruction_quality_score",
    "reconstruction_quality_tier",
    "publisher_full_text_available",
    "automatic_relevance",
    "relevance_score",
    "relevance_reasons",
    "event_category",
    "disclosure_type",
    "entity_confidence",
    "text_quality_score",
    "sentiment_model_id",
    "sentiment_runtime_config_sha256",
    "sentiment_model_variant_sha256",
    "sentiment_component_artifacts_sha256",
    "sentiment_scored_at",
    "label_validation_status",
    "llm_assisted_label",
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "article_id": ("article_id", "id", "source_id", "source.id", "guid", "uuid"),
    "ticker": ("ticker", "query_ticker", "symbol", "stock_ticker"),
    "company_name": ("company_name", "query_company", "company", "issuer"),
    "provider": ("provider", "source_provider"),
    "source_name": ("source_name", "source.name", "source", "publisher", "site_name"),
    "source_domain": ("source_domain", "domain", "website"),
    "published_at": ("published_at", "publishedAt", "published", "date", "datetime", "timestamp"),
    "retrieved_at": ("retrieved_at", "retrievedAt", "collected_at", "collection_timestamp"),
    "title": ("title", "headline", "article_title"),
    "description": ("description", "summary", "snippet", "content", "text"),
    "content_snippet": (
        "content_snippet",
        "content",
        "body",
        "article_content",
        "article_text",
    ),
    "url": ("url", "link", "article_url", "url_to_article", "source_url"),
    "sentiment_label": ("sentiment_label", "sentiment", "label", "tone"),
    "sentiment_score": ("sentiment_score", "score", "tone_score"),
    "sentiment_probability": ("sentiment_probability", "confidence", "sentiment_confidence"),
    "probability_negative": ("probability_negative", "negative_probability", "negative_prob"),
    "probability_neutral": ("probability_neutral", "neutral_probability", "neutral_prob"),
    "probability_positive": ("probability_positive", "positive_probability", "positive_prob"),
    "sentiment_uncertain": ("sentiment_uncertain", "uncertain", "low_confidence"),
    "gdelt_organizations": ("gdelt_organizations", "V2Organizations"),
    "commoncrawl_index": ("commoncrawl_index",),
    "commoncrawl_capture_at": ("commoncrawl_capture_at",),
    "commoncrawl_capture_lag_days": ("commoncrawl_capture_lag_days",),
    "content_sha256": ("content_sha256",),
    "text_source": ("text_source",),
    "is_pseudo_document": ("is_pseudo_document",),
    "recovery_status": ("recovery_status",),
    "recovery_failure_reason": ("recovery_failure_reason",),
    "recovered_at": ("recovered_at",),
    "ngram_file_timestamps": ("ngram_file_timestamps",),
    "ngram_record_count": ("ngram_record_count",),
    "deciles_present": ("deciles_present",),
    "unique_token_count": ("unique_token_count",),
    "overlap_attachment_rate": ("overlap_attachment_rate",),
    "reconstruction_quality_score": ("reconstruction_quality_score",),
    "reconstruction_quality_tier": ("reconstruction_quality_tier",),
    "publisher_full_text_available": ("publisher_full_text_available",),
    "automatic_relevance": ("automatic_relevance",),
    "relevance_score": ("relevance_score",),
    "relevance_reasons": ("relevance_reasons",),
    "event_category": ("event_category", "announcement_category", "material_event_category"),
    "disclosure_type": ("disclosure_type", "source_type", "document_type"),
    "entity_confidence": ("entity_confidence", "entity_relevance_confidence"),
    "text_quality_score": ("text_quality_score", "reconstruction_quality_score"),
    "sentiment_model_id": ("sentiment_model_id", "sentiment_model_variant"),
    "sentiment_runtime_config_sha256": ("sentiment_runtime_config_sha256",),
    "sentiment_model_variant_sha256": ("sentiment_model_variant_sha256",),
    "sentiment_component_artifacts_sha256": ("sentiment_component_artifacts_sha256",),
    "sentiment_scored_at": ("sentiment_scored_at",),
    "label_validation_status": ("label_validation_status",),
    "llm_assisted_label": ("llm_assisted_label",),
}


@dataclass(frozen=True)
class ImportResult:
    """Immutable summary of a normalized historical-news import."""

    output_dir: Path
    manifest_path: Path
    rows: int
    tickers: tuple[str, ...]
    providers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "rows": self.rows,
            "tickers": list(self.tickers),
            "providers": list(self.providers),
        }


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", clean_text(value).casefold()).strip("-") or "unknown"


def _normalise_ticker(value: Any) -> str:
    return clean_text(value).upper()


def _normalise_url(value: Any) -> str:
    raw = clean_text(value)
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        query = "&".join(
            item for item in parsed.query.split("&")
            if item and not item.casefold().startswith(("utm_", "fbclid=", "gclid="))
        )
        return urlunparse((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path.rstrip("/"), "", query, ""))
    except ValueError:
        return raw


def _parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialize mixed list/dict provenance columns deterministically.

    Recovery exports sometimes represent a provenance field as a JSON string
    in one row and a decoded list in another.  Arrow requires one physical
    type per column, so complex values are encoded at the Parquet boundary.
    """

    output = frame.copy()
    for column in output.select_dtypes(include=["object", "string"]).columns:
        values = output[column].dropna()
        contains_complex = values.map(
            lambda value: isinstance(value, (list, tuple, dict, set))
        ).any()
        type_count = len({type(value) for value in values})
        if contains_complex:
            output[column] = output[column].map(
                lambda value: json.dumps(
                    sorted(value) if isinstance(value, set) else value,
                    ensure_ascii=False,
                    default=str,
                )
                if isinstance(value, (list, tuple, dict, set))
                else clean_text(value)
            )
        elif type_count > 1:
            output[column] = output[column].map(clean_text)
    return output


def _first_column(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    for alias in aliases:
        if alias in frame.columns:
            return frame[alias]
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def _source_domain(url: Any, value: Any) -> str:
    explicit = clean_text(value).casefold().removeprefix("www.")
    if explicit:
        return explicit
    raw = clean_text(url)
    try:
        return clean_text(urlparse(raw).netloc).casefold().removeprefix("www.")
    except ValueError:
        return ""


def _stable_article_id(row: Mapping[str, Any]) -> str:
    candidate = clean_text(row.get("article_id"))
    if candidate:
        return candidate
    basis = "|".join(
        clean_text(row.get(key)).casefold()
        for key in ("ticker", "published_at", "title", "url")
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest() if basis.strip("|") else ""


def _read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            records = payload.get("articles") or payload.get("data") or payload.get("results")
            if isinstance(records, list):
                return pd.json_normalize(records)
            result = payload.get("result")
            if isinstance(result, dict) and isinstance(result.get("articles"), list):
                return pd.json_normalize(result["articles"])
            return pd.json_normalize([payload])
    raise ValueError(f"Unsupported historical-news file type: {path}")


def _coerce_boolean(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "yes", "y", "t"}
    return series.map(lambda value: str(value).strip().casefold() in truthy if pd.notna(value) else False)


def normalize_historical_news_frame(
    frame: pd.DataFrame,
    *,
    provider: str | None = None,
    ticker_aliases: Mapping[str, str] | None = None,
    default_company_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize provider exports into the versioned historical-news schema.

    Ticker and publication timestamp are mandatory.  Rows without article
    text are dropped because they cannot be scored by the sentiment backend.
    Existing sentiment columns are retained when supplied; this function does
    not run a sentiment model.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    aliases = ticker_aliases or {}
    defaults = default_company_names or {}
    output = pd.DataFrame(index=frame.index)
    for canonical, names in _ALIASES.items():
        output[canonical] = _first_column(frame, names)
    if provider:
        output["provider"] = clean_text(provider)
    output["ticker"] = output["ticker"].map(_normalise_ticker).replace(aliases)
    output["company_name"] = output["company_name"].map(clean_text)
    output["company_name"] = output.apply(
        lambda row: row["company_name"] or clean_text(defaults.get(row["ticker"])), axis=1
    )
    output["provider"] = output["provider"].map(clean_text).replace("", "unknown")
    output["source_name"] = output["source_name"].map(clean_text)
    output["url"] = output["url"].map(_normalise_url)
    output["source_domain"] = [
        _source_domain(url, domain)
        for url, domain in zip(output["url"], output["source_domain"])
    ]
    # The corpus can combine ISO timestamps from provider exports with the
    # space-separated timestamps in the frozen historical partitions.  Pandas
    # otherwise infers one format from the first row and silently coerces the
    # other valid rows to NaT, dropping newly appended disclosures.
    output["published_at"] = pd.to_datetime(
        output["published_at"], utc=True, errors="coerce", format="mixed"
    )
    output["retrieved_at"] = pd.to_datetime(
        output["retrieved_at"], utc=True, errors="coerce", format="mixed"
    )
    output["title"] = output["title"].map(clean_text)
    output["description"] = output["description"].map(clean_text)
    output["content_snippet"] = output["content_snippet"].map(clean_text)
    output["article_id"] = [
        _stable_article_id(row)
        for row in output.to_dict(orient="records")
    ]
    output["sentiment_label"] = output["sentiment_label"].map(lambda value: clean_text(value).casefold())
    output["sentiment_score"] = pd.to_numeric(output["sentiment_score"], errors="coerce")
    output["sentiment_score"] = output["sentiment_score"].where(
        output["sentiment_score"].notna(),
        output["sentiment_label"].map(SENTIMENT_SCORE_MAP),
    )
    output["sentiment_probability"] = pd.to_numeric(output["sentiment_probability"], errors="coerce")
    for column in ("probability_negative", "probability_neutral", "probability_positive"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["sentiment_uncertain"] = _coerce_boolean(output["sentiment_uncertain"])
    output["sentiment_uncertain"] = output["sentiment_uncertain"] | output["sentiment_probability"].lt(0.55).fillna(True)
    output["gdelt_organizations"] = output["gdelt_organizations"].map(clean_text)
    output["commoncrawl_index"] = output["commoncrawl_index"].map(clean_text)
    output["commoncrawl_capture_at"] = pd.to_datetime(
        output["commoncrawl_capture_at"], utc=True, errors="coerce", format="mixed"
    )
    output["commoncrawl_capture_lag_days"] = pd.to_numeric(
        output["commoncrawl_capture_lag_days"], errors="coerce"
    )
    output["content_sha256"] = output["content_sha256"].map(clean_text)
    output["automatic_relevance"] = output["automatic_relevance"].map(clean_text)
    output["relevance_score"] = pd.to_numeric(output["relevance_score"], errors="coerce")
    output["relevance_reasons"] = output["relevance_reasons"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, tuple))
        else clean_text(value)
    )
    output["event_category"] = output["event_category"].map(clean_text).str.casefold()
    output["disclosure_type"] = output["disclosure_type"].map(clean_text).str.casefold()
    output["entity_confidence"] = pd.to_numeric(
        output["entity_confidence"], errors="coerce"
    ).clip(0.0, 1.0)
    output["text_quality_score"] = pd.to_numeric(
        output["text_quality_score"], errors="coerce"
    ).clip(0.0, 1.0)
    output["sentiment_model_id"] = output["sentiment_model_id"].map(clean_text)
    for column in (
        "sentiment_runtime_config_sha256",
        "sentiment_model_variant_sha256",
        "sentiment_component_artifacts_sha256",
        "label_validation_status",
    ):
        output[column] = output[column].map(clean_text)
    output["sentiment_scored_at"] = pd.to_datetime(
        output["sentiment_scored_at"], utc=True, errors="coerce", format="mixed"
    )
    output["llm_assisted_label"] = _coerce_boolean(output["llm_assisted_label"])
    output["content_snippet"] = output["content_snippet"].where(
        output["content_snippet"].ne(""), output["description"]
    )
    required = output["ticker"].ne("") & output["published_at"].notna() & (output["title"].ne("") | output["description"].ne(""))
    output = output.loc[required, list(NORMALIZED_COLUMNS)].copy()
    output["ticker"] = output["ticker"].astype(str)
    output = output[output["article_id"].ne("")]
    output = output.drop_duplicates(["ticker", "article_id"], keep="last")
    return output.sort_values(["ticker", "published_at", "article_id"]).reset_index(drop=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_historical_news(
    source_paths: Path | str | Sequence[Path | str],
    *,
    output_dir: Path = DEFAULT_FUSION_DATA_DIR,
    provider: str | None = None,
    ticker_aliases: Mapping[str, str] | None = None,
    default_company_names: Mapping[str, str] | None = None,
) -> ImportResult:
    """Normalize source files and write ticker/year-partitioned Parquet.

    The operation is intentionally explicit: no source paths means an error,
    and no generated sample rows are written.  Existing partitions with the
    same ticker/year are replaced by the normalized import.
    """
    if isinstance(source_paths, (str, Path)):
        paths = [Path(source_paths)]
    else:
        paths = [Path(item) for item in source_paths]
    if not paths:
        raise ValueError("At least one historical-news source path is required")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Historical-news source files not found: {missing}")
    if default_company_names is not None:
        stock_metadata = default_company_names
    else:
        try:
            stock_metadata = load_stock_metadata()
        except FileNotFoundError:
            # Provider exports can be normalized independently of the local
            # ten-company metadata file; company_name is optional in the
            # normalized contract.
            stock_metadata = {}
    frames = [normalize_historical_news_frame(_read_source(path), provider=provider, ticker_aliases=ticker_aliases, default_company_names=stock_metadata) for path in paths]
    normalized = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NORMALIZED_COLUMNS)
    normalized = normalized.drop_duplicates(["ticker", "article_id"], keep="last")
    output_dir = Path(output_dir)
    partition_root = output_dir / "normalized"
    partition_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not normalized.empty:
        normalized["_year"] = normalized["published_at"].dt.year.astype(int)
        for (ticker, year), partition in normalized.groupby(["ticker", "_year"], sort=True):
            path = partition_root / f"ticker={_slug(ticker)}" / f"year={year}" / "articles.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            _parquet_safe_frame(
                partition.drop(columns=["_year"])
            ).to_parquet(path, index=False)
            written.append(path)
        normalized = normalized.drop(columns=["_year"])
    manifest = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "normalized_columns": list(NORMALIZED_COLUMNS),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [{"path": str(path), "sha256": _file_sha256(path)} for path in paths],
        "row_count": int(len(normalized)),
        "tickers": sorted(normalized["ticker"].dropna().unique().tolist()),
        "providers": sorted(normalized["provider"].dropna().unique().tolist()),
        "published_min": normalized["published_at"].min().isoformat() if not normalized.empty else None,
        "published_max": normalized["published_at"].max().isoformat() if not normalized.empty else None,
        "partitions": [str(path.relative_to(output_dir)) for path in written],
    }
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return ImportResult(output_dir, manifest_path, len(normalized), tuple(manifest["tickers"]), tuple(manifest["providers"]))


def load_normalized_historical_news(data_dir: Path = DEFAULT_FUSION_DATA_DIR) -> pd.DataFrame:
    """Load all normalized Parquet partitions written by the import operation."""
    paths = sorted((Path(data_dir) / "normalized").glob("ticker=*/year=*/articles.parquet"))
    if not paths:
        raise FileNotFoundError(f"No normalized historical-news partitions found in {Path(data_dir) / 'normalized'}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    return frame.sort_values(["ticker", "published_at", "article_id"]).reset_index(drop=True)


def _date_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(LOCAL_TIMEZONE).tz_localize(None)
    return timestamp.normalize()


def _cutoff_timestamp(observation_date: Any) -> pd.Timestamp:
    local = _date_timestamp(observation_date) + pd.Timedelta(hours=MARKET_CUTOFF_HOUR)
    return local.tz_localize(LOCAL_TIMEZONE).tz_convert("UTC")


def _age_days(published_at: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    return ((cutoff - pd.to_datetime(published_at, utc=True, errors="coerce")) / pd.Timedelta(days=1)).clip(lower=0)


def _score_news(
    rows: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    *,
    window_days: int | None = None,
    half_life_days: float | None = None,
) -> dict[str, Any]:
    """Aggregate historical articles with the same event cap as runtime."""

    window = int(window_days if window_days is not None else CONTEXT_WINDOWS[horizon])
    half_life = float(
        half_life_days if half_life_days is not None else CONTEXT_HALF_LIVES[horizon]
    )
    candidates = rows.loc[rows["published_at"].le(cutoff)].copy()
    candidates["age_days"] = _age_days(candidates["published_at"], cutoff)
    if window == 0:
        # "Current session" means information arriving after the preceding
        # Bursa close and no later than this observation's 17:00 MYT cutoff.
        # A simple age<=0 test would incorrectly discard the entire session.
        from pipeline.market_calendar import previous_market_session

        local_session = cutoff.tz_convert(LOCAL_TIMEZONE).date()
        previous_session = previous_market_session(local_session)
        previous_cutoff = _cutoff_timestamp(previous_session)
        candidates = candidates.loc[candidates["published_at"].gt(previous_cutoff)].copy()
    else:
        candidates = candidates.loc[candidates["age_days"].le(window)].copy()
    usable = candidates.loc[
        (~candidates["sentiment_uncertain"].astype(bool))
        & candidates["sentiment_score"].isin([-1, 0, 1])
        & pd.to_numeric(candidates["sentiment_probability"], errors="coerce").notna()
    ].copy()
    if not usable.empty:
        usable["_event_id"] = [
            stable_event_id(row) for row in usable.to_dict(orient="records")
        ]
    source_series = usable["source_domain"].where(usable["source_domain"].ne(""), usable["source_name"])
    sources = {clean_text(item).casefold() for item in source_series if clean_text(item)}
    events = {
        clean_text(item)
        for item in usable.get("_event_id", pd.Series(dtype=str))
        if clean_text(item)
    }
    event_rows_summary: list[dict[str, Any]] = []
    if usable.empty:
        score = None
    else:
        usable["recency_weight"] = usable["age_days"].map(lambda age: calculate_recency_weight(float(age), half_life))
        confidence = pd.to_numeric(
            usable["sentiment_probability"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        entity_confidence = pd.to_numeric(
            usable.get("entity_confidence", pd.Series(1.0, index=usable.index)),
            errors="coerce",
        ).fillna(1.0).clip(0.0, 1.0)
        text_quality = pd.to_numeric(
            usable.get("text_quality_score", pd.Series(1.0, index=usable.index)),
            errors="coerce",
        ).fillna(1.0).clip(0.0, 1.0)
        usable["fusion_weight"] = (
            usable["recency_weight"]
            * confidence
            * entity_confidence
            * text_quality
        )
        event_weight_total = 0.0
        event_weighted_score = 0.0
        for event_id, event_rows in usable.groupby("_event_id", sort=False):
            # A syndicated event is one information event, not one vote per
            # copied article.  Retain the strongest independent representation
            # for both its weight and tone so duplicates cannot move the score.
            strongest = event_rows.loc[event_rows["fusion_weight"].idxmax()]
            event_weight = float(strongest["fusion_weight"])
            event_score = float(strongest["sentiment_score"])
            event_weight_total += event_weight
            event_weighted_score += event_score * event_weight
            event_rows_summary.append(
                {
                    "event_id": str(event_id),
                    "weight": event_weight,
                    "score": event_score,
                    "confidence": float(confidence.loc[strongest.name]),
                    "entity_confidence": float(
                        entity_confidence.loc[strongest.name]
                    ),
                    "text_quality": float(text_quality.loc[strongest.name]),
                    "age_days": float(strongest["age_days"]),
                    "category": clean_text(strongest.get("event_category")).casefold(),
                    "disclosure_type": clean_text(strongest.get("disclosure_type")).casefold(),
                }
            )
        score = (
            float(event_weighted_score / event_weight_total)
            if event_weight_total > 0
            else None
        )
    label_counts = usable["sentiment_label"].value_counts() if not usable.empty else pd.Series(dtype=int)
    coverage = (
        "High"
        if len(usable) >= 5 and len(sources) >= 3 and len(events) >= 3
        else "Medium"
        if len(usable) >= 3 and len(sources) >= 2 and len(events) >= 2
        else "Low"
        if len(usable)
        else "Unavailable"
    )
    event_weight_total = sum(float(item["weight"]) for item in event_rows_summary)
    category_shares = {
        category: (
            sum(
                float(item["weight"])
                for item in event_rows_summary
                if item["category"] == category
            )
            / event_weight_total
            if event_weight_total > 0
            else 0.0
        )
        for category in MATERIAL_EVENT_CATEGORIES
    }
    official_weight = sum(
        float(item["weight"])
        for item in event_rows_summary
        if item["disclosure_type"] in {
            "official",
            "official_disclosure",
            "bursa_announcement",
            "issuer_ir",
            "press_release",
        }
    )
    event_count = len(event_rows_summary)
    positive_event_weight = sum(
        float(item["weight"])
        for item in event_rows_summary
        if float(item["score"]) > 0
    )
    neutral_event_weight = sum(
        float(item["weight"])
        for item in event_rows_summary
        if float(item["score"]) == 0
    )
    negative_event_weight = sum(
        float(item["weight"])
        for item in event_rows_summary
        if float(item["score"]) < 0
    )
    event_confidences = [float(item["confidence"]) for item in event_rows_summary]
    event_entity_confidences = [
        float(item["entity_confidence"]) for item in event_rows_summary
    ]
    event_text_quality = [float(item["text_quality"]) for item in event_rows_summary]
    event_ages = [float(item["age_days"]) for item in event_rows_summary]
    result = {
        "candidate_article_count": int(len(candidates)),
        "usable_article_count": int(len(usable)),
        "news_usable_article_count": int(len(usable)),
        "uncertain_article_count": int(len(candidates) - len(usable)),
        "unique_source_count": int(len(sources)),
        "news_unique_source_count": int(len(sources)),
        "news_independent_event_count": int(len(events)),
        "weighted_sentiment_score": score,
        # Stable aliases consumed by the experimental late-fusion trainer.
        "news_weighted_sentiment": score,
        "positive_count": int(label_counts.get("positive", 0)),
        "neutral_count": int(label_counts.get("neutral", 0)),
        "negative_count": int(label_counts.get("negative", 0)),
        "news_positive_share": positive_event_weight / event_weight_total if event_weight_total > 0 else 0.0,
        "news_neutral_share": neutral_event_weight / event_weight_total if event_weight_total > 0 else 0.0,
        "news_negative_share": negative_event_weight / event_weight_total if event_weight_total > 0 else 0.0,
        "news_positive_event_impact": positive_event_weight / event_count if event_count else 0.0,
        "news_negative_event_impact": negative_event_weight / event_count if event_count else 0.0,
        "news_event_reliability_mean": event_weight_total / event_count if event_count else 0.0,
        "news_mean_confidence": float(np.mean(event_confidences)) if event_confidences else np.nan,
        "news_max_confidence": float(np.max(event_confidences)) if event_confidences else np.nan,
        "news_mean_entity_confidence": float(np.mean(event_entity_confidences)) if event_entity_confidences else np.nan,
        "news_mean_text_quality": float(np.mean(event_text_quality)) if event_text_quality else np.nan,
        "news_mean_age_days": float(np.mean(event_ages)) if event_ages else np.nan,
        "news_newest_age_days": float(np.min(event_ages)) if event_ages else np.nan,
        "news_coverage": coverage,
        "coverage_confidence": coverage,
        "news_coverage_score": {"High": 1.0, "Medium": 0.5, "Low": 0.25, "Unavailable": 0.0}[coverage],
        "news_available": bool(len(usable)),
        "news_window_days": window,
        "news_half_life_days": half_life,
        "news_latest_article_at": usable["published_at"].max().isoformat() if not usable.empty else None,
        "news_official_disclosure_share": (
            official_weight / event_weight_total if event_weight_total > 0 else 0.0
        ),
        "news_event_ids": json.dumps(
            sorted(item["event_id"] for item in event_rows_summary),
            separators=(",", ":"),
        ),
        "news_material_event_ids": json.dumps(
            sorted(
                item["event_id"]
                for item in event_rows_summary
                if item["category"] in MATERIAL_EVENT_CATEGORIES
            ),
            separators=(",", ":"),
        ),
    }
    result.update(
        {
            f"news_event_category_{category}_share": float(value)
            for category, value in category_shares.items()
        }
    )
    return result


def build_point_in_time_feature_table(
    stock_frame: pd.DataFrame,
    news_frame: pd.DataFrame,
    *,
    stock_probability_frame: pd.DataFrame | None = None,
    horizons: Sequence[int] = TARGET_HORIZONS,
    context_windows: Mapping[int, int] | None = None,
    context_half_lives: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    """Build leakage-safe daily fusion features without fitting a model.

    ``stock_frame`` must contain ``Date`` and ``Ticker`` plus any target
    columns.  ``stock_probability_frame`` is optional and can contain one row
    per ticker/date/horizon with ``probability_up`` (or
    ``stock_probability_up``), threshold margin, and evidence strength.  If
    omitted, stock probability columns remain missing by design.
    """
    if not isinstance(stock_frame, pd.DataFrame) or not isinstance(news_frame, pd.DataFrame):
        raise TypeError("stock_frame and news_frame must be pandas DataFrames")
    for column in ("Date", "Ticker"):
        if column not in stock_frame.columns:
            raise ValueError(f"stock_frame is missing required column: {column}")
    if not {"ticker", "published_at"}.issubset(news_frame.columns):
        raise ValueError("news_frame must be normalized with ticker and published_at columns")
    unknown_horizons = set(horizons) - set(TARGET_HORIZONS)
    if unknown_horizons:
        raise ValueError(f"Unsupported horizons: {sorted(unknown_horizons)}")
    stock = stock_frame.copy()
    stock["Date"] = pd.to_datetime(stock["Date"], errors="coerce").dt.normalize()
    stock["Ticker"] = stock["Ticker"].map(_normalise_ticker)
    stock = stock.dropna(subset=["Date", "Ticker"]).drop_duplicates(["Ticker", "Date"], keep="last")
    news = news_frame.copy()
    news["ticker"] = news["ticker"].map(_normalise_ticker)
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    news = news.dropna(subset=["ticker", "published_at"])
    probability = None
    probability_lookup: dict[tuple[str, pd.Timestamp, int], Mapping[str, Any]] = {}
    if stock_probability_frame is not None:
        probability = stock_probability_frame.copy()
        for column in ("Date", "Ticker"):
            if column not in probability.columns:
                raise ValueError(f"stock_probability_frame is missing required column: {column}")
        probability["Date"] = pd.to_datetime(probability["Date"], errors="coerce").dt.normalize()
        probability["Ticker"] = probability["Ticker"].map(_normalise_ticker)
        probability["horizon"] = pd.to_numeric(probability.get("horizon", pd.Series(pd.NA, index=probability.index)), errors="coerce")
        probability_series = (
            probability["probability_up"]
            if "probability_up" in probability.columns
            else probability.get("stock_probability_up", pd.Series(np.nan, index=probability.index))
        )
        probability["probability_up"] = pd.to_numeric(probability_series, errors="coerce")
        probability["decision_threshold"] = pd.to_numeric(
            probability.get(
                "decision_threshold", pd.Series(0.5, index=probability.index)
            ),
            errors="coerce",
        ).fillna(0.5)
        probability = probability.dropna(subset=["Date", "Ticker", "horizon"])
        probability["horizon"] = probability["horizon"].astype(int)
        probability_lookup = {
            (
                str(row["Ticker"]),
                pd.Timestamp(row["Date"]),
                int(row["horizon"]),
            ): row
            for row in probability.to_dict(orient="records")
        }
        available_pairs = {
            (ticker, observation_date)
            for ticker, observation_date, _ in probability_lookup
        }
        stock = stock.loc[
            [
                (str(ticker), pd.Timestamp(observation_date)) in available_pairs
                for ticker, observation_date in zip(stock["Ticker"], stock["Date"])
            ]
        ].copy()
    news_by_ticker = {
        str(ticker): group.copy()
        for ticker, group in news.groupby("ticker", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for stock_row in stock.to_dict(orient="records"):
        ticker = str(stock_row["Ticker"])
        observation_date = pd.Timestamp(stock_row["Date"])
        ticker_news = news_by_ticker.get(ticker, news.iloc[0:0])
        for horizon in horizons:
            row: dict[str, Any] = {
                "Date": observation_date.date().isoformat(),
                "Ticker": ticker,
                "ticker": ticker,
                "horizon": int(horizon),
                # Aliases match the experimental fusion-model contract.
                "observation_date": observation_date.date().isoformat(),
                "horizon_days": int(horizon),
                "feature_schema_version": FUSION_FEATURE_SCHEMA_VERSION,
                "news_cutoff_at": _cutoff_timestamp(observation_date).isoformat(),
                "stock_probability_up": np.nan,
                "stock_threshold_margin": np.nan,
                "stock_decision_threshold": np.nan,
                "stock_evidence_strength": np.nan,
                f"Target_{horizon}D": stock_row.get(f"Target_{horizon}D", np.nan),
                f"Target_Date_{horizon}D": stock_row.get(f"Target_Date_{horizon}D", pd.NaT),
            }
            if probability is not None:
                p = probability_lookup.get((ticker, observation_date, int(horizon)))
                if p is not None:
                    row["stock_probability_up"] = p.get("probability_up", np.nan)
                    row["stock_decision_threshold"] = p.get("decision_threshold", 0.5)
                    row["stock_threshold_margin"] = (
                        float(row["stock_probability_up"])
                        - float(row["stock_decision_threshold"])
                        if pd.notna(row["stock_probability_up"])
                        else np.nan
                    )
                    evidence_strength = p.get("evidence_strength", pd.NA)
                    row["stock_evidence_strength"] = EVIDENCE_STRENGTH_SCORES.get(
                        str(evidence_strength).strip().casefold(), evidence_strength
                    )
                    for provenance_name in (
                        "prediction_role",
                        "training_end",
                        "calibration_end",
                        "validation_start",
                        "fold_id",
                        "stock_model_name",
                        "stock_model_hyperparameters",
                        "stock_registry_run_id",
                        "stock_registry_dataset_sha256",
                        "stock_feature_schema_sha256",
                    ):
                        if provenance_name in p:
                            row[provenance_name] = p.get(provenance_name)
            row.update(
                _score_news(
                    ticker_news,
                    _cutoff_timestamp(observation_date),
                    horizon,
                    window_days=(context_windows or {}).get(horizon),
                    half_life_days=(context_half_lives or {}).get(horizon),
                )
            )
            row["target_up"] = row.get(f"Target_{horizon}D", np.nan)
            row["target_date"] = row.get(f"Target_Date_{horizon}D", pd.NaT)
            score = row.get("weighted_sentiment_score")
            if pd.isna(row["stock_probability_up"]) or score is None:
                row["stock_news_agreement"] = np.nan
            elif float(score) > 0.05:
                row["stock_news_agreement"] = float(
                    float(row["stock_probability_up"])
                    >= float(row.get("stock_decision_threshold", 0.5))
                )
            elif float(score) < -0.05:
                row["stock_news_agreement"] = float(
                    float(row["stock_probability_up"])
                    < float(row.get("stock_decision_threshold", 0.5))
                )
            else:
                row["stock_news_agreement"] = 0.5
            rows.append(row)
    return pd.DataFrame(rows)


def assess_fusion_data_readiness(
    news_frame: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame | None = None,
    required_tickers: Sequence[str] | None = None,
    minimum_date: pd.Timestamp | str = DEFAULT_MINIMUM_DATE,
    maximum_date: pd.Timestamp | str = DEFAULT_MAXIMUM_DATE,
    min_news_present_observations: int = DEFAULT_MIN_NEWS_PRESENT_OBSERVATIONS,
    min_source_domains: int = DEFAULT_MIN_SOURCE_DOMAINS,
) -> dict[str, Any]:
    """Return an auditable readiness report; this function never promotes data."""
    minimum = pd.Timestamp(minimum_date).tz_localize(None).normalize()
    maximum = pd.Timestamp(maximum_date).tz_localize(None).normalize()
    tickers = tuple(sorted(_normalise_ticker(item) for item in (required_tickers or load_stock_metadata().keys())))
    frame = news_frame.copy()
    if not {"ticker", "published_at"}.issubset(frame.columns):
        raise ValueError("news_frame must be normalized with ticker and published_at columns")
    frame["ticker"] = frame["ticker"].map(_normalise_ticker)
    published = pd.to_datetime(frame["published_at"], utc=True, errors="coerce").dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None).dt.normalize()
    frame = frame.assign(_published_date=published).dropna(subset=["_published_date"])
    frame = frame.loc[frame["_published_date"].between(minimum, maximum)]
    per_ticker: dict[str, Any] = {}
    for ticker in tickers:
        subset = frame.loc[frame["ticker"].eq(ticker)]
        domains = set(subset.get("source_domain", pd.Series(dtype=str)).dropna().astype(str).str.casefold()) - {""}
        stats = {
            "article_rows": int(len(subset)),
            "first_date": subset["_published_date"].min().date().isoformat() if not subset.empty else None,
            "last_date": subset["_published_date"].max().date().isoformat() if not subset.empty else None,
            "source_domain_count": int(len(domains)),
        }
        if feature_frame is not None and not feature_frame.empty:
            features = feature_frame.loc[feature_frame["Ticker"].astype(str).str.upper().eq(ticker)] if "Ticker" in feature_frame.columns else pd.DataFrame()
            if "news_available" in features.columns:
                stats["news_present_observations"] = int(features["news_available"].fillna(False).astype(bool).sum())
            else:
                stats["news_present_observations"] = 0
            horizon_column = "horizon" if "horizon" in features.columns else "horizon_days" if "horizon_days" in features.columns else None
            if horizon_column and "news_available" in features.columns:
                by_horizon = features.groupby(horizon_column)["news_available"].apply(
                    lambda values: int(values.fillna(False).astype(bool).sum())
                )
                stats["news_present_by_horizon"] = {str(key): int(value) for key, value in by_horizon.to_dict().items()}
            else:
                stats["news_present_by_horizon"] = {}
            target_columns = [
                column for column in features.columns
                if re.fullmatch(r"Target_\d+D", str(column))
            ]
            stats["target_class_counts"] = {column: {str(key): int(value) for key, value in features[column].value_counts(dropna=True).to_dict().items()} for column in target_columns}
            if "target_up" in features.columns:
                stats["target_class_counts"]["target_up"] = {
                    str(key): int(value)
                    for key, value in features["target_up"].value_counts(dropna=True).to_dict().items()
                }
        else:
            stats["news_present_observations"] = None
            stats["news_present_by_horizon"] = {}
            stats["target_class_counts"] = {}
        stats["passes_coverage"] = bool(stats["first_date"] and stats["last_date"] and stats["source_domain_count"] >= min_source_domains)
        if feature_frame is None:
            stats["passes_news_observations"] = False
        elif stats["news_present_by_horizon"]:
            stats["passes_news_observations"] = all(
                value >= min_news_present_observations
                for key, value in stats["news_present_by_horizon"].items()
            ) and set(stats["news_present_by_horizon"]) >= {str(horizon) for horizon in TARGET_HORIZONS}
        else:
            stats["passes_news_observations"] = stats["news_present_observations"] >= min_news_present_observations
        per_ticker[ticker] = stats
    missing_tickers = [ticker for ticker in tickers if not per_ticker[ticker]["article_rows"]]
    coverage_passed = all(per_ticker[ticker]["passes_coverage"] for ticker in tickers)
    observations_known = feature_frame is not None
    observation_passed = bool(observations_known and all(per_ticker[ticker]["passes_news_observations"] for ticker in tickers))
    class_passed = True
    if feature_frame is not None:
        for ticker in tickers:
            for counts in per_ticker[ticker]["target_class_counts"].values():
                if not {"0", "1"}.issubset(counts):
                    class_passed = False
    ready = bool(tickers and not missing_tickers and coverage_passed and observation_passed and class_passed)
    return {
        "ready": ready,
        "schema_version": FUSION_SCHEMA_VERSION,
        "feature_schema_version": FUSION_FEATURE_SCHEMA_VERSION,
        "minimum_date": minimum.date().isoformat(),
        "maximum_date": maximum.date().isoformat(),
        "required_tickers": list(tickers),
        "missing_tickers": missing_tickers,
        "minimum_news_present_observations": min_news_present_observations,
        "minimum_source_domains": min_source_domains,
        "coverage_passed": coverage_passed,
        "observation_requirement_checked": observations_known,
        "observation_passed": observation_passed,
        "target_class_requirement_passed": class_passed,
        "per_ticker": per_ticker,
    }


def run_import_historical_news_pipeline(source_paths: Sequence[str], output_dir: str | None = None, provider: str | None = None) -> dict[str, Any]:
    """CLI-friendly wrapper for :func:`import_historical_news`."""
    result = import_historical_news(source_paths, output_dir=Path(output_dir) if output_dir else DEFAULT_FUSION_DATA_DIR, provider=provider)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))
    return payload


def run_fusion_data_readiness(data_dir: str | None = None) -> dict[str, Any]:
    """CLI-friendly readiness report for an imported normalized dataset."""
    target = Path(data_dir) if data_dir else DEFAULT_FUSION_DATA_DIR
    try:
        frame = load_normalized_historical_news(target)
        report = assess_fusion_data_readiness(frame)
    except FileNotFoundError as exc:
        report = {
            "ready": False,
            "schema_version": FUSION_SCHEMA_VERSION,
            "feature_schema_version": FUSION_FEATURE_SCHEMA_VERSION,
            "data_dir": str(target),
            "reason": str(exc),
            "errors": [str(exc)],
        }
    print(json.dumps(report, indent=2))
    return report


def _read_offline_table(path: Path | str) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(source, lines=True)
    if suffix == ".json":
        return pd.read_json(source)
    raise ValueError(f"Unsupported offline news table: {source}")


def score_historical_news_with_production_finbert(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Score reviewed historical articles using the deployed FinBERT artifact.

    LLM-assisted labels are reference labels only.  Fusion features must use
    the same model-generated probabilities available at runtime, so this
    function overwrites sentiment fields with production FinBERT outputs while
    retaining the original labels in explicit audit columns.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    output = frame.copy()
    if "entity_relevant" in output:
        entity = output["entity_relevant"].map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"1", "true", "yes", "y"}
        )
        output = output.loc[entity].copy()
    if output.empty:
        return normalize_historical_news_frame(output)
    if "description" not in output:
        output["description"] = output.get(
            "article_text", output.get("content_snippet", "")
        )
    if "url" not in output and "source_url" in output:
        output["url"] = output["source_url"]
    if "article_id" not in output:
        output["article_id"] = [
            _stable_article_id(row)
            for row in output.to_dict(orient="records")
        ]
    output["reference_sentiment_label"] = output.get(
        "sentiment_label", pd.Series("", index=output.index)
    )
    source_series = output.get(
        "label_source", pd.Series("", index=output.index)
    )
    output["reference_label_source_original"] = source_series
    output["reference_label_source"] = source_series.map(_canonical_label_source)
    sentiment_config = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
    sentiment_identity = _sentiment_runtime_identity(sentiment_config)
    scored_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    records = output.to_dict(orient="records")
    add_sentiment_scores(records, load_sentiment_model())
    scored = pd.DataFrame(records)
    scored["event_category"] = [
        canonical_material_event_category(
            row.get("event_category"),
            row.get("title"),
            row.get("description"),
        )
        for row in records
    ]
    scored["sentiment_model_id"] = sentiment_identity["variant"]
    scored["sentiment_runtime_config_sha256"] = sentiment_identity["config_sha256"]
    scored["sentiment_model_variant_sha256"] = sentiment_identity["variant_sha256"]
    scored["sentiment_component_artifacts_sha256"] = sentiment_identity[
        "component_artifacts_sha256"
    ]
    scored["sentiment_scored_at"] = scored_at
    # Scoring an article with FinBERT is not the same as obtaining a reviewed
    # reference label.  Existing reviewed rows retain the explicit
    # ``llm_assisted_label`` provenance; newly imported official/issuer rows
    # remain ``unlabelled_reference`` until a separate labelling workflow is
    # completed.  This prevents a model prediction from being misrepresented
    # as human or LLM annotation.
    existing_status = scored.get(
        "label_validation_status", pd.Series("", index=scored.index)
    ).map(clean_text)
    existing_flag = _coerce_boolean(
        scored.get("llm_assisted_label", pd.Series(False, index=scored.index))
    )
    scored["label_validation_status"] = existing_status.where(
        existing_status.ne(""),
        existing_flag.map(
            lambda value: "llm_assisted_label" if bool(value) else "unlabelled_reference"
        ),
    )
    scored["llm_assisted_label"] = existing_flag
    normalized = normalize_historical_news_frame(scored)
    audit = scored[
        [
            "article_id",
            "reference_sentiment_label",
            "reference_label_source",
            "reference_label_source_original",
        ]
    ].drop_duplicates("article_id", keep="last")
    normalized = normalized.merge(
        audit,
        on="article_id",
        how="left",
        validate="many_to_one",
    )
    normalized["sentiment_model_role"] = "production_finbert_historical_score"
    return normalized


def rewrite_historical_label_provenance_as_llm_assisted(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Migrate audit provenance without changing FinBERT probabilities."""

    output = frame.copy()
    if "reference_label_source" not in output:
        output["reference_label_source"] = ""
    if "reference_label_source_original" not in output:
        output["reference_label_source_original"] = output[
            "reference_label_source"
        ]
    output["reference_label_source"] = output["reference_label_source"].map(
        _canonical_label_source
    )
    output["label_validation_status"] = "llm_assisted_label"
    output["llm_assisted_label"] = True
    return output


def run_score_historical_news_for_fusion(
    source_path: str,
    *,
    output_path: str,
) -> dict[str, Any]:
    source = _read_offline_table(source_path)
    result = score_historical_news_with_production_finbert(source)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.casefold() in {".parquet", ".pq"}:
        _parquet_safe_frame(result).to_parquet(target, index=False)
    elif target.suffix.casefold() == ".csv":
        result.to_csv(target, index=False)
    elif target.suffix.casefold() in {".jsonl", ".ndjson"}:
        result.to_json(target, orient="records", lines=True, date_format="iso")
    else:
        raise ValueError("Scored historical news output must be Parquet, CSV or JSONL")
    payload = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "source_path": str(Path(source_path)),
        "output_path": str(target),
        "rows": int(len(result)),
        "sentiment_model_role": "production_finbert_historical_score",
        "sentiment_model_id": (
            result["sentiment_model_id"].dropna().astype(str).unique().tolist()
            if "sentiment_model_id" in result
            else []
        ),
        "sentiment_runtime_config_sha256": (
            result["sentiment_runtime_config_sha256"].dropna().astype(str).unique().tolist()
            if "sentiment_runtime_config_sha256" in result
            else []
        ),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    print(json.dumps(payload, indent=2))
    return payload


def run_build_point_in_time_features(
    stock_path: str,
    news_dir: str | None = None,
    output_path: str | None = None,
    stock_probability_path: str | None = None,
) -> dict[str, Any]:
    """CLI-friendly feature-table builder; stock probabilities remain optional."""
    from pipeline.fusion_stock_oof import read_stock_feature_input

    stock = read_stock_feature_input(stock_path)
    news = load_normalized_historical_news(Path(news_dir) if news_dir else DEFAULT_FUSION_DATA_DIR)
    probability_source = Path(stock_probability_path) if stock_probability_path else None
    probabilities = (
        pd.read_parquet(probability_source)
        if probability_source and probability_source.suffix.casefold() in {".parquet", ".pq"}
        else pd.read_csv(probability_source)
        if probability_source
        else None
    )
    if probabilities is not None:
        accepted_roles = {
            "nested_out_of_fold",
            "runtime_compatibility_nested_out_of_fold",
            "scientific_nested_model_selection_oof",
        }
        if "prediction_role" not in probabilities.columns or not set(
            probabilities["prediction_role"].astype(str).unique()
        ).issubset(accepted_roles):
            raise ValueError(
                "Fusion stock probabilities must use a recognized nested OOF contract"
            )
    result = build_point_in_time_feature_table(
        stock, news, stock_probability_frame=probabilities
    )
    # The fusion trainer accepts only rows whose stock probability was created
    # by the nested OOF contract and whose future target has matured.  Keeping
    # pre-OOF or right-censored rows makes the persisted dataset fail its own
    # leakage/readiness contract.
    result["target_up"] = pd.to_numeric(result["target_up"], errors="coerce")
    result = result.loc[
        result["stock_probability_up"].notna()
        & result["target_up"].isin([0, 1])
    ].copy()
    target = Path(output_path) if output_path else DATA_DIR / "fusion" / "fusion-features.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    _parquet_safe_frame(result).to_parquet(target, index=False)
    payload = {"output_path": str(target), "rows": int(len(result)), "feature_schema_version": FUSION_FEATURE_SCHEMA_VERSION}
    print(json.dumps(payload, indent=2))
    return payload


def run_build_fusion_window_candidates(
    stock_path: str,
    *,
    stock_probability_path: str,
    output_path: str,
    policy_path: str,
    news_dir: str | None = None,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build every policy-approved window without looking at benchmark labels."""

    from pipeline.fusion_stock_oof import read_stock_feature_input

    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    policy_schema = str(policy.get("schema_version") or "")
    if policy_schema not in {"fusion-evaluation-v2", "fusion-evaluation-v3"}:
        raise ValueError(
            "Window-candidate generation requires fusion-evaluation-v2 or fusion-evaluation-v3"
        )
    stock = read_stock_feature_input(stock_path)
    news_root = Path(news_dir) if news_dir else DEFAULT_FUSION_DATA_DIR
    news = load_normalized_historical_news(news_root)
    probability_source = Path(stock_probability_path)
    probabilities = (
        pd.read_parquet(probability_source)
        if probability_source.suffix.casefold() in {".parquet", ".pq"}
        else pd.read_csv(probability_source)
    )
    if "prediction_role" not in probabilities or not probabilities[
        "prediction_role"
    ].astype(str).eq("scientific_nested_model_selection_oof").all():
        raise ValueError(
            "The v2 headline dataset requires scientific_nested_model_selection_oof probabilities"
        )
    frames: list[pd.DataFrame] = []
    requested_horizons = tuple(
        int(value) for value in (horizons or policy["supported_horizons"])
    )
    unsupported = set(requested_horizons) - set(
        int(value) for value in policy["supported_horizons"]
    )
    if unsupported:
        raise ValueError(f"Unsupported fusion window horizons: {sorted(unsupported)}")
    for horizon in requested_horizons:
        generated_variants: set[tuple[int, float]] = set()
        for window in policy["window_candidates"][str(horizon)]:
            for fraction in policy["half_life_fractions"]:
                # A zero-day candidate denotes the current Bursa session.  It
                # still needs a positive decay constant for stable weighting.
                half_life = max(0.25, float(window) * float(fraction))
                variant = (int(window), float(half_life))
                if variant in generated_variants:
                    continue
                generated_variants.add(variant)
                candidate = build_point_in_time_feature_table(
                    stock,
                    news,
                    stock_probability_frame=probabilities,
                    horizons=(int(horizon),),
                    context_windows={int(horizon): int(window)},
                    context_half_lives={int(horizon): half_life},
                )
                candidate["window_candidate_id"] = (
                    f"{int(horizon)}d-{int(window)}d-hl-{half_life:g}"
                )
                frames.append(candidate)
    result = pd.concat(frames, ignore_index=True)
    result["target_up"] = pd.to_numeric(result["target_up"], errors="coerce")
    result = result.loc[
        result["stock_probability_up"].notna()
        & result["target_up"].isin([0, 1])
    ].copy()
    sentiment_config = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
    sentiment_identity = _sentiment_runtime_identity(sentiment_config)
    article_identity_columns = {
        "sentiment_model_id": sentiment_identity["variant"],
        "sentiment_runtime_config_sha256": sentiment_identity["config_sha256"],
        "sentiment_model_variant_sha256": sentiment_identity["variant_sha256"],
        "sentiment_component_artifacts_sha256": sentiment_identity[
            "component_artifacts_sha256"
        ],
    }
    article_identity_checks: dict[str, bool] = {}
    for column, expected in article_identity_columns.items():
        observed = sorted(
            {
                clean_text(value)
                for value in news.get(column, pd.Series(dtype=str)).dropna()
                if clean_text(value)
            }
        )
        article_identity_checks[column] = bool(expected) and observed == [str(expected)]
    article_identity_verified = all(article_identity_checks.values())
    if policy_schema == "fusion-evaluation-v3" and not article_identity_verified:
        raise ValueError(
            "Fusion v3 requires article-level sentiment identities that exactly match "
            f"the active runtime ensemble: {article_identity_checks}. Rescore the "
            "historical corpus before rebuilding window candidates."
        )
    result["sentiment_runtime_config_sha256"] = sentiment_identity["config_sha256"]
    result["sentiment_model_variant"] = sentiment_identity["variant"]
    result["sentiment_model_variant_sha256"] = sentiment_identity["variant_sha256"]
    result["sentiment_component_artifacts_sha256"] = sentiment_identity[
        "component_artifacts_sha256"
    ]
    result["article_sentiment_identity_verified"] = article_identity_verified
    result["article_sentiment_identity_checks"] = json.dumps(
        article_identity_checks, sort_keys=True, separators=(",", ":")
    )
    news_files = sorted((news_root / "normalized").glob("ticker=*/year=*/articles.parquet"))
    corpus_digest = hashlib.sha256()
    for path in news_files:
        corpus_digest.update(path.relative_to(news_root).as_posix().encode("utf-8"))
        corpus_digest.update(hashlib.sha256(path.read_bytes()).digest())
    result["article_corpus_sha256"] = corpus_digest.hexdigest()
    aggregation_contract = {
        "cutoff": "17:00 Asia/Kuala_Lumpur",
        "after_close": "next Bursa session",
        "uncertain_articles": "excluded",
        "syndicated_event_cap": "maximum member weight",
        "missing_news": "explicit zero influence",
    }
    result["news_aggregation_policy_sha256"] = hashlib.sha256(
        json.dumps(aggregation_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _parquet_safe_frame(result).to_parquet(target, index=False)
    payload = {
        "schema_version": FUSION_FEATURE_SCHEMA_VERSION,
        "output_path": str(target),
        "rows": int(len(result)),
        "horizons": list(requested_horizons),
        "window_candidate_count": int(result["window_candidate_id"].nunique()),
        "stock_probability_sha256": hashlib.sha256(
            probability_source.read_bytes()
        ).hexdigest(),
        "article_corpus_sha256": corpus_digest.hexdigest(),
        "policy_sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
        "sentiment_model_variant": sentiment_identity["variant"],
        "sentiment_model_variant_sha256": sentiment_identity["variant_sha256"],
        "sentiment_component_artifacts_sha256": sentiment_identity[
            "component_artifacts_sha256"
        ],
        "article_sentiment_identity_verified": article_identity_verified,
        "article_sentiment_identity_checks": article_identity_checks,
    }
    print(json.dumps(payload, indent=2))
    return payload
