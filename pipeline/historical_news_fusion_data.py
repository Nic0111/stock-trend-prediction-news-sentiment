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
    calculate_recency_weight,
    clean_text,
    load_stock_metadata,
)
from pipeline.fusion_model import EVIDENCE_STRENGTH_SCORES
from pipeline.stock_features import TARGET_HORIZONS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FUSION_DATA_DIR = DATA_DIR / "historical-news-fusion"
FUSION_SCHEMA_VERSION = "historical-news-fusion-v1"
FUSION_FEATURE_SCHEMA_VERSION = "point-in-time-news-features-v1"
DEFAULT_MINIMUM_DATE = pd.Timestamp("2019-01-01")
DEFAULT_MAXIMUM_DATE = pd.Timestamp("2026-12-31")
DEFAULT_MIN_NEWS_PRESENT_OBSERVATIONS = 100
DEFAULT_MIN_SOURCE_DOMAINS = 2
LOCAL_TIMEZONE = "Asia/Kuala_Lumpur"
MARKET_CUTOFF_HOUR = 17

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
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "article_id": ("article_id", "id", "source_id", "source.id", "guid", "uuid"),
    "ticker": ("ticker", "query_ticker", "symbol", "stock_ticker"),
    "company_name": ("company_name", "query_company", "company", "issuer"),
    "provider": ("provider", "source_provider"),
    "source_name": ("source_name", "source.name", "source", "publisher", "site_name"),
    "source_domain": ("source_domain", "domain", "website", "source_url"),
    "published_at": ("published_at", "publishedAt", "published", "date", "datetime", "timestamp"),
    "retrieved_at": ("retrieved_at", "retrievedAt", "collected_at", "collection_timestamp"),
    "title": ("title", "headline", "article_title"),
    "description": ("description", "summary", "snippet", "content", "text", "article_text"),
    "content_snippet": ("content_snippet", "content", "body", "article_content"),
    "url": ("url", "link", "article_url", "url_to_article"),
    "sentiment_label": ("sentiment_label", "sentiment", "label", "tone"),
    "sentiment_score": ("sentiment_score", "score", "tone_score"),
    "sentiment_probability": ("sentiment_probability", "confidence", "sentiment_confidence"),
    "probability_negative": ("probability_negative", "negative_probability", "negative_prob"),
    "probability_neutral": ("probability_neutral", "neutral_probability", "neutral_prob"),
    "probability_positive": ("probability_positive", "positive_probability", "positive_prob"),
    "sentiment_uncertain": ("sentiment_uncertain", "uncertain", "low_confidence"),
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
    output["published_at"] = pd.to_datetime(output["published_at"], utc=True, errors="coerce")
    output["retrieved_at"] = pd.to_datetime(output["retrieved_at"], utc=True, errors="coerce")
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
            partition.drop(columns=["_year"]).to_parquet(path, index=False)
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


def _score_news(rows: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> dict[str, Any]:
    window = CONTEXT_WINDOWS[horizon]
    half_life = CONTEXT_HALF_LIVES[horizon]
    candidates = rows.loc[rows["published_at"].le(cutoff)].copy()
    candidates["age_days"] = _age_days(candidates["published_at"], cutoff)
    candidates = candidates.loc[candidates["age_days"].le(window)].copy()
    usable = candidates.loc[
        (~candidates["sentiment_uncertain"].astype(bool))
        & candidates["sentiment_score"].isin([-1, 0, 1])
        & pd.to_numeric(candidates["sentiment_probability"], errors="coerce").notna()
    ].copy()
    source_series = usable["source_domain"].where(usable["source_domain"].ne(""), usable["source_name"])
    sources = {clean_text(item).casefold() for item in source_series if clean_text(item)}
    if usable.empty:
        score = None
    else:
        usable["recency_weight"] = usable["age_days"].map(lambda age: calculate_recency_weight(float(age), half_life))
        usable["fusion_weight"] = usable["recency_weight"] * pd.to_numeric(usable["sentiment_probability"], errors="coerce")
        total = float(usable["fusion_weight"].sum())
        score = float((usable["sentiment_score"] * usable["fusion_weight"]).sum() / total) if total > 0 else None
    label_counts = usable["sentiment_label"].value_counts() if not usable.empty else pd.Series(dtype=int)
    coverage = "High" if len(usable) >= 5 and len(sources) >= 3 else "Medium" if len(usable) >= 3 and len(sources) >= 2 else "Low" if len(usable) else "Unavailable"
    return {
        "candidate_article_count": int(len(candidates)),
        "usable_article_count": int(len(usable)),
        "news_usable_article_count": int(len(usable)),
        "uncertain_article_count": int(len(candidates) - len(usable)),
        "unique_source_count": int(len(sources)),
        "news_unique_source_count": int(len(sources)),
        "weighted_sentiment_score": score,
        # Stable aliases consumed by the experimental late-fusion trainer.
        "news_weighted_sentiment": score,
        "positive_count": int(label_counts.get("positive", 0)),
        "neutral_count": int(label_counts.get("neutral", 0)),
        "negative_count": int(label_counts.get("negative", 0)),
        "news_positive_share": float(label_counts.get("positive", 0) / len(usable)) if len(usable) else 0.0,
        "news_neutral_share": float(label_counts.get("neutral", 0) / len(usable)) if len(usable) else 0.0,
        "news_negative_share": float(label_counts.get("negative", 0) / len(usable)) if len(usable) else 0.0,
        "news_mean_confidence": float(pd.to_numeric(usable["sentiment_probability"], errors="coerce").mean()) if len(usable) else np.nan,
        "news_max_confidence": float(pd.to_numeric(usable["sentiment_probability"], errors="coerce").max()) if len(usable) else np.nan,
        "news_mean_age_days": float(usable["age_days"].mean()) if len(usable) else np.nan,
        "news_newest_age_days": float(usable["age_days"].min()) if len(usable) else np.nan,
        "news_coverage": coverage,
        "coverage_confidence": coverage,
        "news_coverage_score": {"High": 1.0, "Medium": 0.5, "Low": 0.25, "Unavailable": 0.0}[coverage],
        "news_available": bool(len(usable)),
        "news_window_days": window,
        "news_half_life_days": half_life,
        "news_latest_article_at": usable["published_at"].max().isoformat() if not usable.empty else None,
    }


def build_point_in_time_feature_table(
    stock_frame: pd.DataFrame,
    news_frame: pd.DataFrame,
    *,
    stock_probability_frame: pd.DataFrame | None = None,
    horizons: Sequence[int] = TARGET_HORIZONS,
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
        probability = probability.dropna(subset=["Date", "Ticker", "horizon"])
    rows: list[dict[str, Any]] = []
    for stock_row in stock.to_dict(orient="records"):
        ticker = str(stock_row["Ticker"])
        observation_date = pd.Timestamp(stock_row["Date"])
        ticker_news = news.loc[news["ticker"].eq(ticker)]
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
                "stock_evidence_strength": np.nan,
                f"Target_{horizon}D": stock_row.get(f"Target_{horizon}D", np.nan),
                f"Target_Date_{horizon}D": stock_row.get(f"Target_Date_{horizon}D", pd.NaT),
            }
            if probability is not None:
                matched = probability.loc[
                    probability["Ticker"].eq(ticker)
                    & probability["Date"].eq(observation_date)
                    & probability["horizon"].eq(horizon)
                ]
                if not matched.empty:
                    p = matched.iloc[-1]
                    row["stock_probability_up"] = p.get("probability_up", np.nan)
                    row["stock_threshold_margin"] = p.get("threshold_margin", np.nan)
                    evidence_strength = p.get("evidence_strength", pd.NA)
                    row["stock_evidence_strength"] = EVIDENCE_STRENGTH_SCORES.get(
                        str(evidence_strength).strip().casefold(), evidence_strength
                    )
            row.update(_score_news(ticker_news, _cutoff_timestamp(observation_date), horizon))
            row["target_up"] = row.get(f"Target_{horizon}D", np.nan)
            score = row.get("weighted_sentiment_score")
            row["stock_news_agreement"] = (
                np.nan
                if pd.isna(row["stock_probability_up"]) or score is None
                else float((row["stock_probability_up"] >= 0.5) == (score >= 0))
            )
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


def run_build_point_in_time_features(stock_path: str, news_dir: str | None = None, output_path: str | None = None) -> dict[str, Any]:
    """CLI-friendly feature-table builder; stock probabilities remain optional."""
    stock = pd.read_csv(stock_path)
    news = load_normalized_historical_news(Path(news_dir) if news_dir else DEFAULT_FUSION_DATA_DIR)
    result = build_point_in_time_feature_table(stock, news)
    target = Path(output_path) if output_path else DEFAULT_FUSION_DATA_DIR / "point-in-time-features.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    payload = {"output_path": str(target), "rows": int(len(result)), "feature_schema_version": FUSION_FEATURE_SCHEMA_VERSION}
    print(json.dumps(payload, indent=2))
    return payload
