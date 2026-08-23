from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
from sklearn.pipeline import Pipeline

from pipeline.finbert_sentiment_model import (
    AveragedFinBERTEnsemble,
    FinBERTArtifactError,
    WeightedFinBERTEnsemble,
    load_finbert_sentiment_model,
)
from pipeline.sentiment_text import TEXT_MODE_DEFAULT, build_sentiment_text
from pipeline.news_relevance import (
    classify_entity_relevance,
    classify_non_news_source,
    stable_article_id,
    stable_event_id,
)
from pipeline.official_disclosure_news import classify_material_event
from pipeline.runtime_provenance import (
    sentiment_artifact_hashes_for_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FINANCIAL_NEWS_DATA_DIR = DATA_DIR / "financial-news-dataset"
FINANCIAL_NEWS_JSON_DIR = FINANCIAL_NEWS_DATA_DIR / "json"
FINANCIAL_NEWS_CSV_DIR = FINANCIAL_NEWS_DATA_DIR / "csv"
CONFIG_DIR = PROJECT_ROOT / "config"
METADATA_FILE = DATA_DIR / "stock-metadata.csv"
FINBERT_SENTIMENT_MODEL_DIR = MODELS_DIR / "experimental" / "finbert-sentiment"
SENTIMENT_RUNTIME_CONFIG_FILE = MODELS_DIR / "sentiment-runtime-config.json"
NEWSAPI_KEY_FILE = CONFIG_DIR / "newsapi_key.txt"
SENTIMENT_VARIANT_ENV_VAR = "STOCKSENSE_SENTIMENT_VARIANT"
DEFAULT_SENTIMENT_VARIANT = "finbert_phrasebank_v1"

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
NEWSAPI_KEY_ENV_VAR = "NEWSAPI_KEY"
SEARCH_IN = "title,description"
LANGUAGE = "en"
SORT_BY = "publishedAt"
PAGE_SIZE = 100
NEWS_RETRIEVAL_LOOKBACK_DAYS = 30
# Bump when provider date-window semantics change so cached contexts are not
# mistaken for results produced under the current NewsAPI boundary contract.
# Increment when the serialized context shape changes. The contribution trail
# is required for the evidence visualisations and must not be read from an
# older cache that cannot reproduce the displayed score.
NEWS_CONTEXT_CONFIG_VERSION = 12
NEWS_HISTORICAL_CONTEXT_START_DAYS = NEWS_RETRIEVAL_LOOKBACK_DAYS + 1
NEWS_HISTORICAL_CONTEXT_END_DAYS = 90
SENTIMENT_PROBABILITY_THRESHOLD = 0.55
DASHBOARD_ARTICLES = 5

CONTEXT_WINDOWS = {1: 7, 5: 14, 10: 30}
CONTEXT_HALF_LIVES = {1: 3.5, 5: 7.0, 10: 15.0}
SENTIMENT_SCORE_MAP = {"positive": 1, "neutral": 0, "negative": -1}
BURSA_TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")
BURSA_SESSION_CLOSE = time(17, 0)

KNOWN_COMPANY_ALIASES = {
    "1155.KL": ["Maybank", "Malayan Banking", "Maybank Group"],
    "1295.KL": ["Public Bank", "PBBank", "Public Bank Group"],
    "1023.KL": ["CIMB", "CIMB Group", "CIMB Bank"],
    "5347.KL": ["Tenaga Nasional", "TNB"],
    "5225.KL": ["IHH Healthcare", "IHH"],
    "6947.KL": ["CelcomDigi", "Celcom Digi", "CelcomDigi Berhad"],
    "5183.KL": ["PETRONAS Chemicals", "PCG", "PetChem"],
    "8869.KL": ["Press Metal", "Press Metal Aluminium"],
    "3816.KL": ["MISC", "MISC Berhad"],
    "4677.KL": ["YTL", "YTL Corporation"],
}

ARTICLE_COLUMNS = [
    "query_ticker",
    "query_company",
    "query_text",
    "display_rank",
    "matched_alias",
    "provider",
    "source_id",
    "source_name",
    "source_domain",
    "author",
    "title",
    "description",
    "url",
    "url_to_image",
    "published_at",
    "content_snippet",
    "sentiment_label",
    "sentiment_score",
    "sentiment_probability",
    "probability_negative",
    "probability_neutral",
    "probability_positive",
    "phrasebank_probability_negative",
    "phrasebank_probability_neutral",
    "phrasebank_probability_positive",
    "bursa_probability_negative",
    "bursa_probability_neutral",
    "bursa_probability_positive",
    "sentiment_model_variant",
    "sentiment_model_weights",
    "sentiment_uncertain",
    "article_id",
    "event_id",
    "automatic_relevance",
    "relevance_score",
    "relevance_reasons",
    "matched_entity_terms",
    "collision_signals",
    "automatic_included",
    "effective_included",
    "user_override",
    "contributes_to_horizons",
]


@dataclass
class NewsContextResult:
    ticker: str
    company_name: str
    query_text: str
    as_of_date: str
    retrieved_at: str
    raw_article_count: int
    relevant_article_count: int
    results_truncated: bool
    contexts: Dict[str, Dict[str, Any]]
    articles: List[Dict[str, Any]]
    output_paths: Dict[str, str]
    provider_mode: str = "newsapi"
    provider_article_counts: Dict[str, int] = field(default_factory=dict)
    provider_status: Dict[str, str] = field(default_factory=dict)
    provider_metadata: Dict[str, Any] = field(default_factory=dict)
    # Articles outside the current sentiment retrieval window are deliberately
    # kept separate so they cannot enter contexts or be mistaken for current
    # evidence by downstream consumers.
    historical_articles: List[Dict[str, Any]] = field(default_factory=list)
    review_articles: List[Dict[str, Any]] = field(default_factory=list)
    automatic_articles: List[Dict[str, Any]] = field(default_factory=list)
    automatic_review_articles: List[Dict[str, Any]] = field(default_factory=list)
    automatic_contexts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    user_overrides: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_news_context_metadata() -> Dict[str, Any]:
    """Return the immutable retrieval/context contract attached to news results."""
    variant = os.getenv(SENTIMENT_VARIANT_ENV_VAR, "").strip() or DEFAULT_SENTIMENT_VARIANT
    weights: Dict[str, float] = {"phrasebank": 1.0}
    if SENTIMENT_RUNTIME_CONFIG_FILE.exists():
        try:
            runtime_config = json.loads(
                SENTIMENT_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8")
            )
            challenger = (runtime_config.get("challengers") or {}).get(variant) or {}
            if isinstance(challenger.get("weights"), dict):
                weights = {
                    str(name): float(value)
                    for name, value in challenger["weights"].items()
                }
        except (OSError, ValueError, TypeError):
            # Runtime loading performs the fail-closed validation. Metadata
            # generation remains safe for cached/provider error responses.
            pass
    return {
        "news_context_config_version": NEWS_CONTEXT_CONFIG_VERSION,
        "retrieval_lookback_days": NEWS_RETRIEVAL_LOOKBACK_DAYS,
        "historical_context_start_days": NEWS_HISTORICAL_CONTEXT_START_DAYS,
        "historical_context_end_days": NEWS_HISTORICAL_CONTEXT_END_DAYS,
        "context_windows": {str(horizon): window for horizon, window in CONTEXT_WINDOWS.items()},
        "context_half_lives": {str(horizon): half_life for horizon, half_life in CONTEXT_HALF_LIVES.items()},
        "sentiment_weighting_policy": "recency_x_finbert_confidence_v1",
        "sentiment_model_variant": variant,
        "sentiment_model_weights": weights,
        "coverage_policy": "article_source_event_diversity_v1",
        "prediction_cutoff_time": "17:00:00+08:00",
    }


def ensure_data_structure() -> None:
    FINANCIAL_NEWS_JSON_DIR.mkdir(parents=True, exist_ok=True)
    FINANCIAL_NEWS_CSV_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def normalize_match_text(value: str) -> str:
    normalized = re.sub(r"[^\w]+", " ", clean_text(value).casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_title(title: str) -> str:
    return normalize_match_text(title)


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query_pairs = [
        pair
        for pair in parsed.query.split("&")
        if pair and not pair.casefold().startswith(("utm_", "fbclid=", "gclid="))
    ]
    return urlunparse(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path.rstrip("/"), "", "&".join(query_pairs), "")
    )


def get_newsapi_key() -> str:
    if NEWSAPI_KEY_FILE.exists():
        key = NEWSAPI_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    key = os.getenv(NEWSAPI_KEY_ENV_VAR, "").strip()
    if key:
        return key
    raise RuntimeError(
        "NewsAPI key is not set. Add config/newsapi_key.txt or set NEWSAPI_KEY."
    )


def load_stock_metadata() -> Dict[str, str]:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Stock metadata not found: {METADATA_FILE}")
    metadata = pd.read_csv(METADATA_FILE)
    return {
        clean_text(row["Ticker"]).upper(): clean_text(row["Company_Name"])
        for _, row in metadata.iterrows()
        if clean_text(row.get("Ticker")) and clean_text(row.get("Company_Name"))
    }


def build_company_query_variants(
    ticker: str,
    company_name: str,
    user_aliases: Sequence[str] | None = None,
) -> List[str]:
    legal_suffix_pattern = re.compile(
        r"\b(berhad|bhd|corporation|corp|holdings?)\b\.?", re.IGNORECASE
    )
    without_berhad = re.sub(r"\bberhad\b\.?", "", company_name, flags=re.IGNORECASE)
    without_legal_suffixes = legal_suffix_pattern.sub("", company_name)
    candidates = [
        company_name,
        without_berhad,
        without_legal_suffixes,
        *KNOWN_COMPANY_ALIASES.get(ticker.upper(), []),
        *(user_aliases or []),
    ]
    variants = unique_preserve_order(candidates)
    return [variant for variant in variants if len(normalize_match_text(variant)) >= 3]


def build_newsapi_query(query_variants: Sequence[str]) -> str:
    query = " OR ".join(f'"{variant}"' for variant in query_variants)
    if len(query) <= 500:
        return query
    accepted: List[str] = []
    for variant in query_variants:
        candidate = " OR ".join([*(f'"{item}"' for item in accepted), f'"{variant}"'])
        if len(candidate) > 500:
            break
        accepted.append(variant)
    return " OR ".join(f'"{variant}"' for variant in accepted)


def get_news_date_range(
    lookback_days: int = NEWS_RETRIEVAL_LOOKBACK_DAYS,
    as_of_date: date | None = None,
) -> Tuple[str, str]:
    end_date = as_of_date or date.today()
    if lookback_days < 1:
        raise ValueError("lookback_days must be at least one day")
    # NewsAPI's free rolling window is timestamp-based.  Subtracting 30 days
    # from the date creates a 31-date inclusive range and can be rejected when
    # the observation date is one day behind today.  Return exactly 30 calendar
    # dates, keeping the request safely inside the provider boundary.
    start_date = end_date - timedelta(days=lookback_days - 1)
    return start_date.isoformat(), end_date.isoformat()


def build_request_url(
    query_text: str,
    lookback_days: int = NEWS_RETRIEVAL_LOOKBACK_DAYS,
    as_of_date: date | None = None,
) -> str:
    from_date, to_date = get_news_date_range(lookback_days, as_of_date)
    params = {
        "q": query_text,
        "searchIn": SEARCH_IN,
        "from": from_date,
        "to": to_date,
        "language": LANGUAGE,
        "sortBy": SORT_BY,
        "pageSize": str(PAGE_SIZE),
        "page": "1",
    }
    return f"{NEWSAPI_ENDPOINT}?{urlencode(params)}"


def fetch_newsapi_response(
    api_key: str,
    query_text: str,
    lookback_days: int = NEWS_RETRIEVAL_LOOKBACK_DAYS,
    as_of_date: date | None = None,
) -> Dict[str, Any]:
    request = Request(
        build_request_url(query_text, lookback_days, as_of_date),
        headers={"X-Api-Key": api_key, "User-Agent": "stock-investment-analytics/2.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NewsAPI HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"NewsAPI request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("NewsAPI request timed out") from exc

    data = json.loads(payload)
    if data.get("status") != "ok":
        raise RuntimeError(
            f"NewsAPI returned {data.get('code', 'unknown_error')}: "
            f"{data.get('message', 'No message')}"
        )
    return data


def match_relevant_alias(
    title: str,
    description: str,
    query_variants: Sequence[str],
) -> str:
    searchable = normalize_match_text(f"{title} {description}")
    for variant in query_variants:
        normalized_alias = normalize_match_text(variant)
        if not normalized_alias:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(
            re.escape(part) for part in normalized_alias.split()
        ) + r"(?!\w)"
        if re.search(pattern, searchable, flags=re.UNICODE):
            return variant
    return ""


def is_relevant_article(
    title: str,
    description: str,
    query_variants: Sequence[str],
) -> bool:
    return bool(match_relevant_alias(title, description, query_variants))


def is_removed_article(*values: str) -> bool:
    combined = " ".join(clean_text(value).casefold() for value in values)
    return "[removed]" in combined or combined.strip() in {"removed", "null"}


def filter_articles(
    data: Dict[str, Any],
    query_variants: Sequence[str],
    *,
    diagnostics: MutableMapping[str, int] | None = None,
    ticker: str | None = None,
    review_articles: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for article in data.get("articles", []):
        source = article.get("source") or {}
        title = clean_text(article.get("title"))
        description = clean_text(article.get("description"))
        content = clean_text(article.get("content"))
        url = clean_text(article.get("url"))
        published_at = clean_text(article.get("publishedAt"))
        if not title or not url or not published_at or (not description and not content):
            continue
        if is_removed_article(title, description, content):
            continue
        exclusion_reason = classify_non_news_source(
            url,
            (source.get("name") if isinstance(source, dict) else ""),
        )
        if exclusion_reason:
            if diagnostics is not None:
                diagnostics[exclusion_reason] = diagnostics.get(exclusion_reason, 0) + 1
            continue
        matched_alias = match_relevant_alias(title, description, query_variants)
        if not matched_alias:
            continue
        normalized_url = normalize_url(url)
        normalized_title = normalize_title(title)
        if normalized_url in seen_urls or normalized_title in seen_titles:
            continue
        seen_urls.add(normalized_url)
        seen_titles.add(normalized_title)
        row = {
                "matched_alias": matched_alias,
                "provider": "newsapi",
                "source_id": clean_text(source.get("id")),
                "source_name": clean_text(source.get("name")),
                "source_domain": clean_text(urlparse(url).netloc).casefold().removeprefix("www."),
                "author": clean_text(article.get("author")),
                "title": title,
                "description": description,
                "url": url,
                "url_to_image": clean_text(article.get("urlToImage")),
                "published_at": published_at,
                "content_snippet": content,
            }
        if ticker and review_articles is not None:
            entity_decision = classify_entity_relevance(
                ticker,
                title,
                description,
                matched_alias=matched_alias,
            )
            row.update(entity_decision.to_dict())
            if entity_decision.status != "included":
                review_articles.append(row)
                continue
        rows.append(row)
    return rows


def get_article_sentiment_text(
    article: Dict[str, Any], *, mode: str = TEXT_MODE_DEFAULT
) -> str:
    """Build the same bounded article text used by the deployed FinBERT recipe."""

    return build_sentiment_text(article, mode=mode)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_sentiment_runtime_config() -> Dict[str, Any]:
    if not SENTIMENT_RUNTIME_CONFIG_FILE.exists():
        return {}
    try:
        loaded_config = json.loads(SENTIMENT_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded_config if isinstance(loaded_config, dict) else {}


def sentiment_runtime_variant() -> str:
    """Return the explicit override, configured active variant, or stable default.

    Environment overrides remain useful for an isolated challenger check.  The
    persisted ``active_variant`` is only changed by the guarded promotion or
    rollback workflow, so a restart cannot silently select whichever
    challenger happens to be listed first in the JSON file.
    """
    override = os.getenv(SENTIMENT_VARIANT_ENV_VAR, "").strip()
    if override:
        return override
    configured = str(_read_sentiment_runtime_config().get("active_variant", "")).strip()
    return configured or DEFAULT_SENTIMENT_VARIANT


def _validate_configured_hashes(
    expected_hashes: Any,
    model_directory: Path,
    *,
    label: str,
) -> None:
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        return
    actual_hashes = sentiment_artifact_hashes_for_directory(model_directory)
    mismatches = [
        key for key, expected in expected_hashes.items()
        if actual_hashes.get(key) != expected
    ]
    if mismatches:
        raise FinBERTArtifactError(
            f"FinBERT {label} artifact hashes do not match the deployed configuration: "
            + ", ".join(mismatches)
        )


def load_sentiment_model() -> object:
    """Load the approved FinBERT runtime backend.

    The TF-IDF classifier remains available to the offline training and
    benchmark pipelines, but runtime evidence is intentionally standardized on
    FinBERT so provider comparisons measure retrieval rather than classifier
    changes.
    """
    runtime_config = _read_sentiment_runtime_config()
    variant = sentiment_runtime_variant()
    if variant != DEFAULT_SENTIMENT_VARIANT:
        challenger = (runtime_config.get("challengers") or {}).get(variant)
        if not isinstance(challenger, dict):
            raise ValueError(
                f"Unknown sentiment challenger {variant!r}. "
                "Add it to models/sentiment-runtime-config.json before enabling it."
            )
        if str(challenger.get("backend", "")).casefold() != "weighted_finbert_ensemble":
            raise ValueError(f"Sentiment challenger {variant!r} is not a weighted FinBERT ensemble")
        components_config = challenger.get("components")
        weights = challenger.get("weights")
        if not isinstance(components_config, dict) or not isinstance(weights, dict):
            raise ValueError(f"Sentiment challenger {variant!r} has incomplete component configuration")
        components: Dict[str, object] = {}
        for name, component_config in components_config.items():
            if not isinstance(component_config, dict):
                raise ValueError(f"Sentiment challenger component {name!r} is invalid")
            seed_models = component_config.get("seed_models")
            if seed_models is not None:
                if not isinstance(seed_models, list) or len(seed_models) < 2:
                    raise ValueError(
                        f"Sentiment challenger seed-average component {name!r} needs at least two seed models"
                    )
                seed_components: Dict[str, object] = {}
                for index, seed_config in enumerate(seed_models):
                    if not isinstance(seed_config, dict):
                        raise ValueError(
                            f"Sentiment challenger seed model {name!r}[{index}] is invalid"
                        )
                    seed_directory_value = seed_config.get("model_directory")
                    if not seed_directory_value:
                        raise ValueError(
                            f"Sentiment challenger seed model {name!r}[{index}] has no model directory"
                        )
                    seed_directory = _resolve_project_path(seed_directory_value)
                    _validate_configured_hashes(
                        seed_config.get("artifact_hashes"),
                        seed_directory,
                        label=f"challenger component {name} seed {index}",
                    )
                    seed_name = str(seed_config.get("seed", index))
                    if seed_name in seed_components:
                        raise ValueError(
                            f"Sentiment challenger component {name!r} repeats seed {seed_name!r}"
                        )
                    seed_components[seed_name] = load_finbert_sentiment_model(
                        str(seed_directory),
                        temperature=float(seed_config.get("temperature", 1.0)),
                    )
                components[str(name)] = AveragedFinBERTEnsemble(
                    seed_components,
                    variant_id=f"{variant}:{name}:seed-average",
                )
                continue
            directory_value = component_config.get("model_directory")
            if not directory_value:
                raise ValueError(f"Sentiment challenger component {name!r} has no model directory")
            directory = _resolve_project_path(directory_value)
            _validate_configured_hashes(
                component_config.get("artifact_hashes"),
                directory,
                label=f"challenger component {name}",
            )
            components[str(name)] = load_finbert_sentiment_model(
                str(directory),
                temperature=float(component_config.get("temperature", 1.0)),
            )
        model = WeightedFinBERTEnsemble(
            components,
            {str(name): float(value) for name, value in weights.items()},
            variant_id=variant,
            ensemble_temperature=float(challenger.get("ensemble_temperature", 1.0)),
        )
        setattr(
            model,
            "sentiment_text_mode",
            str(challenger.get("text_mode") or TEXT_MODE_DEFAULT),
        )
        return model
    backend = str(runtime_config.get("backend", "finbert")).strip().casefold()
    if backend != "finbert":
        raise ValueError(
            "The runtime sentiment backend must be FinBERT. "
            "TF-IDF is retained only for offline benchmarking."
        )
    configured_directory = runtime_config.get("model_directory")
    model_directory = _resolve_project_path(
        configured_directory or FINBERT_SENTIMENT_MODEL_DIR
    )
    _validate_configured_hashes(
        runtime_config.get("artifact_hashes"),
        model_directory,
        label="runtime",
    )
    temperature = float(runtime_config.get("temperature", 1.0))
    model = load_finbert_sentiment_model(str(model_directory), temperature=temperature)
    # PhraseBank's historical representation is headline + lead.  A challenger
    # may opt into a target-aware representation, which must travel with its
    # immutable runtime recipe rather than being inferred from the variant name.
    setattr(model, "sentiment_text_mode", TEXT_MODE_DEFAULT)
    return model


def load_or_train_sentiment_model() -> object:
    """Backward-compatible name; runtime training has intentionally been removed."""
    return load_sentiment_model()


def add_sentiment_scores(rows: List[Dict[str, Any]], model: object) -> int:
    if not rows:
        return 0
    text_mode = str(getattr(model, "sentiment_text_mode", TEXT_MODE_DEFAULT))
    texts = [get_article_sentiment_text(row, mode=text_mode) for row in rows]
    classes = [str(value).lower() for value in model.classes_]
    component_probabilities = {}
    if hasattr(model, "predict_component_proba") and hasattr(model, "blend_component_proba"):
        component_probabilities = model.predict_component_proba(texts)
        probabilities = model.blend_component_proba(component_probabilities)
    else:
        probabilities = model.predict_proba(texts)
    variant = str(getattr(model, "sentiment_model_variant", DEFAULT_SENTIMENT_VARIANT))
    component_weights = dict(getattr(model, "component_weights", {"phrasebank": 1.0}))
    for row, probability_row in zip(rows, probabilities):
        class_probabilities = dict(zip(classes, probability_row))
        label = max(class_probabilities, key=class_probabilities.get)
        confidence = float(class_probabilities[label])
        update = {
                "sentiment_label": label,
                "sentiment_score": SENTIMENT_SCORE_MAP[label],
                "sentiment_probability": confidence,
                "probability_negative": float(class_probabilities.get("negative", 0.0)),
                "probability_neutral": float(class_probabilities.get("neutral", 0.0)),
                "probability_positive": float(class_probabilities.get("positive", 0.0)),
                "sentiment_uncertain": confidence < SENTIMENT_PROBABILITY_THRESHOLD,
                "sentiment_model_variant": variant,
                "sentiment_model_weights": component_weights,
            }
        row.update(update)
    if component_probabilities:
        for row_index, row in enumerate(rows):
            for component_name, component_matrix in component_probabilities.items():
                component_values = dict(zip(classes, component_matrix[row_index]))
                for class_name in ("negative", "neutral", "positive"):
                    row[f"{component_name}_probability_{class_name}"] = float(
                        component_values.get(class_name, 0.0)
                    )
    return len(rows)


def prediction_cutoff_at(as_of_date: date) -> datetime:
    """Return the completed Bursa session cutoff for a market observation."""

    return datetime.combine(as_of_date, BURSA_SESSION_CLOSE, tzinfo=BURSA_TIMEZONE)


def article_age_days(
    published_at: str,
    as_of_date: date,
    cutoff_at: datetime | None = None,
) -> float:
    published = pd.to_datetime(published_at, utc=True, errors="coerce")
    if pd.isna(published):
        return math.inf
    as_of = cutoff_at or datetime.combine(as_of_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    published_at_utc = published.to_pydatetime()
    age = (as_of.astimezone(timezone.utc) - published_at_utc).total_seconds() / 86400.0
    # Future articles are not candidates for a point-in-time context. Returning
    # infinity keeps the existing caller contract while making the exclusion
    # explicit and leakage-safe.
    return age if age >= 0 else math.inf


def calculate_recency_weight(age_days: float, half_life_days: float) -> float:
    """Return exponential time decay for an article, with safe invalid-age handling."""
    if not math.isfinite(age_days) or half_life_days <= 0:
        return 0.0
    return math.exp(-math.log(2) * age_days / half_life_days)


def build_news_contexts(
    rows: Sequence[Dict[str, Any]],
    as_of_date: date,
    cutoff_at: datetime | None = None,
) -> Dict[str, Dict[str, Any]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    for horizon, window_days in CONTEXT_WINDOWS.items():
        half_life = CONTEXT_HALF_LIVES[horizon]
        candidates = [
            row
            for row in rows
            if article_age_days(str(row["published_at"]), as_of_date, cutoff_at) <= window_days
        ]
        usable = [
            row
            for row in candidates
            if not bool(row.get("sentiment_uncertain", True))
            and row.get("sentiment_score") in {-1, 0, 1}
        ]
        sources = {
            (clean_text(row.get("source_domain")) or clean_text(row.get("source_name"))).casefold()
            for row in usable
            if clean_text(row.get("source_domain")) or clean_text(row.get("source_name"))
        }
        events = {
            str(row.get("event_id") or stable_event_id(row))
            for row in usable
        }
        predominantly_uncertain = bool(candidates) and len(usable) < (len(candidates) / 2)
        if not usable:
            coverage = "Unavailable"
        elif (
            len(usable) >= 5
            and len(sources) >= 3
            and len(events) >= 3
            and not predominantly_uncertain
        ):
            coverage = "High"
        elif (
            len(usable) >= 3
            and len(sources) >= 2
            and len(events) >= 2
            and not predominantly_uncertain
        ):
            coverage = "Medium"
        else:
            coverage = "Low"

        usable_ages: list[float] = []
        usable_confidences: list[float] = []
        contribution_rows: list[dict[str, Any]] = []
        for row in usable:
            age_days = article_age_days(str(row["published_at"]), as_of_date, cutoff_at)
            usable_ages.append(float(age_days))
            sentiment_confidence = float(row["sentiment_probability"])
            usable_confidences.append(sentiment_confidence)
            recency_weight = calculate_recency_weight(age_days, half_life)
            weight = recency_weight * sentiment_confidence
            sentiment_score = float(row["sentiment_score"])
            contribution_rows.append(
                {
                    "article_id": str(row.get("article_id") or row.get("url") or row.get("title") or ""),
                    "event_id": str(row.get("event_id") or stable_event_id(row)),
                    "age_days": float(age_days),
                    "recency_weight": float(recency_weight),
                    "sentiment_confidence": float(sentiment_confidence),
                    "raw_weight": float(weight),
                    "sentiment_score": sentiment_score,
                    "sentiment_label": str(row.get("sentiment_label") or "neutral"),
                    "event_category": str(
                        row.get("event_category")
                        or classify_material_event(
                            row.get("title", ""), row.get("description", "")
                        )
                    ).casefold(),
                    "disclosure_type": str(row.get("disclosure_type") or "").casefold(),
                }
            )
        # A syndicated story is one information event. Copies remain visible,
        # but together receive no more influence than the strongest
        # recent/confident representation of that event.
        event_rows: Dict[str, list[dict[str, Any]]] = {}
        for contribution in contribution_rows:
            event_rows.setdefault(str(contribution["event_id"]), []).append(contribution)
        event_total_weight = 0.0
        weighted_sum = 0.0
        event_representatives: list[dict[str, Any]] = []
        for event_contributions in event_rows.values():
            raw_total = sum(float(item["raw_weight"]) for item in event_contributions)
            event_weight = max(
                (float(item["raw_weight"]) for item in event_contributions),
                default=0.0,
            )
            event_score = (
                sum(
                    float(item["sentiment_score"]) * float(item["raw_weight"])
                    for item in event_contributions
                )
                / raw_total
                if raw_total
                else 0.0
            )
            event_total_weight += event_weight
            weighted_sum += event_score * event_weight
            if event_contributions:
                event_representatives.append(
                    max(event_contributions, key=lambda item: float(item["raw_weight"]))
                )
            for item in event_contributions:
                within_event = float(item["raw_weight"]) / raw_total if raw_total else 0.0
                item["event_weight"] = event_weight
                item["effective_weight"] = event_weight * within_event
        weighted_score = weighted_sum / event_total_weight if event_total_weight else None
        if event_total_weight:
            for contribution in contribution_rows:
                contribution["normalized_weight"] = (
                    contribution["effective_weight"] / event_total_weight
                )
                contribution["signed_contribution"] = contribution["sentiment_score"] * contribution["normalized_weight"]
        else:
            for contribution in contribution_rows:
                contribution["normalized_weight"] = 0.0
                contribution["signed_contribution"] = 0.0
        labels = [str(row["sentiment_label"]) for row in usable]
        contexts[str(horizon)] = {
            "horizon_days": horizon,
            "window_days": window_days,
            "half_life_days": half_life,
            "candidate_article_count": len(candidates),
            "usable_article_count": len(usable),
            "uncertain_article_count": len(candidates) - len(usable),
            "unique_source_count": len(sources),
            "independent_event_count": len(events),
            "weighted_sentiment_score": weighted_score,
            "positive_count": labels.count("positive"),
            "neutral_count": labels.count("neutral"),
            "negative_count": labels.count("negative"),
            "mean_sentiment_confidence": (
                sum(usable_confidences) / len(usable_confidences)
                if usable_confidences
                else None
            ),
            "max_sentiment_confidence": (
                max(usable_confidences) if usable_confidences else None
            ),
            "mean_article_age_days": (
                sum(usable_ages) / len(usable_ages) if usable_ages else None
            ),
            "newest_article_age_days": (
                min(usable_ages) if usable_ages else None
            ),
            "article_contributions": contribution_rows,
            "coverage_confidence": coverage,
            "low_confidence_flag": coverage in {"Low", "Unavailable"},
            "weighting_policy_version": "event_capped_recency_x_finbert_confidence_v2",
        }
        event_count = len(event_rows)
        contexts[str(horizon)].update(
            {
                "news_event_reliability_mean": (
                    event_total_weight / event_count if event_count else 0.0
                ),
                "news_positive_event_impact": (
                    sum(
                        float(item.get("event_weight", 0.0))
                        for item in event_representatives
                        if float(item["sentiment_score"]) > 0
                    )
                    / event_total_weight
                    if event_total_weight
                    else 0.0
                ),
                "news_negative_event_impact": (
                    sum(
                        float(item.get("event_weight", 0.0))
                        for item in event_representatives
                        if float(item["sentiment_score"]) < 0
                    )
                    / event_total_weight
                    if event_total_weight
                    else 0.0
                ),
                "news_official_disclosure_share": (
                    sum(
                        float(item.get("event_weight", 0.0))
                        for item in event_representatives
                        if item.get("disclosure_type")
                        in {
                            "official",
                            "official_disclosure",
                            "bursa_announcement",
                            "issuer_ir",
                            "press_release",
                        }
                    )
                    / event_total_weight
                    if event_total_weight
                    else 0.0
                ),
            }
        )
        for category in (
            "financial_results_guidance",
            "dividends_distributions",
            "contracts_order_wins",
            "acquisitions_disposals_investments",
            "financing_capital_changes",
            "regulatory_legal_governance",
            "operational_product_strategy",
            "management_changes",
        ):
            category_weight = sum(
                float(item.get("event_weight", 0.0))
                for item in event_representatives
                if item.get("event_category") == category
            )
            contexts[str(horizon)][f"news_event_category_{category}_share"] = (
                category_weight / event_total_weight if event_total_weight else 0.0
            )
    return contexts


def annotate_contribution_horizons(
    rows: Sequence[Dict[str, Any]],
    as_of_date: date,
    cutoff_at: datetime | None = None,
) -> None:
    """Attach server-authoritative horizon contribution flags to article rows."""

    for row in rows:
        horizons: list[int] = []
        usable = not bool(row.get("sentiment_uncertain", True)) and row.get("sentiment_score") in {-1, 0, 1}
        if usable:
            age = article_age_days(str(row.get("published_at", "")), as_of_date, cutoff_at)
            for horizon, window_days in CONTEXT_WINDOWS.items():
                if age <= window_days:
                    horizons.append(horizon)
        row["contributes_to_horizons"] = horizons


def build_output_paths(ticker: str) -> Dict[str, Path]:
    ticker_slug = slugify(ticker)
    return {
        "raw_json": FINANCIAL_NEWS_JSON_DIR / f"newsapi-{ticker_slug}-raw.json",
        "filtered_articles": FINANCIAL_NEWS_CSV_DIR / f"newsapi-{ticker_slug}-filtered-articles.csv",
        "context_summary": FINANCIAL_NEWS_CSV_DIR / f"newsapi-{ticker_slug}-context-summary.csv",
        "dashboard_top5": FINANCIAL_NEWS_CSV_DIR / f"newsapi-{ticker_slug}-dashboard-top5.csv",
    }


def enrich_output_rows(
    ticker: str,
    company_name: str,
    query_text: str,
    rows: Sequence[Dict[str, Any]],
    *,
    automatic_included: bool = True,
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        enriched_row = {
            "query_ticker": ticker,
            "query_company": company_name,
            "query_text": query_text,
            "display_rank": index,
            **row,
        }
        enriched_row["article_id"] = stable_article_id(enriched_row)
        enriched_row["event_id"] = stable_event_id(enriched_row)
        enriched_row.setdefault("automatic_relevance", "included" if automatic_included else "review")
        enriched_row.setdefault("automatic_included", automatic_included)
        enriched_row.setdefault("effective_included", automatic_included)
        enriched_row.setdefault("user_override", None)
        enriched.append(enriched_row)
    return enriched


def _save_article_csv(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ARTICLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_context_csv(contexts: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    pd.DataFrame(contexts.values()).to_csv(output_path, index=False)


def retrieve_financial_news(
    ticker: str,
    as_of_date: date | str | None = None,
    user_aliases: Sequence[str] | None = None,
    api_key: str | None = None,
    response_data: Dict[str, Any] | None = None,
    *,
    persist_outputs: bool = True,
) -> NewsContextResult:
    ensure_data_structure()
    metadata = load_stock_metadata()
    normalized_ticker = clean_text(ticker).upper()
    if normalized_ticker not in metadata:
        raise ValueError(f"Unsupported ticker: {normalized_ticker}")
    company_name = metadata[normalized_ticker]
    effective_date = pd.Timestamp(as_of_date).date() if as_of_date is not None else date.today()
    variants = build_company_query_variants(normalized_ticker, company_name, user_aliases)
    query_text = build_newsapi_query(variants)
    # Preflight the shared sentiment engine before spending a provider request.
    # NewsAPI and Marketaux cannot recover from the same FinBERT runtime failure.
    model = load_sentiment_model()
    data = response_data if response_data is not None else fetch_newsapi_response(
        api_key or get_newsapi_key(), query_text, NEWS_RETRIEVAL_LOOKBACK_DAYS, effective_date
    )
    filter_diagnostics: Dict[str, int] = {}
    review_rows: List[Dict[str, Any]] = []
    rows = filter_articles(
        data,
        variants,
        diagnostics=filter_diagnostics,
        ticker=normalized_ticker,
        review_articles=review_rows,
    )
    add_sentiment_scores(rows, model)
    enriched_rows = enrich_output_rows(normalized_ticker, company_name, query_text, rows)
    enriched_review_rows = enrich_output_rows(
        normalized_ticker,
        company_name,
        query_text,
        review_rows,
        automatic_included=False,
    )
    cutoff_at = prediction_cutoff_at(effective_date)
    annotate_contribution_horizons(enriched_rows, effective_date, cutoff_at)
    contexts = build_news_contexts(enriched_rows, effective_date, cutoff_at)
    paths = build_output_paths(normalized_ticker) if persist_outputs else {}
    if persist_outputs:
        paths["raw_json"].write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        _save_article_csv(enriched_rows, paths["filtered_articles"])
        _save_article_csv(enriched_rows[:DASHBOARD_ARTICLES], paths["dashboard_top5"])
        _save_context_csv(contexts, paths["context_summary"])

    return NewsContextResult(
        ticker=normalized_ticker,
        company_name=company_name,
        query_text=query_text,
        as_of_date=effective_date.isoformat(),
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        raw_article_count=len(data.get("articles", [])),
        relevant_article_count=len(rows),
        results_truncated=int(data.get("totalResults", 0)) > PAGE_SIZE,
        contexts=contexts,
        articles=enriched_rows,
        output_paths={key: str(value) for key, value in paths.items()},
        provider_mode="newsapi",
        provider_article_counts={"newsapi": len(rows)},
        provider_status={"newsapi": "ok"},
        provider_metadata={
            **build_news_context_metadata(),
            "prediction_cutoff_at": cutoff_at.isoformat(),
            "excluded_non_news_count": sum(filter_diagnostics.values()),
            "excluded_non_news_reasons": filter_diagnostics,
            "entity_review_article_count": len(enriched_review_rows),
        },
        review_articles=enriched_review_rows,
        automatic_articles=list(enriched_rows),
        automatic_review_articles=list(enriched_review_rows),
        automatic_contexts=dict(contexts),
    )


def retrieve_with_fallback(
    api_key: str,
    query_text: str,
    query_variants: Sequence[str],
) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]], str, bool]:
    """Compatibility wrapper for the replaced fallback API."""
    data = fetch_newsapi_response(api_key, query_text, NEWS_RETRIEVAL_LOOKBACK_DAYS)
    rows = filter_articles(data, query_variants)
    coverage = "High" if len(rows) >= 5 else "Low"
    return NEWS_RETRIEVAL_LOOKBACK_DAYS, data, rows, coverage, coverage == "Low"


def run_financial_news_download_pipeline(
    ticker: str | None = None,
    as_of_date: date | str | None = None,
) -> NewsContextResult:
    selected_ticker = clean_text(ticker).upper() if ticker else input("Enter stock ticker: ").strip().upper()
    from pipeline.marketaux_news_provider import retrieve_news_with_marketaux_fallback

    result = retrieve_news_with_marketaux_fallback(selected_ticker, 5, as_of_date=as_of_date)
    print(f"\nFinancial news context for {result.ticker} - {result.company_name}")
    print(f"Relevant articles: {result.relevant_article_count}")
    for horizon in (1, 5, 10):
        context = result.contexts[str(horizon)]
        score = context["weighted_sentiment_score"]
        score_text = "N/A" if score is None else f"{score:.4f}"
        print(
            f"{horizon}D context ({context['window_days']} days): "
            f"score={score_text}, coverage={context['coverage_confidence']}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve financial news sentiment context")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    run_financial_news_download_pipeline(args.ticker, args.as_of)


if __name__ == "__main__":
    main()
