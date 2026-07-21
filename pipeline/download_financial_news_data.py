from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.pipeline import Pipeline

from pipeline.finbert_sentiment_model import load_finbert_sentiment_model


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

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
NEWSAPI_KEY_ENV_VAR = "NEWSAPI_KEY"
SEARCH_IN = "title,description"
LANGUAGE = "en"
SORT_BY = "publishedAt"
PAGE_SIZE = 100
NEWS_RETRIEVAL_LOOKBACK_DAYS = 30
NEWS_CONTEXT_CONFIG_VERSION = 3
NEWS_HISTORICAL_CONTEXT_START_DAYS = NEWS_RETRIEVAL_LOOKBACK_DAYS + 1
NEWS_HISTORICAL_CONTEXT_END_DAYS = 90
SENTIMENT_PROBABILITY_THRESHOLD = 0.55
DASHBOARD_ARTICLES = 5

CONTEXT_WINDOWS = {1: 7, 5: 14, 10: 30}
CONTEXT_HALF_LIVES = {1: 3.5, 5: 7.0, 10: 15.0}
SENTIMENT_SCORE_MAP = {"positive": 1, "neutral": 0, "negative": -1}

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
    "sentiment_uncertain",
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_news_context_metadata() -> Dict[str, Any]:
    """Return the immutable retrieval/context contract attached to news results."""
    return {
        "news_context_config_version": NEWS_CONTEXT_CONFIG_VERSION,
        "retrieval_lookback_days": NEWS_RETRIEVAL_LOOKBACK_DAYS,
        "historical_context_start_days": NEWS_HISTORICAL_CONTEXT_START_DAYS,
        "historical_context_end_days": NEWS_HISTORICAL_CONTEXT_END_DAYS,
        "context_windows": {str(horizon): window for horizon, window in CONTEXT_WINDOWS.items()},
        "context_half_lives": {str(horizon): half_life for horizon, half_life in CONTEXT_HALF_LIVES.items()},
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
    return (end_date - timedelta(days=lookback_days)).isoformat(), end_date.isoformat()


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
        matched_alias = match_relevant_alias(title, description, query_variants)
        if not matched_alias:
            continue
        normalized_url = normalize_url(url)
        normalized_title = normalize_title(title)
        if normalized_url in seen_urls or normalized_title in seen_titles:
            continue
        seen_urls.add(normalized_url)
        seen_titles.add(normalized_title)
        rows.append(
            {
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
        )
    return rows


def get_article_sentiment_text(article: Dict[str, Any]) -> str:
    return clean_text(f"{article.get('title', '')} {article.get('description', '')}")


def load_sentiment_model() -> object:
    """Load the approved FinBERT runtime backend.

    The TF-IDF classifier remains available to the offline training and
    benchmark pipelines, but runtime evidence is intentionally standardized on
    FinBERT so provider comparisons measure retrieval rather than classifier
    changes.
    """
    runtime_config: Dict[str, Any] = {}
    if SENTIMENT_RUNTIME_CONFIG_FILE.exists():
        try:
            loaded_config = json.loads(SENTIMENT_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded_config, dict):
                runtime_config = loaded_config
        except (OSError, json.JSONDecodeError):
            runtime_config = {}
    backend = str(runtime_config.get("backend", "finbert")).strip().casefold()
    if backend != "finbert":
        raise ValueError(
            "The runtime sentiment backend must be FinBERT. "
            "TF-IDF is retained only for offline benchmarking."
        )
    configured_directory = runtime_config.get("model_directory")
    model_directory = Path(configured_directory) if configured_directory else FINBERT_SENTIMENT_MODEL_DIR
    if not model_directory.is_absolute():
        model_directory = PROJECT_ROOT / model_directory
    temperature = float(runtime_config.get("temperature", 1.0))
    return load_finbert_sentiment_model(str(model_directory), temperature=temperature)


def load_or_train_sentiment_model() -> object:
    """Backward-compatible name; runtime training has intentionally been removed."""
    return load_sentiment_model()


def add_sentiment_scores(rows: List[Dict[str, Any]], model: Pipeline) -> int:
    if not rows:
        return 0
    texts = [get_article_sentiment_text(row) for row in rows]
    probabilities = model.predict_proba(texts)
    classes = [str(value).lower() for value in model.classes_]
    for row, probability_row in zip(rows, probabilities):
        class_probabilities = dict(zip(classes, probability_row))
        label = max(class_probabilities, key=class_probabilities.get)
        confidence = float(class_probabilities[label])
        row.update(
            {
                "sentiment_label": label,
                "sentiment_score": SENTIMENT_SCORE_MAP[label],
                "sentiment_probability": confidence,
                "probability_negative": float(class_probabilities.get("negative", 0.0)),
                "probability_neutral": float(class_probabilities.get("neutral", 0.0)),
                "probability_positive": float(class_probabilities.get("positive", 0.0)),
                "sentiment_uncertain": confidence < SENTIMENT_PROBABILITY_THRESHOLD,
            }
        )
    return len(rows)


def article_age_days(published_at: str, as_of_date: date) -> float:
    published = pd.to_datetime(published_at, utc=True, errors="coerce")
    if pd.isna(published):
        return math.inf
    as_of = datetime.combine(as_of_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    return max(0.0, (as_of - published.to_pydatetime()).total_seconds() / 86400.0)


def calculate_recency_weight(age_days: float, half_life_days: float) -> float:
    """Return exponential time decay for an article, with safe invalid-age handling."""
    if not math.isfinite(age_days) or half_life_days <= 0:
        return 0.0
    return math.exp(-math.log(2) * age_days / half_life_days)


def build_news_contexts(
    rows: Sequence[Dict[str, Any]],
    as_of_date: date,
) -> Dict[str, Dict[str, Any]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    for horizon, window_days in CONTEXT_WINDOWS.items():
        half_life = CONTEXT_HALF_LIVES[horizon]
        candidates = [
            row for row in rows if article_age_days(str(row["published_at"]), as_of_date) <= window_days
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
        predominantly_uncertain = bool(candidates) and len(usable) < (len(candidates) / 2)
        if not usable:
            coverage = "Unavailable"
        elif len(usable) >= 5 and len(sources) >= 3 and not predominantly_uncertain:
            coverage = "High"
        elif len(usable) >= 3 and len(sources) >= 2 and not predominantly_uncertain:
            coverage = "Medium"
        else:
            coverage = "Low"

        weighted_sum = 0.0
        total_weight = 0.0
        for row in usable:
            age_days = article_age_days(str(row["published_at"]), as_of_date)
            recency_weight = calculate_recency_weight(age_days, half_life)
            weight = recency_weight * float(row["sentiment_probability"])
            weighted_sum += float(row["sentiment_score"]) * weight
            total_weight += weight
        weighted_score = weighted_sum / total_weight if total_weight else None
        labels = [str(row["sentiment_label"]) for row in usable]
        contexts[str(horizon)] = {
            "horizon_days": horizon,
            "window_days": window_days,
            "half_life_days": half_life,
            "candidate_article_count": len(candidates),
            "usable_article_count": len(usable),
            "uncertain_article_count": len(candidates) - len(usable),
            "unique_source_count": len(sources),
            "weighted_sentiment_score": weighted_score,
            "positive_count": labels.count("positive"),
            "neutral_count": labels.count("neutral"),
            "negative_count": labels.count("negative"),
            "coverage_confidence": coverage,
            "low_confidence_flag": coverage in {"Low", "Unavailable"},
        }
    return contexts


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
) -> List[Dict[str, Any]]:
    return [
        {
            "query_ticker": ticker,
            "query_company": company_name,
            "query_text": query_text,
            "display_rank": index,
            **row,
        }
        for index, row in enumerate(rows, start=1)
    ]


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
    rows = filter_articles(data, variants)
    add_sentiment_scores(rows, model)
    contexts = build_news_contexts(rows, effective_date)
    enriched_rows = enrich_output_rows(normalized_ticker, company_name, query_text, rows)
    paths = build_output_paths(normalized_ticker)
    paths["raw_json"].write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_article_csv(enriched_rows, paths["filtered_articles"])
    _save_article_csv(enriched_rows[:DASHBOARD_ARTICLES], paths["dashboard_top5"])
    _save_context_csv(contexts, paths["context_summary"])

    return NewsContextResult(
        ticker=normalized_ticker,
        company_name=company_name,
        query_text=query_text,
        as_of_date=effective_date.isoformat(),
        retrieved_at=datetime.now().isoformat(timespec="seconds"),
        raw_article_count=len(data.get("articles", [])),
        relevant_article_count=len(rows),
        results_truncated=int(data.get("totalResults", 0)) > PAGE_SIZE,
        contexts=contexts,
        articles=enriched_rows,
        output_paths={key: str(value) for key, value in paths.items()},
        provider_mode="newsapi",
        provider_article_counts={"newsapi": len(rows)},
        provider_status={"newsapi": "ok"},
        provider_metadata=build_news_context_metadata(),
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
