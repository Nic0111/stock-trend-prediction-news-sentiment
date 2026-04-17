from __future__ import annotations

import csv
import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# Project configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FINANCIAL_NEWS_DATA_DIR = DATA_DIR / "financial-news-dataset"
FINANCIAL_NEWS_JSON_DIR = FINANCIAL_NEWS_DATA_DIR / "json"
FINANCIAL_NEWS_CSV_DIR = FINANCIAL_NEWS_DATA_DIR / "csv"
CONFIG_DIR = PROJECT_ROOT / "config"

METADATA_FILE = DATA_DIR / "stock-metadata.csv"
SENTIMENT_DATASET_FILE = (
    DATA_DIR
    / "financial-news-sentiment-dataset"
    / "financial-news-sentiment-dataset.csv"
)
SENTIMENT_MODEL_FILE = MODELS_DIR / "financial-news-sentiment-model.joblib"
NEWSAPI_KEY_FILE = CONFIG_DIR / "newsapi_key.txt"

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
NEWSAPI_KEY_ENV_VAR = "NEWSAPI_KEY"

SEARCH_IN = "title,description"
LANGUAGE = "en"
SORT_BY = "publishedAt"
PAGE_SIZE = 20
PAGE = 1

INITIAL_LOOKBACK_DAYS = 7
FALLBACK_LOOKBACK_DAYS = 14
EXTENDED_LOOKBACK_DAYS = 21
LOOKBACK_SEQUENCE = [INITIAL_LOOKBACK_DAYS, FALLBACK_LOOKBACK_DAYS, EXTENDED_LOOKBACK_DAYS]
MIN_FILTERED_ARTICLES = 5
MAX_SENTIMENT_ARTICLES = 15
DASHBOARD_ARTICLES = 5

SENTIMENT_SCORE_MAP = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}

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
    "lookback_days_used",
    "display_rank",
    "source_id",
    "source_name",
    "author",
    "title",
    "description",
    "url",
    "url_to_image",
    "published_at",
    "content_snippet",
    "sentiment_label",
    "sentiment_score",
]

SUMMARY_COLUMNS = [
    "query_ticker",
    "query_company",
    "query_text",
    "from_date",
    "to_date",
    "lookback_days_used",
    "newsapi_total_results",
    "raw_article_count",
    "filtered_article_count",
    "sentiment_article_count",
    "average_sentiment_score",
    "positive_count",
    "neutral_count",
    "negative_count",
    "coverage_confidence",
    "weak_coverage_flag",
    "low_confidence_flag",
    "low_confidence_reasons",
    "raw_json_path",
    "filtered_articles_path",
    "dashboard_top5_path",
    "retrieved_at",
]


# ============================================================
# Generic helpers
# ============================================================
def ensure_data_structure() -> None:
    FINANCIAL_NEWS_JSON_DIR.mkdir(parents=True, exist_ok=True)
    FINANCIAL_NEWS_CSV_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "unknown"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique_values: List[str] = []

    for value in values:
        cleaned = clean_text(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            unique_values.append(cleaned)

    return unique_values


def normalize_title(title: str) -> str:
    title = clean_text(title).casefold()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return clean_text(title)


def normalize_url(url: str) -> str:
    parsed = urlparse(clean_text(url))
    if not parsed.scheme or not parsed.netloc:
        return clean_text(url).casefold()

    normalized_path = parsed.path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            normalized_path,
            "",
            "",
            "",
        )
    )


def get_newsapi_key() -> str:
    if NEWSAPI_KEY_FILE.exists():
        api_key = NEWSAPI_KEY_FILE.read_text(encoding="utf-8").strip()
        if api_key:
            return api_key

    api_key = os.getenv(NEWSAPI_KEY_ENV_VAR, "").strip()
    if api_key:
        return api_key

    raise RuntimeError(
        "NewsAPI key is not set. Add it to config/newsapi_key.txt or set "
        "the NEWSAPI_KEY environment variable before running "
        "`python main.py download_financial_news`."
    )


# ============================================================
# Query input and NewsAPI request
# ============================================================
def load_stock_metadata() -> Dict[str, str]:
    if not METADATA_FILE.exists():
        return {}

    metadata_df = pd.read_csv(METADATA_FILE)
    return {
        str(row["Ticker"]).strip().upper(): str(row["Company_Name"]).strip()
        for _, row in metadata_df.iterrows()
        if clean_text(row.get("Ticker")) and clean_text(row.get("Company_Name"))
    }


def prompt_for_query_details() -> Tuple[str, str, List[str], str]:
    metadata = load_stock_metadata()

    ticker = input("Enter stock ticker: ").strip().upper()
    while not ticker:
        ticker = input("Stock ticker is required. Enter stock ticker: ").strip().upper()

    company_name = metadata.get(ticker, "")
    if company_name:
        print(f"Matched company from metadata: {company_name}")
    else:
        company_name = input("Ticker not found. Enter company name: ").strip()
        while not company_name:
            company_name = input("Company name is required. Enter company name: ").strip()

    try:
        aliases_input = input(
            "Enter aliases separated by comma, or press Enter to skip: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo aliases provided. Continuing with company-name variants only.")
        aliases_input = ""
    user_aliases = [
        alias.strip()
        for alias in aliases_input.split(",")
        if alias.strip()
    ]

    query_variants = build_company_query_variants(ticker, company_name, user_aliases)
    query_text = build_newsapi_query(query_variants)

    return ticker, company_name, query_variants, query_text


def build_company_query_variants(
    ticker: str,
    company_name: str,
    user_aliases: List[str],
) -> List[str]:
    legal_suffix_pattern = re.compile(
        r"\b(berhad|bhd|corporation|corp|holdings?)\b\.?",
        flags=re.IGNORECASE,
    )

    words = clean_text(company_name).split()
    variants = [company_name]

    without_berhad = re.sub(r"\bberhad\b\.?", "", company_name, flags=re.IGNORECASE)
    variants.append(without_berhad)

    without_legal_suffixes = legal_suffix_pattern.sub("", company_name)
    variants.append(without_legal_suffixes)

    meaningful_words = [
        word
        for word in words
        if not legal_suffix_pattern.fullmatch(word)
    ]

    if len(meaningful_words) >= 2:
        variants.append(" ".join(meaningful_words[:2]))
    if len(meaningful_words) >= 3:
        variants.append(" ".join(meaningful_words[:3]))

    initials = "".join(word[0] for word in meaningful_words if word)
    if 2 <= len(initials) <= 8:
        variants.append(initials.upper())

    compact_name = "".join(meaningful_words)
    if 4 <= len(compact_name) <= 30:
        variants.append(compact_name)

    if meaningful_words and meaningful_words[-1].casefold() == "banking":
        variants.append(f"{' '.join(meaningful_words[:-1])} Bank")
    if "bank" in " ".join(meaningful_words).casefold():
        variants.append(f"{meaningful_words[0]} Bank")

    first_token = words[0] if words else ""
    if first_token.isupper() and 2 <= len(first_token) <= 8:
        variants.append(first_token)

    variants.extend(KNOWN_COMPANY_ALIASES.get(ticker.upper(), []))
    variants.extend(user_aliases)
    return unique_preserve_order(variants)


def build_newsapi_query(query_variants: List[str]) -> str:
    quoted_variants = [f'"{variant}"' for variant in query_variants]
    query_text = " OR ".join(quoted_variants)

    if len(query_text) <= 500:
        return query_text

    trimmed_variants: List[str] = []
    for variant in query_variants:
        candidate = " OR ".join([*(f'"{item}"' for item in trimmed_variants), f'"{variant}"'])
        if len(candidate) > 500:
            break
        trimmed_variants.append(variant)

    return " OR ".join(f'"{variant}"' for variant in trimmed_variants)


def get_news_date_range(lookback_days: int) -> Tuple[str, str]:
    to_date = date.today()
    from_date = to_date - timedelta(days=lookback_days)
    return from_date.isoformat(), to_date.isoformat()


def build_request_url(query_text: str, lookback_days: int) -> str:
    from_date, to_date = get_news_date_range(lookback_days)
    params = {
        "q": query_text,
        "searchIn": SEARCH_IN,
        "from": from_date,
        "to": to_date,
        "language": LANGUAGE,
        "sortBy": SORT_BY,
        "pageSize": str(PAGE_SIZE),
        "page": str(PAGE),
    }
    return f"{NEWSAPI_ENDPOINT}?{urlencode(params)}"


def fetch_newsapi_response(
    api_key: str,
    query_text: str,
    lookback_days: int,
) -> Dict[str, Any]:
    request = Request(
        build_request_url(query_text, lookback_days),
        headers={
            "X-Api-Key": api_key,
            "User-Agent": "stock-investment-analytics-news-retrieval/1.0",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"NewsAPI request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"NewsAPI request failed: {exc.reason}") from exc

    data = json.loads(response_body)
    if data.get("status") != "ok":
        code = data.get("code", "unknown_error")
        message = data.get("message", "No error message returned.")
        raise RuntimeError(f"NewsAPI returned {code}: {message}")

    return data


# ============================================================
# Article filtering and sentiment scoring
# ============================================================
def is_removed_article(*values: str) -> bool:
    combined_text = " ".join(clean_text(value).casefold() for value in values)
    return "[removed]" in combined_text or combined_text in {"removed", "null"}


def is_relevant_article(
    title: str,
    description: str,
    query_variants: List[str],
) -> bool:
    searchable_text = f"{title} {description}".casefold()
    return any(variant.casefold() in searchable_text for variant in query_variants)


def filter_articles(
    data: Dict[str, Any],
    query_variants: List[str],
) -> List[Dict[str, str]]:
    filtered_rows: List[Dict[str, str]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for article in data.get("articles", []):
        source = article.get("source") or {}
        title = clean_text(article.get("title"))
        description = clean_text(article.get("description"))
        content = clean_text(article.get("content"))
        url = clean_text(article.get("url"))
        published_at = clean_text(article.get("publishedAt"))

        if not title or not url or not published_at:
            continue
        if not description and not content:
            continue
        if is_removed_article(title, description, content):
            continue
        if not is_relevant_article(title, description, query_variants):
            continue

        normalized_url = normalize_url(url)
        normalized_title = normalize_title(title)
        if normalized_url in seen_urls or normalized_title in seen_titles:
            continue

        seen_urls.add(normalized_url)
        seen_titles.add(normalized_title)

        filtered_rows.append(
            {
                "source_id": clean_text(source.get("id")),
                "source_name": clean_text(source.get("name")),
                "author": clean_text(article.get("author")),
                "title": title,
                "description": description,
                "url": url,
                "url_to_image": clean_text(article.get("urlToImage")),
                "published_at": published_at,
                "content_snippet": content,
            }
        )

    return filtered_rows


def get_article_sentiment_text(article: Dict[str, str]) -> str:
    return clean_text(f"{article.get('title', '')} {article.get('description', '')}")


def load_or_train_sentiment_model() -> Pipeline:
    if SENTIMENT_MODEL_FILE.exists():
        return joblib.load(SENTIMENT_MODEL_FILE)

    if not SENTIMENT_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Sentiment dataset not found: {SENTIMENT_DATASET_FILE}"
        )

    sentiment_df = read_sentiment_dataset()
    sentiment_df["sentiment"] = sentiment_df["sentiment"].astype(str).str.strip().str.lower()
    sentiment_df["sentence"] = sentiment_df["sentence"].astype(str).str.strip()
    sentiment_df = sentiment_df[
        sentiment_df["sentiment"].isin(SENTIMENT_SCORE_MAP)
        & sentiment_df["sentence"].ne("")
    ]

    if sentiment_df.empty:
        raise ValueError("No valid labelled sentiment rows were found.")

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(sentiment_df["sentence"], sentiment_df["sentiment"])
    joblib.dump(model, SENTIMENT_MODEL_FILE)

    return model


def read_sentiment_dataset() -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "cp1252"]
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                SENTIMENT_DATASET_FILE,
                header=0,
                names=["sentiment", "sentence"],
                encoding=encoding,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        "Unable to decode sentiment dataset with utf-8, latin1, or cp1252.",
    )


def add_sentiment_scores(
    filtered_rows: List[Dict[str, str]],
    model: Pipeline,
) -> int:
    sentiment_texts = [get_article_sentiment_text(row) for row in filtered_rows]
    usable_indexes = [
        index
        for index, text in enumerate(sentiment_texts)
        if text
    ]

    if not usable_indexes:
        for row in filtered_rows:
            row["sentiment_label"] = ""
            row["sentiment_score"] = ""
        return 0

    predictions = model.predict([sentiment_texts[index] for index in usable_indexes])

    for row in filtered_rows:
        row["sentiment_label"] = ""
        row["sentiment_score"] = ""

    for index, label in zip(usable_indexes, predictions):
        label_text = str(label).strip().lower()
        filtered_rows[index]["sentiment_label"] = label_text
        filtered_rows[index]["sentiment_score"] = str(SENTIMENT_SCORE_MAP[label_text])

    return len(usable_indexes)


def build_low_confidence_reasons(
    filtered_rows: List[Dict[str, str]],
    sentiment_article_count: int,
    coverage_confidence: str,
) -> List[str]:
    reasons: List[str] = []

    if coverage_confidence == "Low":
        reasons.append("filtered_article_count_below_5")
    if sentiment_article_count < 3:
        reasons.append("sentiment_article_count_below_3")

    source_names = {
        row.get("source_name", "").casefold()
        for row in filtered_rows
        if row.get("source_name")
    }
    if filtered_rows and len(source_names) <= 1:
        reasons.append("single_news_source")

    return reasons


def build_summary_row(
    ticker: str,
    company_name: str,
    query_text: str,
    lookback_days_used: int,
    data: Dict[str, Any],
    filtered_rows: List[Dict[str, str]],
    sentiment_article_count: int,
    coverage_confidence: str,
    weak_coverage_flag: bool,
    paths: Dict[str, Path],
) -> Dict[str, str]:
    from_date, to_date = get_news_date_range(lookback_days_used)
    aggregation_rows = filtered_rows[:MAX_SENTIMENT_ARTICLES]
    sentiment_labels = [
        row.get("sentiment_label", "")
        for row in aggregation_rows
        if row.get("sentiment_label")
    ]
    sentiment_scores = [
        int(row["sentiment_score"])
        for row in aggregation_rows
        if row.get("sentiment_score") not in {"", None}
    ]
    average_score = (
        sum(sentiment_scores) / len(sentiment_scores)
        if sentiment_scores
        else 0.0
    )
    low_confidence_reasons = build_low_confidence_reasons(
        filtered_rows,
        sentiment_article_count,
        coverage_confidence,
    )

    return {
        "query_ticker": ticker,
        "query_company": company_name,
        "query_text": query_text,
        "from_date": from_date,
        "to_date": to_date,
        "lookback_days_used": str(lookback_days_used),
        "newsapi_total_results": str(data.get("totalResults", 0)),
        "raw_article_count": str(len(data.get("articles", []))),
        "filtered_article_count": str(len(filtered_rows)),
        "sentiment_article_count": str(sentiment_article_count),
        "average_sentiment_score": f"{average_score:.6f}",
        "positive_count": str(sentiment_labels.count("positive")),
        "neutral_count": str(sentiment_labels.count("neutral")),
        "negative_count": str(sentiment_labels.count("negative")),
        "coverage_confidence": coverage_confidence,
        "weak_coverage_flag": str(weak_coverage_flag),
        "low_confidence_flag": str(bool(low_confidence_reasons)),
        "low_confidence_reasons": ";".join(low_confidence_reasons),
        "raw_json_path": str(paths["raw_json"]),
        "filtered_articles_path": str(paths["filtered_articles"]),
        "dashboard_top5_path": str(paths["dashboard_top5"]),
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
    }


# ============================================================
# File outputs
# ============================================================
def build_output_paths(ticker: str) -> Dict[str, Path]:
    ticker_slug = slugify(ticker)
    return {
        "raw_json": FINANCIAL_NEWS_JSON_DIR / f"newsapi-{ticker_slug}-raw.json",
        "filtered_articles": (
            FINANCIAL_NEWS_CSV_DIR / f"newsapi-{ticker_slug}-filtered-articles.csv"
        ),
        "sentiment_summary": (
            FINANCIAL_NEWS_CSV_DIR / f"newsapi-{ticker_slug}-sentiment-summary.csv"
        ),
        "dashboard_top5": (
            FINANCIAL_NEWS_CSV_DIR / f"newsapi-{ticker_slug}-dashboard-top5.csv"
        ),
    }


def save_raw_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_article_csv(
    rows: List[Dict[str, str]],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ARTICLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(summary_row: Dict[str, str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(summary_row)


def enrich_output_rows(
    ticker: str,
    company_name: str,
    query_text: str,
    lookback_days_used: int,
    rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    enriched_rows: List[Dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        enriched_rows.append(
            {
                "query_ticker": ticker,
                "query_company": company_name,
                "query_text": query_text,
                "lookback_days_used": str(lookback_days_used),
                "display_rank": str(index),
                **row,
            }
        )

    return enriched_rows


def print_terminal_summary(summary_row: Dict[str, str], dashboard_rows: List[Dict[str, str]]) -> None:
    print("\n[Financial news sentiment summary]")
    print(f"Ticker: {summary_row['query_ticker']}")
    print(f"Company: {summary_row['query_company']}")
    print(f"Date range: {summary_row['from_date']} to {summary_row['to_date']}")
    print(f"Lookback used: {summary_row['lookback_days_used']} days")
    print(f"Filtered articles: {summary_row['filtered_article_count']}")
    print(f"Average sentiment score: {summary_row['average_sentiment_score']}")
    print(
        "Sentiment distribution: "
        f"positive={summary_row['positive_count']}, "
        f"neutral={summary_row['neutral_count']}, "
        f"negative={summary_row['negative_count']}"
    )
    print(f"Coverage confidence: {summary_row['coverage_confidence']}")
    print(f"Weak coverage: {summary_row['weak_coverage_flag']}")
    print(f"Low confidence: {summary_row['low_confidence_flag']}")
    if summary_row["low_confidence_reasons"]:
        print(f"Low confidence reasons: {summary_row['low_confidence_reasons']}")

    print("\n[Top 5 dashboard articles]")
    if not dashboard_rows:
        print("No dashboard articles available.")
        return

    for row in dashboard_rows:
        print(
            f"{row['display_rank']}. "
            f"[{row.get('sentiment_label', '')}] "
            f"{row.get('title', '')} "
            f"({row.get('source_name', '')})"
        )


def retrieve_with_fallback(
    api_key: str,
    query_text: str,
    query_variants: List[str],
) -> Tuple[int, Dict[str, Any], List[Dict[str, str]], str, bool]:
    final_lookback_days = LOOKBACK_SEQUENCE[-1]
    selected_data: Dict[str, Any] = {}
    selected_rows: List[Dict[str, str]] = []
    selected_lookback_days = final_lookback_days

    for lookback_days in LOOKBACK_SEQUENCE:
        print(f"Requesting NewsAPI articles with {lookback_days}-day lookback...")
        data = fetch_newsapi_response(api_key, query_text, lookback_days)
        filtered_rows = filter_articles(data, query_variants)
        print(f"Relevant filtered articles found: {len(filtered_rows)}")

        selected_data = data
        selected_rows = filtered_rows
        selected_lookback_days = lookback_days

        if len(filtered_rows) >= MIN_FILTERED_ARTICLES:
            break

    if len(selected_rows) >= MIN_FILTERED_ARTICLES:
        coverage_confidence = (
            "Medium"
            if selected_lookback_days == final_lookback_days
            else "High"
        )
        weak_coverage_flag = False
    else:
        coverage_confidence = "Low"
        weak_coverage_flag = True

    return (
        selected_lookback_days,
        selected_data,
        selected_rows,
        coverage_confidence,
        weak_coverage_flag,
    )


# ============================================================
# Main execution
# ============================================================
def run_financial_news_download_pipeline() -> None:
    ensure_data_structure()

    ticker, company_name, query_variants, query_text = prompt_for_query_details()
    api_key = get_newsapi_key()

    print("\n[Starting financial news retrieval]")
    print(f"Ticker: {ticker}")
    print(f"Company: {company_name}")
    print(f"Query variants: {', '.join(query_variants)}")
    print(f"NewsAPI query: {query_text}")
    print(f"Initial lookback: {INITIAL_LOOKBACK_DAYS} days")
    print(f"Fallback lookbacks: {FALLBACK_LOOKBACK_DAYS}, {EXTENDED_LOOKBACK_DAYS} days")
    print(f"Page size: {PAGE_SIZE}\n")

    (
        lookback_days_used,
        data,
        filtered_rows,
        coverage_confidence,
        weak_coverage_flag,
    ) = retrieve_with_fallback(api_key, query_text, query_variants)

    model = load_or_train_sentiment_model()
    aggregation_rows = filtered_rows[:MAX_SENTIMENT_ARTICLES]
    sentiment_article_count = add_sentiment_scores(aggregation_rows, model)

    paths = build_output_paths(ticker)
    save_raw_json(data, paths["raw_json"])

    enriched_filtered_rows = enrich_output_rows(
        ticker,
        company_name,
        query_text,
        lookback_days_used,
        filtered_rows,
    )
    dashboard_rows = enriched_filtered_rows[:DASHBOARD_ARTICLES]

    summary_row = build_summary_row(
        ticker,
        company_name,
        query_text,
        lookback_days_used,
        data,
        filtered_rows,
        sentiment_article_count,
        coverage_confidence,
        weak_coverage_flag,
        paths,
    )

    save_article_csv(enriched_filtered_rows, paths["filtered_articles"])
    save_summary_csv(summary_row, paths["sentiment_summary"])
    save_article_csv(dashboard_rows, paths["dashboard_top5"])

    print_terminal_summary(summary_row, dashboard_rows)

    print("\n[Output files]")
    print(f"Raw JSON: {paths['raw_json']}")
    print(f"Filtered articles CSV: {paths['filtered_articles']}")
    print(f"Sentiment summary CSV: {paths['sentiment_summary']}")
    print(f"Dashboard top 5 CSV: {paths['dashboard_top5']}")
    print("\n[Done]")


def main() -> None:
    try:
        run_financial_news_download_pipeline()
    except KeyboardInterrupt:
        print("\nNews retrieval cancelled while waiting for terminal input.")


if __name__ == "__main__":
    main()
