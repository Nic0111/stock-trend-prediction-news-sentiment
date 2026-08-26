from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "news-evidence-benchmark"
    / "bursa-news-llm-assisted-label.jsonl"
)
SUMMARY_DIR = PROJECT_ROOT / "summary" / "data-understanding"
EDA_CHARTS_DIR = SUMMARY_DIR / "bursa-news-eda-charts"

LABEL_ORDER = ["negative", "neutral", "positive", "unlabeled"]
KEY_FIELDS = {
    "article_id": "Unique identifier for the historical news article.",
    "ticker": "Bursa Malaysia ticker associated with the article.",
    "published_at": "Publication timestamp of the article.",
    "source_domain": "Domain of the article publisher.",
    "title": "Article title used as part of the news text.",
    "entity_relevant": "Indicates whether the article is relevant to the associated company.",
    "sentiment_label": "Historical sentiment label or an empty value when no valid label is available.",
    "split": "Chronological dataset partition assigned using the publication date.",
    "event_category": "Category assigned to the financial event described by the article.",
    "confidence": "Confidence associated with the historical label.",
    "label_provenance": "Source and provenance category of the sentiment label.",
}


def ensure_summary_structure() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    EDA_CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def format_number(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not rows:
        return "_No rows to report._"

    string_rows = [[format_number(value) for value in row] for row in rows]
    widths = [
        max(3, len(str(header)), *(len(row[index]) for row in string_rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values: List[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    lines = [
        format_row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(format_row(row) for row in string_rows)
    return "\n".join(lines)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def get_matplotlib_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate Bursa Malaysia news EDA charts."
        ) from exc
    return plt


def load_bursa_news_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Bursa Malaysia news dataset was not found: {DATASET_FILE}")

    records = []
    with DATASET_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["label_status"] = df["sentiment_label"].fillna("").astype(str).str.strip().str.lower()
    df["label_status"] = df["label_status"].where(
        df["label_status"].isin(["negative", "neutral", "positive"]),
        "unlabeled",
    )
    df["publication_year"] = df["published_at"].dt.year
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce")
    return df


def write_variables_summary(df: pd.DataFrame) -> None:
    rows = []
    for field, meaning in KEY_FIELDS.items():
        if field not in df.columns:
            continue
        rows.append([
            field,
            str(df[field].dtype),
            meaning,
            int(df[field].isna().sum()),
            int(df[field].nunique(dropna=True)),
        ])

    content = f"""# 4.4.5.1 Bursa Malaysia Historical-News Variables Summary

## Purpose
This section documents the principal variables examined in the Bursa Malaysia-specific historical news corpus.

## Variable Dictionary
{markdown_table(["Variable", "Data Type", "Meaning", "Missing Count", "Unique Count"], rows)}
"""
    write_markdown(SUMMARY_DIR / "bursa-news-variables-summary.md", content)


def write_observations_summary(df: pd.DataFrame) -> None:
    labelled = df["label_status"].isin(["negative", "neutral", "positive"])
    dates = df["published_at"].dropna()
    rows = [
        ["Total records", len(df)],
        ["Unique article identifiers", int(df["article_id"].nunique(dropna=True))],
        ["Duplicate article identifiers", int(df["article_id"].duplicated().sum())],
        ["Unique tickers", int(df["ticker"].nunique(dropna=True))],
        ["Unique source domains", int(df["source_domain"].nunique(dropna=True))],
        ["Entity-relevant records", int(df["entity_relevant"].fillna(False).astype(bool).sum())],
        ["Records with valid sentiment labels", int(labelled.sum())],
        ["Records without valid sentiment labels", int((~labelled).sum())],
        ["First publication date", dates.min() if not dates.empty else "N/A"],
        ["Last publication date", dates.max() if not dates.empty else "N/A"],
    ]

    content = f"""# 4.4.5.2 Bursa Malaysia Historical-News Observations Summary

## Purpose
This section documents the observation coverage, article identity, publisher coverage, entity relevance, and sentiment-label availability of the historical news corpus.

## Observation Summary
{markdown_table(["Metric", "Value"], rows)}
"""
    write_markdown(SUMMARY_DIR / "bursa-news-observations-summary.md", content)


def save_label_distribution_chart(df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "bursa-news-label-distribution.png"
    distribution = df["label_status"].value_counts().reindex(LABEL_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        distribution.index,
        distribution.values,
        color=["#b45f5f", "#7f8c8d", "#376c5a", "#d9a441"],
    )
    ax.set_title("Bursa Malaysia Historical-News Label Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number Of Records")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, distribution.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_publication_year_chart(df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "bursa-news-publication-year-distribution.png"
    yearly = df["publication_year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(yearly.index.astype(str), yearly.values, color="#2f6f9f")
    ax.set_title("Bursa Malaysia Historical-News Records By Publication Year")
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number Of Records")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_ticker_distribution_chart(df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "bursa-news-ticker-distribution.png"
    ticker_counts = df["ticker"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ticker_counts.index, ticker_counts.values, color="#805ad5")
    ax.set_title("Bursa Malaysia Historical-News Records By Ticker")
    ax.set_xlabel("Number Of Records")
    ax.set_ylabel("Ticker")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_confidence_distribution_chart(df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "bursa-news-label-confidence-distribution.png"
    confidence = df.loc[
        df["label_status"].isin(["negative", "neutral", "positive"]),
        "confidence",
    ].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidence, bins=20, range=(0, 1), color="#376c5a", edgecolor="white")
    ax.set_title("Bursa Malaysia Historical-News Label-Confidence Distribution")
    ax.set_xlabel("Label Confidence")
    ax.set_ylabel("Number Of Records")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_eda_summary(df: pd.DataFrame) -> None:
    charts = [
        (
            "Bursa Malaysia Historical-News Label Distribution",
            save_label_distribution_chart(df),
            "Shows the distribution of positive, neutral, negative, and unlabeled records.",
        ),
        (
            "Bursa Malaysia Historical-News Records By Publication Year",
            save_publication_year_chart(df),
            "Shows the chronological coverage of the historical corpus.",
        ),
        (
            "Bursa Malaysia Historical-News Records By Ticker",
            save_ticker_distribution_chart(df),
            "Shows the number of historical-news records associated with each ticker.",
        ),
        (
            "Bursa Malaysia Historical-News Label-Confidence Distribution",
            save_confidence_distribution_chart(df),
            "Shows the confidence distribution among records with valid sentiment labels.",
        ),
    ]

    chart_sections = []
    for title, chart_path, description in charts:
        relative_path = chart_path.relative_to(SUMMARY_DIR).as_posix()
        chart_sections.append(
            f"### {title}\n![{title}]({relative_path})\n\n{description}"
        )

    split_rows = [
        [split, int(count)]
        for split, count in df["split"].fillna("unassigned").value_counts().items()
    ]
    label_rows = [
        [label, int(count), (int(count) / len(df) * 100) if len(df) else 0.0]
        for label, count in df["label_status"].value_counts().reindex(LABEL_ORDER, fill_value=0).items()
    ]
    ticker_rows = [
        [ticker, int(count)]
        for ticker, count in df["ticker"].value_counts().items()
    ]
    source_rows = [
        [source, int(count)]
        for source, count in df["source_domain"].fillna("unknown").value_counts().head(10).items()
    ]

    content = f"""# 4.4.5.3 Bursa Malaysia Historical-News Exploratory Data Analysis Summary

## Purpose
This section examines the chronological coverage, ticker representation, publisher distribution, sentiment-label status, and label-confidence characteristics of the Bursa Malaysia-specific historical news corpus.

## Chronological Split Distribution
{markdown_table(["Dataset Split", "Records"], split_rows)}

## Sentiment-Label Distribution
{markdown_table(["Label Status", "Records", "Percentage"], label_rows)}

## Ticker Distribution
{markdown_table(["Ticker", "Records"], ticker_rows)}

## Main Publisher Domains
{markdown_table(["Source Domain", "Records"], source_rows)}

## Exploratory Data Analysis Charts
{chr(10).join(chart_sections)}

## Interpretation
The historical-news analysis describes how records are distributed across time, companies, publishers, and sentiment-label statuses. The presence of unlabeled records is reported separately from the three valid sentiment classes so that unavailable labels are not interpreted as neutral sentiment. The chronological distribution also supports the separation of historical records into training, validation, and test periods.
"""
    write_markdown(SUMMARY_DIR / "bursa-news-eda-summary.md", content)


def main() -> None:
    ensure_summary_structure()
    df = load_bursa_news_dataset()
    write_variables_summary(df)
    write_observations_summary(df)
    write_eda_summary(df)
    print(f"Bursa Malaysia news records analyzed: {len(df):,}")
    print(f"Summaries saved to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
