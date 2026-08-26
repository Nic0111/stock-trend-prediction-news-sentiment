from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEWS_CSV_DIR = PROJECT_ROOT / "data" / "financial-news-dataset" / "csv"
SUMMARY_DIR = PROJECT_ROOT / "summary" / "data-understanding"
EDA_CHARTS_DIR = SUMMARY_DIR / "recent-financial-news-eda-charts"

SENTIMENT_ORDER = ["negative", "neutral", "positive", "unlabeled"]


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
            "matplotlib is required to generate recent financial-news EDA charts."
        ) from exc
    return plt


def load_filtered_articles() -> pd.DataFrame:
    files = sorted(NEWS_CSV_DIR.glob("*-filtered-articles.csv"))
    frames = []
    for file_path in files:
        frame = pd.read_csv(file_path, low_memory=False)
        frame["eda_file"] = file_path.name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No filtered recent-news CSV files were found in: {NEWS_CSV_DIR}"
        )

    df = pd.concat(frames, ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["sentiment_probability"] = pd.to_numeric(
        df.get("sentiment_probability"), errors="coerce"
    )
    df["relevance_score"] = pd.to_numeric(df.get("relevance_score"), errors="coerce")
    df["sentiment_label"] = (
        df["sentiment_label"].fillna("").astype(str).str.strip().str.lower()
    )
    df["sentiment_label"] = df["sentiment_label"].where(
        df["sentiment_label"].isin(["negative", "neutral", "positive"]),
        "unlabeled",
    )
    df["sentiment_uncertain"] = (
        df["sentiment_uncertain"].fillna(False).astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    )
    df["article_key"] = (
        df["query_ticker"].fillna("").astype(str)
        + "|"
        + df["article_id"].fillna(df["url"]).fillna("").astype(str)
    )
    return df


def load_context_summaries() -> pd.DataFrame:
    files = sorted(NEWS_CSV_DIR.glob("*-context-summary.csv"))
    frames = []
    for file_path in files:
        frame = pd.read_csv(file_path, low_memory=False)
        frame["eda_file"] = file_path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for column in [
        "horizon_days",
        "window_days",
        "candidate_article_count",
        "usable_article_count",
        "uncertain_article_count",
        "unique_source_count",
        "independent_event_count",
        "weighted_sentiment_score",
        "mean_sentiment_confidence",
        "max_sentiment_confidence",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def save_article_count_by_ticker_chart(df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "recent-news-article-count-by-ticker.png"
    counts = df["query_ticker"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts.index, counts.values, color="#2f6f9f")
    ax.set_title("Recent Financial-News Articles By Ticker")
    ax.set_xlabel("Article Records")
    ax.set_ylabel("Ticker")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_article_sentiment_distribution_chart(df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "recent-news-sentiment-distribution.png"
    counts = df["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=["#b45f5f", "#7f8c8d", "#376c5a", "#d9a441"],
    )
    ax.set_title("Recent Financial-News Sentiment Distribution")
    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Article Records")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, counts.values):
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


def save_article_publication_trend_chart(df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "recent-news-publication-trend.png"
    monthly = (
        df.dropna(subset=["published_at"])
        .set_index("published_at")
        .resample("ME")
        .size()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly.index, monthly.values, marker="o", color="#805ad5", linewidth=1.4)
    ax.set_title("Recent Financial-News Publication Trend")
    ax.set_xlabel("Publication Month")
    ax.set_ylabel("Article Records")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_context_coverage_chart(context_df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "recent-news-context-coverage.png"
    grouped = (
        context_df.groupby("horizon_days")[["candidate_article_count", "usable_article_count"]]
        .mean()
        .reindex([1, 5, 10])
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = list(range(len(grouped.index)))
    width = 0.35
    ax.bar(
        [position - width / 2 for position in x_positions],
        grouped["candidate_article_count"].fillna(0),
        width=width,
        label="Candidate Articles",
        color="#2f6f9f",
    )
    ax.bar(
        [position + width / 2 for position in x_positions],
        grouped["usable_article_count"].fillna(0),
        width=width,
        label="Usable Articles",
        color="#376c5a",
    )
    ax.set_title("Recent Financial-News Context Coverage By Horizon")
    ax.set_xlabel("Forecasting Horizon")
    ax.set_ylabel("Mean Article Count")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(value)}-Day" for value in grouped.index])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_sentiment_confidence_chart(df: pd.DataFrame) -> Path:
    plt = get_matplotlib_pyplot()
    path = EDA_CHARTS_DIR / "recent-news-sentiment-confidence.png"
    confidence = df["sentiment_probability"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidence, bins=20, range=(0, 1), color="#376c5a", edgecolor="white")
    ax.set_title("Recent Financial-News Sentiment-Confidence Distribution")
    ax.set_xlabel("Sentiment Probability")
    ax.set_ylabel("Article Records")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_eda_summary(df: pd.DataFrame, context_df: pd.DataFrame) -> None:
    charts = [
        (
            "Recent Financial-News Articles By Ticker",
            save_article_count_by_ticker_chart(df),
            "Shows the number of filtered recent-news records available for each ticker.",
        ),
        (
            "Recent Financial-News Sentiment Distribution",
            save_article_sentiment_distribution_chart(df),
            "Shows the distribution of sentiment labels assigned to filtered articles.",
        ),
        (
            "Recent Financial-News Publication Trend",
            save_article_publication_trend_chart(df),
            "Shows the publication-month distribution of the available recent-news outputs.",
        ),
        (
            "Recent Financial-News Sentiment-Confidence Distribution",
            save_sentiment_confidence_chart(df),
            "Shows the distribution of sentiment probabilities recorded for filtered articles.",
        ),
    ]
    if not context_df.empty:
        charts.append(
            (
                "Recent Financial-News Context Coverage By Horizon",
                save_context_coverage_chart(context_df),
                "Compares candidate and usable article counts across the one-day, five-day, and ten-day contexts.",
            )
        )

    chart_sections = []
    for title, chart_path, description in charts:
        relative_path = chart_path.relative_to(SUMMARY_DIR).as_posix()
        chart_sections.append(
            f"### {title}\n![{title}]({relative_path})\n\n{description}"
        )

    dates = df["published_at"].dropna()
    sentiment_rows = [
        [label, int(count), (int(count) / len(df) * 100) if len(df) else 0.0]
        for label, count in df["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0).items()
    ]
    provider_rows = [
        [provider, int(count)]
        for provider, count in df["provider"].fillna("unknown").value_counts().items()
    ]
    ticker_rows = [
        [ticker, int(count)]
        for ticker, count in df["query_ticker"].value_counts().items()
    ]
    context_rows = []
    if not context_df.empty:
        grouped = context_df.groupby("horizon_days").agg(
            Context_Files=("eda_file", "nunique"),
            Mean_Candidate_Articles=("candidate_article_count", "mean"),
            Mean_Usable_Articles=("usable_article_count", "mean"),
            Mean_Uncertain_Articles=("uncertain_article_count", "mean"),
        )
        for horizon, row in grouped.sort_index().iterrows():
            context_rows.append([
                f"{int(horizon)}-day",
                int(row["Context_Files"]),
                row["Mean_Candidate_Articles"],
                row["Mean_Usable_Articles"],
                row["Mean_Uncertain_Articles"],
            ])

    content = f"""# 4.4.6 Recent Financial-News Data Understanding

## Purpose
This section examines the available recent financial-news outputs according to ticker coverage, provider, publication timing, sentiment labels, sentiment confidence, and forecasting-horizon context.

## Observation Summary
{markdown_table([
        "Metric",
        "Value",
    ], [
        ["Filtered article rows before article-level deduplication", len(df)],
        ["Unique ticker-article records", int(df["article_key"].nunique())],
        ["Filtered-news output files", int(df["eda_file"].nunique())],
        ["Tickers represented", int(df["query_ticker"].nunique())],
        ["First publication date", dates.min() if not dates.empty else "N/A"],
        ["Last publication date", dates.max() if not dates.empty else "N/A"],
        ["Uncertain sentiment predictions", int(df["sentiment_uncertain"].sum())],
    ])}

## Ticker Coverage
{markdown_table(["Ticker", "Article Records"], ticker_rows)}

## Provider Coverage
{markdown_table(["Provider", "Article Records"], provider_rows)}

## Sentiment Distribution
{markdown_table(["Sentiment Label", "Records", "Percentage"], sentiment_rows)}

## Context Coverage By Forecasting Horizon
{markdown_table([
        "Forecasting Horizon",
        "Context Files",
        "Mean Candidate Articles",
        "Mean Usable Articles",
        "Mean Uncertain Articles",
    ], context_rows)}

## Exploratory Data Analysis Charts
{chr(10).join(chart_sections)}

## Interpretation
The recent-news analysis describes the availability and characteristics of the retrieved financial-news outputs rather than representing a fixed historical population. The results depend on the retrieval date, provider response, company coverage, article filters, sentiment confidence, and configured forecasting windows. Candidate, usable, and uncertain article counts are therefore interpreted as retrieval-context measures for the available output files.
"""
    write_markdown(SUMMARY_DIR / "recent-financial-news-eda-summary.md", content)


def main() -> None:
    ensure_summary_structure()
    articles = load_filtered_articles()
    context = load_context_summaries()
    write_eda_summary(articles, context)
    print(f"Recent financial-news rows analyzed: {len(articles):,}")
    print(f"Recent context rows analyzed: {len(context):,}")
    print(f"Summaries saved to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
