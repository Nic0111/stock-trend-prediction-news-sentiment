from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ============================================================
# Project configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_SENTIMENT_DATASET_FILE = (
    DATA_DIR
    / "preprocessed-financial-news-sentiment-dataset"
    / "financial-news-sentiment-preprocessed.csv"
)

SUMMARY_DIR = PROJECT_ROOT / "summary"
DATA_UNDERSTANDING_SUMMARY_DIR = SUMMARY_DIR / "data-understanding"

SENTIMENT_MEANINGS = {
    "Sentiment": "Target sentiment label assigned to the financial news sentence.",
    "Sentence": "Financial news sentence used as the input text for sentiment classification.",
    "Sentiment_Score": "Numeric sentiment score used for aggregation: positive=1, neutral=0, negative=-1.",
    "Character_Count": "Number of characters in the cleaned sentence text.",
    "Word_Count": "Number of words in the cleaned sentence text.",
}


# ============================================================
# Helpers
# ============================================================
def ensure_summary_structure() -> None:
    DATA_UNDERSTANDING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


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
    column_widths = [
        max(3, len(str(header)), *(len(row[index]) for row in string_rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values: List[str]) -> str:
        padded_values = [
            value.ljust(column_widths[index])
            for index, value in enumerate(values)
        ]
        return "| " + " | ".join(padded_values) + " |"

    lines = [
        format_row(headers),
        "| " + " | ".join("-" * width for width in column_widths) + " |",
    ]
    for row in string_rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_preprocessed_sentiment_dataset() -> pd.DataFrame:
    if not PREPROCESSED_SENTIMENT_DATASET_FILE.exists():
        raise FileNotFoundError(
            "Preprocessed sentiment dataset was not found. Run "
            "`python main.py preprocess_financial_news_sentiment` first."
        )

    df = pd.read_csv(PREPROCESSED_SENTIMENT_DATASET_FILE)
    for column in ["Sentiment", "Sentence"]:
        if column in df.columns:
            df[column] = df[column].astype("string")
    for column in ["Sentiment_Score", "Character_Count", "Word_Count"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def describe_numeric_series(series: pd.Series) -> List[List[Any]]:
    stats = series.describe()
    return [
        ["Count", stats.get("count", 0)],
        ["Mean", stats.get("mean", "N/A")],
        ["Standard deviation", stats.get("std", "N/A")],
        ["Minimum", stats.get("min", "N/A")],
        ["Q1", series.quantile(0.25)],
        ["Median", series.quantile(0.50)],
        ["Q3", series.quantile(0.75)],
        ["Maximum", stats.get("max", "N/A")],
    ]


# ============================================================
# 3.4.4.1 Variables
# ============================================================
def write_variables_summary(df: pd.DataFrame) -> None:
    rows: List[List[Any]] = []

    for column in df.columns:
        series = df[column]
        min_value: Any = "N/A"
        max_value: Any = "N/A"
        if pd.api.types.is_numeric_dtype(series):
            min_value = series.min()
            max_value = series.max()

        rows.append([
            column,
            str(series.dtype),
            SENTIMENT_MEANINGS.get(column, "Generated or supporting field."),
            int(series.isna().sum()),
            int(series.nunique(dropna=True)),
            min_value,
            max_value,
        ])

    content = f"""# 3.4.4.1 Financial News Sentiment Variables Summary

## Purpose
This section documents the variables available in the preprocessed financial news sentiment dataset.

## Variable Dictionary
{markdown_table(
        [
            "Variable",
            "Data Type",
            "Meaning",
            "Missing Count",
            "Unique Count",
            "Minimum",
            "Maximum",
        ],
        rows,
    )}
"""
    write_markdown(
        DATA_UNDERSTANDING_SUMMARY_DIR / "financial-news-sentiment-variables-summary.md",
        content,
    )


# ============================================================
# 3.4.4.2 Observations
# ============================================================
def write_observations_summary(df: pd.DataFrame) -> None:
    rows = [
        ["Total observations", len(df)],
        ["Unique sentences", int(df["Sentence"].nunique(dropna=True))],
        ["Duplicate full rows", int(df.duplicated().sum())],
        ["Duplicate sentences", int(df.duplicated(subset=["Sentence"]).sum())],
        ["Missing values", int(df.isna().sum().sum())],
    ]

    content = f"""# 3.4.4.2 Financial News Sentiment Observations Summary

## Purpose
This section documents the observation coverage and basic record quality of the preprocessed financial news sentiment dataset.

## Observation Summary
{markdown_table(["Metric", "Value"], rows)}
"""
    write_markdown(
        DATA_UNDERSTANDING_SUMMARY_DIR / "financial-news-sentiment-observations-summary.md",
        content,
    )


# ============================================================
# 3.4.4.3 Sentiment Class Distribution
# ============================================================
def write_class_distribution_summary(df: pd.DataFrame) -> None:
    distribution = df["Sentiment"].value_counts().reindex(["neutral", "positive", "negative"], fill_value=0)
    total = len(df)
    rows = [
        [label, int(count), (int(count) / total * 100) if total else 0.0]
        for label, count in distribution.items()
    ]

    content = f"""# 3.4.4.3 Financial News Sentiment Class Distribution Summary

## Purpose
This section describes the distribution of sentiment labels in the preprocessed financial news sentiment dataset.

## Sentiment Distribution
{markdown_table(["Sentiment", "Count", "Percentage"], rows)}

## Interpretation
The class distribution helps identify whether the dataset is balanced. A large neutral class can make the sentiment model more likely to predict neutral if class imbalance is not considered during training.
"""
    write_markdown(
        DATA_UNDERSTANDING_SUMMARY_DIR / "financial-news-sentiment-class-distribution-summary.md",
        content,
    )


# ============================================================
# 3.4.4.4 Sentence Length Distribution
# ============================================================
def write_sentence_length_summary(df: pd.DataFrame) -> None:
    character_rows = describe_numeric_series(df["Character_Count"])
    word_rows = describe_numeric_series(df["Word_Count"])

    content = f"""# 3.4.4.4 Financial News Sentiment Sentence Length Summary

## Purpose
This section describes the sentence length distribution of the financial news sentiment dataset.

## Character Count Statistics
{markdown_table(["Statistic", "Value"], character_rows)}

## Word Count Statistics
{markdown_table(["Statistic", "Value"], word_rows)}

## Interpretation
Sentence length helps determine whether the text records are suitable for common text classification methods. Short to medium-length sentences are generally suitable for TF-IDF vectorization and Logistic Regression.
"""
    write_markdown(
        DATA_UNDERSTANDING_SUMMARY_DIR / "financial-news-sentiment-sentence-length-summary.md",
        content,
    )


# ============================================================
# 3.4.4.5 Overall Sentiment Dataset Insights
# ============================================================
def write_dataset_insights_summary(df: pd.DataFrame) -> None:
    total_rows = len(df)
    missing_total = int(df.isna().sum().sum())
    class_distribution = df["Sentiment"].value_counts()
    majority_label = str(class_distribution.idxmax()) if not class_distribution.empty else "N/A"
    minority_label = str(class_distribution.idxmin()) if not class_distribution.empty else "N/A"
    median_words = float(df["Word_Count"].median()) if not df.empty else 0.0

    content = f"""# 3.4.4.5 Financial News Sentiment Dataset Insights Summary

## Purpose
This section summarizes the quality and modelling implications of the preprocessed financial news sentiment dataset.

## Dataset Quality
- The preprocessed sentiment dataset contains {total_rows:,} records.
- Total missing values remaining: {missing_total:,}.
- Duplicate full rows and duplicate sentences have been removed during preprocessing.

## Sentiment and Text Insights
- The majority sentiment class is `{majority_label}`.
- The smallest sentiment class is `{minority_label}`.
- The median sentence length is {median_words:.0f} words.

## Modelling Implications
- The sentence text is used as the input feature for sentiment classification.
- The sentiment label is used as the target class.
- The dataset can be used with TF-IDF vectorization because the text records are short to medium length.
- Class imbalance should be considered because the majority class may influence classifier behaviour.
"""
    write_markdown(
        DATA_UNDERSTANDING_SUMMARY_DIR / "financial-news-sentiment-dataset-insights-summary.md",
        content,
    )


def run_financial_news_sentiment_understanding_pipeline() -> None:
    ensure_summary_structure()
    df = load_preprocessed_sentiment_dataset()

    write_variables_summary(df)
    write_observations_summary(df)
    write_class_distribution_summary(df)
    write_sentence_length_summary(df)
    write_dataset_insights_summary(df)

    print("[Financial news sentiment understanding finished]")
    print(f"Rows analyzed: {len(df):,}")
    print(f"Summaries saved to: {DATA_UNDERSTANDING_SUMMARY_DIR}")


def main() -> None:
    run_financial_news_sentiment_understanding_pipeline()


if __name__ == "__main__":
    main()
