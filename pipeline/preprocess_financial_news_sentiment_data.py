from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# ============================================================
# Project configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_SENTIMENT_DATASET_FILE = (
    DATA_DIR
    / "financial-news-sentiment-dataset"
    / "financial-news-sentiment-dataset.csv"
)
PREPROCESSED_SENTIMENT_DATA_DIR = DATA_DIR / "preprocessed-financial-news-sentiment-dataset"
PREPROCESSED_SENTIMENT_DATASET_FILE = (
    PREPROCESSED_SENTIMENT_DATA_DIR
    / "financial-news-sentiment-preprocessed.csv"
)

SUMMARY_DIR = PROJECT_ROOT / "summary"
DATA_PREPROCESSING_SUMMARY_DIR = SUMMARY_DIR / "data-preprocessing"

VALID_SENTIMENT_LABELS = ["positive", "neutral", "negative"]
SENTIMENT_SCORE_MAP = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}


# ============================================================
# Helpers
# ============================================================
def ensure_data_structure() -> None:
    PREPROCESSED_SENTIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PREPROCESSING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


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


def read_raw_sentiment_dataset() -> Tuple[pd.DataFrame, str]:
    encodings = ["utf-8", "latin1", "cp1252"]
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                RAW_SENTIMENT_DATASET_FILE,
                header=0,
                names=["Sentiment", "Sentence"],
                encoding=encoding,
            )
            return df, encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        "Unable to decode sentiment dataset with utf-8, latin1, or cp1252.",
    )


def count_words(text: str) -> int:
    words = [word for word in text.split(" ") if word]
    return len(words)


# ============================================================
# Preprocessing
# ============================================================
def clean_sentiment_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    original_rows = len(raw_df)
    df = raw_df.copy()

    df["Sentiment"] = df["Sentiment"].astype("string")
    df["Sentence"] = df["Sentence"].astype("string")

    missing_sentiment_before = int(df["Sentiment"].isna().sum())
    missing_sentence_before = int(df["Sentence"].isna().sum())

    df["Sentiment"] = df["Sentiment"].fillna("").str.strip().str.lower()
    df["Sentence"] = df["Sentence"].fillna("").str.strip()
    df["Sentence"] = df["Sentence"].str.replace(r"\s+", " ", regex=True)

    empty_sentiment_after_strip = int(df["Sentiment"].eq("").sum())
    empty_sentence_after_strip = int(df["Sentence"].eq("").sum())
    invalid_label_rows = int((~df["Sentiment"].isin(VALID_SENTIMENT_LABELS)).sum())
    valid_label_mask = df["Sentiment"].isin(VALID_SENTIMENT_LABELS)
    non_empty_mask = df["Sentiment"].ne("") & df["Sentence"].ne("")
    rows_removed_missing_or_invalid = int((~(valid_label_mask & non_empty_mask)).sum())

    df = df[valid_label_mask & non_empty_mask].reset_index(drop=True)

    duplicate_full_rows = int(df.duplicated(subset=["Sentiment", "Sentence"]).sum())
    df = df.drop_duplicates(subset=["Sentiment", "Sentence"], keep="first").reset_index(drop=True)

    duplicate_sentence_rows = int(df.duplicated(subset=["Sentence"]).sum())
    df = df.drop_duplicates(subset=["Sentence"], keep="first").reset_index(drop=True)

    df["Sentiment_Score"] = df["Sentiment"].map(SENTIMENT_SCORE_MAP).astype(int)
    df["Character_Count"] = df["Sentence"].str.len().astype(int)
    df["Word_Count"] = df["Sentence"].apply(count_words).astype(int)

    summary = {
        "original_rows": original_rows,
        "missing_sentiment_before": missing_sentiment_before,
        "missing_sentence_before": missing_sentence_before,
        "empty_sentiment_after_strip": empty_sentiment_after_strip,
        "empty_sentence_after_strip": empty_sentence_after_strip,
        "invalid_label_rows": invalid_label_rows,
        "rows_removed_missing_or_invalid": rows_removed_missing_or_invalid,
        "duplicate_full_rows_removed": duplicate_full_rows,
        "duplicate_sentence_rows_removed": duplicate_sentence_rows,
        "final_rows": len(df),
        "positive_count": int((df["Sentiment"] == "positive").sum()),
        "neutral_count": int((df["Sentiment"] == "neutral").sum()),
        "negative_count": int((df["Sentiment"] == "negative").sum()),
    }
    return df, summary


def save_preprocessed_dataset(df: pd.DataFrame) -> None:
    df.to_csv(PREPROCESSED_SENTIMENT_DATASET_FILE, index=False, encoding="utf-8")


# ============================================================
# Summaries
# ============================================================
def write_missing_values_summary(summary: Dict[str, Any], encoding_used: str) -> None:
    content = f"""# 3.4.3.1 Financial News Sentiment Missing Values Summary

## Purpose
This step checks whether the sentiment label and sentence text fields are available before sentiment model training.

## Method Applied
- The Kaggle CSV is read with encoding fallback because the source file may not be encoded as UTF-8.
- The malformed source header is normalized into `Sentiment` and `Sentence`.
- Empty labels and empty sentences are removed because both fields are required for supervised sentiment classification.

## Overall Result
- Encoding used: {encoding_used}
- Original records: {summary["original_rows"]:,}
- Missing sentiment values before cleaning: {summary["missing_sentiment_before"]:,}
- Missing sentence values before cleaning: {summary["missing_sentence_before"]:,}
- Empty sentiment values after stripping: {summary["empty_sentiment_after_strip"]:,}
- Empty sentence values after stripping: {summary["empty_sentence_after_strip"]:,}
- Rows removed due to missing or invalid values: {summary["rows_removed_missing_or_invalid"]:,}
- Rows after missing-value handling and label validation: {summary["original_rows"] - summary["rows_removed_missing_or_invalid"]:,}
"""
    write_markdown(
        DATA_PREPROCESSING_SUMMARY_DIR / "financial-news-sentiment-missing-values-summary.md",
        content,
    )


def write_duplicate_records_summary(summary: Dict[str, Any]) -> None:
    content = f"""# 3.4.3.2 Financial News Sentiment Duplicate Records Summary

## Purpose
This step removes repeated records so that duplicated sentences do not overrepresent the same text during sentiment model training.

## Method Applied
- Exact duplicate full rows are removed using both `Sentiment` and `Sentence`.
- Duplicate sentence rows are then removed by keeping the first occurrence.

## Overall Result
- Exact duplicate full rows removed: {summary["duplicate_full_rows_removed"]:,}
- Duplicate sentence rows removed after full duplicate removal: {summary["duplicate_sentence_rows_removed"]:,}
- Final records after duplicate handling: {summary["final_rows"]:,}
"""
    write_markdown(
        DATA_PREPROCESSING_SUMMARY_DIR / "financial-news-sentiment-duplicate-records-summary.md",
        content,
    )


def write_text_label_cleaning_summary(summary: Dict[str, Any]) -> None:
    rows = [
        ["positive", summary["positive_count"], SENTIMENT_SCORE_MAP["positive"]],
        ["neutral", summary["neutral_count"], SENTIMENT_SCORE_MAP["neutral"]],
        ["negative", summary["negative_count"], SENTIMENT_SCORE_MAP["negative"]],
    ]
    content = f"""# 3.4.3.3 Financial News Sentiment Text and Label Cleaning Summary

## Purpose
This step standardizes the sentiment labels and sentence text before model-input preparation.

## Method Applied
- Sentiment labels are stripped, converted to lowercase, and restricted to `positive`, `neutral`, and `negative`.
- Sentence text is stripped and repeated whitespace is collapsed.
- A numeric sentiment score is added for later aggregation.

## Label Validation
- Invalid sentiment label rows removed: {summary["invalid_label_rows"]:,}

## Final Label Distribution
{markdown_table(["Sentiment", "Count", "Sentiment Score"], rows)}
"""
    write_markdown(
        DATA_PREPROCESSING_SUMMARY_DIR / "financial-news-sentiment-text-label-cleaning-summary.md",
        content,
    )


def write_preprocessed_dataset_summary(summary: Dict[str, Any]) -> None:
    content = f"""# 3.4.3.4 Final Preprocessed Financial News Sentiment Dataset Summary

## Purpose
This section documents the cleaned sentiment dataset used as the source for financial news sentiment model training.

## Dataset Location
`data/preprocessed-financial-news-sentiment-dataset/financial-news-sentiment-preprocessed.csv`

## Overall Result
- Original records: {summary["original_rows"]:,}
- Final preprocessed records: {summary["final_rows"]:,}
- Total records removed: {summary["original_rows"] - summary["final_rows"]:,}
- Final columns: `Sentiment`, `Sentence`, `Sentiment_Score`, `Character_Count`, `Word_Count`
"""
    write_markdown(
        DATA_PREPROCESSING_SUMMARY_DIR / "financial-news-sentiment-preprocessed-dataset-summary.md",
        content,
    )


def write_model_input_summary(summary: Dict[str, Any]) -> None:
    content = f"""# 3.4.3.5 Financial News Sentiment Model-Input Summary

## Purpose
This section explains how the cleaned sentiment dataset is prepared for sentiment classification.

## Model-Input Structure
- Input feature: `Sentence`
- Target label: `Sentiment`
- Numeric aggregation support: `Sentiment_Score`
- Text-length support fields: `Character_Count`, `Word_Count`

## Overall Result
- Model-input records: {summary["final_rows"]:,}
- Target classes: positive, neutral, negative
- Positive records: {summary["positive_count"]:,}
- Neutral records: {summary["neutral_count"]:,}
- Negative records: {summary["negative_count"]:,}

## Note
The sentence text is converted into numerical features during model training using TF-IDF vectorization. The cleaned CSV remains human-readable so that the text and labels can still be inspected easily.
"""
    write_markdown(
        DATA_PREPROCESSING_SUMMARY_DIR / "financial-news-sentiment-model-input-summary.md",
        content,
    )


def run_financial_news_sentiment_preprocess_pipeline() -> None:
    ensure_data_structure()

    if not RAW_SENTIMENT_DATASET_FILE.exists():
        raise FileNotFoundError(f"Raw sentiment dataset not found: {RAW_SENTIMENT_DATASET_FILE}")

    raw_df, encoding_used = read_raw_sentiment_dataset()
    preprocessed_df, summary = clean_sentiment_dataset(raw_df)
    save_preprocessed_dataset(preprocessed_df)

    write_missing_values_summary(summary, encoding_used)
    write_duplicate_records_summary(summary)
    write_text_label_cleaning_summary(summary)
    write_preprocessed_dataset_summary(summary)
    write_model_input_summary(summary)

    print("[Financial news sentiment preprocessing finished]")
    print(f"Raw records: {summary['original_rows']:,}")
    print(f"Preprocessed records: {summary['final_rows']:,}")
    print(f"Output saved to: {PREPROCESSED_SENTIMENT_DATASET_FILE}")


def main() -> None:
    run_financial_news_sentiment_preprocess_pipeline()


if __name__ == "__main__":
    main()
