from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pipeline.stock_features import (
    ALL_NUMERIC_FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    STATIONARY_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    TARGET_DATE_COLUMNS,
    TARGET_HORIZONS,
    add_stock_targets,
    build_stock_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed-stock-dataset"
MODEL_INPUT_DATA_DIR = DATA_DIR / "model-input-stock-dataset"
MARKET_CONTEXT_FILE = (
    DATA_DIR / "raw-market-context-dataset" / "ftse-bursa-malaysia-klci-raw.csv"
)
SUMMARY_DIR = PROJECT_ROOT / "summary"
DATA_PREPROCESSING_SUMMARY_DIR = SUMMARY_DIR / "data-preprocessing"
SUMMARY_FILE = DATA_DIR / "model-input-summary.csv"

FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)
MODEL_INPUT_COLUMNS = [
    "Date",
    "Ticker",
    *ALL_NUMERIC_FEATURE_COLUMNS,
    *TARGET_COLUMNS,
    *TARGET_DATE_COLUMNS,
]


def ensure_data_structure() -> None:
    MODEL_INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PREPROCESSING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def get_preprocessed_csv_files() -> List[Path]:
    return sorted(PREPROCESSED_DATA_DIR.glob("*.csv"))


def load_market_context() -> pd.DataFrame | None:
    if not MARKET_CONTEXT_FILE.exists():
        return None
    return pd.read_csv(MARKET_CONTEXT_FILE)


def build_output_filename(preprocessed_file: Path) -> str:
    if preprocessed_file.name.endswith("-preprocessed.csv"):
        return preprocessed_file.name.replace("-preprocessed.csv", "-model-input.csv")
    return f"{preprocessed_file.stem}-model-input.csv"


def add_derived_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper around the shared feature builder."""
    return build_stock_features(df, market_df=market_df)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper that also records each target's future date."""
    return add_stock_targets(df)


def finalize_model_input(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Keep every feature-complete row, including rows with unavailable future targets."""
    frame = df.copy().replace([float("inf"), float("-inf")], pd.NA)
    original_rows = len(frame)
    stock_feature_columns = list(
        dict.fromkeys((*BASE_FEATURE_COLUMNS, *STATIONARY_FEATURE_COLUMNS))
    )
    missing_columns = [column for column in MODEL_INPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing model-input columns: {missing_columns}")

    frame = frame.dropna(subset=stock_feature_columns).copy()
    rows_dropped = original_rows - len(frame)

    for target_column in TARGET_COLUMNS:
        frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce").astype("Int64")

    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    for target_date_column in TARGET_DATE_COLUMNS:
        frame[target_date_column] = pd.to_datetime(
            frame[target_date_column], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    frame = frame[MODEL_INPUT_COLUMNS].reset_index(drop=True)
    summary: Dict[str, int] = {
        "original_rows": original_rows,
        "rows_dropped_due_to_feature_nan_or_inf": rows_dropped,
        "final_rows": len(frame),
    }
    for horizon in TARGET_HORIZONS:
        target = pd.to_numeric(frame[f"Target_{horizon}D"], errors="coerce")
        summary[f"trainable_rows_{horizon}d"] = int(target.notna().sum())
        summary[f"positive_rows_{horizon}d"] = int((target == 1).sum())
        summary[f"negative_rows_{horizon}d"] = int((target == 0).sum())
    return frame, summary


def process_single_file(
    file_path: Path,
    market_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Dict[str, int | str]]:
    raw_df = pd.read_csv(file_path)
    features = add_derived_features(raw_df, market_df=market_df)
    targeted = add_targets(features)
    model_df, summary = finalize_model_input(targeted)
    return model_df, {"file_name": file_path.name, **summary}


def _atomic_csv_write(df: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        df.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
    except PermissionError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise PermissionError(
            f"Could not replace {output_path}. Close it in other applications and retry."
        ) from exc


def save_model_input_csv(df: pd.DataFrame, preprocessed_file: Path) -> Path:
    output_path = MODEL_INPUT_DATA_DIR / build_output_filename(preprocessed_file)
    _atomic_csv_write(df, output_path)
    return output_path


def save_summary(summary_rows: List[Dict[str, int | str]]) -> None:
    _atomic_csv_write(pd.DataFrame(summary_rows), SUMMARY_FILE)


def write_model_input_ready_summary(
    summary_rows: List[Dict[str, int | str]],
    market_context_available: bool,
) -> None:
    totals = {
        horizon: sum(int(row[f"trainable_rows_{horizon}d"]) for row in summary_rows)
        for horizon in TARGET_HORIZONS
    }
    positives = {
        horizon: sum(int(row[f"positive_rows_{horizon}d"]) for row in summary_rows)
        for horizon in TARGET_HORIZONS
    }
    total_feature_rows = sum(int(row["final_rows"]) for row in summary_rows)
    feature_rows = "\n".join(
        f"- Round 1 baseline: `{column}`" for column in BASE_FEATURE_COLUMNS
    )
    stationary_rows = "\n".join(
        f"- Round 2 stationary: `{column}`" for column in STATIONARY_FEATURE_COLUMNS
    )
    market_rows = "\n".join(
        f"- Round 3 market context: `{column}`" for column in MARKET_FEATURE_COLUMNS
    )
    horizon_rows = "\n".join(
        (
            f"- {horizon}-day: {totals[horizon]:,} labelled rows "
            f"({positives[horizon]:,} up, {totals[horizon] - positives[horizon]:,} down)"
        )
        for horizon in TARGET_HORIZONS
    )

    content = f"""# Model Input-Ready Dataset Summary

## Corrected Construction
- Feature-complete rows are retained even when a future target is unavailable.
- Each target includes its actual future trading date for leakage-free purging.
- Training drops missing labels only for the horizon currently being trained.
- Feature-complete rows across all stocks: {total_feature_rows:,}.
- Bursa KLCI market context available: {market_context_available}.

## Horizon-Specific Label Coverage
{horizon_rows}

## Feature Sets
{feature_rows}

{stationary_rows}

{market_rows}
"""
    output_path = DATA_PREPROCESSING_SUMMARY_DIR / "model-input-ready-dataset-summary.md"
    output_path.write_text(content, encoding="utf-8")


def run_model_input_pipeline() -> None:
    ensure_data_structure()
    preprocessed_files = get_preprocessed_csv_files()
    if not preprocessed_files:
        raise FileNotFoundError(f"No preprocessed CSV files found in: {PREPROCESSED_DATA_DIR}")

    market_df = load_market_context()
    summary_rows: List[Dict[str, int | str]] = []
    print("[Starting corrected model-input dataset pipeline]\n")
    if market_df is None:
        print(f"Warning: market context file not found: {MARKET_CONTEXT_FILE}")

    for preprocessed_file in preprocessed_files:
        model_df, summary = process_single_file(preprocessed_file, market_df=market_df)
        output_path = save_model_input_csv(model_df, preprocessed_file)
        summary_rows.append(summary)
        print(
            f"Saved {output_path.name}: feature rows={summary['final_rows']}, "
            + ", ".join(
                f"{horizon}D labels={summary[f'trainable_rows_{horizon}d']}"
                for horizon in TARGET_HORIZONS
            )
        )

    save_summary(summary_rows)
    write_model_input_ready_summary(summary_rows, market_context_available=market_df is not None)
    print(f"\nModel-input summary saved to: {SUMMARY_FILE}")


def main() -> None:
    run_model_input_pipeline()


if __name__ == "__main__":
    main()
