from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json
import warnings

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# XGBoost is part of the locked strategy.
from xgboost import XGBClassifier


# ============================================================
# Project configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_INPUT_DATA_DIR = DATA_DIR / "model-input-stock-dataset"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification-reports"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion-matrices"

SUMMARY_FILE = RESULTS_DIR / "model-comparison-summary.csv"

# Locked horizon choices
TARGET_HORIZONS = [1, 5, 10]

# Locked report focus
PRIMARY_REPORT_HORIZON = 5

# Locked date split
DEVELOPMENT_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"

# Locked baseline features
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Return_1D",
    "Return_5D",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_5",
    "MA_5",
    "MA_20",
    "MA_60",
    "Volatility_5D",
    "Volatility_20D",
    "High_Low_Spread",
    "Open_Close_Change",
    "Volume_Change_1D",
    "Volume_MA_5",
]

IDENTITY_COLUMNS = ["Date", "Ticker"]


# ============================================================
# Helper functions
# ============================================================
def ensure_output_structure() -> None:
    """
    Ensure model and results folders exist.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)


def get_model_input_csv_files() -> List[Path]:
    return sorted(MODEL_INPUT_DATA_DIR.glob("*.csv"))


def load_and_combine_model_input_data() -> pd.DataFrame:
    """
    Read all model-input CSV files and combine them into one DataFrame.
    """
    files = get_model_input_csv_files()
    if not files:
        raise FileNotFoundError(
            f"No model-input CSV files found in: {MODEL_INPUT_DATA_DIR}"
        )

    dataframes: List[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
    combined_df = combined_df.dropna(subset=["Date"]).sort_values(
        by=["Date", "Ticker"], ascending=[True, True]
    ).reset_index(drop=True)

    return combined_df


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = IDENTITY_COLUMNS + FEATURE_COLUMNS + [
        "Target_1D",
        "Target_5D",
        "Target_10D",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in combined dataset: {missing}")


def split_development_and_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by locked dates:
    - development: Date <= 2024-12-31
    - test: Date >= 2025-01-01
    """
    dev_df = df[df["Date"] <= pd.Timestamp(DEVELOPMENT_END_DATE)].copy()
    test_df = df[df["Date"] >= pd.Timestamp(TEST_START_DATE)].copy()

    if dev_df.empty:
        raise ValueError("Development dataset is empty after date split.")
    if test_df.empty:
        raise ValueError("Test dataset is empty after date split.")

    return dev_df, test_df


def get_models() -> Dict[str, object]:
    """
    Build the locked model set:
    - Logistic Regression (with StandardScaler in a Pipeline)
    - Random Forest
    - XGBoost
    """
    models: Dict[str, object] = {
        "logistic-regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "random-forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models


def time_series_cv_f1(
    model,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Evaluate mean/std F1 on the development set using TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1_scores: List[float] = []

    for train_idx, val_idx in tscv.split(X_dev):
        X_train_fold = X_dev.iloc[train_idx]
        y_train_fold = y_dev.iloc[train_idx]
        X_val_fold = X_dev.iloc[val_idx]
        y_val_fold = y_dev.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)
        y_val_pred = fold_model.predict(X_val_fold)

        fold_f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
        f1_scores.append(fold_f1)

    return {
        "cv_f1_mean": float(pd.Series(f1_scores).mean()),
        "cv_f1_std": float(pd.Series(f1_scores).std(ddof=0)),
    }


def evaluate_on_test_set(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    """
    Produce test metrics, classification report, and confusion matrix.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report_dict": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
        "classification_report_text": classification_report(
            y_test, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def save_classification_report(
    model_name: str,
    horizon: int,
    report_text: str,
) -> Path:
    file_path = (
        CLASSIFICATION_REPORTS_DIR
        / f"{model_name}-{horizon}d-classification-report.txt"
    )
    file_path.write_text(report_text, encoding="utf-8")
    return file_path


def save_confusion_matrix(
    model_name: str,
    horizon: int,
    matrix: List[List[int]],
) -> Path:
    file_path = (
        CONFUSION_MATRICES_DIR
        / f"{model_name}-{horizon}d-confusion-matrix.csv"
    )
    cm_df = pd.DataFrame(matrix, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
    cm_df.to_csv(file_path, index=True)
    return file_path


def save_model(
    model,
    model_name: str,
    horizon: int,
) -> Path:
    file_path = MODELS_DIR / f"{model_name}-{horizon}d.joblib"
    joblib.dump(model, file_path)
    return file_path


def train_for_single_horizon(
    df: pd.DataFrame,
    horizon: int,
) -> List[Dict[str, object]]:
    """
    Train all locked models for one target horizon.
    """
    target_column = f"Target_{horizon}D"

    feature_df = df[IDENTITY_COLUMNS + FEATURE_COLUMNS + [target_column]].copy()
    feature_df = feature_df.dropna().sort_values(by=["Date", "Ticker"]).reset_index(drop=True)

    dev_df, test_df = split_development_and_test(feature_df)

    X_dev = dev_df[FEATURE_COLUMNS]
    y_dev = dev_df[target_column].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_column].astype(int)

    if len(dev_df) < 10:
        raise ValueError(f"Development data for {horizon}D is too small.")
    if len(test_df) < 10:
        raise ValueError(f"Test data for {horizon}D is too small.")

    models = get_models()
    results_rows: List[Dict[str, object]] = []

    for model_name, model in models.items():
        print(f"Training {model_name} for {horizon}D horizon...")

        # Cross-validated F1 on development period
        cv_metrics = time_series_cv_f1(model, X_dev, y_dev, n_splits=5)

        # Train final model on full development period
        final_model = clone(model)
        final_model.fit(X_dev, y_dev)

        # Evaluate on final 2025 holdout
        test_metrics = evaluate_on_test_set(final_model, X_test, y_test)

        # Save outputs
        model_path = save_model(final_model, model_name, horizon)
        report_path = save_classification_report(
            model_name,
            horizon,
            test_metrics["classification_report_text"],
        )
        cm_path = save_confusion_matrix(
            model_name,
            horizon,
            test_metrics["confusion_matrix"],
        )

        result_row = {
            "model_name": model_name,
            "horizon_days": horizon,
            "development_rows": len(dev_df),
            "test_rows": len(test_df),
            "cv_f1_mean": cv_metrics["cv_f1_mean"],
            "cv_f1_std": cv_metrics["cv_f1_std"],
            "test_accuracy": test_metrics["test_accuracy"],
            "test_precision": test_metrics["test_precision"],
            "test_recall": test_metrics["test_recall"],
            "test_f1": test_metrics["test_f1"],
            "saved_model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "classification_report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_path": str(cm_path.relative_to(PROJECT_ROOT)),
            "is_primary_report_horizon": horizon == PRIMARY_REPORT_HORIZON,
        }
        results_rows.append(result_row)

    return results_rows


def save_summary(summary_rows: List[Dict[str, object]]) -> None:
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["horizon_days", "model_name"]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_FILE, index=False)


# ============================================================
# Main pipeline
# ============================================================
def run_training_pipeline() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    ensure_output_structure()

    print("[Loading combined model-input dataset]\n")
    combined_df = load_and_combine_model_input_data()
    validate_required_columns(combined_df)

    print(f"Combined rows: {len(combined_df)}")
    print(f"Date range: {combined_df['Date'].min().date()} to {combined_df['Date'].max().date()}\n")

    all_summary_rows: List[Dict[str, object]] = []

    for horizon in TARGET_HORIZONS:
        print(f"==============================")
        print(f"Processing horizon: {horizon}D")
        print(f"==============================")
        horizon_rows = train_for_single_horizon(combined_df, horizon)
        all_summary_rows.extend(horizon_rows)
        print()

    save_summary(all_summary_rows)
    print(f"Model comparison summary saved to: {SUMMARY_FILE}")
    print("\n[Training pipeline finished]")


def main() -> None:
    run_training_pipeline()


if __name__ == "__main__":
    main()