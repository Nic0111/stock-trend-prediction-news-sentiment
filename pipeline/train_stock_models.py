from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import hashlib
import importlib.util
import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from pipeline.model_artifacts import CalibratedStockModel, probability_to_logit
from pipeline.stock_features import (
    ALL_NUMERIC_FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    MARKET_FEATURE_COLUMNS,
    STATIONARY_FEATURE_COLUMNS,
    TARGET_HORIZONS,
    feature_columns_for_round,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_INPUT_DATA_DIR = DATA_DIR / "model-input-stock-dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification-reports"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion-matrices"

SUMMARY_FILE = RESULTS_DIR / "model-comparison-summary.csv"
CANDIDATE_FILE = RESULTS_DIR / "stock-model-candidates.csv"
BASELINE_FILE = RESULTS_DIR / "stock-model-baselines.csv"
PER_TICKER_FILE = RESULTS_DIR / "stock-model-per-ticker-metrics.csv"
REGISTRY_FILE = MODELS_DIR / "model-registry.json"

# The downloaded history starts in 2015. Keep this boundary explicit so a
# stray older row cannot silently enter development, while 2025 remains the
# fixed observed benchmark holdout.
DEVELOPMENT_START_DATE = pd.Timestamp("2015-01-01")
DEVELOPMENT_END_DATE = pd.Timestamp("2024-12-31")
TEST_START_DATE = pd.Timestamp("2025-01-01")
TEST_END_DATE = pd.Timestamp("2025-12-31")
N_SPLITS = 5
RANDOM_STATE = 42
MIN_BALANCED_ACCURACY_GAIN = 0.02
MIN_DIRECTION_RECALL = 0.35
MIN_PREDICTED_CLASS_SHARE = 0.20
MAX_PREDICTED_CLASS_SHARE = 0.80
MODEL_NAMES = (
    "logistic-regression",
    "elastic-net-logistic",
    "random-forest",
    "extra-trees",
    "hist-gradient-boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "soft-voting-ensemble",
)

FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)
IDENTITY_COLUMNS = ["Date", "Ticker"]


def available_model_names() -> tuple[str, ...]:
    """Return the expanded benchmark while keeping optional learners optional."""
    names = [
        "logistic-regression",
        "elastic-net-logistic",
        "random-forest",
        "extra-trees",
        "hist-gradient-boosting",
        "xgboost",
        "soft-voting-ensemble",
    ]
    if importlib.util.find_spec("lightgbm") is not None:
        names.append("lightgbm")
    if importlib.util.find_spec("catboost") is not None:
        try:
            from catboost import CatBoostClassifier
            from sklearn.utils._tags import get_tags

            get_tags(CatBoostClassifier(verbose=False, allow_writing_files=False))
        except Exception:
            # CatBoost 1.2.8 is not compatible with scikit-learn 1.8's tag
            # protocol. Keep it optional rather than allowing one candidate to
            # abort the complete benchmark.
            pass
        else:
            names.append("catboost")
    return tuple(names)


def search_plan_for_profile(
    profile: str,
    available_models: Sequence[str],
) -> Dict[int, Dict[int, tuple[str, ...]]]:
    """Choose a bounded evidence-led search without changing validation rules."""
    available = set(available_models)
    if profile != "focused":
        models = tuple(model for model in MODEL_NAMES if model in available)
        return {
            horizon: {round_number: models for round_number in (1, 2, 3, 4)}
            for horizon in TARGET_HORIZONS
        }

    requested = {
        1: {
            3: ("logistic-regression", "elastic-net-logistic", "hist-gradient-boosting", "xgboost", "lightgbm"),
            4: ("logistic-regression", "hist-gradient-boosting", "xgboost", "lightgbm", "soft-voting-ensemble"),
        },
        5: {
            2: ("logistic-regression", "elastic-net-logistic", "hist-gradient-boosting", "lightgbm"),
            4: ("logistic-regression", "elastic-net-logistic", "random-forest", "hist-gradient-boosting", "lightgbm", "soft-voting-ensemble"),
        },
        10: {
            2: ("logistic-regression", "elastic-net-logistic", "extra-trees"),
            4: ("logistic-regression", "elastic-net-logistic", "extra-trees", "soft-voting-ensemble"),
        },
    }
    return {
        horizon: {
            round_number: tuple(model for model in models if model in available)
            for round_number, models in rounds.items()
        }
        for horizon, rounds in requested.items()
    }


def ensure_output_structure() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)


def get_model_input_csv_files() -> List[Path]:
    return sorted(MODEL_INPUT_DATA_DIR.glob("*.csv"))


def hash_file_set(files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(files):
        digest.update(file_path.name.encode("utf-8"))
        with file_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_and_combine_model_input_data() -> pd.DataFrame:
    files = get_model_input_csv_files()
    if not files:
        raise FileNotFoundError(f"No model-input CSV files found in: {MODEL_INPUT_DATA_DIR}")

    combined = pd.concat([pd.read_csv(file_path) for file_path in files], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined["Ticker"] = combined["Ticker"].astype("string").str.strip().str.upper()
    for horizon in TARGET_HORIZONS:
        combined[f"Target_{horizon}D"] = pd.to_numeric(
            combined[f"Target_{horizon}D"], errors="coerce"
        )
        combined[f"Target_Date_{horizon}D"] = pd.to_datetime(
            combined[f"Target_Date_{horizon}D"], errors="coerce"
        )
    for column in ALL_NUMERIC_FEATURE_COLUMNS:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    return combined.dropna(subset=["Date", "Ticker"]).sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)


def validate_required_columns(df: pd.DataFrame) -> None:
    required = ["Date", "Ticker", *ALL_NUMERIC_FEATURE_COLUMNS]
    for horizon in TARGET_HORIZONS:
        required.extend([f"Target_{horizon}D", f"Target_Date_{horizon}D"])
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in model input: {missing}")


def prepare_horizon_frame(
    df: pd.DataFrame,
    horizon: int,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    target_column = f"Target_{horizon}D"
    target_date_column = f"Target_Date_{horizon}D"
    supporting_columns = ["Return_1D", "Return_5D", "Return_10D", "MA_5", "MA_20"]
    selected = list(
        dict.fromkeys(
            ["Date", "Ticker", *feature_columns, *supporting_columns, target_column, target_date_column]
        )
    )
    frame = df[selected].copy().replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[*feature_columns, target_column, target_date_column])
    frame[target_column] = frame[target_column].astype(int)
    return frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def split_development_and_test(
    df: pd.DataFrame,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_date_column = f"Target_Date_{horizon}D"
    development = df[
        (df["Date"] >= DEVELOPMENT_START_DATE)
        & (df["Date"] <= DEVELOPMENT_END_DATE)
        & (df[target_date_column] < TEST_START_DATE)
    ].copy()
    test = df[
        (df["Date"] >= TEST_START_DATE)
        & (df["Date"] <= TEST_END_DATE)
        & (df[target_date_column] <= TEST_END_DATE)
    ].copy()
    if development.empty or test.empty:
        raise ValueError(f"Empty development or test data for {horizon}D horizon")
    return development.reset_index(drop=True), test.reset_index(drop=True)


def build_purged_date_splits(
    development_df: pd.DataFrame,
    target_date_column: str,
    n_splits: int = N_SPLITS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.array(sorted(development_df["Date"].dropna().unique()))
    if len(unique_dates) < n_splits + 1:
        raise ValueError("Not enough unique dates for purged expanding validation")

    date_blocks = np.array_split(unique_dates, n_splits + 1)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for validation_dates in date_blocks[1:]:
        if len(validation_dates) == 0:
            continue
        validation_start = pd.Timestamp(validation_dates[0])
        train_mask = (
            (development_df["Date"] < validation_start)
            & (development_df[target_date_column] < validation_start)
        )
        validation_mask = development_df["Date"].isin(validation_dates)
        train_indexes = np.flatnonzero(train_mask.to_numpy())
        validation_indexes = np.flatnonzero(validation_mask.to_numpy())
        if len(train_indexes) == 0 or len(validation_indexes) == 0:
            raise ValueError("Purged split produced an empty training or validation fold")
        splits.append((train_indexes, validation_indexes))

    if len(splits) != n_splits:
        raise ValueError(f"Expected {n_splits} purged folds, created {len(splits)}")
    validate_purged_splits(development_df, target_date_column, splits)
    return splits


def validate_purged_splits(
    df: pd.DataFrame,
    target_date_column: str,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> None:
    for fold_number, (train_indexes, validation_indexes) in enumerate(splits, start=1):
        train = df.iloc[train_indexes]
        validation = df.iloc[validation_indexes]
        if train[target_date_column].max() >= validation["Date"].min():
            raise AssertionError(f"Target leakage detected in fold {fold_number}")
        if set(train["Date"]).intersection(set(validation["Date"])):
            raise AssertionError(f"A feature date appears on both sides of fold {fold_number}")
        expected_validation_rows = df[df["Date"].isin(validation["Date"].unique())]
        if len(expected_validation_rows) != len(validation):
            raise AssertionError(f"Fold {fold_number} splits stocks from the same date")


def _build_preprocessor(
    numeric_features: Sequence[str],
    include_ticker: bool,
    scale_numeric: bool,
) -> ColumnTransformer | None:
    if not include_ticker and not scale_numeric:
        return None

    numeric_transformer: Any = StandardScaler() if scale_numeric else "passthrough"
    transformers: List[Tuple[str, Any, Sequence[str]]] = [
        ("numeric", numeric_transformer, list(numeric_features))
    ]
    if include_ticker:
        transformers.append(
            (
                "ticker",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Ticker"],
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    # Keep names through future inference so feature-aware estimators such as
    # LightGBM receive the same schema at fit and predict time.
    return preprocessor.set_output(transform="pandas")


def build_estimator(
    model_name: str,
    refinement_round: int,
    numeric_features: Sequence[str],
) -> Pipeline:
    include_ticker = refinement_round >= 2
    if model_name == "logistic-regression":
        classifier: Any = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
        scale_numeric = True
    elif model_name == "elastic-net-logistic":
        classifier = LogisticRegression(
            max_iter=4000,
            solver="saga",
            penalty="elasticnet",
            random_state=RANDOM_STATE,
        )
        scale_numeric = True
    elif model_name == "random-forest":
        classifier = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
        scale_numeric = False
    elif model_name == "extra-trees":
        classifier = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1)
        scale_numeric = False
    elif model_name == "hist-gradient-boosting":
        classifier = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        scale_numeric = False
    elif model_name == "soft-voting-ensemble":
        classifier = VotingClassifier(
            estimators=[
                (
                    "logistic",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "extra_trees",
                    ExtraTreesClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
                (
                    "hist_gradient",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        min_samples_leaf=25,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ],
            voting="soft",
            n_jobs=1,
        )
        scale_numeric = True
    elif model_name == "xgboost":
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        scale_numeric = False
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install lightgbm to run the LightGBM candidate") from exc
        classifier = LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
        )
        scale_numeric = False
    elif model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Install catboost to run the CatBoost candidate") from exc
        classifier = CatBoostClassifier(
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        )
        scale_numeric = False
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    preprocessor = _build_preprocessor(numeric_features, include_ticker, scale_numeric)
    steps: List[Tuple[str, Any]] = []
    if preprocessor is not None:
        steps.append(("preprocess", preprocessor))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def get_parameter_space(model_name: str, profile: str = "full") -> tuple[Dict[str, List[Any]], int]:
    if profile == "quick":
        if model_name in {"logistic-regression", "elastic-net-logistic"}:
            return {
                "classifier__C": [1.0],
                "classifier__class_weight": ["balanced"],
                **({"classifier__l1_ratio": [0.5]} if model_name == "elastic-net-logistic" else {}),
            }, 1
        if model_name == "random-forest":
            return {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [10],
                "classifier__min_samples_leaf": [5],
                "classifier__max_features": ["sqrt"],
            }, 1
        if model_name == "xgboost":
            return {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [3],
                "classifier__learning_rate": [0.05],
                "classifier__min_child_weight": [1],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
            }, 1
        if model_name == "extra-trees":
            return {
                "classifier__n_estimators": [200],
                "classifier__max_depth": [10],
                "classifier__min_samples_leaf": [5],
                "classifier__max_features": ["sqrt"],
            }, 1
        if model_name == "hist-gradient-boosting":
            return {
                "classifier__max_iter": [150],
                "classifier__learning_rate": [0.05],
                "classifier__max_leaf_nodes": [15],
                "classifier__min_samples_leaf": [25],
            }, 1
        if model_name == "lightgbm":
            return {
                "classifier__n_estimators": [200],
                "classifier__num_leaves": [15],
                "classifier__learning_rate": [0.05],
                "classifier__min_child_samples": [25],
            }, 1
        if model_name == "catboost":
            return {
                "classifier__iterations": [200],
                "classifier__depth": [5],
                "classifier__learning_rate": [0.05],
            }, 1
        if model_name == "soft-voting-ensemble":
            return {"classifier__weights": [(2, 1, 1)]}, 1
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name == "logistic-regression":
        space = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        }
        return space, 8
    if model_name == "elastic-net-logistic":
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__l1_ratio": [0.1, 0.5, 0.9],
            "classifier__class_weight": [None, "balanced"],
        }, 12
    if model_name == "random-forest":
        space = {
            "classifier__n_estimators": [150, 300],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 5, 20],
            "classifier__max_features": ["sqrt", 0.5],
        }
        return space, 12
    if model_name == "xgboost":
        space = {
            "classifier__n_estimators": [150, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__min_child_weight": [1, 5],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        }
        return space, 16
    if model_name == "extra-trees":
        return {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 5, 20],
            "classifier__max_features": ["sqrt", 0.5, 1.0],
            "classifier__class_weight": [None, "balanced"],
        }, 20
    if model_name == "hist-gradient-boosting":
        return {
            "classifier__max_iter": [100, 200, 400],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_leaf_nodes": [7, 15, 31],
            "classifier__l2_regularization": [0.0, 0.1, 1.0],
            "classifier__min_samples_leaf": [10, 25, 50],
        }, 24
    if model_name == "lightgbm":
        return {
            "classifier__n_estimators": [150, 300],
            "classifier__num_leaves": [7, 15, 31],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__min_child_samples": [10, 25, 50],
            "classifier__reg_lambda": [0.0, 0.1, 1.0],
        }, 24
    if model_name == "catboost":
        return {
            "classifier__iterations": [150, 300],
            "classifier__depth": [3, 5, 7],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__l2_leaf_reg": [1.0, 3.0, 10.0],
        }, 24
    if model_name == "soft-voting-ensemble":
        return {
            "classifier__weights": [
                (1, 1, 1),
                (2, 1, 1),
                (3, 1, 1),
                (2, 2, 1),
                (1, 2, 2),
                (1, 1, 2),
            ]
        }, 6
    raise ValueError(f"Unsupported model: {model_name}")


def _best_result_index(cv_results: Dict[str, Any]) -> int:
    indexes = list(range(len(cv_results["params"])))
    non_degenerate = [
        index
        for index in indexes
        if cv_results["mean_test_up_recall"][index] >= MIN_DIRECTION_RECALL
        and cv_results["mean_test_down_recall"][index] >= MIN_DIRECTION_RECALL
    ]
    if non_degenerate:
        indexes = non_degenerate
    return max(
        indexes,
        key=lambda index: (
            cv_results["mean_test_balanced_accuracy"][index],
            cv_results["mean_test_mcc"][index],
            cv_results["mean_test_macro_f1"][index],
        ),
    )


def _scoring() -> Dict[str, Any]:
    from sklearn.metrics import make_scorer

    return {
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "mcc": make_scorer(matthews_corrcoef),
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "up_recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "down_recall": make_scorer(recall_score, pos_label=0, zero_division=0),
    }


def run_candidate_search(
    development_df: pd.DataFrame,
    horizon: int,
    refinement_round: int,
    model_name: str,
    feature_columns: Sequence[str],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    profile: str,
) -> tuple[Pipeline, Dict[str, Any], List[Dict[str, Any]]]:
    include_ticker = refinement_round >= 2
    input_columns = [*(("Ticker",) if include_ticker else ()), *feature_columns]
    X = development_df[input_columns]
    y = development_df[f"Target_{horizon}D"].astype(int)
    estimator = build_estimator(model_name, refinement_round, feature_columns)
    parameter_space, budget = get_parameter_space(model_name, profile=profile)
    search_jobs = int(os.getenv("MODEL_SEARCH_JOBS", "-1"))

    common = dict(
        estimator=estimator,
        scoring=_scoring(),
        refit=_best_result_index,
        cv=list(splits),
        n_jobs=search_jobs,
        error_score="raise",
        return_train_score=False,
    )
    if model_name == "logistic-regression" and profile in {"full", "focused"}:
        search: Any = GridSearchCV(param_grid=parameter_space, **common)
    else:
        search = RandomizedSearchCV(
            param_distributions=parameter_space,
            n_iter=budget,
            random_state=RANDOM_STATE,
            **common,
        )

    search.fit(X, y)
    best_index = int(search.best_index_)
    fold_balanced_accuracy = [
        float(search.cv_results_[f"split{fold}_test_balanced_accuracy"][best_index])
        for fold in range(len(splits))
    ]
    fold_mcc = [
        float(search.cv_results_[f"split{fold}_test_mcc"][best_index])
        for fold in range(len(splits))
    ]
    best_summary = {
        "horizon_days": horizon,
        "refinement_round": refinement_round,
        "model_name": model_name,
        "development_rows": len(development_df),
        "cv_balanced_accuracy": float(search.cv_results_["mean_test_balanced_accuracy"][best_index]),
        "cv_mcc": float(search.cv_results_["mean_test_mcc"][best_index]),
        "cv_macro_f1": float(search.cv_results_["mean_test_macro_f1"][best_index]),
        "cv_up_recall": float(search.cv_results_["mean_test_up_recall"][best_index]),
        "cv_down_recall": float(search.cv_results_["mean_test_down_recall"][best_index]),
        "cv_fold_balanced_accuracy": fold_balanced_accuracy,
        "cv_fold_mcc": fold_mcc,
        "cv_positive_mcc_folds": int(sum(value > 0 for value in fold_mcc)),
        "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        "input_columns": list(input_columns),
        "feature_columns": list(feature_columns),
    }

    candidate_rows: List[Dict[str, Any]] = []
    for index, parameters in enumerate(search.cv_results_["params"]):
        candidate_rows.append(
            {
                "horizon_days": horizon,
                "refinement_round": refinement_round,
                "model_name": model_name,
                "candidate_index": index,
                "cv_balanced_accuracy": search.cv_results_["mean_test_balanced_accuracy"][index],
                "cv_mcc": search.cv_results_["mean_test_mcc"][index],
                "cv_macro_f1": search.cv_results_["mean_test_macro_f1"][index],
                "cv_up_recall": search.cv_results_["mean_test_up_recall"][index],
                "cv_down_recall": search.cv_results_["mean_test_down_recall"][index],
                "parameters": json.dumps(parameters, sort_keys=True),
            }
        )
    return search.best_estimator_, best_summary, candidate_rows


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    probability_up: np.ndarray | None = None,
) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "predicted_up_rate": float(np.mean(np.asarray(y_pred) == 1)),
        "predicted_down_rate": float(np.mean(np.asarray(y_pred) == 0)),
    }
    if probability_up is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probability_up))
        metrics["average_precision"] = float(average_precision_score(y_true, probability_up))
    return metrics


def select_directional_threshold(
    y_true: np.ndarray,
    probability_up: np.ndarray,
) -> tuple[float, Dict[str, float], bool]:
    """Select a threshold without accepting an all-Up/all-Down classifier."""
    probabilities = np.asarray(probability_up, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    quantiles = np.quantile(probabilities, np.linspace(0.02, 0.98, 97))
    thresholds = np.unique(np.concatenate([quantiles, np.array([0.5])]))

    best_valid: tuple[tuple[float, ...], float, Dict[str, float]] | None = None
    best_any: tuple[tuple[float, ...], float, Dict[str, float]] | None = None
    prevalence = float(labels.mean())
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = calculate_metrics(labels, predictions, probabilities)
        key = (
            metrics["balanced_accuracy"],
            metrics["mcc"],
            metrics["macro_f1"],
            -abs(metrics["predicted_up_rate"] - prevalence),
            -abs(float(threshold) - 0.5),
        )
        candidate = (key, float(threshold), metrics)
        if best_any is None or candidate[0] > best_any[0]:
            best_any = candidate
        constraints_passed = (
            metrics["recall"] >= MIN_DIRECTION_RECALL
            and metrics["specificity"] >= MIN_DIRECTION_RECALL
            and MIN_PREDICTED_CLASS_SHARE
            <= metrics["predicted_up_rate"]
            <= MAX_PREDICTED_CLASS_SHARE
        )
        if constraints_passed and (best_valid is None or candidate[0] > best_valid[0]):
            best_valid = candidate

    selected = best_valid or best_any
    if selected is None:
        raise ValueError("No threshold candidates were generated")
    _, threshold, metrics = selected
    return threshold, metrics, best_valid is not None


def fit_calibrated_champion(
    estimator: Pipeline,
    development_df: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    input_columns: Sequence[str],
    feature_columns: Sequence[str],
    refinement_round: int,
    model_name: str,
    horizon: int,
) -> tuple[CalibratedStockModel, Dict[str, float]]:
    X = development_df[list(input_columns)]
    y = development_df[f"Target_{horizon}D"].astype(int)
    raw_probabilities: List[np.ndarray] = []
    oof_labels: List[np.ndarray] = []

    for train_indexes, validation_indexes in splits:
        fold_estimator = clone(estimator)
        fold_estimator.fit(X.iloc[train_indexes], y.iloc[train_indexes])
        raw_probabilities.append(
            fold_estimator.predict_proba(X.iloc[validation_indexes])[:, 1]
        )
        oof_labels.append(y.iloc[validation_indexes].to_numpy())

    raw_oof = np.concatenate(raw_probabilities)
    y_oof = np.concatenate(oof_labels)
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=RANDOM_STATE)
    calibrator.fit(probability_to_logit(raw_oof), y_oof)
    calibrated_oof = calibrator.predict_proba(probability_to_logit(raw_oof))[:, 1]

    best_threshold, best_metrics, threshold_constraints_passed = select_directional_threshold(
        y_oof, calibrated_oof
    )

    final_estimator = clone(estimator)
    final_estimator.fit(X, y)
    feature_reference_values: Dict[str, float] = {}
    for column in feature_columns:
        median = pd.to_numeric(development_df[column], errors="coerce").median()
        if pd.notna(median):
            feature_reference_values[column] = float(median)
    artifact = CalibratedStockModel(
        estimator=final_estimator,
        calibrator=calibrator,
        threshold=best_threshold,
        input_columns=list(input_columns),
        numeric_feature_columns=list(feature_columns),
        refinement_round=refinement_round,
        model_name=model_name,
        horizon_days=horizon,
        threshold_constraints_passed=threshold_constraints_passed,
        feature_reference_values=feature_reference_values,
    )
    return artifact, {
        "threshold": best_threshold,
        "threshold_constraints_passed": threshold_constraints_passed,
        **best_metrics,
    }


def build_baseline_predictions(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
) -> Dict[str, np.ndarray]:
    target_column = f"Target_{horizon}D"
    majority_class = int(development_df[target_column].mean() >= 0.5)
    return {
        "majority-class": np.full(len(test_df), majority_class, dtype=int),
        "previous-horizon-direction": (
            test_df[f"Return_{horizon}D"].to_numpy() > 0
        ).astype(int),
        "ma5-vs-ma20": (test_df["MA_5"].to_numpy() > test_df["MA_20"].to_numpy()).astype(int),
    }


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def _atomic_json(value: Any, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _atomic_joblib(value: Any, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temp_path)
    temp_path.replace(path)


def _save_horizon_reports(
    horizon: int,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> tuple[Path, Path]:
    report_path = CLASSIFICATION_REPORTS_DIR / f"champion-{horizon}d-classification-report.txt"
    report_path.write_text(classification_report(y_true, y_pred, zero_division=0), encoding="utf-8")
    matrix_path = CONFUSION_MATRICES_DIR / f"champion-{horizon}d-confusion-matrix.csv"
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(matrix, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]).to_csv(
        matrix_path
    )
    return report_path, matrix_path


def run_training_pipeline(search_profile: str | None = None) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_output_structure()
    profile = search_profile or os.getenv("MODEL_SEARCH_PROFILE", "full").strip().lower()
    if profile not in {"full", "focused", "quick"}:
        raise ValueError("MODEL_SEARCH_PROFILE must be 'full', 'focused', or 'quick'")

    files = get_model_input_csv_files()
    dataset_hash = hash_file_set(files)
    combined = load_and_combine_model_input_data()
    validate_required_columns(combined)

    candidate_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    per_ticker_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    created_at = datetime.now()
    run_id = created_at.strftime("%Y%m%dT%H%M%S")
    model_names = available_model_names()
    search_plan = search_plan_for_profile(profile, model_names)
    registry: Dict[str, Any] = {
        "schema_version": 2,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "created_at": created_at.isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset_sha256": dataset_hash,
        "development_start_date": DEVELOPMENT_START_DATE.date().isoformat(),
        "development_end_date": DEVELOPMENT_END_DATE.date().isoformat(),
        "test_start_date": TEST_START_DATE.date().isoformat(),
        "test_end_date": TEST_END_DATE.date().isoformat(),
        "search_profile": profile,
        "model_candidates": list(model_names),
        "search_plan": {
            str(horizon): {
                str(round_number): list(models)
                for round_number, models in rounds.items()
            }
            for horizon, rounds in search_plan.items()
        },
        "benchmark_context": {
            "test_status": "fixed observed benchmark",
            "test_warning": "2025 results were inspected during subsequent design review; new choices are selected from development evidence only.",
            "acceptance_rule": "balanced accuracy gain >= 0.02, positive MCC, macro-F1 no worse than strongest baseline",
        },
        "horizons": {},
    }

    for horizon in TARGET_HORIZONS:
        print(
            f"\n[Selecting {horizon}D champion using "
            f"{DEVELOPMENT_START_DATE.year}-{DEVELOPMENT_END_DATE.year} development data]"
        )
        best_candidates: List[tuple[Pipeline, Dict[str, Any], pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]] = []

        for refinement_round, round_models in search_plan[horizon].items():
            feature_columns = feature_columns_for_round(refinement_round)
            horizon_frame = prepare_horizon_frame(combined, horizon, feature_columns)
            development, _ = split_development_and_test(horizon_frame, horizon)
            splits = build_purged_date_splits(
                development, f"Target_Date_{horizon}D", n_splits=N_SPLITS
            )

            for model_name in round_models:
                print(f"Round {refinement_round}: {model_name}")
                estimator, best_summary, rows = run_candidate_search(
                    development,
                    horizon,
                    refinement_round,
                    model_name,
                    feature_columns,
                    splits,
                    profile,
                )
                candidate_rows.extend(rows)
                best_candidates.append((estimator, best_summary, development, splits))

        estimator, best_summary, development, splits = max(
            best_candidates,
            key=lambda item: (
                item[1]["cv_balanced_accuracy"],
                item[1]["cv_mcc"],
                item[1]["cv_macro_f1"],
            ),
        )
        refinement_round = int(best_summary["refinement_round"])
        feature_columns = list(best_summary["feature_columns"])
        input_columns = list(best_summary["input_columns"])

        champion_frame = prepare_horizon_frame(combined, horizon, feature_columns)
        development, test = split_development_and_test(champion_frame, horizon)
        splits = build_purged_date_splits(
            development, f"Target_Date_{horizon}D", n_splits=N_SPLITS
        )
        artifact, oof_metrics = fit_calibrated_champion(
            estimator,
            development,
            splits,
            input_columns,
            feature_columns,
            refinement_round,
            str(best_summary["model_name"]),
            horizon,
        )

        target_column = f"Target_{horizon}D"
        y_test = test[target_column].astype(int)
        probability_up = artifact.predict_proba(test[input_columns])[:, 1]
        predictions = (probability_up >= artifact.threshold).astype(int)
        test_metrics = calculate_metrics(y_test, predictions, probability_up)

        baseline_predictions = build_baseline_predictions(development, test, horizon)
        evaluated_baselines: List[tuple[str, Dict[str, float]]] = []
        for baseline_name, baseline_pred in baseline_predictions.items():
            metrics = calculate_metrics(y_test, baseline_pred)
            evaluated_baselines.append((baseline_name, metrics))
            baseline_rows.append(
                {"horizon_days": horizon, "baseline_name": baseline_name, **metrics}
            )
        strongest_baseline_name, strongest_baseline_metrics = max(
            evaluated_baselines,
            key=lambda item: (
                item[1]["balanced_accuracy"],
                item[1]["mcc"],
                item[1]["macro_f1"],
            ),
        )

        stability_passed = bool(best_summary.get("cv_positive_mcc_folds", 0) >= 4)
        threshold_constraints_passed = bool(
            oof_metrics.get("threshold_constraints_passed", False)
        )
        test_directional_balance_passed = bool(
            test_metrics["recall"] >= MIN_DIRECTION_RECALL
            and test_metrics["specificity"] >= MIN_DIRECTION_RECALL
            and MIN_PREDICTED_CLASS_SHARE
            <= test_metrics["predicted_up_rate"]
            <= MAX_PREDICTED_CLASS_SHARE
        )
        threshold_passed = bool(
            test_metrics["balanced_accuracy"]
            >= strongest_baseline_metrics["balanced_accuracy"] + MIN_BALANCED_ACCURACY_GAIN
            and test_metrics["mcc"] > 0
            and test_metrics["macro_f1"] >= strongest_baseline_metrics["macro_f1"]
            and stability_passed
            and threshold_constraints_passed
            and test_directional_balance_passed
        )
        status = "accepted" if threshold_passed else "experimental"

        model_path = MODELS_DIR / f"stock-trend-{horizon}d-{run_id}.joblib"
        _atomic_joblib(artifact, model_path)
        report_path, matrix_path = _save_horizon_reports(horizon, y_test, predictions)

        for ticker, ticker_frame in test.assign(_prediction=predictions, _probability=probability_up).groupby(
            "Ticker"
        ):
            ticker_metrics = calculate_metrics(
                ticker_frame[target_column].astype(int),
                ticker_frame["_prediction"].to_numpy(),
                ticker_frame["_probability"].to_numpy(),
            )
            per_ticker_rows.append(
                {"horizon_days": horizon, "ticker": ticker, "rows": len(ticker_frame), **ticker_metrics}
            )

        summary = {
            "horizon_days": horizon,
            "model_name": best_summary["model_name"],
            "refinement_round": refinement_round,
            "development_rows": len(development),
            "test_rows": len(test),
            "cv_balanced_accuracy": best_summary["cv_balanced_accuracy"],
            "cv_mcc": best_summary["cv_mcc"],
            "cv_macro_f1": best_summary["cv_macro_f1"],
            "cv_up_recall": best_summary["cv_up_recall"],
            "cv_down_recall": best_summary["cv_down_recall"],
            "cv_fold_balanced_accuracy": best_summary["cv_fold_balanced_accuracy"],
            "cv_fold_mcc": best_summary["cv_fold_mcc"],
            "cv_positive_mcc_folds": best_summary["cv_positive_mcc_folds"],
            "locked_threshold": artifact.threshold,
            "strongest_baseline": strongest_baseline_name,
            "baseline_balanced_accuracy": strongest_baseline_metrics["balanced_accuracy"],
            "balanced_accuracy_gain": test_metrics["balanced_accuracy"]
            - strongest_baseline_metrics["balanced_accuracy"],
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "threshold_passed": threshold_passed,
            "stability_passed": stability_passed,
            "threshold_constraints_passed": threshold_constraints_passed,
            "test_directional_balance_passed": test_directional_balance_passed,
            "status": status,
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_path": str(matrix_path.relative_to(PROJECT_ROOT)),
        }
        summary_rows.append(summary)
        registry["horizons"][str(horizon)] = {
            "artifact_path": str(model_path.relative_to(PROJECT_ROOT)),
            "model_name": best_summary["model_name"],
            "refinement_round": refinement_round,
            "input_columns": input_columns,
            "numeric_feature_columns": feature_columns,
            "calibrated": True,
            "decision_threshold": artifact.threshold,
            "threshold_passed": threshold_passed,
            "stability_passed": stability_passed,
            "threshold_constraints_passed": threshold_constraints_passed,
            "test_directional_balance_passed": test_directional_balance_passed,
            "cv_up_recall": best_summary["cv_up_recall"],
            "cv_down_recall": best_summary["cv_down_recall"],
            "cv_fold_balanced_accuracy": best_summary["cv_fold_balanced_accuracy"],
            "cv_fold_mcc": best_summary["cv_fold_mcc"],
            "cv_positive_mcc_folds": best_summary["cv_positive_mcc_folds"],
            "status": status,
            "strongest_baseline": strongest_baseline_name,
            "metrics": test_metrics,
            "baseline_metrics": strongest_baseline_metrics,
            "oof_metrics": oof_metrics,
        }
        print(
            f"{horizon}D champion: {best_summary['model_name']} round {refinement_round}; "
            f"balanced accuracy={test_metrics['balanced_accuracy']:.4f}; status={status}"
        )

    _atomic_csv(pd.DataFrame(candidate_rows), CANDIDATE_FILE)
    _atomic_csv(pd.DataFrame(baseline_rows), BASELINE_FILE)
    _atomic_csv(pd.DataFrame(per_ticker_rows), PER_TICKER_FILE)
    _atomic_csv(pd.DataFrame(summary_rows), SUMMARY_FILE)
    _atomic_json(registry, REGISTRY_FILE)
    print(f"\nModel registry saved to: {REGISTRY_FILE}")


def main() -> None:
    run_training_pipeline()


if __name__ == "__main__":
    main()
