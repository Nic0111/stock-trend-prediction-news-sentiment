from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import hashlib
import importlib.util
import json
import os
import shutil
import warnings
from datetime import datetime
from importlib import metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
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
from pipeline.horizon_acceptance import (
    COMMON_POLICY,
    POLICY as HORIZON_ACCEPTANCE_POLICY,
    evaluate_acceptance,
    weaker_status,
)
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
STAGED_RUNS_DIR = MODELS_DIR / "staged-stock-runs"
SELECTION_POLICY_FILE = PROJECT_ROOT / "config" / "stock_model_selection_policy.json"

# The downloaded history starts in 2015. Keep this boundary explicit so a
# stray older row cannot silently enter development, while 2025 remains the
# fixed observed benchmark holdout.
DEVELOPMENT_START_DATE = pd.Timestamp("2015-01-01")
DEVELOPMENT_END_DATE = pd.Timestamp("2024-12-31")
TEST_START_DATE = pd.Timestamp("2025-01-01")
TEST_END_DATE = pd.Timestamp("2025-12-31")
N_SPLITS = 5
RANDOM_STATE = 42
OFFICIAL_PROFILE = "official"
OFFICIAL_SEEDS = (11, 42, 73, 101, 202)
OFFICIAL_MODEL_UNIVERSE = (
    "logistic-regression",
    "elastic-net-logistic",
    "random-forest",
    "extra-trees",
    "hist-gradient-boosting",
    "xgboost",
    "lightgbm",
    "soft-voting-ensemble",
)
OFFICIAL_FEATURE_ROUNDS = {1: 3, 5: 4, 10: 4}
MIN_BALANCED_ACCURACY_GAIN = float(COMMON_POLICY["minimum_balanced_accuracy_gain"])
MIN_DIRECTION_RECALL = float(COMMON_POLICY["minimum_direction_recall"])
MIN_PREDICTED_CLASS_SHARE = float(COMMON_POLICY["minimum_predicted_class_share"])
MAX_PREDICTED_CLASS_SHARE = float(COMMON_POLICY["maximum_predicted_class_share"])
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


def load_selection_policy() -> Dict[str, Any]:
    """Load the versioned reproducibility contract used by official training."""
    return json.loads(SELECTION_POLICY_FILE.read_text(encoding="utf-8"))


class SklearnCompatibleCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """CatBoost adapter for scikit-learn 1.8's estimator/tag protocol.

    CatBoost 1.2.8 still exposes the older scikit-learn integration. Wrapping
    it in a native BaseEstimator keeps cloning, parameter search, pipelines,
    and tags working without changing the underlying CatBoost algorithm.
    """

    def __init__(
        self,
        iterations: int = 300,
        depth: int = 5,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        loss_function: str = "Logloss",
        random_seed: int = RANDOM_STATE,
        verbose: bool = False,
        thread_count: int = 1,
        allow_writing_files: bool = False,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.thread_count = thread_count
        self.allow_writing_files = allow_writing_files

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "SklearnCompatibleCatBoostClassifier":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("Install catboost to run the CatBoost candidate") from exc
        self._catboost = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            thread_count=self.thread_count,
            allow_writing_files=self.allow_writing_files,
        )
        self._catboost.fit(X, y, **fit_params)
        self.classes_ = np.asarray(self._catboost.classes_)
        self.n_features_in_ = getattr(self._catboost, "n_features_in_", X.shape[1])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self._catboost.predict(X)).reshape(-1)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self._catboost.predict_proba(X))


def catboost_compatibility_report() -> Dict[str, Any]:
    """Report whether CatBoost can run through the current sklearn pipeline."""
    if importlib.util.find_spec("catboost") is None:
        return {"installed": False, "compatible": False, "reason": "catboost is not installed"}
    try:
        from sklearn.utils._tags import get_tags

        get_tags(SklearnCompatibleCatBoostClassifier())
        return {
            "installed": True,
            "compatible": True,
            "adapter": "SklearnCompatibleCatBoostClassifier",
        }
    except Exception as exc:
        return {"installed": True, "compatible": False, "reason": str(exc)}


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
    if catboost_compatibility_report()["compatible"]:
        names.append("catboost")
    return tuple(names)


def search_plan_for_profile(
    profile: str,
    available_models: Sequence[str],
) -> Dict[int, Dict[int, tuple[str, ...]]]:
    """Choose a bounded evidence-led search without changing validation rules."""
    available = set(available_models)
    if profile == OFFICIAL_PROFILE:
        missing = [model for model in OFFICIAL_MODEL_UNIVERSE if model not in available]
        if missing:
            raise RuntimeError(
                "Official stock-model selection requires the fixed candidate universe. "
                f"Missing candidates: {', '.join(missing)}"
            )
        return {
            horizon: {OFFICIAL_FEATURE_ROUNDS[horizon]: OFFICIAL_MODEL_UNIVERSE}
            for horizon in TARGET_HORIZONS
        }

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


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


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
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    include_ticker = refinement_round >= 2
    if model_name == "logistic-regression":
        classifier: Any = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state)
        scale_numeric = True
    elif model_name == "elastic-net-logistic":
        classifier = LogisticRegression(
            max_iter=4000,
            solver="saga",
            penalty="elasticnet",
            random_state=random_state,
        )
        scale_numeric = True
    elif model_name == "random-forest":
        classifier = RandomForestClassifier(random_state=random_state, n_jobs=1)
        scale_numeric = False
    elif model_name == "extra-trees":
        classifier = ExtraTreesClassifier(random_state=random_state, n_jobs=1)
        scale_numeric = False
    elif model_name == "hist-gradient-boosting":
        classifier = HistGradientBoostingClassifier(random_state=random_state)
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
                        random_state=random_state,
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
                        random_state=random_state,
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
                        random_state=random_state,
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
            random_state=random_state,
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
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )
        scale_numeric = False
    elif model_name == "catboost":
        if not catboost_compatibility_report()["compatible"]:
            raise RuntimeError(
                "CatBoost is unavailable or incompatible with the installed scikit-learn version"
            )
        classifier = SklearnCompatibleCatBoostClassifier(
            loss_function="Logloss",
            random_seed=random_state,
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
    if profile in {"quick", OFFICIAL_PROFILE}:
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
            -index,
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
    *,
    seed: int = RANDOM_STATE,
    search_seed: int = RANDOM_STATE,
    fixed_parameters: Dict[str, Any] | None = None,
) -> tuple[Pipeline, Dict[str, Any], List[Dict[str, Any]]]:
    include_ticker = refinement_round >= 2
    input_columns = [*(("Ticker",) if include_ticker else ()), *feature_columns]
    X = development_df[input_columns]
    y = development_df[f"Target_{horizon}D"].astype(int)
    estimator = build_estimator(
        model_name, refinement_round, feature_columns, random_state=seed
    )
    parameter_space, budget = get_parameter_space(model_name, profile=profile)
    search_jobs = 1 if profile == OFFICIAL_PROFILE else int(os.getenv("MODEL_SEARCH_JOBS", "-1"))

    common = dict(
        estimator=estimator,
        scoring=_scoring(),
        refit=_best_result_index,
        cv=list(splits),
        n_jobs=search_jobs,
        error_score="raise",
        return_train_score=False,
    )
    if fixed_parameters is not None:
        fixed_grid = {key: [value] for key, value in fixed_parameters.items()}
        search: Any = GridSearchCV(param_grid=fixed_grid, **common)
    elif model_name == "logistic-regression" and profile in {"full", "focused", OFFICIAL_PROFILE}:
        search: Any = GridSearchCV(param_grid=parameter_space, **common)
    else:
        search = RandomizedSearchCV(
            param_distributions=parameter_space,
            n_iter=budget,
            random_state=search_seed,
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
        "seed": seed,
        "fixed_parameters": fixed_parameters is not None,
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
                "seed": seed,
                "fixed_parameters": fixed_parameters is not None,
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


OFFICIAL_MODEL_COMPLEXITY = {
    "logistic-regression": 1,
    "elastic-net-logistic": 2,
    "hist-gradient-boosting": 3,
    "random-forest": 4,
    "extra-trees": 5,
    "xgboost": 6,
    "lightgbm": 6,
    "soft-voting-ensemble": 7,
}


def _official_candidate_sort_key(summary: Dict[str, Any]) -> tuple[Any, ...]:
    """Stable ranking: quality, stability, then simpler model and name."""
    return (
        -float(summary["cv_balanced_accuracy"]),
        -float(summary["cv_mcc"]),
        -float(summary["cv_macro_f1"]),
        float(summary.get("cv_seed_balanced_accuracy_std", 0.0)),
        int(OFFICIAL_MODEL_COMPLEXITY.get(str(summary["model_name"]), 99)),
        str(summary["model_name"]),
    )


def run_official_candidate_selection(
    development_df: pd.DataFrame,
    horizon: int,
    refinement_round: int,
    model_name: str,
    feature_columns: Sequence[str],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> tuple[Pipeline, Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one fixed model/configuration over the official seed panel.

    A single seed-42 search establishes the hyperparameter manifest. Every
    other seed evaluates that exact manifest, so retraining cannot silently
    change both the algorithm and the random draw at once.
    """
    anchor_estimator, anchor_summary, anchor_rows = run_candidate_search(
        development_df,
        horizon,
        refinement_round,
        model_name,
        feature_columns,
        splits,
        OFFICIAL_PROFILE,
        seed=RANDOM_STATE,
        search_seed=RANDOM_STATE,
    )
    fixed_parameters = json.loads(anchor_summary["best_parameters"])
    seed_summaries: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    representative_estimator: Pipeline | None = None

    for seed in OFFICIAL_SEEDS:
        estimator, summary, rows = run_candidate_search(
            development_df,
            horizon,
            refinement_round,
            model_name,
            feature_columns,
            splits,
            OFFICIAL_PROFILE,
            seed=seed,
            search_seed=RANDOM_STATE,
            fixed_parameters=fixed_parameters,
        )
        seed_summaries.append(summary)
        seed_rows.extend(rows)
        if seed == RANDOM_STATE:
            representative_estimator = estimator

    if representative_estimator is None:
        raise RuntimeError("Official seed panel did not include the representative seed")

    fold_ba = np.asarray(
        [summary["cv_fold_balanced_accuracy"] for summary in seed_summaries], dtype=float
    )
    fold_mcc = np.asarray([summary["cv_fold_mcc"] for summary in seed_summaries], dtype=float)
    seed_ba = np.asarray([summary["cv_balanced_accuracy"] for summary in seed_summaries], dtype=float)
    seed_mcc = np.asarray([summary["cv_mcc"] for summary in seed_summaries], dtype=float)
    seed_f1 = np.asarray([summary["cv_macro_f1"] for summary in seed_summaries], dtype=float)
    seed_up_recall = np.asarray([summary["cv_up_recall"] for summary in seed_summaries], dtype=float)
    seed_down_recall = np.asarray([summary["cv_down_recall"] for summary in seed_summaries], dtype=float)
    positive_seed_fraction = float(
        np.mean(
            [
                summary["cv_positive_mcc_folds"] >= int(0.8 * len(splits))
                for summary in seed_summaries
            ]
        )
    )
    summary = {
        "horizon_days": horizon,
        "refinement_round": refinement_round,
        "model_name": model_name,
        "development_rows": len(development_df),
        "cv_balanced_accuracy": float(seed_ba.mean()),
        "cv_mcc": float(seed_mcc.mean()),
        "cv_macro_f1": float(seed_f1.mean()),
        "cv_up_recall": float(seed_up_recall.mean()),
        "cv_down_recall": float(seed_down_recall.mean()),
        "cv_fold_balanced_accuracy": [float(value) for value in fold_ba.mean(axis=0)],
        "cv_fold_mcc": [float(value) for value in fold_mcc.mean(axis=0)],
        "cv_positive_mcc_folds": int(round(float(np.mean([summary["cv_positive_mcc_folds"] for summary in seed_summaries])))),
        "cv_seed_balanced_accuracy": [float(value) for value in seed_ba],
        "cv_seed_mcc": [float(value) for value in seed_mcc],
        "cv_seed_macro_f1": [float(value) for value in seed_f1],
        "cv_seed_balanced_accuracy_std": float(seed_ba.std(ddof=0)),
        "positive_mcc_seed_fraction": positive_seed_fraction,
        "official_seeds": list(OFFICIAL_SEEDS),
        "search_seed": RANDOM_STATE,
        "fixed_hyperparameters": True,
        "best_parameters": json.dumps(fixed_parameters, sort_keys=True),
        "input_columns": list(anchor_summary["input_columns"]),
        "feature_columns": list(feature_columns),
    }
    rows: List[Dict[str, Any]] = []
    for row in anchor_rows:
        row = dict(row)
        row["selection_stage"] = "seed-42-search"
        rows.append(row)
    for row in seed_rows:
        row = dict(row)
        row["selection_stage"] = "official-fixed-manifest"
        rows.append(row)
    return representative_estimator, summary, rows


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
    *,
    majority_class: int | None = None,
) -> Dict[str, np.ndarray]:
    target_column = f"Target_{horizon}D"
    if majority_class is None:
        majority_class = int(development_df[target_column].mean() >= 0.5)
    baselines = {
        "majority-class": np.full(len(test_df), majority_class, dtype=int),
        "previous-horizon-direction": (
            test_df[f"Return_{horizon}D"].to_numpy() > 0
        ).astype(int),
        "ma5-vs-ma20": (test_df["MA_5"].to_numpy() > test_df["MA_20"].to_numpy()).astype(int),
    }
    market_column = f"Market_Return_{horizon}D"
    if market_column in test_df.columns:
        baselines["klci-horizon-direction"] = (
            test_df[market_column].to_numpy() > 0
        ).astype(int)
    return baselines


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


def run_training_pipeline(
    search_profile: str | None = None,
    *,
    promote: bool | None = None,
) -> Path:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_output_structure()
    profile = search_profile or os.getenv("MODEL_SEARCH_PROFILE", "full").strip().lower()
    if profile not in {"full", "focused", "quick", OFFICIAL_PROFILE}:
        raise ValueError("MODEL_SEARCH_PROFILE must be 'full', 'focused', 'quick', or 'official'")
    if promote is None:
        # Official runs are challengers by default. Promotion is an explicit,
        # reviewed action so a retrain cannot silently replace production.
        promote = profile != OFFICIAL_PROFILE
    if profile == OFFICIAL_PROFILE and promote:
        raise ValueError(
            "Official runs must be staged first; evaluate prospective shadow evidence, "
            "then use promote_stock_models --approve"
        )

    files = get_model_input_csv_files()
    dataset_hash = hash_file_set(files)
    combined = load_and_combine_model_input_data()
    validate_required_columns(combined)

    candidate_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    per_ticker_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    created_at = datetime.now()
    run_id = created_at.strftime("%Y%m%dT%H%M%S%f")
    model_names = available_model_names()
    search_plan = search_plan_for_profile(profile, model_names)
    selection_policy = load_selection_policy()
    artifact_dir = MODELS_DIR
    if profile == OFFICIAL_PROFILE and not promote:
        artifact_dir = STAGED_RUNS_DIR / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
    registry_target = REGISTRY_FILE if promote else artifact_dir / "model-registry.json"
    registry: Dict[str, Any] = {
        "schema_version": 4,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "created_at": created_at.isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset_sha256": dataset_hash,
        "development_start_date": DEVELOPMENT_START_DATE.date().isoformat(),
        "development_end_date": DEVELOPMENT_END_DATE.date().isoformat(),
        "test_start_date": TEST_START_DATE.date().isoformat(),
        "test_end_date": TEST_END_DATE.date().isoformat(),
        "search_profile": profile,
        "publication_status": "validated" if promote else "staged",
        "promotion_required": bool(profile == OFFICIAL_PROFILE and not promote),
        "selection_policy": selection_policy,
        "selection_policy_sha256": hash_file(SELECTION_POLICY_FILE),
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
            "acceptance_policy_version": HORIZON_ACCEPTANCE_POLICY["policy_version"],
            "acceptance_policy_path": str(Path("config") / "horizon_acceptance_policy.json"),
            "status_tiers": ["accepted", "limited", "experimental"],
            "official_selection": {
                "fixed_feature_rounds": selection_policy["feature_round_by_horizon"],
                "candidate_universe": selection_policy["official_model_universe"],
                "seeds": selection_policy["evaluation_seeds"],
                "search_seed": selection_policy["search_seed"],
                "practical_balanced_accuracy_gain": selection_policy[
                    "minimum_practical_balanced_accuracy_gain"
                ],
                "requires_prospective_shadow": selection_policy[
                    "requires_prospective_shadow"
                ],
                "manual_promotion_required": selection_policy["manual_promotion_required"],
            },
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
                if profile == OFFICIAL_PROFILE:
                    estimator, best_summary, rows = run_official_candidate_selection(
                        development,
                        horizon,
                        refinement_round,
                        model_name,
                        feature_columns,
                        splits,
                    )
                else:
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

        if profile == OFFICIAL_PROFILE:
            ranked_candidates = sorted(best_candidates, key=lambda item: _official_candidate_sort_key(item[1]))
            estimator, best_summary, development, splits = ranked_candidates[0]
            top_score = float(best_summary["cv_balanced_accuracy"])
            confidence_set = [
                str(item[1]["model_name"])
                for item in ranked_candidates
                if top_score - float(item[1]["cv_balanced_accuracy"])
                <= float(selection_policy["minimum_practical_balanced_accuracy_gain"])
            ]
            best_summary["official_confidence_set"] = confidence_set
            best_summary["official_ranked_candidates"] = [
                {
                    "model_name": str(item[1]["model_name"]),
                    "cv_balanced_accuracy": float(item[1]["cv_balanced_accuracy"]),
                    "cv_mcc": float(item[1]["cv_mcc"]),
                    "cv_macro_f1": float(item[1]["cv_macro_f1"]),
                    "cv_seed_balanced_accuracy_std": float(
                        item[1].get("cv_seed_balanced_accuracy_std", 0.0)
                    ),
                }
                for item in ranked_candidates
            ]
        else:
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
        development_baselines = build_baseline_predictions(development, development, horizon)
        development_baseline_metrics = [
            (
                baseline_name,
                calculate_metrics(
                    development[f"Target_{horizon}D"].astype(int),
                    baseline_pred,
                ),
            )
            for baseline_name, baseline_pred in development_baselines.items()
        ]
        strongest_baseline_name, strongest_development_baseline_metrics = max(
            development_baseline_metrics,
            key=lambda item: (
                item[1]["balanced_accuracy"],
                item[1]["mcc"],
                item[1]["macro_f1"],
            ),
        )
        strongest_baseline_predictions = baseline_predictions[strongest_baseline_name]
        strongest_baseline_metrics = calculate_metrics(y_test, strongest_baseline_predictions)
        for baseline_name, baseline_pred in baseline_predictions.items():
            baseline_metrics = calculate_metrics(y_test, baseline_pred)
            baseline_rows.append(
                {"horizon_days": horizon, "baseline_name": baseline_name, **baseline_metrics}
            )

        acceptance = evaluate_acceptance(
            horizon=horizon,
            y_true=y_test,
            predictions=predictions,
            probability_up=probability_up,
            baseline_predictions=strongest_baseline_predictions,
            dates=test["Date"].to_numpy(),
            metrics=test_metrics,
            baseline_metrics=strongest_baseline_metrics,
            oof_metrics=oof_metrics,
            cv_positive_mcc_folds=int(best_summary.get("cv_positive_mcc_folds", 0)),
            phase="observed",
        )
        stability_passed = bool(acceptance["accepted_checks"]["development_stability"])
        threshold_constraints_passed = bool(
            acceptance["accepted_checks"]["threshold_constraints"]
        )
        test_directional_balance_passed = bool(
            acceptance["accepted_checks"]["directional_recall"]
            and acceptance["accepted_checks"]["predicted_class_share"]
        )
        threshold_passed = acceptance["status"] == "accepted"
        status = str(acceptance["status"])

        model_path = artifact_dir / f"stock-trend-{horizon}d-{run_id}.joblib"
        _atomic_joblib(artifact, model_path)
        artifact_sha256 = hash_file(model_path)
        report_path, matrix_path = _save_horizon_reports(horizon, y_test, predictions)

        ticker_acceptance: Dict[str, Dict[str, Any]] = {}
        evaluated_test = test.assign(
            _prediction=predictions,
            _probability=probability_up,
            _baseline=strongest_baseline_predictions,
        )
        for ticker, ticker_frame in evaluated_test.groupby(
            "Ticker"
        ):
            ticker_metrics = calculate_metrics(
                ticker_frame[target_column].astype(int),
                ticker_frame["_prediction"].to_numpy(),
                ticker_frame["_probability"].to_numpy(),
            )
            ticker_baseline = ticker_frame["_baseline"].to_numpy()
            ticker_acceptance_result = evaluate_acceptance(
                horizon=horizon,
                y_true=ticker_frame[target_column].astype(int).to_numpy(),
                predictions=ticker_frame["_prediction"].to_numpy(),
                probability_up=ticker_frame["_probability"].to_numpy(),
                baseline_predictions=ticker_baseline,
                dates=ticker_frame["Date"].to_numpy(),
                metrics=ticker_metrics,
                baseline_metrics=calculate_metrics(
                    ticker_frame[target_column].astype(int).to_numpy(), ticker_baseline
                ),
                oof_metrics=oof_metrics,
                cv_positive_mcc_folds=int(best_summary.get("cv_positive_mcc_folds", 0)),
                phase="observed",
                ticker=str(ticker),
            )
            ticker_status = weaker_status(status, str(ticker_acceptance_result["status"]))
            ticker_acceptance_result["status"] = ticker_status
            ticker_acceptance[str(ticker)] = ticker_acceptance_result
            per_ticker_rows.append(
                {
                    "horizon_days": horizon,
                    "ticker": ticker,
                    "rows": len(ticker_frame),
                    "status": ticker_status,
                    **ticker_metrics,
                    "bootstrap_gain_lower": ticker_acceptance_result["bootstrap"]["lower"],
                    "bootstrap_gain_upper": ticker_acceptance_result["bootstrap"]["upper"],
                    "failed_checks": ";".join(ticker_acceptance_result["failed_checks"]),
                }
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
            "cv_seed_balanced_accuracy": best_summary.get("cv_seed_balanced_accuracy"),
            "cv_seed_mcc": best_summary.get("cv_seed_mcc"),
            "cv_seed_macro_f1": best_summary.get("cv_seed_macro_f1"),
            "cv_seed_balanced_accuracy_std": best_summary.get("cv_seed_balanced_accuracy_std"),
            "official_confidence_set": best_summary.get("official_confidence_set"),
            "official_ranked_candidates": best_summary.get("official_ranked_candidates"),
            "positive_mcc_seed_fraction": best_summary.get("positive_mcc_seed_fraction"),
            "official_seeds": best_summary.get("official_seeds"),
            "fixed_hyperparameters": best_summary.get("fixed_hyperparameters", False),
            "search_seed": best_summary.get("search_seed", RANDOM_STATE),
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
            "acceptance_policy_version": HORIZON_ACCEPTANCE_POLICY["policy_version"],
            "acceptance": acceptance,
            "status": status,
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "report_path": str(report_path.relative_to(PROJECT_ROOT)),
            "confusion_matrix_path": str(matrix_path.relative_to(PROJECT_ROOT)),
        }
        summary_rows.append(summary)
        registry["horizons"][str(horizon)] = {
            "artifact_path": str(model_path.relative_to(PROJECT_ROOT)),
            "artifact_sha256": artifact_sha256,
            "training_run_id": run_id,
            "model_name": best_summary["model_name"],
            "refinement_round": refinement_round,
            "input_columns": input_columns,
            "numeric_feature_columns": feature_columns,
            "calibrated": True,
            "decision_threshold": artifact.threshold,
            "threshold_passed": threshold_passed,
            "global_status": status,
            "observed_acceptance": acceptance,
            "shadow_acceptance": None,
            "ticker_acceptance": ticker_acceptance,
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
            "baseline_majority_class": int(
                development[f"Target_{horizon}D"].astype(int).mean() >= 0.5
            ),
            "metrics": test_metrics,
            "baseline_metrics": strongest_baseline_metrics,
            "oof_metrics": oof_metrics,
            "official_selection": {
                "confidence_set": best_summary.get("official_confidence_set"),
                "ranked_candidates": best_summary.get("official_ranked_candidates"),
                "positive_mcc_seed_fraction": best_summary.get("positive_mcc_seed_fraction"),
                "seed_balanced_accuracy_std": best_summary.get("cv_seed_balanced_accuracy_std"),
                "fixed_hyperparameters": best_summary.get("fixed_hyperparameters", False),
                "manual_promotion_required": bool(profile == OFFICIAL_PROFILE),
            }
        }
        print(
            f"{horizon}D champion: {best_summary['model_name']} round {refinement_round}; "
            f"balanced accuracy={test_metrics['balanced_accuracy']:.4f}; status={status}"
        )

    _atomic_csv(pd.DataFrame(candidate_rows), CANDIDATE_FILE)
    _atomic_csv(pd.DataFrame(baseline_rows), BASELINE_FILE)
    _atomic_csv(pd.DataFrame(per_ticker_rows), PER_TICKER_FILE)
    _atomic_csv(pd.DataFrame(summary_rows), SUMMARY_FILE)
    bundle_dir = RESULTS_DIR / "training-runs" / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_files = [SUMMARY_FILE, CANDIDATE_FILE, BASELINE_FILE, PER_TICKER_FILE]
    for source in bundle_files:
        shutil.copy2(source, bundle_dir / source.name)
    registry["report_bundle_path"] = str(bundle_dir.relative_to(PROJECT_ROOT))
    registry["provenance"] = {
        "dataset_sha256": dataset_hash,
        "artifact_sha256": {str(h): value.get("artifact_sha256") for h, value in registry["horizons"].items()},
        "report_sha256": {source.name: hash_file(bundle_dir / source.name) for source in bundle_files},
        "publication_status": registry["publication_status"],
        "training_code_sha256": hash_file(Path(__file__).resolve()),
        "python_version": os.sys.version,
        "package_versions": {
            package: _installed_version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost")
        },
    }
    _atomic_json(registry, registry_target)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "registry_signature": hash_file(registry_target),
                "dataset_sha256": dataset_hash,
                "artifacts": registry["provenance"]["artifact_sha256"],
                "reports": registry["provenance"]["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nModel registry saved to: {registry_target}")
    return registry_target


def _staged_registry_path(run_id: str) -> Path:
    path = STAGED_RUNS_DIR / run_id / "model-registry.json"
    if not path.exists():
        raise FileNotFoundError(f"No staged official run found for run id: {run_id}")
    return path


def promote_stock_models(run_id: str, horizons: Sequence[int] | None = None, *, approve: bool = False) -> Path:
    """Promote a reviewed staged official run and retain rollback metadata."""
    if not approve:
        raise PermissionError("Promotion is manual: pass --approve after reviewing the shadow report")
    staged_path = _staged_registry_path(run_id)
    staged = json.loads(staged_path.read_text(encoding="utf-8"))
    if staged.get("search_profile") != OFFICIAL_PROFILE:
        raise ValueError("Only official profile runs may be promoted")
    selected_horizons = [str(value) for value in (horizons or TARGET_HORIZONS)]
    missing = [horizon for horizon in selected_horizons if horizon not in staged.get("horizons", {})]
    if missing:
        raise ValueError(f"Staged run has no horizon entries for: {', '.join(missing)}")
    shadow = staged.get("shadow_evaluation", {})
    policy = staged.get("selection_policy", {})
    if policy.get("requires_prospective_shadow", True):
        if shadow.get("phase") != "prospective_shadow":
            raise ValueError("Official promotion requires a prospective shadow evaluation")
        minimum_sessions = int(policy.get("minimum_matured_shadow_sessions", 100))
        for horizon in selected_horizons:
            shadow_result = staged["horizons"][horizon].get("shadow_acceptance") or {}
            if int(shadow_result.get("rows", 0)) < minimum_sessions:
                raise ValueError(
                    f"{horizon}D has only {shadow_result.get('rows', 0)} matured shadow sessions; "
                    f"{minimum_sessions} are required before promotion"
                )

    backup_dir = MODELS_DIR / "registry-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if REGISTRY_FILE.exists():
        shutil.copy2(REGISTRY_FILE, backup_dir / f"model-registry-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}.json")

    existing = json.loads(REGISTRY_FILE.read_text(encoding="utf-8")) if REGISTRY_FILE.exists() else {}
    production = dict(staged if len(selected_horizons) == len(TARGET_HORIZONS) else existing)
    if not production:
        production = dict(staged)
    production["publication_status"] = "validated"
    production["promotion_required"] = False
    production["promoted_at"] = datetime.now().isoformat(timespec="seconds")
    production["promoted_from_run_id"] = run_id
    production["promotion_approval"] = "manual --approve"
    production_horizons = dict(production.get("horizons", {}))
    for horizon in selected_horizons:
        entry = dict(production_horizons[horizon])
        source = PROJECT_ROOT / entry["artifact_path"]
        if not source.exists():
            raise FileNotFoundError(f"Staged artifact is missing: {source}")
        destination = MODELS_DIR / f"stock-trend-{horizon}d-{run_id}.joblib"
        shutil.copy2(source, destination)
        entry["artifact_path"] = str(destination.relative_to(PROJECT_ROOT))
        entry["artifact_sha256"] = hash_file(destination)
        entry["training_run_id"] = run_id
        production_horizons[horizon] = entry
    production["horizons"] = production_horizons
    _atomic_json(production, REGISTRY_FILE)
    return REGISTRY_FILE


def verify_stock_reproducibility(run_a: str, run_b: str) -> Dict[str, Any]:
    """Compare two official manifests without changing production state."""
    first = json.loads(_staged_registry_path(run_a).read_text(encoding="utf-8"))
    second = json.loads(_staged_registry_path(run_b).read_text(encoding="utf-8"))
    fields = (
        "dataset_sha256",
        "feature_schema_version",
        "selection_policy_sha256",
        "search_profile",
        "search_plan",
    )
    mismatches = {
        field: {"run_a": first.get(field), "run_b": second.get(field)}
        for field in fields
        if first.get(field) != second.get(field)
    }
    horizon_comparison: Dict[str, Any] = {}
    for horizon in sorted(set(first.get("horizons", {})) | set(second.get("horizons", {}))):
        a = first.get("horizons", {}).get(horizon, {})
        b = second.get("horizons", {}).get(horizon, {})
        horizon_comparison[horizon] = {
            "same_model": a.get("model_name") == b.get("model_name"),
            "same_features": a.get("numeric_feature_columns") == b.get("numeric_feature_columns"),
            "same_threshold": a.get("decision_threshold") == b.get("decision_threshold"),
            "artifact_sha256_equal": a.get("artifact_sha256") == b.get("artifact_sha256"),
        }
    return {
        "run_a": run_a,
        "run_b": run_b,
        "specification_equal": not mismatches,
        "mismatches": mismatches,
        "horizons": horizon_comparison,
        "reproducible_artifacts": all(
            result["artifact_sha256_equal"] for result in horizon_comparison.values()
        ),
    }


def main() -> None:
    run_training_pipeline()


if __name__ == "__main__":
    main()
