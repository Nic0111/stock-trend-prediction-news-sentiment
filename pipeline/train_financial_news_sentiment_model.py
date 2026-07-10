from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hashlib
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "preprocessed-financial-news-sentiment-dataset"
    / "financial-news-sentiment-preprocessed.csv"
)
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_FILE = MODELS_DIR / "financial-news-sentiment-model.joblib"
METADATA_FILE = MODELS_DIR / "financial-news-sentiment-model-metadata.json"
EVALUATION_FILE = RESULTS_DIR / "financial-news-sentiment-evaluation.csv"
REPORT_FILE = RESULTS_DIR / "classification-reports" / "financial-news-sentiment-report.txt"
CONFUSION_FILE = RESULTS_DIR / "confusion-matrices" / "financial-news-sentiment-confusion-matrix.csv"

RANDOM_STATE = 42
LABEL_ORDER = ["negative", "neutral", "positive"]


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sentiment_training_data() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessed sentiment dataset not found: {DATASET_FILE}. "
            "Run `python main.py preprocess_financial_news_sentiment` first."
        )
    df = pd.read_csv(DATASET_FILE)
    missing = [column for column in ["Sentiment", "Sentence"] if column not in df.columns]
    if missing:
        raise ValueError(f"Sentiment dataset is missing columns: {missing}")
    df = df[["Sentiment", "Sentence"]].copy()
    df["Sentiment"] = df["Sentiment"].astype("string").str.strip().str.lower()
    df["Sentence"] = df["Sentence"].astype("string").str.strip()
    df = df[
        df["Sentiment"].isin(LABEL_ORDER)
        & df["Sentence"].notna()
        & df["Sentence"].ne("")
    ]
    return df.drop_duplicates(subset=["Sentence"], keep="first").reset_index(drop=True)


def build_sentiment_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def sentiment_parameter_space() -> Dict[str, list[Any]]:
    return {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 5],
        "tfidf__max_features": [10_000, 20_000, None],
        "tfidf__sublinear_tf": [True, False],
        "classifier__C": [0.1, 1.0, 10.0],
    }


def _atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_joblib(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_sentiment_training_pipeline(search_profile: str = "full") -> Dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = load_sentiment_training_data()
    development, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["Sentiment"],
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=build_sentiment_pipeline(),
        param_distributions=sentiment_parameter_space(),
        n_iter=30 if search_profile == "full" else 2,
        scoring="f1_macro",
        refit=True,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(development["Sentence"], development["Sentiment"])

    predictions = search.best_estimator_.predict(test["Sentence"])
    report_dict = classification_report(
        test["Sentiment"], predictions, labels=LABEL_ORDER, zero_division=0, output_dict=True
    )
    report_text = classification_report(
        test["Sentiment"], predictions, labels=LABEL_ORDER, zero_division=0
    )
    macro_f1 = float(f1_score(test["Sentiment"], predictions, average="macro", zero_division=0))
    balanced_accuracy = float(balanced_accuracy_score(test["Sentiment"], predictions))

    majority_label = development["Sentiment"].value_counts().idxmax()
    majority_predictions = np.full(len(test), majority_label, dtype=object)
    majority_macro_f1 = float(
        f1_score(test["Sentiment"], majority_predictions, average="macro", zero_division=0)
    )
    per_class_recall_passed = all(
        float(report_dict[label]["recall"]) >= 0.40 for label in LABEL_ORDER
    )
    threshold_passed = bool(
        macro_f1 >= majority_macro_f1 + 0.10 and per_class_recall_passed
    )

    evaluation = {
        "development_rows": len(development),
        "test_rows": len(test),
        "cv_macro_f1": float(search.best_score_),
        "test_balanced_accuracy": balanced_accuracy,
        "test_macro_f1": macro_f1,
        "majority_macro_f1": majority_macro_f1,
        "per_class_recall_passed": per_class_recall_passed,
        "threshold_passed": threshold_passed,
        "status": "accepted" if threshold_passed else "experimental",
    }
    for label in LABEL_ORDER:
        evaluation[f"{label}_precision"] = float(report_dict[label]["precision"])
        evaluation[f"{label}_recall"] = float(report_dict[label]["recall"])
        evaluation[f"{label}_f1"] = float(report_dict[label]["f1-score"])

    pd.DataFrame([evaluation]).to_csv(EVALUATION_FILE, index=False)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    matrix = confusion_matrix(test["Sentiment"], predictions, labels=LABEL_ORDER)
    pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(CONFUSION_FILE)

    deployment_model = build_sentiment_pipeline().set_params(**search.best_params_)
    deployment_model.fit(df["Sentence"], df["Sentiment"])
    _atomic_joblib(deployment_model, MODEL_FILE)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_sha256": _dataset_hash(DATASET_FILE),
        "dataset_rows": len(df),
        "class_order": deployment_model.named_steps["classifier"].classes_.tolist(),
        "best_parameters": search.best_params_,
        "evaluation": evaluation,
        "artifact_path": _display_path(MODEL_FILE),
    }
    _atomic_json(metadata, METADATA_FILE)
    print(
        f"Sentiment model saved to {MODEL_FILE}; macro-F1={macro_f1:.4f}; "
        f"status={evaluation['status']}"
    )
    return metadata


def main() -> None:
    run_sentiment_training_pipeline()


if __name__ == "__main__":
    main()
