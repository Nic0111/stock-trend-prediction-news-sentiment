from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
)

from pipeline.model_artifacts import CalibratedStockModel, probability_to_logit
from pipeline.train_stock_models import (
    build_purged_date_splits,
    load_and_combine_model_input_data,
    prepare_horizon_frame,
    split_development_and_test,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "models" / "model-registry.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "model-performance" / "model-performance.json"
COMPANY_NAMES = {
    "1023.KL": "CIMB Group Holdings Berhad",
    "1155.KL": "Malayan Banking Berhad",
    "1295.KL": "Public Bank Berhad",
    "3816.KL": "MISC Berhad",
    "4677.KL": "YTL Corporation Berhad",
    "5183.KL": "PETRONAS Chemicals Group Berhad",
    "5225.KL": "IHH Healthcare Berhad",
    "5347.KL": "Tenaga Nasional Berhad",
    "6947.KL": "CelcomDigi Berhad",
    "8869.KL": "Press Metal Aluminium Holdings Berhad",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _curve_points(x: np.ndarray, y: np.ndarray, *, x_key: str, y_key: str) -> list[dict[str, float]]:
    pairs = list(zip(x, y))
    if len(pairs) > 101:
        indexes = np.linspace(0, len(pairs) - 1, 101, dtype=int)
        pairs = [pairs[index] for index in sorted(set(indexes.tolist()))]
    return [{x_key: float(a), y_key: float(b)} for a, b in pairs]


def _calibration_points(y_true: np.ndarray, probability: np.ndarray) -> list[dict[str, float]]:
    observed, predicted = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    return _curve_points(predicted, observed, x_key="predicted", y_key="observed")


def _candidate_family_summary(bundle: Path, horizon: int) -> list[dict[str, Any]]:
    rows = [row for row in _read_csv(bundle / "stock-model-candidates.csv") if int(row["horizon_days"]) == horizon]
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("model_name", "Unavailable")
        if name not in best or float(row.get("cv_balanced_accuracy") or -1) > float(best[name].get("cv_balanced_accuracy") or -1):
            best[name] = row
    return sorted(
        [
            {
                "model_name": name,
                "balanced_accuracy": _float(row.get("cv_balanced_accuracy")),
                "macro_f1": _float(row.get("cv_macro_f1")),
                "mcc": _float(row.get("cv_mcc")),
                "refinement_round": int(row.get("refinement_round") or 0),
            }
            for name, row in best.items()
        ],
        key=lambda item: item["balanced_accuracy"] or -1,
        reverse=True,
    )


def _hyperparameter_trials(bundle: Path, horizon: int) -> list[dict[str, Any]]:
    rows = [
        row for row in _read_csv(bundle / "stock-model-candidates.csv")
        if int(row["horizon_days"]) == horizon
        and row.get("selection_stage", "seed-42-search") == "seed-42-search"
        and str(row.get("fixed_parameters", "False")).casefold() != "true"
    ]
    return [
        {
            "model_name": row.get("model_name") or "Unavailable",
            "candidate_index": int(row.get("candidate_index") or 0),
            "selected": str(row.get("selected_configuration", "False")).casefold() == "true",
            "balanced_accuracy": _float(row.get("cv_balanced_accuracy")),
            "balanced_accuracy_std": _float(row.get("cv_balanced_accuracy_std")),
            "macro_f1": _float(row.get("cv_macro_f1")),
            "mcc": _float(row.get("cv_mcc")),
            "parameters": row.get("parameters") or "{}",
            "search_strategy": row.get("search_strategy") or "unrecorded",
            "search_budget": int(row.get("search_budget") or 0),
        }
        for row in rows
    ]


def _seed_stability(bundle: Path, horizon: int, model_name: str) -> list[dict[str, Any]]:
    rows = [
        row for row in _read_csv(bundle / "stock-model-candidates.csv")
        if int(row["horizon_days"]) == horizon
        and row.get("model_name") == model_name
        and row.get("selection_stage") == "official-fixed-manifest"
    ]
    return sorted(
        [
            {
                "seed": int(row.get("seed") or 0),
                "balanced_accuracy": _float(row.get("cv_balanced_accuracy")),
                "macro_f1": _float(row.get("cv_macro_f1")),
                "mcc": _float(row.get("cv_mcc")),
            }
            for row in rows
        ],
        key=lambda row: row["seed"],
    )


def _threshold_curve(target: np.ndarray, probability: np.ndarray, selected: float) -> list[dict[str, Any]]:
    thresholds = np.unique(np.concatenate([np.linspace(0.30, 0.70, 41), np.asarray([selected])]))
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)
        output.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
                "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
                "up_recall": float(recall_score(target, prediction, pos_label=1, zero_division=0)),
                "down_recall": float(recall_score(target, prediction, pos_label=0, zero_division=0)),
                "selected": bool(np.isclose(threshold, selected)),
            }
        )
    return output


def _probability_distribution(target: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, 11)
    output: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        output.append(
            {
                "bin": f"{lower:.1f}–{upper:.1f}",
                "midpoint": float((lower + upper) / 2),
                "actual_down": int(np.sum(mask & (target == 0))),
                "actual_up": int(np.sum(mask & (target == 1))),
            }
        )
    return output


def _permutation_importance(
    model: CalibratedStockModel,
    test: pd.DataFrame,
    target: str,
) -> list[dict[str, Any]]:
    selected = test[list(model.input_columns)].copy()
    actual = test[target].astype(int).to_numpy()
    baseline = balanced_accuracy_score(actual, model.predict(selected))
    rng = np.random.default_rng(42)
    importance_rows: list[tuple[str, float, float]] = []
    for column in model.input_columns:
        scores: list[float] = []
        original = selected[column].to_numpy(copy=True)
        for _ in range(5):
            permuted = selected.copy()
            permuted[column] = rng.permutation(original)
            scores.append(baseline - balanced_accuracy_score(actual, model.predict(permuted)))
        importance_rows.append((column, float(np.mean(scores)), float(np.std(scores))))
    rows = sorted(
        importance_rows,
        key=lambda row: row[1],
        reverse=True,
    )
    return [
        {"feature": str(name), "importance_mean": float(mean), "importance_std": float(std)}
        for name, mean, std in rows[:15]
    ]


def _per_company_summary(bundle: Path, horizon: int) -> list[dict[str, Any]]:
    rows = [row for row in _read_csv(bundle / "stock-model-per-ticker-metrics.csv") if int(row["horizon_days"]) == horizon]
    return [
        {
            "ticker": row.get("ticker"),
            "company_name": COMPANY_NAMES.get(str(row.get("ticker")), str(row.get("ticker"))),
            "rows": int(row.get("rows") or 0),
            "status": row.get("status") or "unavailable",
            "balanced_accuracy": _float(row.get("balanced_accuracy")),
            "macro_f1": _float(row.get("macro_f1")),
            "mcc": _float(row.get("mcc")),
            "roc_auc": _float(row.get("roc_auc")),
        }
        for row in rows
    ]


def _learning_curve(
    model: CalibratedStockModel,
    development: pd.DataFrame,
    horizon: int,
) -> list[dict[str, Any]]:
    target = f"Target_{horizon}D"
    target_date = f"Target_Date_{horizon}D"
    splits = build_purged_date_splits(development, target_date)
    fractions = (0.2, 0.4, 0.6, 0.8, 1.0)
    output: list[dict[str, Any]] = []
    for fraction in fractions:
        train_scores: list[float] = []
        validation_scores: list[float] = []
        sample_sizes: list[int] = []
        for train_indexes, validation_indexes in splits:
            fold_train = development.iloc[train_indexes].copy()
            validation = development.iloc[validation_indexes].copy()
            dates = np.array(sorted(fold_train["Date"].unique()))
            selected_dates = dates[: max(2, int(np.ceil(len(dates) * fraction)))]
            subset = fold_train[fold_train["Date"].isin(selected_dates)].copy()
            calibration_start_index = max(1, int(len(selected_dates) * 0.8))
            calibration_dates = selected_dates[calibration_start_index:]
            if not len(calibration_dates):
                continue
            calibration_start = pd.Timestamp(calibration_dates[0])
            base = subset[(subset["Date"] < calibration_start) & (subset[target_date] < calibration_start)].copy()
            calibration = subset[subset["Date"].isin(calibration_dates)].copy()
            if base[target].nunique() < 2 or calibration[target].nunique() < 2 or validation[target].nunique() < 2:
                continue
            estimator = clone(model.estimator)
            estimator.fit(base[list(model.input_columns)], base[target])
            raw_calibration = estimator.predict_proba(calibration[list(model.input_columns)])[:, 1]
            calibrator = clone(model.calibrator)
            calibrator.fit(probability_to_logit(raw_calibration), calibration[target])

            def predict(frame: pd.DataFrame) -> np.ndarray:
                raw = estimator.predict_proba(frame[list(model.input_columns)])[:, 1]
                probability = calibrator.predict_proba(probability_to_logit(raw))[:, 1]
                return (probability >= model.threshold).astype(int)

            train_scores.append(float(balanced_accuracy_score(base[target], predict(base))))
            validation_scores.append(float(balanced_accuracy_score(validation[target], predict(validation))))
            sample_sizes.append(len(subset))
        if validation_scores:
            output.append(
                {
                    "fraction": fraction,
                    "mean_training_rows": int(round(float(np.mean(sample_sizes)))),
                    "training_mean": float(np.mean(train_scores)),
                    "training_std": float(np.std(train_scores)),
                    "validation_mean": float(np.mean(validation_scores)),
                    "validation_std": float(np.std(validation_scores)),
                    "folds": len(validation_scores),
                }
            )
    return output


def generate(
    output_path: Path = OUTPUT_PATH,
    *,
    include_learning_curves: bool = True,
    registry_path: Path = REGISTRY_PATH,
    include_permutation_importance: bool = True,
) -> Path:
    registry = _read_json(registry_path)
    bundle = (PROJECT_ROOT / str(registry["report_bundle_path"])).resolve()
    combined = load_and_combine_model_input_data()
    horizons: list[dict[str, Any]] = []
    for horizon_text, record in sorted(registry["horizons"].items(), key=lambda item: int(item[0])):
        horizon = int(horizon_text)
        artifact_path = (PROJECT_ROOT / str(record["artifact_path"])).resolve()
        model: CalibratedStockModel = joblib.load(artifact_path)
        frame = prepare_horizon_frame(combined, horizon, model.input_columns)
        development, test = split_development_and_test(frame, horizon)
        target = f"Target_{horizon}D"
        probability = model.predict_proba(test)[:, 1]
        prediction = (probability >= model.threshold).astype(int)
        target_values = test[target].astype(int).to_numpy()
        matrix = confusion_matrix(test[target], prediction, labels=[0, 1])
        false_positive_rate, true_positive_rate, _ = roc_curve(test[target], probability)
        precision, recall, _ = precision_recall_curve(test[target], probability)
        horizons.append(
            {
                "horizon_days": horizon,
                "learning_curve": _learning_curve(model, development, horizon) if include_learning_curves else [],
                "confusion_matrix": {
                    "labels": ["Down", "Up"],
                    "values": matrix.astype(int).tolist(),
                    "normalized": (matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)).tolist(),
                },
                "calibration_curve": _calibration_points(test[target].to_numpy(), probability),
                "roc_curve": _curve_points(false_positive_rate, true_positive_rate, x_key="false_positive_rate", y_key="true_positive_rate"),
                "precision_recall_curve": _curve_points(recall, precision, x_key="recall", y_key="precision"),
                "threshold_curve": _threshold_curve(target_values, probability, float(model.threshold)),
                "probability_distribution": _probability_distribution(target_values, probability),
                "permutation_importance": (
                    _permutation_importance(model, test, target)
                    if include_permutation_importance else []
                ),
                "candidate_families": _candidate_family_summary(bundle, horizon),
                "hyperparameter_trials": _hyperparameter_trials(bundle, horizon),
                "seed_stability": _seed_stability(bundle, horizon, str(record.get("model_name") or model.model_name)),
                "per_company": _per_company_summary(bundle, horizon),
            }
        )
    payload = {
        "schema_version": "stocksense-model-performance-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stock_run_id": registry.get("run_id"),
        "registry_path": str(registry_path),
        "search_profile": registry.get("search_profile"),
        "methodology": {
            "learning_curve": "Purged chronological five-fold development learning curve. Each fold uses an expanding earliest-date subset, an internal chronological calibration tail and the frozen champion threshold.",
            "holdout_curves": "Calibration, ROC, precision-recall and confusion matrices use the locked 2025 retrospective test only.",
        },
        "horizons": horizons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate read-only presentation diagnostics for the active stock registry.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--skip-learning-curves", action="store_true")
    parser.add_argument("--skip-permutation-importance", action="store_true")
    args = parser.parse_args()
    print(
        generate(
            args.output,
            include_learning_curves=not args.skip_learning_curves,
            registry_path=args.registry,
            include_permutation_importance=not args.skip_permutation_importance,
        )
    )


if __name__ == "__main__":
    main()
