"""Matched, point-in-time stock/news ablation for presentation and governance.

This module is diagnostics-only.  It never writes a production registry and
never promotes either a stock or sentiment artifact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fusion_model import (
    DEFAULT_FUSION_DATASET,
    FUSION_FEATURE_NAMES,
    FUSION_SUPPORTED_HORIZONS,
    NEWS_FUSION_FEATURES,
    dataset_manifest_hash,
    load_fusion_dataset,
    prepare_fusion_features,
)
from pipeline.fusion_training import (
    C_GRID,
    _eligible_rows,
    _fit_calibrator,
    _metrics_with_variable_threshold,
    _select_threshold,
    classification_metrics,
    expanding_purged_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "results" / "fusion" / "fusion-ablation.json"
NEWS_ONLY_FEATURES = tuple(
    name for name in FUSION_FEATURE_NAMES
    if name in NEWS_FUSION_FEATURES and name != "stock_news_agreement"
)
NEWS_COLUMNS_FOR_PLACEBO = tuple(name for name in FUSION_FEATURE_NAMES if name in NEWS_FUSION_FEATURES)


def _model(features: Sequence[str], C: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    solver="lbfgs",
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _select_c(development: pd.DataFrame, horizon: int, features: Sequence[str]) -> tuple[float, list[dict[str, Any]]]:
    splits = expanding_purged_folds(development, horizon)
    trials: list[dict[str, Any]] = []
    for C in C_GRID:
        probabilities: list[float] = []
        targets: list[int] = []
        fold_scores: list[float] = []
        for train_index, validation_index in splits:
            train = development.iloc[train_index]
            validation = development.iloc[validation_index]
            estimator = _model(features, C)
            estimator.fit(prepare_fusion_features(train, features), train["target_up"].astype(int))
            probability = estimator.predict_proba(prepare_fusion_features(validation, features))[:, 1]
            target = validation["target_up"].astype(int).to_numpy()
            metrics = classification_metrics(target, probability, 0.5)
            probabilities.extend(probability.tolist())
            targets.extend(target.tolist())
            fold_scores.append(metrics["balanced_accuracy"])
        metrics = classification_metrics(targets, probabilities, 0.5)
        trials.append({"C": C, "metrics": metrics, "fold_balanced_accuracy": fold_scores})
    selected = max(
        trials,
        key=lambda row: (
            row["metrics"]["balanced_accuracy"],
            row["metrics"]["mcc"],
            row["metrics"]["macro_f1"],
            -row["C"],
        ),
    )
    return float(selected["C"]), trials


def _fit_arm(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    horizon: int,
    features: Sequence[str],
) -> dict[str, Any]:
    C, trials = _select_c(development, horizon, features)
    estimator = _model(features, C)
    estimator.fit(prepare_fusion_features(development, features), development["target_up"].astype(int))
    calibration_raw = estimator.predict_proba(prepare_fusion_features(calibration, features))[:, 1]
    calibrator = _fit_calibrator(calibration_raw, calibration["target_up"])
    calibration_probability = calibrator.predict_proba(calibration_raw.reshape(-1, 1))[:, 1]
    threshold = _select_threshold(calibration["target_up"], calibration_probability)
    benchmark_raw = estimator.predict_proba(prepare_fusion_features(benchmark, features))[:, 1]
    probability = calibrator.predict_proba(benchmark_raw.reshape(-1, 1))[:, 1]
    target = benchmark["target_up"].astype(int).to_numpy()
    return {
        "probability": probability,
        "prediction": (probability >= threshold).astype(int),
        "threshold": threshold,
        "selected_C": C,
        "development_trials": trials,
        "metrics": classification_metrics(target, probability, threshold),
    }


def _shuffled_news(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    shuffled = frame.copy()
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(len(shuffled))
    shuffled.loc[:, list(NEWS_COLUMNS_FOR_PLACEBO)] = (
        shuffled.loc[:, list(NEWS_COLUMNS_FOR_PLACEBO)].iloc[permutation].to_numpy()
    )
    return shuffled


def _date_block_bootstrap(
    benchmark: pd.DataFrame,
    target: np.ndarray,
    challenger_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    *,
    iterations: int = 2_000,
) -> dict[str, float]:
    dates = pd.to_datetime(benchmark["observation_date"]).dt.normalize().to_numpy()
    unique_dates = np.unique(dates)
    generator = np.random.default_rng(42)
    gains: list[float] = []
    for _ in range(iterations):
        sampled_dates = generator.choice(unique_dates, size=len(unique_dates), replace=True)
        sampled_indexes = np.concatenate([np.flatnonzero(dates == date) for date in sampled_dates])
        if np.unique(target[sampled_indexes]).size < 2:
            continue
        challenger = float(
            ((challenger_prediction[sampled_indexes][target[sampled_indexes] == 0] == 0).mean()
             + (challenger_prediction[sampled_indexes][target[sampled_indexes] == 1] == 1).mean()) / 2
        )
        baseline = float(
            ((baseline_prediction[sampled_indexes][target[sampled_indexes] == 0] == 0).mean()
             + (baseline_prediction[sampled_indexes][target[sampled_indexes] == 1] == 1).mean()) / 2
        )
        gains.append(float(challenger - baseline))
    return {
        "lower": float(np.quantile(gains, 0.025)) if gains else -1.0,
        "median": float(np.quantile(gains, 0.5)) if gains else -1.0,
        "upper": float(np.quantile(gains, 0.975)) if gains else -1.0,
        "iterations": len(gains),
        "block": "observation_date",
    }


def _paired_diagnostics(
    benchmark: pd.DataFrame,
    target: np.ndarray,
    stock_prediction: np.ndarray,
    fusion_prediction: np.ndarray,
) -> dict[str, Any]:
    stock_correct = stock_prediction == target
    fusion_correct = fusion_prediction == target
    stock_wrong_fusion_right = int((~stock_correct & fusion_correct).sum())
    stock_right_fusion_wrong = int((stock_correct & ~fusion_correct).sum())
    discordant = stock_wrong_fusion_right + stock_right_fusion_wrong
    p_value = (
        float(binomtest(min(stock_wrong_fusion_right, stock_right_fusion_wrong), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    per_company: dict[str, Any] = {}
    for ticker, indexes in benchmark.groupby("ticker", sort=True).indices.items():
        index = np.asarray(indexes, dtype=int)
        if np.unique(target[index]).size < 2:
            continue
        stock_score = classification_metrics(target[index], stock_prediction[index], 0.5)["balanced_accuracy"]
        fusion_score = classification_metrics(target[index], fusion_prediction[index], 0.5)["balanced_accuracy"]
        per_company[str(ticker)] = {
            "rows": int(len(index)),
            "stock_balanced_accuracy": stock_score,
            "fusion_balanced_accuracy": fusion_score,
            "gain": float(fusion_score - stock_score),
        }
    return {
        "error_transitions": {
            "stock_wrong_fusion_right": stock_wrong_fusion_right,
            "stock_right_fusion_wrong": stock_right_fusion_wrong,
            "both_correct": int((stock_correct & fusion_correct).sum()),
            "both_wrong": int((~stock_correct & ~fusion_correct).sum()),
        },
        "mcnemar_exact_p_value": p_value,
        "per_company": per_company,
    }


def run_fusion_ablation(
    dataset_path: Path = DEFAULT_FUSION_DATASET,
    *,
    output_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    frame = load_fusion_dataset(dataset_path)
    report: dict[str, Any] = {
        "schema_version": "stocksense-fusion-ablation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_manifest_hash(dataset_path),
        "periods": {"development": "2019-2023", "calibration": "2024", "benchmark": "2025"},
        "arms": ["stock_only", "news_only", "stock_plus_news", "stock_plus_shuffled_news"],
        "horizons": {},
    }
    for horizon in FUSION_SUPPORTED_HORIZONS:
        eligible = _eligible_rows(frame, horizon)
        years = pd.to_datetime(eligible["observation_date"]).dt.year
        development = eligible.loc[years.between(2019, 2023)].reset_index(drop=True)
        calibration = eligible.loc[years.eq(2024)].reset_index(drop=True)
        benchmark = eligible.loc[years.eq(2025)].reset_index(drop=True)
        if any(part.empty for part in (development, calibration, benchmark)):
            raise ValueError(f"Horizon {horizon} lacks a matched development/calibration/benchmark period")

        target = benchmark["target_up"].astype(int).to_numpy()
        stock_probability = np.clip(benchmark["stock_probability_up"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
        stock_threshold = benchmark.get("stock_decision_threshold", pd.Series(0.5, index=benchmark.index)).astype(float).fillna(0.5).to_numpy()
        stock_prediction = (stock_probability >= stock_threshold).astype(int)
        stock_metrics = _metrics_with_variable_threshold(
            target, stock_probability, stock_threshold
        )

        news = _fit_arm(development, calibration, benchmark, horizon, NEWS_ONLY_FEATURES)
        fusion = _fit_arm(development, calibration, benchmark, horizon, FUSION_FEATURE_NAMES)
        shuffled = _fit_arm(
            _shuffled_news(development, 4200 + horizon),
            _shuffled_news(calibration, 4300 + horizon),
            _shuffled_news(benchmark, 4400 + horizon),
            horizon,
            FUSION_FEATURE_NAMES,
        )
        paired = _paired_diagnostics(benchmark, target, stock_prediction, fusion["prediction"])
        bootstrap = _date_block_bootstrap(benchmark, target, fusion["prediction"], stock_prediction)
        gain = float(fusion["metrics"]["balanced_accuracy"] - stock_metrics["balanced_accuracy"])
        placebo_gain = float(fusion["metrics"]["balanced_accuracy"] - shuffled["metrics"]["balanced_accuracy"])
        company_losses = [value["gain"] for value in paired["per_company"].values()]
        checks = {
            "balanced_accuracy_gain_at_least_0_02": gain >= 0.02,
            "block_bootstrap_lower_above_zero": bootstrap["lower"] > 0.0,
            "mcnemar_p_at_most_0_05": paired["mcnemar_exact_p_value"] <= 0.05,
            "macro_f1_not_worse": fusion["metrics"]["macro_f1"] >= stock_metrics["macro_f1"],
            "ece_not_worse_by_more_than_0_01": fusion["metrics"]["ece"] <= stock_metrics["ece"] + 0.01,
            "no_company_loss_below_minus_0_05": not company_losses or min(company_losses) >= -0.05,
            "beats_shuffled_news_by_0_01": placebo_gain >= 0.01,
        }
        report["horizons"][str(horizon)] = {
            "matched_benchmark_rows": int(len(benchmark)),
            "ticker_count": int(benchmark["ticker"].nunique()),
            "arms": {
                "stock_only": {"metrics": stock_metrics, "threshold": "row-specific frozen stock threshold"},
                "news_only": {key: value for key, value in news.items() if key not in {"probability", "prediction"}},
                "stock_plus_news": {key: value for key, value in fusion.items() if key not in {"probability", "prediction"}},
                "stock_plus_shuffled_news": {key: value for key, value in shuffled.items() if key not in {"probability", "prediction"}},
            },
            "stock_plus_news_gain": gain,
            "actual_minus_placebo_gain": placebo_gain,
            "block_bootstrap_gain_ci": bootstrap,
            **paired,
            "acceptance_checks": checks,
            "accepted": all(checks.values()),
            "teaching_conclusion": (
                "News provides statistically supported incremental directional evidence."
                if all(checks.values())
                else "The comparison is educational evidence only; incremental news benefit is not established."
            ),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_fusion_ablation()
