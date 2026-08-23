"""Leakage-safe, capacity-matched stock/news fusion evaluation.

This module is deliberately isolated from the live runtime.  It writes a staged
v2 registry and retrospective report; the active fusion registry is never
changed here.  Promotion is only possible when the report's all-horizon gate is
true and every recorded artifact identity still matches.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.fusion_model import DEFAULT_FUSION_DATASET, dataset_manifest_hash, load_fusion_dataset
from pipeline.fusion_training import _expected_calibration_error, _select_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "config" / "fusion_evaluation_v2.json"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "fusion" / "fusion-evaluation-v2.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "models" / "fusion" / "staged-fusion-v2-registry.json"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "fusion" / "staged-v2"
ACTIVE_STOCK_REGISTRY = PROJECT_ROOT / "models" / "model-registry.json"
SENTIMENT_RUNTIME_CONFIG = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"

STOCK_FEATURES = (
    "stock_logit",
    "stock_threshold_margin",
    "stock_evidence_strength",
    "stock_uncertainty",
)
NEWS_FEATURES = (
    "gated_sentiment",
    "gated_positive_share",
    "gated_neutral_share",
    "gated_negative_share",
    "gated_sentiment_dispersion",
    "gated_conflict",
    "gated_log_article_count",
    "gated_log_event_count",
    "gated_log_source_count",
    "gated_volume_surprise",
    "gated_sentiment_surprise",
    "gated_newest_age",
    "gated_mean_age",
    "gated_stock_news_agreement",
    "reliability_x_stock_uncertainty",
    "news_available",
    "news_reliability",
)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _project_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_policy(path: Path = DEFAULT_POLICY) -> tuple[dict[str, Any], str]:
    policy = _read_json(path)
    if policy.get("schema_version") != "fusion-evaluation-v2":
        raise ValueError(f"Invalid fusion v2 policy: {path}")
    return policy, _sha256_payload(policy)


def _safe_number(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _row_series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in frame:
        return _safe_number(frame[name], default)
    return pd.Series(default, index=frame.index, dtype=float)


def _enhance_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time occurrence, tone, reliability and surprise features."""

    result = frame.copy()
    result["observation_date"] = pd.to_datetime(result["observation_date"], errors="coerce").dt.normalize()
    result["target_date"] = pd.to_datetime(result.get("target_date"), errors="coerce").dt.normalize()
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    result["target_up"] = pd.to_numeric(result["target_up"], errors="coerce")
    probability = _row_series(result, "stock_probability_up", 0.5).clip(1e-6, 1 - 1e-6)
    result["stock_logit"] = np.log(probability / (1.0 - probability))
    result["stock_uncertainty"] = 1.0 - (2.0 * (probability - 0.5).abs()).clip(0.0, 1.0)
    result["stock_threshold_margin"] = _row_series(result, "stock_threshold_margin")
    result["stock_evidence_strength"] = _row_series(result, "stock_evidence_strength")

    available = _row_series(result, "news_available").clip(0.0, 1.0)
    article_count = _row_series(result, "news_usable_article_count")
    event_count = _row_series(result, "news_independent_event_count")
    source_count = _row_series(result, "news_unique_source_count")
    confidence = _row_series(result, "news_mean_confidence").clip(0.0, 1.0)
    coverage = _row_series(result, "news_coverage_score").clip(0.0, 1.0)
    newest_age = _row_series(result, "news_newest_age_days", 999.0).clip(lower=0.0)
    mean_age = _row_series(result, "news_mean_age_days", 999.0).clip(lower=0.0)
    half_life = _row_series(result, "news_half_life_days", 1.0).clip(lower=0.25)
    freshness = np.exp(-math.log(2.0) * newest_age / half_life)
    diversity = np.minimum(1.0, source_count / 3.0)
    event_depth = np.minimum(1.0, event_count / 3.0)
    reliability = available * coverage * confidence * freshness * np.sqrt(diversity * event_depth)
    result["news_reliability"] = reliability.clip(0.0, 1.0)

    positive = _row_series(result, "news_positive_share").clip(0.0, 1.0)
    neutral = _row_series(result, "news_neutral_share").clip(0.0, 1.0)
    negative = _row_series(result, "news_negative_share").clip(0.0, 1.0)
    shares = np.column_stack([negative, neutral, positive])
    entropy = -(shares * np.log(np.clip(shares, 1e-12, 1.0))).sum(axis=1) / math.log(3.0)
    result["sentiment_dispersion"] = entropy
    result["conflicting_news"] = ((positive > 0.0) & (negative > 0.0)).astype(float)
    result["log_article_count"] = np.log1p(article_count)
    result["log_event_count"] = np.log1p(event_count)
    result["log_source_count"] = np.log1p(source_count)
    result["news_weighted_sentiment"] = _row_series(result, "news_weighted_sentiment")
    result["stock_news_agreement"] = _row_series(result, "stock_news_agreement", 0.5)
    result["news_newest_age_days"] = newest_age.where(available.gt(0), 0.0)
    result["news_mean_age_days"] = mean_age.where(available.gt(0), 0.0)

    variant_keys = ["ticker", "horizon_days"]
    for candidate_key in ("news_window_days", "news_half_life_days"):
        if candidate_key in result:
            variant_keys.append(candidate_key)
    result = result.sort_values([*variant_keys, "observation_date"]).reset_index(
        drop=True
    )
    grouped = result.groupby(variant_keys, sort=False, observed=True, dropna=False)
    trailing_volume = grouped["log_event_count"].transform(
        lambda values: values.shift(1).rolling(252, min_periods=20).mean()
    )
    trailing_tone = grouped["news_weighted_sentiment"].transform(
        lambda values: values.shift(1).rolling(252, min_periods=20).mean()
    )
    result["volume_surprise"] = (result["log_event_count"] - trailing_volume).fillna(0.0)
    result["sentiment_surprise"] = (result["news_weighted_sentiment"] - trailing_tone).fillna(0.0)

    gated_sources = {
        "sentiment": "news_weighted_sentiment",
        "positive_share": "news_positive_share",
        "neutral_share": "news_neutral_share",
        "negative_share": "news_negative_share",
        "sentiment_dispersion": "sentiment_dispersion",
        "conflict": "conflicting_news",
        "log_article_count": "log_article_count",
        "log_event_count": "log_event_count",
        "log_source_count": "log_source_count",
        "volume_surprise": "volume_surprise",
        "sentiment_surprise": "sentiment_surprise",
        "newest_age": "news_newest_age_days",
        "mean_age": "news_mean_age_days",
        "stock_news_agreement": "stock_news_agreement",
    }
    for output_name, source_name in gated_sources.items():
        result[f"gated_{output_name}"] = result["news_reliability"] * _row_series(result, source_name)
    result["reliability_x_stock_uncertainty"] = result["news_reliability"] * result["stock_uncertainty"]
    return result


def _eligible_all_rows(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    subset = frame.loc[pd.to_numeric(frame["horizon_days"], errors="coerce").eq(horizon)].copy()
    subset = subset.dropna(subset=["observation_date", "ticker", "target_up", "stock_probability_up"])
    subset = subset.loc[subset["target_up"].isin([0, 1])]
    return subset.sort_values(["observation_date", "ticker"]).reset_index(drop=True)


def _select_window_variant(
    frame: pd.DataFrame, horizon: int, policy: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lock one lookback/half-life using development rows only."""

    subset = frame.loc[
        pd.to_numeric(frame["horizon_days"], errors="coerce").eq(horizon)
    ].copy()
    combinations = (
        subset[["news_window_days", "news_half_life_days"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["news_window_days", "news_half_life_days"])
    )
    if len(combinations) <= 1:
        return _eligible_all_rows(frame, horizon), {
            "status": "single_runtime_matched_variant",
            "trials": [],
        }
    trials: list[dict[str, Any]] = []
    for combination in combinations.to_dict(orient="records"):
        candidate = subset.loc[
            pd.to_numeric(subset["news_window_days"], errors="coerce").eq(
                float(combination["news_window_days"])
            )
            & pd.to_numeric(subset["news_half_life_days"], errors="coerce").eq(
                float(combination["news_half_life_days"])
            )
        ].copy()
        candidate = _eligible_all_rows(candidate, horizon)
        development = candidate.loc[
            pd.to_datetime(candidate["observation_date"]).dt.year.between(2019, 2023)
            & pd.to_datetime(candidate["target_date"], errors="coerce").lt(
                pd.Timestamp("2024-01-01")
            )
        ].reset_index(drop=True)
        if development.empty:
            continue
        _, tuning, _ = _select_hyperparameters(
            development, (*STOCK_FEATURES, *NEWS_FEATURES), policy
        )
        best = min(
            tuning,
            key=lambda row: (
                row["brier_score"],
                -row["balanced_accuracy"],
                -row["macro_f1"],
            ),
        )
        trials.append(
            {
                "window_days": int(combination["news_window_days"]),
                "half_life_days": float(combination["news_half_life_days"]),
                "development_brier": float(best["brier_score"]),
                "development_balanced_accuracy": float(best["balanced_accuracy"]),
            }
        )
    if not trials:
        raise ValueError(f"No development-eligible news-window variants for {horizon}D")
    selected = min(
        trials,
        key=lambda row: (
            row["development_brier"],
            -row["development_balanced_accuracy"],
            row["window_days"],
            row["half_life_days"],
        ),
    )
    chosen = subset.loc[
        pd.to_numeric(subset["news_window_days"], errors="coerce").eq(
            selected["window_days"]
        )
        & pd.to_numeric(subset["news_half_life_days"], errors="coerce").eq(
            selected["half_life_days"]
        )
    ].copy()
    return _eligible_all_rows(chosen, horizon), {
        "status": "selected_on_2019_2023_development_only",
        "selected": selected,
        "trials": trials,
    }


def _purged_development_splits(frame: pd.DataFrame, fold_count: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(frame["observation_date"]).dt.normalize()
    targets = pd.to_datetime(frame["target_date"], errors="coerce").fillna(dates)
    unique = np.asarray(sorted(dates.unique()))
    blocks = [block for block in np.array_split(unique, fold_count + 1) if len(block)]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks[1:]:
        start = pd.Timestamp(block[0])
        train = np.flatnonzero(((dates < start) & (targets < start)).to_numpy())
        validation = np.flatnonzero(dates.isin(block).to_numpy())
        if len(train) and len(validation):
            splits.append((train, validation))
    if len(splits) < 2:
        raise ValueError("Insufficient target-complete history for purged development folds")
    return splits


def _preprocessor(features: Sequence[str]) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0.0)), ("scale", StandardScaler())])
    try:
        ticker = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        ticker = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([("numeric", numeric, list(features)), ("ticker", ticker, ["ticker"])])


def _model(features: Sequence[str], C: float, l1_ratio: float) -> Pipeline:
    return Pipeline(
        [
            ("features", _preprocessor(features)),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    l1_ratio=float(l1_ratio),
                    solver="saga",
                    class_weight="balanced",
                    max_iter=1500,
                    tol=1e-3,
                    random_state=42,
                ),
            ),
        ]
    )


def _metric_summary(target: Sequence[int], probability: Sequence[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    predicted = (p >= float(threshold)).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    brier = float(brier_score_loss(y, p))
    null_brier = float(brier_score_loss(y, np.full(len(y), y.mean())))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "brier_score": brier,
        "brier_skill": float(1.0 - brier / null_brier) if null_brier > 0 else 0.0,
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "ece": float(_expected_calibration_error(y, p)),
        "down_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        "up_recall": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "threshold": float(threshold),
    }


def _transparent_weighted_diagnostic(
    development: pd.DataFrame,
    benchmark: pd.DataFrame,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Select and report the legacy transparent formula without promoting it."""

    trials: list[dict[str, float]] = []
    development_stock = np.clip(
        development["stock_probability_up"].to_numpy(dtype=float), 1e-6, 1 - 1e-6
    )
    development_logit = np.log(development_stock / (1 - development_stock))
    development_news = (
        development["news_reliability"].to_numpy(dtype=float)
        * development["news_weighted_sentiment"].to_numpy(dtype=float)
    )
    development_target = development["target_up"].astype(int).to_numpy()
    for alpha in policy.get("transparent_news_weights", [0.25, 0.5, 1.0, 2.0]):
        probability = 1.0 / (
            1.0 + np.exp(-(development_logit + float(alpha) * development_news))
        )
        trials.append(
            {
                "alpha": float(alpha),
                "development_brier": float(
                    brier_score_loss(development_target, probability)
                ),
            }
        )
    selected = min(trials, key=lambda row: (row["development_brier"], row["alpha"]))
    stock = np.clip(
        benchmark["stock_probability_up"].to_numpy(dtype=float), 1e-6, 1 - 1e-6
    )
    logit = np.log(stock / (1 - stock))
    news = (
        benchmark["news_reliability"].to_numpy(dtype=float)
        * benchmark["news_weighted_sentiment"].to_numpy(dtype=float)
    )
    probability = 1.0 / (1.0 + np.exp(-(logit + selected["alpha"] * news)))
    unavailable = benchmark["news_available"].to_numpy(dtype=float) <= 0
    probability[unavailable] = stock[unavailable]
    return {
        "metrics": _metric_summary(
            benchmark["target_up"].astype(int), probability, 0.5
        ),
        "selected_alpha": selected["alpha"],
        "development_trials": trials,
        "promotion_candidate": False,
    }


def _select_hyperparameters(
    development: pd.DataFrame,
    features: Sequence[str],
    policy: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, float]]]:
    splits = _purged_development_splits(development)
    trials: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, float]] = []
    for C in policy["elastic_net"]["C"]:
        for ratio in policy["elastic_net"]["l1_ratio"]:
            probabilities: list[float] = []
            targets: list[int] = []
            fold_brier: list[float] = []
            fold_ba: list[float] = []
            for fold, (train_index, validation_index) in enumerate(splits, start=1):
                estimator = _model(features, C, ratio)
                train = development.iloc[train_index]
                validation = development.iloc[validation_index]
                estimator.fit(train[[*features, "ticker"]], train["target_up"].astype(int))
                probability = estimator.predict_proba(validation[[*features, "ticker"]])[:, 1]
                target = validation["target_up"].astype(int).to_numpy()
                probabilities.extend(probability.tolist())
                targets.extend(target.tolist())
                fold_brier.append(float(brier_score_loss(target, probability)))
                fold_ba.append(float(balanced_accuracy_score(target, probability >= 0.5)))
            aggregate = _metric_summary(targets, probabilities, 0.5)
            trials.append(
                {
                    "C": float(C),
                    "l1_ratio": float(ratio),
                    "brier_score": aggregate["brier_score"],
                    "balanced_accuracy": aggregate["balanced_accuracy"],
                    "macro_f1": aggregate["macro_f1"],
                    "fold_brier": fold_brier,
                    "fold_balanced_accuracy": fold_ba,
                }
            )
    selected = min(trials, key=lambda row: (row["brier_score"], -row["balanced_accuracy"], -row["macro_f1"], row["C"], row["l1_ratio"]))
    return {"C": selected["C"], "l1_ratio": selected["l1_ratio"]}, trials, coefficient_rows


@dataclass
class ArmResult:
    estimator: Pipeline
    calibrator: LogisticRegression
    probability: np.ndarray
    prediction: np.ndarray
    threshold: float
    selected_parameters: dict[str, float]
    trials: list[dict[str, Any]]
    metrics: dict[str, Any]


def _fit_arm(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    features: Sequence[str],
    policy: Mapping[str, Any],
) -> ArmResult:
    selected, trials, _ = _select_hyperparameters(development, features, policy)
    estimator = _model(features, **selected)
    estimator.fit(development[[*features, "ticker"]], development["target_up"].astype(int))
    calibration_raw = estimator.predict_proba(calibration[[*features, "ticker"]])[:, 1]
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=42)
    calibration_logit = np.log(
        np.clip(calibration_raw, 1e-6, 1 - 1e-6)
        / np.clip(1 - calibration_raw, 1e-6, 1)
    ).reshape(-1, 1)
    calibrator.fit(calibration_logit, calibration["target_up"].astype(int))
    calibration_probability = calibrator.predict_proba(calibration_logit)[:, 1]
    threshold = float(_select_threshold(calibration["target_up"].astype(int), calibration_probability))
    raw = estimator.predict_proba(benchmark[[*features, "ticker"]])[:, 1]
    logit = np.log(
        np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1)
    ).reshape(-1, 1)
    probability = calibrator.predict_proba(logit)[:, 1]
    target = benchmark["target_up"].astype(int).to_numpy()
    return ArmResult(
        estimator=estimator,
        calibrator=calibrator,
        probability=probability,
        prediction=(probability >= threshold).astype(int),
        threshold=threshold,
        selected_parameters=selected,
        trials=trials,
        metrics=_metric_summary(target, probability, threshold),
    )


def _fit_arm_fixed(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    features: Sequence[str],
    selected: Mapping[str, float],
) -> ArmResult:
    estimator = _model(features, float(selected["C"]), float(selected["l1_ratio"]))
    estimator.fit(development[[*features, "ticker"]], development["target_up"].astype(int))
    calibration_raw = estimator.predict_proba(calibration[[*features, "ticker"]])[:, 1]
    calibration_logit = np.log(
        np.clip(calibration_raw, 1e-6, 1 - 1e-6)
        / np.clip(1 - calibration_raw, 1e-6, 1)
    ).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=42)
    calibrator.fit(calibration_logit, calibration["target_up"].astype(int))
    calibration_probability = calibrator.predict_proba(calibration_logit)[:, 1]
    threshold = float(_select_threshold(calibration["target_up"].astype(int), calibration_probability))
    raw = estimator.predict_proba(benchmark[[*features, "ticker"]])[:, 1]
    logit = np.log(
        np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1)
    ).reshape(-1, 1)
    probability = calibrator.predict_proba(logit)[:, 1]
    target = benchmark["target_up"].astype(int).to_numpy()
    return ArmResult(
        estimator=estimator,
        calibrator=calibrator,
        probability=probability,
        prediction=(probability >= threshold).astype(int),
        threshold=threshold,
        selected_parameters=dict(selected),
        trials=[],
        metrics=_metric_summary(target, probability, threshold),
    )


def _news_coefficient_stability(
    development: pd.DataFrame,
    selected: Mapping[str, float],
) -> dict[str, Any]:
    """Summarise standardized news coefficients across purged development folds."""

    features = (*STOCK_FEATURES, *NEWS_FEATURES)
    rows: list[dict[str, float]] = []
    for fold, (train_index, _) in enumerate(
        _purged_development_splits(development), start=1
    ):
        estimator = _model(
            features, float(selected["C"]), float(selected["l1_ratio"])
        )
        train = development.iloc[train_index]
        estimator.fit(train[[*features, "ticker"]], train["target_up"].astype(int))
        coefficients = estimator.named_steps["classifier"].coef_[0][
            : len(features)
        ]
        rows.append(
            {
                "fold": float(fold),
                **{
                    name: float(coefficients[position])
                    for position, name in enumerate(features)
                    if name in NEWS_FEATURES
                },
            }
        )
    summaries: dict[str, dict[str, float]] = {}
    for feature in NEWS_FEATURES:
        values = np.asarray([row[feature] for row in rows], dtype=float)
        nonzero = np.abs(values) > 1e-8
        if nonzero.any():
            majority_sign = np.sign(np.median(values[nonzero]))
            sign_consistency = float(
                np.mean(np.sign(values[nonzero]) == majority_sign)
            )
        else:
            sign_consistency = 1.0
        summaries[feature] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "nonzero_fold_share": float(nonzero.mean()) if len(values) else 0.0,
            "sign_consistency": sign_consistency,
        }
    return {"folds": len(rows), "features": summaries, "fold_coefficients": rows}


def _probability_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _cap_news_shift(
    stock_probability: np.ndarray,
    fusion_probability: np.ndarray,
    cap: float,
) -> np.ndarray:
    stock_logit = _probability_logit(stock_probability)
    fusion_logit = _probability_logit(fusion_probability)
    result_logit = stock_logit + np.clip(fusion_logit - stock_logit, -float(cap), float(cap))
    return 1.0 / (1.0 + np.exp(-result_logit))


def _select_news_shift_cap(
    development: pd.DataFrame,
    stock_parameters: Mapping[str, float],
    fusion_parameters: Mapping[str, float],
    caps: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    splits = _purged_development_splits(development)
    values: dict[float, list[float]] = {float(cap): [] for cap in caps}
    for train_index, validation_index in splits:
        train = development.iloc[train_index]
        validation = development.iloc[validation_index]
        stock = _model(STOCK_FEATURES, **stock_parameters)
        fusion = _model((*STOCK_FEATURES, *NEWS_FEATURES), **fusion_parameters)
        stock.fit(train[[*STOCK_FEATURES, "ticker"]], train["target_up"].astype(int))
        fusion.fit(train[[*STOCK_FEATURES, *NEWS_FEATURES, "ticker"]], train["target_up"].astype(int))
        stock_probability = stock.predict_proba(validation[[*STOCK_FEATURES, "ticker"]])[:, 1]
        fusion_probability = fusion.predict_proba(validation[[*STOCK_FEATURES, *NEWS_FEATURES, "ticker"]])[:, 1]
        target = validation["target_up"].astype(int).to_numpy()
        unavailable = validation["news_available"].to_numpy(dtype=float) <= 0
        for cap in values:
            capped = _cap_news_shift(stock_probability, fusion_probability, cap)
            capped[unavailable] = stock_probability[unavailable]
            values[cap].append(float(brier_score_loss(target, capped)))
    trials = [
        {"cap": cap, "mean_development_brier": float(np.mean(scores))}
        for cap, scores in sorted(values.items())
    ]
    selected = min(trials, key=lambda row: (row["mean_development_brier"], row["cap"]))
    return float(selected["cap"]), trials


def _calibration_curve(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    edges = np.linspace(0, 1, bins + 1)
    rows: list[dict[str, float | int]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            rows.append({"predicted": float(probability[mask].mean()), "observed": float(target[mask].mean()), "rows": int(mask.sum())})
    return rows


def _calibration_regression(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1)).reshape(-1, 1)
    model = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=42).fit(logit, target)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def _moving_block_indices(dates: np.ndarray, block_size: int, generator: np.random.Generator) -> np.ndarray:
    unique_dates = np.asarray(sorted(np.unique(dates)))
    starts = generator.integers(0, max(1, len(unique_dates) - block_size + 1), size=math.ceil(len(unique_dates) / block_size))
    sampled_dates = np.concatenate([unique_dates[start : start + block_size] for start in starts])[: len(unique_dates)]
    return np.concatenate([np.flatnonzero(dates == value) for value in sampled_dates])


def _block_bootstrap(
    benchmark: pd.DataFrame,
    target: np.ndarray,
    baseline_probability: np.ndarray,
    challenger_probability: np.ndarray,
    baseline_prediction: np.ndarray,
    challenger_prediction: np.ndarray,
    *,
    block_size: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    dates = pd.to_datetime(benchmark["observation_date"]).dt.normalize().to_numpy()
    generator = np.random.default_rng(seed)
    brier_gains: list[float] = []
    ba_gains: list[float] = []
    for _ in range(iterations):
        index = _moving_block_indices(dates, block_size, generator)
        if len(index) == 0 or np.unique(target[index]).size < 2:
            continue
        brier_gains.append(float(brier_score_loss(target[index], baseline_probability[index]) - brier_score_loss(target[index], challenger_probability[index])))
        ba_gains.append(float(balanced_accuracy_score(target[index], challenger_prediction[index]) - balanced_accuracy_score(target[index], baseline_prediction[index])))
    def interval(values: list[float]) -> dict[str, float]:
        return {"lower": float(np.quantile(values, 0.025)), "median": float(np.quantile(values, 0.5)), "upper": float(np.quantile(values, 0.975))} if values else {"lower": -1.0, "median": -1.0, "upper": -1.0}
    return {"block_trading_sessions": block_size, "iterations": len(brier_gains), "brier_improvement": interval(brier_gains), "balanced_accuracy_gain": interval(ba_gains)}


def _permute_news(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy()
    month = pd.to_datetime(result["observation_date"]).dt.to_period("M").astype(str)
    generator = np.random.default_rng(seed)
    for _, index in result.groupby(["ticker", month], sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        if len(positions) > 1:
            shuffled = generator.permutation(positions)
            result.loc[positions, list(NEWS_FEATURES)] = result.loc[shuffled, list(NEWS_FEATURES)].to_numpy()
    return result


def _permutation_placebo(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    policy: Mapping[str, Any],
    actual_brier_improvement: float,
    stock_parameters: Mapping[str, float],
    fusion_parameters: Mapping[str, float],
    news_shift_cap: float,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    values: list[float] = []
    # Refit each placebo using identically grouped news shuffles.  This is
    # intentionally expensive; the CLI exposes an override for smoke tests.
    for iteration in range(iterations):
        shuffled_development = _permute_news(development, seed + iteration * 3)
        shuffled_calibration = _permute_news(calibration, seed + iteration * 3 + 1)
        shuffled_benchmark = _permute_news(benchmark, seed + iteration * 3 + 2)
        stock = _fit_arm_fixed(
            shuffled_development,
            shuffled_calibration,
            shuffled_benchmark,
            STOCK_FEATURES,
            stock_parameters,
        )
        fusion = _fit_arm_fixed(
            shuffled_development,
            shuffled_calibration,
            shuffled_benchmark,
            (*STOCK_FEATURES, *NEWS_FEATURES),
            fusion_parameters,
        )
        target = shuffled_benchmark["target_up"].astype(int).to_numpy()
        unavailable = shuffled_benchmark["news_available"].to_numpy(dtype=float) <= 0
        fusion_probability = _cap_news_shift(
            stock.probability, fusion.probability, news_shift_cap
        )
        fusion_probability[unavailable] = stock.probability[unavailable]
        values.append(float(brier_score_loss(target, stock.probability) - brier_score_loss(target, fusion_probability)))
    percentile = float(np.mean(np.asarray(values) < actual_brier_improvement)) if values else 0.0
    return {"iterations": len(values), "actual_percentile": percentile, "mean": float(np.mean(values)) if values else None, "p95": float(np.quantile(values, 0.95)) if values else None, "distribution": values}


def _clark_west(target: np.ndarray, baseline: np.ndarray, challenger: np.ndarray, dates: pd.Series) -> dict[str, float]:
    adjusted = (target - baseline) ** 2 - ((target - challenger) ** 2 - (baseline - challenger) ** 2)
    by_date = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize(), "value": adjusted}).groupby("date")["value"].mean()
    if len(by_date) < 2 or float(by_date.std(ddof=1)) == 0:
        return {"statistic": 0.0, "one_sided_p_value": 1.0}
    statistic = float(by_date.mean() / (by_date.std(ddof=1) / math.sqrt(len(by_date))))
    return {"statistic": statistic, "one_sided_p_value": float(1.0 - norm.cdf(statistic))}


def _subgroup_metrics(frame: pd.DataFrame, target: np.ndarray, baseline: ArmResult, challenger_probability: np.ndarray, challenger_threshold: float) -> dict[str, Any]:
    challenger_prediction = challenger_probability >= challenger_threshold
    result: dict[str, Any] = {}
    groups = {
        "coverage": frame.get("coverage_confidence", pd.Series("Unavailable", index=frame.index)).fillna("Unavailable").astype(str),
        "agreement": pd.Series(np.where(frame["news_available"].le(0), "Unavailable", np.where(frame["stock_news_agreement"].eq(0.5), "Balanced", np.where(frame["stock_news_agreement"].ge(1), "Agree", "Conflict"))), index=frame.index),
        "news_reliability": pd.cut(
            frame["news_reliability"],
            bins=[-1e-9, 0.0, 0.25, 0.5, 0.75, 1.0],
            labels=["Unavailable", "0-25%", "25-50%", "50-75%", "75-100%"],
            include_lowest=True,
        ).astype(str),
        "stock_confidence": pd.cut(
            (frame["stock_probability_up"] - 0.5).abs() * 2.0,
            bins=[-1e-9, 0.25, 0.5, 0.75, 1.0],
            labels=["0-25%", "25-50%", "50-75%", "75-100%"],
            include_lowest=True,
        ).astype(str),
    }
    for group_name, labels in groups.items():
        rows: dict[str, Any] = {}
        for label, index in labels.groupby(labels).groups.items():
            positions = np.asarray(list(index), dtype=int)
            if len(positions) < 2 or np.unique(target[positions]).size < 2:
                continue
            base_ba = balanced_accuracy_score(target[positions], baseline.prediction[positions])
            fusion_ba = balanced_accuracy_score(target[positions], challenger_prediction[positions])
            rows[str(label)] = {"rows": int(len(positions)), "stock_balanced_accuracy": float(base_ba), "fusion_balanced_accuracy": float(fusion_ba), "gain": float(fusion_ba - base_ba)}
        result[group_name] = rows
    return result


def _paired_details(frame: pd.DataFrame, target: np.ndarray, baseline_prediction: np.ndarray, challenger_prediction: np.ndarray) -> dict[str, Any]:
    base_correct = baseline_prediction == target
    fusion_correct = challenger_prediction == target
    per_company: dict[str, Any] = {}
    for ticker, index in frame.groupby("ticker", sort=True).indices.items():
        positions = np.asarray(index, dtype=int)
        if np.unique(target[positions]).size < 2:
            continue
        base = balanced_accuracy_score(target[positions], baseline_prediction[positions])
        fusion = balanced_accuracy_score(target[positions], challenger_prediction[positions])
        per_company[str(ticker)] = {"rows": int(len(positions)), "stock_balanced_accuracy": float(base), "fusion_balanced_accuracy": float(fusion), "gain": float(fusion - base)}
    return {
        "error_transitions": {
            "news_corrected_stock": int((~base_correct & fusion_correct).sum()),
            "news_harmed_stock": int((base_correct & ~fusion_correct).sum()),
            "both_correct": int((base_correct & fusion_correct).sum()),
            "both_wrong": int((~base_correct & ~fusion_correct).sum()),
        },
        "per_company": per_company,
    }


def _rolling_origin_results(frame: pd.DataFrame, horizon: int, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dates = pd.to_datetime(frame["observation_date"])
    targets = pd.to_datetime(frame["target_date"], errors="coerce")
    for origin in policy["rolling_origins"]:
        calibration_year = int(origin["calibration"])
        evaluation_year = int(origin["evaluation"])
        calibration_start = pd.Timestamp(f"{calibration_year}-01-01")
        development = frame.loc[(dates.dt.year.between(2019, int(origin["train_end"]))) & (targets < calibration_start)].reset_index(drop=True)
        evaluation_start = pd.Timestamp(f"{evaluation_year}-01-01")
        calibration = frame.loc[
            dates.dt.year.eq(calibration_year) & targets.lt(evaluation_start)
        ].reset_index(drop=True)
        evaluation = frame.loc[dates.dt.year.eq(evaluation_year)].reset_index(drop=True)
        if min(len(development), len(calibration), len(evaluation)) == 0:
            continue
        stock = _fit_arm(development, calibration, evaluation, STOCK_FEATURES, policy)
        fusion = _fit_arm(development, calibration, evaluation, (*STOCK_FEATURES, *NEWS_FEATURES), policy)
        unavailable = evaluation["news_available"].to_numpy(dtype=float) <= 0
        selected_cap, _ = _select_news_shift_cap(
            development,
            stock.selected_parameters,
            fusion.selected_parameters,
            policy["news_log_odds_caps"],
        )
        probability = _cap_news_shift(stock.probability, fusion.probability, selected_cap)
        probability[unavailable] = stock.probability[unavailable]
        metrics = _metric_summary(evaluation["target_up"].astype(int), probability, fusion.threshold)
        rows.append({"evaluation_year": evaluation_year, "rows": int(len(evaluation)), "ticker_count": int(evaluation["ticker"].nunique()), "stock_metrics": stock.metrics, "fusion_metrics": metrics, "brier_improvement": float(stock.metrics["brier_score"] - metrics["brier_score"]), "balanced_accuracy_gain": float(metrics["balanced_accuracy"] - stock.metrics["balanced_accuracy"])})
    return rows


def _provenance(frame: pd.DataFrame, dataset_path: Path, policy_hash: str) -> dict[str, Any]:
    active_stock = _read_json(ACTIVE_STOCK_REGISTRY)
    sentiment = _read_json(SENTIMENT_RUNTIME_CONFIG)
    recorded_runs = sorted(set(frame.get("stock_registry_run_id", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_roles = sorted(set(frame.get("prediction_role", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_stock_hashes = sorted(set(frame.get("stock_registry_dataset_sha256", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_feature_hashes = sorted(set(frame.get("stock_feature_schema_sha256", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_sentiment_hashes = sorted(set(frame.get("sentiment_runtime_config_sha256", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_sentiment_variants = sorted(set(frame.get("sentiment_model_variant", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_sentiment_variant_hashes = sorted(set(frame.get("sentiment_model_variant_sha256", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_component_hashes = sorted(set(frame.get("sentiment_component_artifacts_sha256", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_corpus_hashes = sorted(set(frame.get("article_corpus_sha256", pd.Series(dtype=str)).dropna().astype(str)))
    recorded_aggregation_hashes = sorted(set(frame.get("news_aggregation_policy_sha256", pd.Series(dtype=str)).dropna().astype(str)))
    active_run = str(active_stock.get("run_id") or "")
    active_sentiment_hash = (
        hashlib.sha256(SENTIMENT_RUNTIME_CONFIG.read_bytes()).hexdigest()
        if SENTIMENT_RUNTIME_CONFIG.exists()
        else None
    )
    active_variant = str(sentiment.get("active_variant") or "").strip()
    active_variant_payload = dict(
        (sentiment.get("challengers") or {}).get(active_variant) or {}
    )
    active_variant_hash = (
        _sha256_payload(active_variant_payload) if active_variant_payload else None
    )
    active_component_hash = _sha256_payload(
        {
            "top_level_artifact_hashes": sentiment.get("artifact_hashes") or {},
            "components": active_variant_payload.get("components") or {},
            "weights": active_variant_payload.get("weights") or {},
            "ensemble_temperature": active_variant_payload.get("ensemble_temperature"),
            "text_mode": active_variant_payload.get("text_mode"),
        }
    )
    checks = {
        "scientific_nested_oof": recorded_roles == ["scientific_nested_model_selection_oof"],
        "scientific_oof_run_recorded": len(recorded_runs) == 1
        and recorded_runs[0].startswith("scientific-oof-"),
        "stock_dataset_hash_matches_active": recorded_stock_hashes
        == [str(active_stock.get("dataset_sha256") or "")],
        "stock_feature_schema_hash_recorded": bool(recorded_feature_hashes),
        "sentiment_variant_matches_active": recorded_sentiment_variants
        == ([active_variant] if active_variant else []),
        "sentiment_variant_hash_matches_active": recorded_sentiment_variant_hashes
        == ([active_variant_hash] if active_variant_hash else []),
        "sentiment_component_hash_matches_active": recorded_component_hashes
        == [active_component_hash],
        "sentiment_config_hash_matches_active": recorded_sentiment_hashes
        == [active_sentiment_hash],
        "article_corpus_hash_recorded": bool(recorded_corpus_hashes),
        "aggregation_policy_hash_recorded": bool(recorded_aggregation_hashes),
        "feature_schema_recorded": bool(set(frame.get("feature_schema_version", pd.Series(dtype=str)).dropna().astype(str))),
    }
    return {
        "dataset_sha256": dataset_manifest_hash(dataset_path),
        "policy_sha256": policy_hash,
        "active_stock_run_id": active_run or None,
        "recorded_stock_run_ids": recorded_runs,
        "prediction_roles": recorded_roles,
        "stock_dataset_sha256": active_stock.get("dataset_sha256"),
        "recorded_stock_dataset_sha256": recorded_stock_hashes,
        "stock_feature_schema_sha256": recorded_feature_hashes,
        "sentiment_variant": active_variant or None,
        "recorded_sentiment_variants": recorded_sentiment_variants,
        "sentiment_variant_sha256": active_variant_hash,
        "recorded_sentiment_variant_sha256": recorded_sentiment_variant_hashes,
        "sentiment_component_artifacts_sha256": active_component_hash,
        "recorded_sentiment_component_artifacts_sha256": recorded_component_hashes,
        "sentiment_config_sha256": active_sentiment_hash,
        "recorded_sentiment_config_sha256": recorded_sentiment_hashes,
        "article_corpus_sha256": recorded_corpus_hashes,
        "news_aggregation_policy_sha256": recorded_aggregation_hashes,
        "feature_schema_versions": sorted(set(frame.get("feature_schema_version", pd.Series(dtype=str)).dropna().astype(str))),
        "checks": checks,
        "strict_identity_passed": all(checks.values()),
    }


def _holm_adjust(p_values: Mapping[int, float]) -> dict[int, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[int, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, horizon in enumerate(ordered):
        value = min(1.0, float(p_values[horizon]) * (count - rank))
        running = max(running, value)
        adjusted[horizon] = running
    return adjusted


def _write_staged_registry(
    report: Mapping[str, Any], registry_path: Path
) -> dict[str, Any]:
    provenance = dict(report.get("provenance") or {})
    horizon_entries: dict[str, dict[str, Any]] = {}
    for key, raw_value in dict(report.get("horizons") or {}).items():
        value = dict(raw_value)
        reference = str(value.get("artifact_path") or "")
        artifact_path = Path(reference)
        if reference and not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        observed_hash = (
            hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if artifact_path.is_file()
            else None
        )
        horizon_entries[key] = {
            "accepted": bool(value.get("accepted")),
            "artifact_path": reference,
            "artifact_sha256": value.get("artifact_sha256") or observed_hash,
            "selected_window_days": value.get("selected_window_days"),
            "selected_half_life_days": value.get("selected_half_life_days"),
            "retrospective_metrics": dict(value.get("arms") or {})
            .get("stock_plus_news", {})
            .get("metrics", {}),
            "bootstrap": value.get("bootstrap") or {},
        }
    registry = {
        "schema_version": "fusion-registry-v4",
        "created_at": report["created_at"],
        "status": "accepted" if report.get("overall_all_horizon_accepted") else "experimental",
        "claim_scope": report["claim_scope"],
        "overall_all_horizon_accepted": bool(report.get("overall_all_horizon_accepted")),
        "policy_sha256": report["policy_sha256"],
        "dataset_sha256": report["dataset_sha256"],
        "provenance": provenance,
        "report_path": _project_reference(Path(str(report.get("_report_path") or DEFAULT_REPORT))),
        "horizons": horizon_entries,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def run_fusion_evaluation_v2(
    dataset_path: Path = DEFAULT_FUSION_DATASET,
    *,
    policy_path: Path = DEFAULT_POLICY,
    report_path: Path = DEFAULT_REPORT,
    registry_path: Path = DEFAULT_REGISTRY,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    permutation_iterations: int | None = None,
    bootstrap_iterations: int | None = None,
    horizons: Sequence[int] | None = None,
    include_rolling_origins: bool = True,
) -> dict[str, Any]:
    policy, policy_hash = load_policy(policy_path)
    raw = load_fusion_dataset(dataset_path)
    frame = _enhance_features(raw)
    provenance = _provenance(frame, dataset_path, policy_hash)
    report: dict[str, Any] = {
        "schema_version": "stocksense-fusion-evaluation-v2",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": "Retrospective evidence, 2019-2025",
        "periods": policy["periods"],
        "policy_sha256": policy_hash,
        "dataset_sha256": provenance["dataset_sha256"],
        "provenance": provenance,
        "arms": ["frozen_stock", "stock_meta", "news_only", "stock_plus_news", "permuted_news_placebo"],
        "diagnostic_arms": ["transparent_weighted_formula"],
        "horizons": {},
    }
    artifact_payloads: dict[int, dict[str, Any]] = {}
    raw_p_values: dict[int, float] = {}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gates = policy["acceptance_gates"]
    requested_horizons = tuple(
        int(value) for value in (horizons or policy["supported_horizons"])
    )
    unknown = set(requested_horizons) - set(policy["supported_horizons"])
    if unknown:
        raise ValueError(f"Unsupported fusion v2 horizons: {sorted(unknown)}")
    previous = _read_json(report_path)
    if (
        previous.get("schema_version") == "stocksense-fusion-evaluation-v2"
        and previous.get("policy_sha256") == policy_hash
        and previous.get("dataset_sha256") == provenance["dataset_sha256"]
        and set(map(int, previous.get("horizons", {}).keys())) == set(requested_horizons)
        and all(
            (PROJECT_ROOT / value.get("artifact_path", "")).is_file()
            for value in previous.get("horizons", {}).values()
        )
    ):
        previous["_report_path"] = str(report_path)
        _write_staged_registry(previous, registry_path)
        previous.pop("_report_path", None)
        return previous
    for horizon in requested_horizons:
        eligible, window_selection = _select_window_variant(frame, int(horizon), policy)
        years = pd.to_datetime(eligible["observation_date"]).dt.year
        targets = pd.to_datetime(eligible["target_date"], errors="coerce")
        development = eligible.loc[
            years.between(2019, 2023) & targets.lt(pd.Timestamp("2024-01-01"))
        ].reset_index(drop=True)
        calibration = eligible.loc[
            years.eq(2024) & targets.lt(pd.Timestamp("2025-01-01"))
        ].reset_index(drop=True)
        benchmark = eligible.loc[years.eq(2025)].reset_index(drop=True)
        if any(part.empty for part in (development, calibration, benchmark)):
            raise ValueError(f"Horizon {horizon} lacks a required v2 evaluation period")
        target = benchmark["target_up"].astype(int).to_numpy()
        frozen_probability = np.clip(benchmark["stock_probability_up"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        frozen_threshold = _row_series(benchmark, "stock_decision_threshold", 0.5).to_numpy(dtype=float)
        frozen_prediction = frozen_probability >= frozen_threshold
        frozen_metrics = _metric_summary(target, frozen_probability, 0.5)
        frozen_metrics["balanced_accuracy"] = float(balanced_accuracy_score(target, frozen_prediction))

        stock = _fit_arm(development, calibration, benchmark, STOCK_FEATURES, policy)
        news = _fit_arm(development, calibration, benchmark, NEWS_FEATURES, policy)
        fusion = _fit_arm(development, calibration, benchmark, (*STOCK_FEATURES, *NEWS_FEATURES), policy)
        selected_cap, cap_trials = _select_news_shift_cap(
            development,
            stock.selected_parameters,
            fusion.selected_parameters,
            policy["news_log_odds_caps"],
        )
        unavailable = benchmark["news_available"].to_numpy(dtype=float) <= 0
        fusion_probability = _cap_news_shift(
            stock.probability, fusion.probability, selected_cap
        )
        fusion_probability[unavailable] = stock.probability[unavailable]
        fusion_prediction = fusion_probability >= fusion.threshold
        fusion_metrics = _metric_summary(target, fusion_probability, fusion.threshold)
        transparent = _transparent_weighted_diagnostic(
            development, benchmark, policy
        )

        bootstrap = _block_bootstrap(
            benchmark, target, stock.probability, fusion_probability,
            stock.prediction, fusion_prediction,
            block_size=int(policy["bootstrap_blocks"][str(horizon)]),
            iterations=int(bootstrap_iterations or policy["bootstrap_iterations"]),
            seed=int(policy["random_seed"]) + int(horizon),
        )
        brier_improvement = float(stock.metrics["brier_score"] - fusion_metrics["brier_score"])
        ba_gain = float(fusion_metrics["balanced_accuracy"] - stock.metrics["balanced_accuracy"])
        placebo = _permutation_placebo(
            development, calibration, benchmark, policy, brier_improvement,
            stock.selected_parameters,
            fusion.selected_parameters,
            selected_cap,
            iterations=int(permutation_iterations or policy["permutation_iterations"]),
            seed=int(policy["random_seed"]) + int(horizon) * 10_000,
        )
        clark_west = _clark_west(target, stock.probability, fusion_probability, benchmark["observation_date"])
        raw_p_values[int(horizon)] = clark_west["one_sided_p_value"]
        paired = _paired_details(benchmark, target, stock.prediction, fusion_prediction)
        company_gains = [row["gain"] for row in paired["per_company"].values()]
        rolling = (
            _rolling_origin_results(eligible, int(horizon), policy)
            if include_rolling_origins
            else []
        )
        positive_rolling = sum(row["brier_improvement"] > 0 for row in rolling)
        checks = {
            "minimum_rows": len(benchmark) >= int(gates["minimum_benchmark_rows"]),
            "all_tickers": benchmark["ticker"].nunique() >= int(gates["minimum_tickers"]),
            "brier_improves": brier_improvement > 0,
            "brier_bootstrap_lower_above_zero": bootstrap["brier_improvement"]["lower"] > 0,
            "balanced_accuracy_gain": ba_gain >= float(gates["minimum_balanced_accuracy_gain"]),
            "balanced_accuracy_lower_non_negative": bootstrap["balanced_accuracy_gain"]["lower"] >= 0,
            "macro_f1_not_materially_worse": fusion_metrics["macro_f1"] >= stock.metrics["macro_f1"] - float(gates["maximum_macro_f1_loss"]),
            "ece_not_materially_worse": fusion_metrics["ece"] <= stock.metrics["ece"] + float(gates["maximum_ece_increase"]),
            "placebo_percentile": placebo["actual_percentile"] >= float(gates["minimum_placebo_percentile"]),
            "rolling_year_consistency": positive_rolling >= int(gates["minimum_positive_rolling_years"]),
            "rolling_origin_evaluation_complete": include_rolling_origins
            and len(rolling) == len(policy["rolling_origins"]),
            "median_company_gain_non_negative": bool(company_gains) and float(np.median(company_gains)) >= 0,
            "company_non_deterioration": sum(gain >= 0 for gain in company_gains) >= int(gates["minimum_non_deteriorating_companies"]),
            "no_large_company_loss": bool(company_gains) and min(company_gains) >= -float(gates["maximum_company_balanced_accuracy_loss"]),
            "strict_artifact_identity": provenance["strict_identity_passed"],
        }
        artifact_run_dir = artifact_dir / (
            f"{report_path.stem}-{provenance['dataset_sha256'][:12]}-"
            f"{policy_hash[:12]}"
        )
        artifact_path = artifact_run_dir / f"fusion-v2-{horizon}d.joblib"
        artifact_payloads[int(horizon)] = {
            "schema_version": "fusion-model-artifact-v2",
            "horizon_days": int(horizon),
            "policy_sha256": policy_hash,
            "dataset_sha256": provenance["dataset_sha256"],
            "provenance": provenance,
            "stock_estimator": stock.estimator,
            "stock_calibrator": stock.calibrator,
            "stock_threshold": stock.threshold,
            "stock_selected_parameters": stock.selected_parameters,
            "fusion_estimator": fusion.estimator,
            "fusion_calibrator": fusion.calibrator,
            "fusion_threshold": fusion.threshold,
            "fusion_selected_parameters": fusion.selected_parameters,
            "selected_news_log_odds_cap": selected_cap,
            "feature_names": {"stock": STOCK_FEATURES, "news": NEWS_FEATURES},
        }
        report["horizons"][str(horizon)] = {
            "benchmark_rows": int(len(benchmark)),
            "news_present_rows": int(benchmark["news_available"].sum()),
            "ticker_count": int(benchmark["ticker"].nunique()),
            "selected_window_days": int(pd.to_numeric(benchmark["news_window_days"], errors="coerce").mode().iloc[0]),
            "selected_half_life_days": float(pd.to_numeric(benchmark["news_half_life_days"], errors="coerce").mode().iloc[0]),
            "window_selection_status": window_selection["status"],
            "window_selection": window_selection,
            "arms": {
                "frozen_stock": {"metrics": frozen_metrics},
                "stock_meta": {"metrics": stock.metrics, "selected_parameters": stock.selected_parameters},
                "news_only": {"metrics": news.metrics, "selected_parameters": news.selected_parameters},
                "stock_plus_news": {"metrics": fusion_metrics, "selected_parameters": fusion.selected_parameters},
                "permuted_news_placebo": {"summary": {key: value for key, value in placebo.items() if key != "distribution"}},
                "transparent_weighted_formula": transparent,
            },
            "brier_improvement": brier_improvement,
            "balanced_accuracy_gain": ba_gain,
            "selected_news_log_odds_cap": selected_cap,
            "news_log_odds_cap_trials": cap_trials,
            "bootstrap": bootstrap,
            "clark_west": clark_west,
            "placebo": placebo,
            "calibration": {
                "stock_meta": {"curve": _calibration_curve(target, stock.probability), **_calibration_regression(target, stock.probability)},
                "stock_plus_news": {"curve": _calibration_curve(target, fusion_probability), **_calibration_regression(target, fusion_probability)},
            },
            "rolling_origins": rolling,
            "news_coefficient_stability": _news_coefficient_stability(
                development, fusion.selected_parameters
            ),
            "subgroups": _subgroup_metrics(benchmark, target, stock, fusion_probability, fusion.threshold),
            **paired,
            "acceptance_checks": checks,
            "artifact_path": _project_reference(artifact_path),
        }

    adjusted = _holm_adjust(raw_p_values)
    for horizon, p_value in adjusted.items():
        row = report["horizons"][str(horizon)]
        row["holm_adjusted_clark_west_p_value"] = p_value
        row["acceptance_checks"]["holm_adjusted_significance"] = p_value < float(gates["holm_alpha"])
        row["accepted"] = all(row["acceptance_checks"].values())
    complete_horizon_set = set(requested_horizons) == set(policy["supported_horizons"])
    report["overall_all_horizon_accepted"] = (
        complete_horizon_set
        and bool(report["horizons"])
        and all(row["accepted"] for row in report["horizons"].values())
    )
    for horizon, payload in artifact_payloads.items():
        horizon_result = report["horizons"][str(horizon)]
        artifact_path = PROJECT_ROOT / horizon_result["artifact_path"]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload["validation_status"] = (
            "accepted"
            if report["overall_all_horizon_accepted"] and horizon_result["accepted"]
            else "experimental"
        )
        joblib.dump(payload, artifact_path)
        horizon_result["artifact_sha256"] = hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()
    report["complete_horizon_set"] = complete_horizon_set
    report["runtime_action"] = "eligible_for_reviewed_promotion" if report["overall_all_horizon_accepted"] else "retain_stock_led_runtime"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_report_path"] = str(report_path)
    _write_staged_registry(report, registry_path)
    report.pop("_report_path", None)
    return report
