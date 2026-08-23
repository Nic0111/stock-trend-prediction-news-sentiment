"""Leakage-safe evaluation and promotion workflow for stock/news fusion.

This module is deliberately separate from the live dashboard.  It consumes a
point-in-time feature dataset, compares a transparent formula with a trained
late-fusion challenger, calibrates on 2024, evaluates once on the frozen 2025
holdout, and writes a registry that runtime code will only use when accepted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

from pipeline.fusion_model import (
    DEFAULT_FUSION_DATASET,
    DEFAULT_FUSION_REGISTRY,
    DEFAULT_STAGED_FUSION_REGISTRY,
    FUSION_FEATURE_NAMES,
    FUSION_SUPPORTED_HORIZONS,
    FusionDatasetNotReadyError,
    FusionModelArtifact,
    FusionRegistry,
    PromotionDecision,
    TransparentLogOddsEstimator,
    check_fusion_dataset_readiness,
    dataset_manifest_hash,
    evaluate_promotion_gate,
    load_fusion_dataset,
    prepare_fusion_features,
    train_fusion_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FUSION_REPORT = PROJECT_ROOT / "results" / "fusion" / "fusion-evaluation.json"
DEFAULT_FUSION_SHADOW_REPORT = PROJECT_ROOT / "results" / "fusion" / "fusion-shadow.json"
DEFAULT_FUSION_COVERAGE_REPORT = (
    PROJECT_ROOT / "results" / "fusion" / "fusion-coverage.json"
)
DEFAULT_FUSION_SIGNAL_REPORT = (
    PROJECT_ROOT / "results" / "fusion" / "fusion-signal-diagnostics.json"
)
ALPHA_GRID = tuple(round(value, 1) for value in np.arange(0.0, 1.01, 0.1))
C_GRID = (0.01, 0.1, 1.0, 10.0)
ELIGIBLE_COVERAGE_SCORES = (0.5, 1.0)

FUSION_EVALUATION_PERIODS = {
    "development": (pd.Timestamp("2019-01-01"), pd.Timestamp("2023-12-31")),
    "calibration": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    "benchmark": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
}


@dataclass(frozen=True)
class CandidateEvaluation:
    method: str
    parameter: float
    probabilities: np.ndarray
    targets: np.ndarray
    folds: np.ndarray
    metrics: dict[str, float]
    fold_mcc: tuple[float, ...]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _expected_calibration_error(
    target: Sequence[int], probabilities: Sequence[float], bins: int = 10
) -> float:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if not mask.any():
            continue
        result += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def classification_metrics(
    target: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    predicted = (p >= float(threshold)).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "ece": _expected_calibration_error(y, p),
        "up_recall": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        "down_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        "predicted_up_rate": float(predicted.mean()),
    }


def _target_dates(frame: pd.DataFrame, horizon_days: int) -> pd.Series:
    if "target_date" in frame:
        parsed = pd.to_datetime(frame["target_date"], errors="coerce")
        if parsed.notna().any():
            return parsed
    candidate = f"Target_Date_{horizon_days}D"
    if candidate in frame:
        parsed = pd.to_datetime(frame[candidate], errors="coerce")
        if parsed.notna().any():
            return parsed
    return pd.to_datetime(frame["observation_date"], errors="coerce") + pd.to_timedelta(
        max(2, horizon_days * 2), unit="D"
    )


def expanding_purged_folds(
    development: pd.DataFrame, horizon_days: int, fold_count: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create five expanding chronological folds with target-date purging."""

    dates = pd.to_datetime(development["observation_date"], errors="raise").dt.normalize()
    unique_dates = np.asarray(sorted(dates.unique()))
    if len(unique_dates) < fold_count + 1:
        raise ValueError(
            f"Fusion development period has {len(unique_dates)} dates; "
            f"at least {fold_count + 1} are required"
        )
    blocks = [block for block in np.array_split(unique_dates, fold_count + 1) if len(block)]
    if len(blocks) != fold_count + 1:
        raise ValueError("Fusion development period cannot form five chronological folds")
    target_dates = _target_dates(development, horizon_days)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(1, len(blocks)):
        validation_dates = blocks[fold_index]
        validation_start = pd.Timestamp(validation_dates[0])
        train_mask = (dates < validation_start) & (target_dates < validation_start)
        validation_mask = dates.isin(validation_dates)
        train_indices = np.flatnonzero(train_mask.to_numpy())
        validation_indices = np.flatnonzero(validation_mask.to_numpy())
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(f"Fusion fold {fold_index} is empty after purging")
        splits.append((train_indices, validation_indices))
    return splits


def _transparent_probabilities(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    estimator = TransparentLogOddsEstimator(alpha)
    return estimator.predict_proba(prepare_fusion_features(frame))[:, 1]


def _candidate_key(candidate: CandidateEvaluation) -> tuple[float, float, float, int]:
    # Highest balanced accuracy, then lower log loss/Brier.  Transparent fusion
    # wins a true tie because it is easier to explain and audit.
    return (
        candidate.metrics["balanced_accuracy"],
        -candidate.metrics["log_loss"],
        -candidate.metrics["brier_score"],
        1 if candidate.method == "transparent_log_odds" else 0,
    )


def _evaluate_transparent(
    development: pd.DataFrame, horizon_days: int
) -> CandidateEvaluation:
    splits = expanding_purged_folds(development, horizon_days)
    best: CandidateEvaluation | None = None
    for alpha in ALPHA_GRID:
        probabilities: list[float] = []
        targets: list[int] = []
        folds: list[int] = []
        fold_mcc: list[float] = []
        for fold_number, (_, validation_indices) in enumerate(splits, start=1):
            validation = development.iloc[validation_indices]
            probability = _transparent_probabilities(validation, alpha)
            target = validation["target_up"].astype(int).to_numpy()
            probabilities.extend(probability.tolist())
            targets.extend(target.tolist())
            folds.extend([fold_number] * len(validation))
            fold_mcc.append(classification_metrics(target, probability, 0.5)["mcc"])
        candidate = CandidateEvaluation(
            "transparent_log_odds",
            alpha,
            np.asarray(probabilities),
            np.asarray(targets),
            np.asarray(folds),
            classification_metrics(targets, probabilities, 0.5),
            tuple(fold_mcc),
        )
        if best is None or _candidate_key(candidate) > _candidate_key(best):
            best = candidate
    assert best is not None
    return best


def _evaluate_late_fusion(
    development: pd.DataFrame, horizon_days: int
) -> CandidateEvaluation:
    splits = expanding_purged_folds(development, horizon_days)
    best: CandidateEvaluation | None = None
    for regularization in C_GRID:
        probabilities: list[float] = []
        targets: list[int] = []
        folds: list[int] = []
        fold_mcc: list[float] = []
        for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
            training = development.iloc[train_indices]
            validation = development.iloc[validation_indices]
            artifact = train_fusion_model(
                training,
                horizon_days,
                model_version=f"late-fusion-C{regularization:g}",
                C=regularization,
                random_state=42,
            )
            probability = artifact.predict_proba(validation)[:, 1]
            target = validation["target_up"].astype(int).to_numpy()
            probabilities.extend(probability.tolist())
            targets.extend(target.tolist())
            folds.extend([fold_number] * len(validation))
            fold_mcc.append(classification_metrics(target, probability, 0.5)["mcc"])
        candidate = CandidateEvaluation(
            "late_fusion_logistic",
            regularization,
            np.asarray(probabilities),
            np.asarray(targets),
            np.asarray(folds),
            classification_metrics(targets, probabilities, 0.5),
            tuple(fold_mcc),
        )
        if best is None or _candidate_key(candidate) > _candidate_key(best):
            best = candidate
    assert best is not None
    return best


def _fit_candidate(
    development: pd.DataFrame,
    horizon_days: int,
    candidate: CandidateEvaluation,
    dataset_hash: str,
) -> FusionModelArtifact:
    version = (
        f"fusion-{horizon_days}d-{candidate.method}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )
    if candidate.method == "transparent_log_odds":
        return FusionModelArtifact(
            horizon_days=horizon_days,
            model_version=version,
            feature_names=tuple(FUSION_FEATURE_NAMES),
            estimator=TransparentLogOddsEstimator(candidate.parameter),
            dataset_manifest_sha256=dataset_hash,
            trained_at=_now_iso(),
            fusion_method=candidate.method,
            transparent_alpha=candidate.parameter,
        )
    artifact = train_fusion_model(
        development,
        horizon_days,
        model_version=version,
        dataset_manifest_sha256=dataset_hash,
        C=candidate.parameter,
        random_state=42,
    )
    artifact.fusion_method = candidate.method
    return artifact


def _fit_calibrator(raw_probability: np.ndarray, target: pd.Series) -> LogisticRegression:
    if target.nunique() < 2:
        raise ValueError("2024 calibration period must contain both target classes")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
    calibrator.fit(np.asarray(raw_probability).reshape(-1, 1), target.astype(int))
    return calibrator


def _select_threshold(target: Sequence[int], probabilities: Sequence[float]) -> float:
    best_threshold: float | None = None
    best_key: tuple[float, float, float] | None = None
    for threshold in np.arange(0.30, 0.701, 0.005):
        metrics = classification_metrics(target, probabilities, float(threshold))
        if (
            metrics["up_recall"] < 0.35
            or metrics["down_recall"] < 0.35
            or not 0.20 <= metrics["predicted_up_rate"] <= 0.80
        ):
            continue
        key = (
            metrics["balanced_accuracy"],
            metrics["mcc"],
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    if best_threshold is None:
        return 0.5
    return round(best_threshold, 3)


def _baseline_probabilities(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.clip(
        pd.to_numeric(frame["stock_probability_up"], errors="raise").to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )
    if "stock_decision_threshold" in frame:
        thresholds = pd.to_numeric(
            frame["stock_decision_threshold"], errors="coerce"
        ).fillna(0.5).to_numpy(dtype=float)
    else:
        thresholds = np.full(len(frame), 0.5, dtype=float)
    return probabilities, thresholds


def _metrics_with_variable_threshold(
    target: Sequence[int], probabilities: Sequence[float], thresholds: Sequence[float]
) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = (p >= np.asarray(thresholds, dtype=float)).astype(int)
    metrics = classification_metrics(y, p, 0.5)
    metrics.update(
        {
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "mcc": float(matthews_corrcoef(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "up_recall": float(recall_score(y, predicted, pos_label=1, zero_division=0)),
            "down_recall": float(recall_score(y, predicted, pos_label=0, zero_division=0)),
            "predicted_up_rate": float(predicted.mean()),
        }
    )
    return metrics


def _bootstrap_gain_interval(
    target: np.ndarray,
    challenger: np.ndarray,
    challenger_threshold: float,
    baseline: np.ndarray,
    baseline_thresholds: np.ndarray,
    iterations: int = 1000,
) -> tuple[float, float]:
    generator = np.random.default_rng(42)
    gains: list[float] = []
    for _ in range(iterations):
        sample = generator.integers(0, len(target), len(target))
        if np.unique(target[sample]).size < 2:
            continue
        challenger_score = balanced_accuracy_score(
            target[sample], challenger[sample] >= challenger_threshold
        )
        baseline_score = balanced_accuracy_score(
            target[sample], baseline[sample] >= baseline_thresholds[sample]
        )
        gains.append(float(challenger_score - baseline_score))
    if not gains:
        return (-1.0, -1.0)
    return (float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975)))


def _company_gains(
    frame: pd.DataFrame,
    challenger: np.ndarray,
    challenger_threshold: float,
    baseline: np.ndarray,
    baseline_thresholds: np.ndarray,
) -> dict[str, float]:
    gains: dict[str, float] = {}
    for ticker, positions in frame.groupby("ticker", sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        target = frame.iloc[index]["target_up"].astype(int).to_numpy()
        if np.unique(target).size < 2:
            gains[str(ticker)] = -1.0
            continue
        challenger_score = balanced_accuracy_score(
            target, challenger[index] >= challenger_threshold
        )
        baseline_score = balanced_accuracy_score(
            target, baseline[index] >= baseline_thresholds[index]
        )
        gains[str(ticker)] = float(challenger_score - baseline_score)
    return gains


def _eligible_rows(frame: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    subset = frame.loc[
        pd.to_numeric(frame["horizon_days"], errors="coerce").eq(horizon_days)
    ].copy()
    subset["observation_date"] = pd.to_datetime(
        subset["observation_date"], errors="coerce"
    )
    subset["news_coverage_score"] = pd.to_numeric(
        subset["news_coverage_score"], errors="coerce"
    )
    subset["news_available"] = pd.to_numeric(
        subset["news_available"], errors="coerce"
    ).fillna(0)
    subset = subset.loc[
        subset["observation_date"].notna()
        & subset["news_coverage_score"].isin(ELIGIBLE_COVERAGE_SCORES)
        & subset["news_available"].eq(1)
    ].copy()
    return subset.sort_values(["observation_date", "ticker"]).reset_index(drop=True)


def build_fusion_coverage_report(
    frame: pd.DataFrame,
    *,
    supported_tickers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe the exact evidence coverage required by frozen evaluation.

    The report intentionally distinguishes any retrieved news from the
    Medium/High rows that are allowed to influence model fitting.  This makes
    data-collection gaps visible without weakening the evaluator's gates.
    """

    # Validate the numerical feature contract while preserving identifiers and
    # target columns needed by the coverage matrix.
    prepare_fusion_features(frame)
    working = frame.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    dataset_tickers = sorted(working["ticker"].unique())
    required_tickers = sorted(
        {str(value).upper() for value in (supported_tickers or dataset_tickers)}
    )
    working = working.loc[working["ticker"].isin(required_tickers)].copy()
    result: dict[str, Any] = {
        "created_at": _now_iso(),
        "eligible_definition": {
            "news_available": 1,
            "coverage_scores": list(ELIGIBLE_COVERAGE_SCORES),
            "coverage_labels": ["Medium", "High"],
        },
        "required_tickers": required_tickers,
        "dataset_tickers": dataset_tickers,
        "research_subset": set(required_tickers) != set(dataset_tickers),
        "periods": {
            name: {
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
            }
            for name, (start, end) in FUSION_EVALUATION_PERIODS.items()
        },
        "horizons": {},
    }
    complete_by_horizon: dict[int, set[str]] = {}
    all_ready = True
    for horizon_days in FUSION_SUPPORTED_HORIZONS:
        horizon = working.loc[
            pd.to_numeric(working["horizon_days"], errors="coerce").eq(
                horizon_days
            )
        ].copy()
        horizon["observation_date"] = pd.to_datetime(
            horizon["observation_date"], errors="coerce"
        )
        horizon["news_available"] = pd.to_numeric(
            horizon["news_available"], errors="coerce"
        ).fillna(0)
        eligible = _eligible_rows(working, horizon_days)
        ticker_rows: dict[str, Any] = {}
        complete: set[str] = set()
        for ticker in required_tickers:
            ticker_complete = True
            periods: dict[str, Any] = {}
            for period_name, (start, end) in FUSION_EVALUATION_PERIODS.items():
                total_period = horizon.loc[
                    horizon["ticker"].eq(ticker)
                    & horizon["observation_date"].between(start, end)
                ]
                eligible_period = eligible.loc[
                    eligible["ticker"].eq(ticker)
                    & eligible["observation_date"].between(start, end)
                ]
                eligible_count = int(len(eligible_period))
                if eligible_count == 0:
                    ticker_complete = False
                periods[period_name] = {
                    "total_rows": int(len(total_period)),
                    "any_news_rows": int(total_period["news_available"].eq(1).sum()),
                    "eligible_medium_high_rows": eligible_count,
                    "target_up_count": int(
                        pd.to_numeric(eligible_period["target_up"], errors="coerce")
                        .eq(1)
                        .sum()
                    ),
                    "target_down_count": int(
                        pd.to_numeric(eligible_period["target_up"], errors="coerce")
                        .eq(0)
                        .sum()
                    ),
                }
            ticker_rows[ticker] = {
                "complete": ticker_complete,
                "periods": periods,
            }
            if ticker_complete:
                complete.add(ticker)
        missing = sorted(set(required_tickers) - complete)
        if missing:
            all_ready = False
        complete_by_horizon[horizon_days] = complete
        result["horizons"][str(horizon_days)] = {
            "eligible_rows": int(len(eligible)),
            "complete_tickers": sorted(complete),
            "missing_tickers": missing,
            "tickers": ticker_rows,
        }
    complete_all = (
        set.intersection(*complete_by_horizon.values())
        if complete_by_horizon
        else set()
    )
    result["complete_all_horizons"] = sorted(complete_all)
    result["ready_for_requested_universe"] = all_ready
    return result


def write_fusion_coverage_report(
    dataset_path: Path = DEFAULT_FUSION_DATASET,
    *,
    output_path: Path = DEFAULT_FUSION_COVERAGE_REPORT,
    supported_tickers: Sequence[str] | None = None,
) -> dict[str, Any]:
    frame = load_fusion_dataset(dataset_path)
    report = build_fusion_coverage_report(
        frame, supported_tickers=supported_tickers
    )
    report["dataset_path"] = str(dataset_path)
    report["dataset_sha256"] = dataset_manifest_hash(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _signal_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame.copy()
    selected["target_up"] = pd.to_numeric(selected["target_up"], errors="coerce")
    selected["news_weighted_sentiment"] = pd.to_numeric(
        selected["news_weighted_sentiment"], errors="coerce"
    )
    selected["stock_probability_up"] = pd.to_numeric(
        selected["stock_probability_up"], errors="coerce"
    )
    selected = selected.dropna(
        subset=["target_up", "news_weighted_sentiment", "stock_probability_up"]
    )
    selected = selected.loc[selected["target_up"].isin([0, 1])]
    result: dict[str, Any] = {
        "rows": int(len(selected)),
        "target_up_rate": (
            float(selected["target_up"].mean()) if len(selected) else None
        ),
        "mean_news_sentiment": (
            float(selected["news_weighted_sentiment"].mean())
            if len(selected)
            else None
        ),
    }
    if len(selected) < 2 or selected["target_up"].nunique() < 2:
        result.update(
            {
                "sentiment_target_correlation": None,
                "sentiment_roc_auc": None,
                "stock_probability_roc_auc": None,
            }
        )
        return result
    sentiment = selected["news_weighted_sentiment"].to_numpy(dtype=float)
    target = selected["target_up"].to_numpy(dtype=int)
    correlation = (
        float(np.corrcoef(sentiment, target)[0, 1])
        if np.unique(sentiment).size > 1
        else None
    )
    result.update(
        {
            "sentiment_target_correlation": correlation,
            "sentiment_roc_auc": float(roc_auc_score(target, sentiment)),
            "stock_probability_roc_auc": float(
                roc_auc_score(
                    target,
                    selected["stock_probability_up"].to_numpy(dtype=float),
                )
            ),
        }
    )
    return result


def write_fusion_signal_diagnostics(
    dataset_path: Path = DEFAULT_FUSION_DATASET,
    *,
    output_path: Path = DEFAULT_FUSION_SIGNAL_REPORT,
) -> dict[str, Any]:
    """Write a non-promotional check of whether historical tone tracks returns."""

    frame = load_fusion_dataset(dataset_path)
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], errors="coerce"
    )
    frame["news_available"] = pd.to_numeric(
        frame["news_available"], errors="coerce"
    ).fillna(0)
    frame["news_coverage_score"] = pd.to_numeric(
        frame["news_coverage_score"], errors="coerce"
    ).fillna(0)
    report: dict[str, Any] = {
        "created_at": _now_iso(),
        "mode": "research_diagnostic_only",
        "production_promotion_allowed": False,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_manifest_hash(dataset_path),
        "interpretation": (
            "AUC near 0.50 means the signed news tone alone does not rank future "
            "Up outcomes better than chance. This diagnostic is not a model gate."
        ),
        "horizons": {},
    }
    for horizon_days in FUSION_SUPPORTED_HORIZONS:
        available = frame.loc[
            pd.to_numeric(frame["horizon_days"], errors="coerce").eq(horizon_days)
            & frame["news_available"].eq(1)
        ].copy()
        tiers = {
            "all_news_available": available,
            "low": available.loc[available["news_coverage_score"].eq(0.25)],
            "medium_high": available.loc[
                available["news_coverage_score"].isin(ELIGIBLE_COVERAGE_SCORES)
            ],
        }
        report["horizons"][str(horizon_days)] = {
            "coverage_tiers": {
                name: _signal_statistics(values) for name, values in tiers.items()
            },
            "years": {
                str(int(year)): _signal_statistics(values)
                for year, values in available.groupby(
                    available["observation_date"].dt.year, dropna=True
                )
            },
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def evaluate_fusion_dataset(
    dataset_path: Path = DEFAULT_FUSION_DATASET,
    *,
    registry_path: Path = DEFAULT_STAGED_FUSION_REGISTRY,
    report_path: Path = DEFAULT_FUSION_REPORT,
    supported_tickers: Sequence[str] | None = None,
    min_rows_per_horizon: int = 100,
) -> dict[int, FusionModelArtifact]:
    """Evaluate both challengers and write horizon-specific frozen decisions."""

    readiness = check_fusion_dataset_readiness(
        dataset_path,
        supported_tickers=supported_tickers,
        min_rows_per_horizon=min_rows_per_horizon,
        required_end_year=2025,
    )
    if not readiness.ready:
        raise FusionDatasetNotReadyError(readiness.reason)
    frame = load_fusion_dataset(dataset_path)
    dataset_hash = dataset_manifest_hash(dataset_path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    dataset_tickers = set(frame["ticker"].unique())
    required_tickers = {
        str(value).upper()
        for value in (supported_tickers or sorted(dataset_tickers))
    }
    research_subset = required_tickers != dataset_tickers
    # A ticker-restricted run is a genuine research subset: excluded companies
    # must not leak back into training merely because their rows share the same
    # dataset file.
    frame = frame.loc[frame["ticker"].isin(required_tickers)].reset_index(drop=True)
    artifacts: dict[int, FusionModelArtifact] = {}
    registry_entries: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "created_at": _now_iso(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "evaluation_scope": {
            "required_tickers": sorted(required_tickers),
            "dataset_tickers": sorted(dataset_tickers),
            "research_subset": research_subset,
            "production_promotion_allowed": not research_subset,
        },
        "periods": {
            "development": "2019-01-01 through 2023-12-31",
            "calibration": "2024-01-01 through 2024-12-31",
            "benchmark": "2025-01-01 through 2025-12-31",
            "shadow": "2026 onward",
        },
        "horizons": {},
    }
    for horizon_days in FUSION_SUPPORTED_HORIZONS:
        eligible = _eligible_rows(frame, horizon_days)
        periods = {
            year: eligible.loc[
                eligible["observation_date"].dt.year.eq(year)
            ].copy().reset_index(drop=True)
            for year in range(2019, 2026)
        }
        development = pd.concat([periods[year] for year in range(2019, 2024)], ignore_index=True)
        calibration = periods[2024]
        benchmark = periods[2025]
        present = set(eligible["ticker"].astype(str).str.upper())
        missing_tickers = sorted(required_tickers - present)
        if missing_tickers:
            raise FusionDatasetNotReadyError(
                f"Horizon {horizon_days} Medium/High fusion data is missing tickers: {missing_tickers}"
            )
        if any(value.empty for value in (development, calibration, benchmark)):
            raise FusionDatasetNotReadyError(
                f"Horizon {horizon_days} requires non-empty development, calibration, and 2025 benchmark rows"
            )
        for period_name, period_frame in (
            ("development", development),
            ("calibration", calibration),
            ("benchmark", benchmark),
        ):
            period_tickers = set(period_frame["ticker"].astype(str).str.upper())
            missing_period_tickers = sorted(required_tickers - period_tickers)
            if missing_period_tickers:
                raise FusionDatasetNotReadyError(
                    f"Horizon {horizon_days} {period_name} period is missing "
                    f"Medium/High rows for tickers: {missing_period_tickers}"
                )

        transparent = _evaluate_transparent(development, horizon_days)
        late = _evaluate_late_fusion(development, horizon_days)
        selected = max((transparent, late), key=_candidate_key)
        artifact = _fit_candidate(development, horizon_days, selected, dataset_hash)
        raw_calibration = artifact.predict_uncalibrated_proba(calibration)
        artifact.calibrator = _fit_calibrator(raw_calibration, calibration["target_up"])
        calibrated = artifact.predict_proba(calibration)[:, 1]
        artifact.threshold = _select_threshold(calibration["target_up"], calibrated)

        target = benchmark["target_up"].astype(int).to_numpy()
        challenger_probability = artifact.predict_proba(benchmark)[:, 1]
        challenger_metrics = classification_metrics(
            target, challenger_probability, artifact.threshold
        )
        baseline_probability, baseline_thresholds = _baseline_probabilities(benchmark)
        baseline_metrics = _metrics_with_variable_threshold(
            target, baseline_probability, baseline_thresholds
        )
        bootstrap_interval = _bootstrap_gain_interval(
            target,
            challenger_probability,
            artifact.threshold,
            baseline_probability,
            baseline_thresholds,
        )
        company_gains = _company_gains(
            benchmark,
            challenger_probability,
            artifact.threshold,
            baseline_probability,
            baseline_thresholds,
        )
        decision = evaluate_promotion_gate(
            challenger_metrics,
            baseline_metrics,
            fold_mcc=selected.fold_mcc,
            bootstrap_balanced_accuracy_gain_ci=bootstrap_interval,
            per_company_balanced_accuracy_gain=company_gains,
        )
        if research_subset:
            checks = dict(decision.checks)
            checks["full_company_universe"] = False
            decision = PromotionDecision(
                False,
                checks,
                tuple(decision.reasons)
                + ("research subset cannot be promoted to production",),
            )
        artifact.validation_status = "accepted" if decision.accepted else "experimental"
        artifact.metrics = challenger_metrics
        artifact.baseline_metrics = baseline_metrics
        artifact.promotion_decision = decision.to_dict()
        artifact.development_start = "2019-01-01"
        artifact.development_end = "2023-12-31"
        artifact.calibration_start = "2024-01-01"
        artifact.calibration_end = "2024-12-31"
        artifact.benchmark_start = "2025-01-01"
        artifact.benchmark_end = "2025-12-31"
        artifact.shadow_start = "2026-01-01"
        artifacts[horizon_days] = artifact

        artifact_path = registry_path.parent / f"fusion-{horizon_days}d-{artifact.model_version}.joblib"
        artifact.save(artifact_path)
        artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        try:
            artifact_reference = str(artifact_path.relative_to(PROJECT_ROOT))
        except ValueError:
            artifact_reference = str(artifact_path)
        registry_entries[str(horizon_days)] = {
            "artifact_path": artifact_reference,
            "artifact_sha256": artifact_hash,
            "model_version": artifact.model_version,
            "validation_status": artifact.validation_status,
            "fusion_method": artifact.fusion_method,
            "feature_names": list(artifact.feature_names),
            "decision_threshold": artifact.threshold,
            "metrics": challenger_metrics,
            "baseline_metrics": baseline_metrics,
            "promotion_decision": decision.to_dict(),
        }
        report["horizons"][str(horizon_days)] = {
            "eligible_rows": int(len(eligible)),
            "development_rows": int(len(development)),
            "calibration_rows": int(len(calibration)),
            "benchmark_rows": int(len(benchmark)),
            "transparent_development": {
                "alpha": transparent.parameter,
                "metrics": transparent.metrics,
                "fold_mcc": list(transparent.fold_mcc),
            },
            "late_fusion_development": {
                "C": late.parameter,
                "metrics": late.metrics,
                "fold_mcc": list(late.fold_mcc),
            },
            "selected_method": selected.method,
            "selected_parameter": selected.parameter,
            "calibrated_threshold": artifact.threshold,
            "benchmark_metrics": challenger_metrics,
            "stock_baseline_metrics": baseline_metrics,
            "bootstrap_balanced_accuracy_gain_ci": list(bootstrap_interval),
            "per_company_balanced_accuracy_gain": company_gains,
            "promotion": decision.to_dict(),
        }

    FusionRegistry(_now_iso(), dataset_hash, registry_entries).write(registry_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def promote_fusion_registry(
    staged_registry_path: Path = DEFAULT_STAGED_FUSION_REGISTRY,
    *,
    active_registry_path: Path = DEFAULT_FUSION_REGISTRY,
    approve: bool = False,
) -> FusionRegistry:
    """Promote only accepted staged horizons; never promote by evaluation side effect."""

    if not approve:
        raise ValueError("Fusion promotion requires explicit approve=True")
    staged = FusionRegistry.read(staged_registry_path)
    accepted = {
        horizon: dict(entry)
        for horizon, entry in staged.horizons.items()
        if str(entry.get("validation_status", "")).casefold() == "accepted"
    }
    if not accepted:
        raise ValueError("The staged fusion registry contains no accepted horizons")
    active_entries: dict[str, dict[str, Any]] = {}
    if active_registry_path.exists():
        try:
            active_entries.update(FusionRegistry.read(active_registry_path).horizons)
        except (OSError, ValueError, KeyError, TypeError):
            active_entries = {}
    active_entries.update(accepted)
    promoted = FusionRegistry(
        created_at=_now_iso(),
        dataset_manifest_sha256=staged.dataset_manifest_sha256,
        horizons=active_entries,
    )
    promoted.write(active_registry_path)
    return promoted


def evaluate_fusion_shadow(
    dataset_path: Path,
    *,
    registry_path: Path = DEFAULT_FUSION_REGISTRY,
    report_path: Path = DEFAULT_FUSION_SHADOW_REPORT,
    start_date: str = "2026-01-01",
    end_date: str | None = None,
) -> dict[str, Any]:
    """Score frozen artifacts on later labelled rows without tuning or promotion."""

    frame = load_fusion_dataset(dataset_path)
    registry = FusionRegistry.read(registry_path)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.max.normalize()
    report: dict[str, Any] = {
        "created_at": _now_iso(),
        "mode": "prospective_shadow_only",
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat() if end_date else None,
        "registry_created_at": registry.created_at,
        "horizons": {},
    }
    for horizon_days in FUSION_SUPPORTED_HORIZONS:
        entry = registry.entry(horizon_days) or {}
        reference = str(entry.get("artifact_path", "")).strip()
        if not reference:
            report["horizons"][str(horizon_days)] = {"available": False}
            continue
        artifact_path = Path(reference)
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        artifact = FusionModelArtifact.load(artifact_path)
        eligible = _eligible_rows(frame, horizon_days)
        eligible = eligible.loc[
            eligible["observation_date"].between(start, end)
            & pd.to_numeric(eligible["target_up"], errors="coerce").isin([0, 1])
        ].reset_index(drop=True)
        if eligible.empty:
            report["horizons"][str(horizon_days)] = {
                "available": True,
                "labelled_rows": 0,
                "status": "awaiting_labels",
            }
            continue
        target = eligible["target_up"].astype(int).to_numpy()
        challenger = artifact.predict_proba(eligible)[:, 1]
        baseline, baseline_thresholds = _baseline_probabilities(eligible)
        report["horizons"][str(horizon_days)] = {
            "available": True,
            "labelled_rows": int(len(eligible)),
            "fusion_method": artifact.fusion_method,
            "frozen_threshold": artifact.threshold,
            "fusion_metrics": classification_metrics(
                target, challenger, artifact.threshold
            ),
            "stock_baseline_metrics": _metrics_with_variable_threshold(
                target, baseline, baseline_thresholds
            ),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
