"""Horizon-aware, leakage-safe acceptance evaluation.

The stock trainer owns model selection. This module owns the promotion policy so
that a new threshold or evidence tier cannot silently diverge between offline
training, shadow evaluation and runtime inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = PROJECT_ROOT / "config" / "horizon_acceptance_policy.json"
STATUS_ORDER = {"experimental": 0, "limited": 1, "accepted": 2}


def load_acceptance_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Load and validate the tracked acceptance policy."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    common = payload.get("common", {})
    horizons = payload.get("horizons", {})
    required_horizons = {"1", "5", "10"}
    missing = required_horizons.difference(horizons)
    if missing:
        raise ValueError(f"Acceptance policy is missing horizons: {sorted(missing)}")
    for name in (
        "minimum_balanced_accuracy_gain",
        "minimum_direction_recall",
        "limited_minimum_direction_recall",
        "minimum_brier_skill",
    ):
        if float(common[name]) < 0:
            raise ValueError(f"Acceptance policy value must be non-negative: {name}")
    if int(payload.get("bootstrap_repetitions", 0)) < 100:
        raise ValueError("Acceptance policy requires at least 100 bootstrap repetitions")
    return payload


POLICY = load_acceptance_policy()
COMMON_POLICY = POLICY["common"]


def status_rank(status: str) -> int:
    return STATUS_ORDER.get(str(status).lower(), 0)


def weaker_status(*statuses: str) -> str:
    """Return the most conservative status in the supplied statuses."""
    return min((str(status).lower() for status in statuses), key=status_rank, default="experimental")


def _as_arrays(
    y_true: Iterable[int],
    predictions: Iterable[int],
    dates: Iterable[Any],
    baseline_predictions: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(predictions), dtype=int)
    observed_dates = np.asarray(list(dates))
    baseline = np.asarray(list(baseline_predictions), dtype=int)
    if not (len(y) == len(pred) == len(observed_dates) == len(baseline)):
        raise ValueError("Acceptance arrays must have equal lengths")
    if len(y) == 0:
        raise ValueError("Acceptance evaluation requires at least one row")
    return y, pred, observed_dates, baseline


def date_block_bootstrap_gain(
    y_true: Iterable[int],
    predictions: Iterable[int],
    baseline_predictions: Iterable[int],
    dates: Iterable[Any],
    *,
    block_size: int,
    repetitions: int | None = None,
    seed: int | None = None,
) -> dict[str, float | int | list[float]]:
    """Estimate model-vs-baseline balanced-accuracy gain by date blocks.

    Every ticker on a sampled date remains together, preserving the panel's
    cross-sectional dependence. Circular blocks make the resampling stable at
    the beginning/end of the evaluation period.
    """
    y, pred, observed_dates, baseline = _as_arrays(
        y_true, predictions, dates, baseline_predictions
    )
    unique_dates = np.array(sorted(set(observed_dates.tolist())))
    if len(unique_dates) < 2:
        return {
            "unique_dates": int(len(unique_dates)),
            "block_size": int(block_size),
            "repetitions": 0,
            "gain": float(balanced_accuracy_score(y, pred) - balanced_accuracy_score(y, baseline)),
            "lower": 0.0,
            "median": 0.0,
            "upper": 0.0,
            "p_value": 1.0,
        }
    groups = {value: np.flatnonzero(observed_dates == value) for value in unique_dates}
    reps = int(repetitions or POLICY["bootstrap_repetitions"])
    rng = np.random.default_rng(seed if seed is not None else POLICY["random_seed"])
    block = max(1, int(block_size))
    gains: list[float] = []
    for _ in range(reps):
        sampled_dates: list[Any] = []
        while len(sampled_dates) < len(unique_dates):
            start = int(rng.integers(0, len(unique_dates)))
            indexes = (start + np.arange(block)) % len(unique_dates)
            sampled_dates.extend(unique_dates[indexes].tolist())
        indexes = np.concatenate([groups[value] for value in sampled_dates[: len(unique_dates)]])
        sample_y = y[indexes]
        if np.unique(sample_y).size < 2:
            continue
        gains.append(
            float(
                balanced_accuracy_score(sample_y, pred[indexes])
                - balanced_accuracy_score(sample_y, baseline[indexes])
            )
        )
    point_gain = float(
        balanced_accuracy_score(y, pred) - balanced_accuracy_score(y, baseline)
    )
    if not gains:
        return {
            "unique_dates": int(len(unique_dates)),
            "block_size": block,
            "repetitions": 0,
            "gain": point_gain,
            "lower": 0.0,
            "median": 0.0,
            "upper": 0.0,
            "p_value": 1.0,
        }
    values = np.asarray(gains, dtype=float)
    alpha = float(POLICY["significance_level"])
    lower, median, upper = np.quantile(values, [alpha / 2, 0.5, 1 - alpha / 2])
    return {
        "unique_dates": int(len(unique_dates)),
        "block_size": block,
        "repetitions": int(len(values)),
        "gain": point_gain,
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
        "p_value": float((np.count_nonzero(values <= 0) + 1) / (len(values) + 1)),
    }


def _calibration_metrics(
    y_true: np.ndarray,
    probability_up: np.ndarray,
    *,
    baseline_prevalence: float,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability_up, dtype=float), 1e-6, 1 - 1e-6)
    locked_prevalence = float(baseline_prevalence)
    if not 0.0 <= locked_prevalence <= 1.0:
        raise ValueError("Locked baseline prevalence must be between zero and one")
    baseline_probability = np.full(len(y), locked_prevalence, dtype=float)
    brier = float(brier_score_loss(y, probability))
    baseline_brier = float(brier_score_loss(y, baseline_probability))
    skill = float(1.0 - brier / baseline_brier) if baseline_brier > 0 else 0.0
    result: dict[str, float] = {
        "brier_score": brier,
        "brier_baseline": baseline_brier,
        "brier_skill": skill,
        "baseline_prevalence": locked_prevalence,
        "evaluation_prevalence": float(np.mean(y)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }
    order = np.argsort(probability)
    bins = np.array_split(order, min(10, len(order)))
    result["ece_10"] = float(
        sum(len(indexes) / len(y) * abs(float(probability[indexes].mean()) - float(y[indexes].mean())) for indexes in bins if len(indexes))
    )
    if np.unique(y).size == 2 and len(y) >= 30:
        calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=42)
        calibrator.fit(np.log(probability / (1 - probability)).reshape(-1, 1), y)
        result["calibration_intercept"] = float(calibrator.intercept_[0])
        result["calibration_slope"] = float(calibrator.coef_[0, 0])
    else:
        result["calibration_intercept"] = 0.0
        result["calibration_slope"] = 1.0
    return result


def _directional_checks(metrics: dict[str, float], *, limited: bool = False) -> dict[str, bool]:
    recall_floor = float(
        COMMON_POLICY["limited_minimum_direction_recall"]
        if limited
        else COMMON_POLICY["minimum_direction_recall"]
    )
    share_low = float(
        COMMON_POLICY["limited_minimum_predicted_class_share"]
        if limited
        else COMMON_POLICY["minimum_predicted_class_share"]
    )
    share_high = float(
        COMMON_POLICY["limited_maximum_predicted_class_share"]
        if limited
        else COMMON_POLICY["maximum_predicted_class_share"]
    )
    return {
        "directional_recall": bool(
            metrics.get("recall", 0.0) >= recall_floor
            and metrics.get("specificity", 0.0) >= recall_floor
        ),
        "predicted_class_share": bool(
            share_low <= metrics.get("predicted_up_rate", 0.0) <= share_high
        ),
    }


def evaluate_acceptance(
    *,
    horizon: int,
    y_true: Iterable[int],
    predictions: Iterable[int],
    probability_up: Iterable[float],
    baseline_predictions: Iterable[int],
    dates: Iterable[Any],
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    oof_metrics: dict[str, float],
    cv_positive_mcc_folds: int,
    baseline_prevalence: float,
    development_evidence_available: bool = True,
    phase: str = "observed",
    ticker: str | None = None,
) -> dict[str, Any]:
    """Return deterministic gates, status and explanations for one scope."""
    y, pred, observed_dates, baseline = _as_arrays(
        y_true, predictions, dates, baseline_predictions
    )
    probabilities = np.asarray(list(probability_up), dtype=float)
    if len(probabilities) != len(y):
        raise ValueError("Probability and target arrays must have equal lengths")
    horizon_config = POLICY["horizons"][str(horizon)]
    bootstrap = date_block_bootstrap_gain(
        y,
        pred,
        baseline,
        observed_dates,
        block_size=int(horizon_config["bootstrap_block_sessions"]),
    )
    calibration = _calibration_metrics(
        y,
        probabilities,
        baseline_prevalence=float(baseline_prevalence),
    )
    development_stability = bool(
        development_evidence_available
        and cv_positive_mcc_folds >= int(COMMON_POLICY["minimum_positive_mcc_folds"])
    )
    limited_stability = bool(
        development_evidence_available
        and cv_positive_mcc_folds >= int(COMMON_POLICY["limited_minimum_positive_mcc_folds"])
    )
    balanced_gain = float(metrics.get("balanced_accuracy", 0.0) - baseline_metrics.get("balanced_accuracy", 0.0))
    accepted_checks = {
        "balanced_accuracy_gain": balanced_gain >= float(COMMON_POLICY["minimum_balanced_accuracy_gain"]),
        "positive_mcc": metrics.get("mcc", 0.0) > 0,
        "macro_f1_not_worse": metrics.get("macro_f1", 0.0) >= baseline_metrics.get("macro_f1", 0.0),
        "development_stability": development_stability,
        "threshold_constraints": bool(
            development_evidence_available
            and oof_metrics.get("threshold_constraints_passed", False)
        ),
        "directional_recall": _directional_checks(metrics)["directional_recall"],
        "predicted_class_share": _directional_checks(metrics)["predicted_class_share"],
        "brier_skill": calibration["brier_skill"] >= float(COMMON_POLICY["minimum_brier_skill"]),
        "bootstrap_gain": (
            int(bootstrap["unique_dates"]) >= int(horizon_config["minimum_observed_dates"] if phase == "observed" else horizon_config["minimum_shadow_dates"])
            and float(bootstrap["lower"]) > 0
            and float(bootstrap["p_value"]) <= float(POLICY["significance_level"])
        ),
    }
    limited_checks = {
        "positive_mcc": metrics.get("mcc", 0.0) > 0,
        "brier_skill": calibration["brier_skill"] >= float(COMMON_POLICY["minimum_brier_skill"]),
        "bootstrap_gain": float(bootstrap["lower"]) > 0,
        "limited_directional_recall": _directional_checks(metrics, limited=True)["directional_recall"],
        "limited_predicted_class_share": _directional_checks(metrics, limited=True)["predicted_class_share"],
        "limited_development_stability": limited_stability,
    }
    if all(accepted_checks.values()):
        evidence_status = "accepted"
    elif all(limited_checks.values()):
        evidence_status = "limited"
    else:
        evidence_status = "experimental"
    failed = [name for name, passed in accepted_checks.items() if not passed]
    return {
        "status": evidence_status,
        "phase": phase,
        "scope": "ticker" if ticker else "horizon",
        "ticker": ticker,
        "accepted_checks": accepted_checks,
        "limited_checks": limited_checks,
        "failed_checks": failed,
        "metrics": dict(metrics),
        "baseline_metrics": dict(baseline_metrics),
        "balanced_accuracy_gain": balanced_gain,
        "bootstrap": bootstrap,
        "calibration": calibration,
        "unique_dates": int(bootstrap["unique_dates"]),
        "development_evidence_available": bool(development_evidence_available),
    }
