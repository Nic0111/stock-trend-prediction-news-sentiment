"""Retrospective, horizon-specific evaluation for anchored stock/news fusion.

Fusion v3 corrects the v2 threshold defect, requires article-level sentiment
identity, and evaluates news as an incremental correction to frozen stock
probabilities.  It only writes staged artifacts and never activates runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

from pipeline.fusion_evaluation_v2 import (
    NEWS_FEATURES as LEGACY_NEWS_FEATURES,
    STOCK_FEATURES as LEGACY_STOCK_FEATURES,
    _block_bootstrap,
    _calibration_curve,
    _calibration_regression,
    _clark_west,
    _holm_adjust,
    _metric_summary,
    _paired_details,
    _purged_development_splits,
    _read_json,
    _sha256_payload,
    _enhance_features as _enhance_v2_features,
    _fit_arm as _fit_v2_arm,
)
from pipeline.fusion_model import DEFAULT_FUSION_DATASET, dataset_manifest_hash, load_fusion_dataset
from pipeline.fusion_training import _select_threshold
from pipeline.incremental_news_fusion import IncrementalNewsFusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "config" / "fusion_evaluation_v3.json"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "fusion" / "fusion-evaluation-v3.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "models" / "fusion" / "staged-fusion-v3-registry.json"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "fusion" / "staged-v3"
ACTIVE_STOCK_REGISTRY = PROJECT_ROOT / "models" / "model-registry.json"
SENTIMENT_RUNTIME_CONFIG = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
LEGACY_DIAGNOSTIC_POLICY = PROJECT_ROOT / "config" / "fusion_evaluation_v2.json"

OCCURRENCE_FEATURES = (
    "news_log_event_count",
    "news_log_article_count",
    "news_volume_surprise",
    "news_official_disclosure_share",
)
TONE_FEATURES = (
    "news_positive_impact",
    "news_negative_impact",
    "news_sentiment_surprise",
    "news_sentiment_dispersion",
    "news_conflicting_signal",
)
EVENT_FEATURES = (
    "news_event_category_financial_results_guidance_share",
    "news_event_category_dividends_distributions_share",
    "news_event_category_contracts_order_wins_share",
    "news_event_category_acquisitions_disposals_investments_share",
    "news_event_category_financing_capital_changes_share",
    "news_event_category_regulatory_legal_governance_share",
    "news_event_category_operational_product_strategy_share",
    "news_event_category_management_changes_share",
)
RELIABILITY_FEATURES = (
    "news_newest_age_days",
    "news_independent_event_count",
    "news_source_diversity",
    "news_reliability_x_stock_uncertainty",
)
FULL_FEATURES = (*OCCURRENCE_FEATURES, *TONE_FEATURES, *EVENT_FEATURES, *RELIABILITY_FEATURES)
ARM_FEATURES = {
    "occurrence_attention": OCCURRENCE_FEATURES,
    "sentiment_tone": TONE_FEATURES,
    "material_event_category": EVENT_FEATURES,
    "full_fusion": FULL_FEATURES,
}

# The raw scientific stock probability is a valid baseline, but a fair
# incremental comparison also needs a capacity-matched stock-only probability
# recalibration.  These are deliberately limited to information already
# available in the stock model output; no news columns are used.
STOCK_META_FEATURES = (
    "stock_probability_logit",
    "stock_threshold_margin",
    "stock_evidence_score",
)
_EVIDENCE_SCORE = {
    "accepted": 1.0,
    "limited": 0.5,
    "experimental": 0.25,
    "experimental_research_only": 0.25,
    "experimental research only": 0.25,
    "research_only": 0.25,
}


class FusionV3ReadinessError(ValueError):
    """Raised when the promotion-grade v3 benchmark lacks required evidence."""


def _stock_meta_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return deterministic stock-state inputs for the capacity baseline."""

    probability = _number(frame, "stock_probability_up", 0.5).clip(1e-6, 1 - 1e-6)
    logit = np.log(probability / (1.0 - probability))
    margin = _number(frame, "stock_threshold_margin", 0.0)
    evidence = (
        frame.get("stock_evidence_strength", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.replace("·", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .map(_EVIDENCE_SCORE)
        .fillna(0.0)
    )
    return np.column_stack([logit.to_numpy(dtype=float), margin.to_numpy(dtype=float), evidence.to_numpy(dtype=float)])


class _CapacityMatchedStockModel:
    """Small, reproducible stock-only meta-model for the v3 comparison.

    It recalibrates the frozen stock probability using only stock-state fields.
    The model is fitted inside each development fold and never sees news
    features.  It is a benchmark arm, not a replacement for the stock model.
    """

    def __init__(self, penalty_strength: float = 1.0) -> None:
        self.penalty_strength = float(penalty_strength)
        self.location_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.model_: LogisticRegression | None = None

    def _matrix(self, frame: pd.DataFrame, *, fitting: bool = False) -> np.ndarray:
        matrix = _stock_meta_matrix(frame)
        if fitting:
            self.location_ = np.nanmedian(matrix, axis=0)
            q75 = np.nanpercentile(matrix, 75, axis=0)
            q25 = np.nanpercentile(matrix, 25, axis=0)
            self.scale_ = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
        if self.location_ is None or self.scale_ is None:
            raise RuntimeError("CapacityMatchedStockModel has not been fitted")
        return (matrix - self.location_) / self.scale_

    def fit(self, frame: pd.DataFrame, target: Sequence[int]) -> "_CapacityMatchedStockModel":
        target_array = np.asarray(target, dtype=int)
        if len(target_array) != len(frame) or np.unique(target_array).size < 2:
            raise ValueError("Stock meta-model requires matching rows and both target classes")
        self._matrix(frame, fitting=True)
        model = LogisticRegression(
            C=1.0 / max(self.penalty_strength, 1e-9),
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        model.fit(self._matrix(frame), target_array)
        self.model_ = model
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("CapacityMatchedStockModel has not been fitted")
        return self.model_.predict_proba(self._matrix(frame))[:, 1]

    def provenance(self) -> dict[str, Any]:
        if self.model_ is None:
            raise RuntimeError("CapacityMatchedStockModel has not been fitted")
        return {
            "module": "CapacityMatchedStockModel",
            "feature_names": list(STOCK_META_FEATURES),
            "penalty_strength": self.penalty_strength,
            "coefficient": self.model_.coef_[0].astype(float).tolist(),
            "intercept": float(self.model_.intercept_[0]),
            "news_features_used": False,
        }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


SCIENTIFIC_HASH_EXCLUDED_KEYS = frozenset(
    {
        "created_at",
        "artifact_path",
        "registry_path",
        "report_path",
        "checkpoint_path",
        "_report_path",
    }
)


def _scientific_payload(value: Any) -> Any:
    """Remove location/time metadata while preserving scientific results.

    The resulting payload is stable across two runs written to different
    folders.  It deliberately retains policy, dataset and model provenance.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _scientific_payload(item)
            for key, item in value.items()
            if str(key) not in SCIENTIFIC_HASH_EXCLUDED_KEYS
            and str(key) != "scientific_result_sha256"
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_payload(item) for item in value]
    return value


def scientific_result_sha256(report: Mapping[str, Any]) -> str:
    """Return a canonical digest suitable for deterministic rerun checks."""

    payload = _canonical_json(_scientific_payload(report)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _project_reference(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_policy(path: Path = DEFAULT_POLICY) -> tuple[dict[str, Any], str]:
    policy = _read_json(path)
    if policy.get("schema_version") != "fusion-evaluation-v3":
        raise ValueError(f"Invalid fusion v3 policy: {path}")
    return policy, _sha256_payload(policy)


def _number(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return (
        pd.to_numeric(frame[name], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def _enhance_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["observation_date"] = pd.to_datetime(result["observation_date"], errors="coerce").dt.normalize()
    result["target_date"] = pd.to_datetime(result.get("target_date"), errors="coerce").dt.normalize()
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    result["target_up"] = pd.to_numeric(result["target_up"], errors="coerce")
    stock_probability = _number(result, "stock_probability_up", 0.5).clip(1e-6, 1 - 1e-6)
    stock_uncertainty = 1.0 - (2.0 * (stock_probability - 0.5).abs()).clip(0.0, 1.0)
    available = _number(result, "news_available").gt(0).astype(float)
    confidence = _number(result, "news_mean_confidence").clip(0.0, 1.0)
    newest_age = _number(result, "news_newest_age_days", 999.0).clip(lower=0.0)
    half_life = _number(result, "news_half_life_days", 1.0).clip(lower=0.25)
    recency = np.exp(-math.log(2.0) * newest_age / half_life)
    entity_confidence = _number(result, "news_mean_entity_confidence", 1.0).clip(0.0, 1.0)
    text_quality = _number(result, "news_mean_text_quality", 1.0).clip(0.0, 1.0)
    event_weight = available * confidence * entity_confidence * text_quality * recency
    if "news_event_reliability_mean" in result:
        event_weight = available * _number(
            result, "news_event_reliability_mean"
        ).clip(0.0, 1.0)
    positive = _number(result, "news_positive_share").clip(0.0, 1.0)
    neutral = _number(result, "news_neutral_share").clip(0.0, 1.0)
    negative = _number(result, "news_negative_share").clip(0.0, 1.0)
    shares = np.column_stack([negative, neutral, positive])
    dispersion = -(shares * np.log(np.clip(shares, 1e-12, 1.0))).sum(axis=1) / math.log(3.0)
    result["news_available"] = available
    result["news_positive_impact"] = (
        available * _number(result, "news_positive_event_impact").clip(0.0, 1.0)
        if "news_positive_event_impact" in result
        else event_weight * positive
    )
    result["news_negative_impact"] = (
        available * _number(result, "news_negative_event_impact").clip(0.0, 1.0)
        if "news_negative_event_impact" in result
        else event_weight * negative
    )
    result["news_sentiment_dispersion"] = available * dispersion
    result["news_conflicting_signal"] = available * ((positive > 0) & (negative > 0)).astype(float)
    result["news_log_event_count"] = available * np.log1p(_number(result, "news_independent_event_count"))
    result["news_log_article_count"] = available * np.log1p(_number(result, "news_usable_article_count"))
    result["news_independent_event_count"] = available * _number(result, "news_independent_event_count")
    result["news_source_diversity"] = available * np.log1p(_number(result, "news_unique_source_count"))
    result["news_newest_age_days"] = available * newest_age
    result["news_official_disclosure_share"] = available * _number(result, "news_official_disclosure_share")
    result["news_reliability_x_stock_uncertainty"] = event_weight * stock_uncertainty
    result["news_weighted_sentiment"] = _number(result, "news_weighted_sentiment")

    variant_keys = ["ticker", "horizon_days"]
    for name in ("news_window_days", "news_half_life_days"):
        if name in result:
            variant_keys.append(name)
    result = result.sort_values([*variant_keys, "observation_date"]).reset_index(drop=True)
    grouped = result.groupby(variant_keys, observed=True, dropna=False, sort=False)
    trailing_volume = grouped["news_log_event_count"].transform(
        lambda values: values.shift(1).rolling(252, min_periods=20).mean()
    )
    trailing_tone = grouped["news_weighted_sentiment"].transform(
        lambda values: values.shift(1).rolling(252, min_periods=20).mean()
    )
    result["news_volume_surprise"] = available * (result["news_log_event_count"] - trailing_volume).fillna(0.0)
    result["news_sentiment_surprise"] = available * (result["news_weighted_sentiment"] - trailing_tone).fillna(0.0)
    for feature in EVENT_FEATURES:
        result[feature] = available * _number(result, feature)
    return result


def _eligible(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    result = frame.loc[_number(frame, "horizon_days").eq(int(horizon))].copy()
    result = result.dropna(subset=["observation_date", "target_date", "ticker", "stock_probability_up", "target_up"])
    result = result.loc[result["target_up"].isin([0, 1])]
    return result.sort_values(["observation_date", "ticker"]).reset_index(drop=True)


def _variants(frame: pd.DataFrame, horizon: int) -> Iterable[tuple[dict[str, float], pd.DataFrame]]:
    subset = _eligible(frame, horizon)
    columns = [name for name in ("news_window_days", "news_half_life_days") if name in subset]
    if not columns:
        yield {}, subset
        return
    for key, rows in subset.groupby(columns, observed=True, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        yield {name: float(value) for name, value in zip(columns, values)}, rows.reset_index(drop=True)


def _fit_model(
    frame: pd.DataFrame,
    features: Sequence[str],
    candidate: Mapping[str, Any],
    cap: float,
) -> IncrementalNewsFusion:
    return IncrementalNewsFusion(
        features,
        penalty_strength=float(candidate["penalty_strength"]),
        l1_ratio=float(candidate["l1_ratio"]),
        adjustment_cap=float(cap),
        random_seed=42,
    ).fit(frame, frame["target_up"].astype(int))


def _fit_stock_meta_model(
    frame: pd.DataFrame,
    candidate: Mapping[str, Any],
) -> _CapacityMatchedStockModel:
    return _CapacityMatchedStockModel(
        penalty_strength=float(candidate["penalty_strength"])
    ).fit(frame, frame["target_up"].astype(int))


def _select_stock_meta_configuration(
    development: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune the stock-only calibration using the same purged folds.

    The stock-only arm is intentionally capacity-matched to the news arm, but
    its search never reads any news feature.  Only ridge candidates are used
    because the stock-state feature count is fixed and the one-standard-error
    rule prefers the more stable regularised family.
    """

    trials: list[dict[str, Any]] = []
    splits = _purged_development_splits(development)
    candidates = [
        item for item in policy["anchored_candidates"]
        if str(item.get("family")) == "ridge"
    ]
    for candidate in candidates:
        baseline_losses: list[float] = []
        fold_losses: list[float] = []
        for train_index, validation_index in splits:
            train = development.iloc[train_index].reset_index(drop=True)
            validation = development.iloc[validation_index].reset_index(drop=True)
            model = _fit_stock_meta_model(train, candidate)
            probability = model.predict(validation)
            target = validation["target_up"].astype(int).to_numpy()
            baseline = validation["stock_probability_up"].to_numpy(dtype=float)
            baseline_losses.extend(((target - baseline) ** 2).tolist())
            fold_losses.append(float(np.mean((target - probability) ** 2)))
        if not fold_losses:
            continue
        trials.append(
            {
                **dict(candidate),
                "development_brier": float(np.mean(fold_losses)),
                "development_brier_improvement": float(
                    np.mean(baseline_losses) - np.mean(fold_losses)
                ),
                "fold_brier": fold_losses,
                "development_improvement_standard_error": (
                    float(np.std(
                        [
                            np.mean((development.iloc[val]["target_up"].astype(int).to_numpy() - development.iloc[val]["stock_probability_up"].to_numpy(dtype=float)) ** 2)
                            - loss
                            for (_, val), loss in zip(splits, fold_losses)
                        ],
                        ddof=1,
                    ) / math.sqrt(len(fold_losses)))
                    if len(fold_losses) > 1
                    else 0.0
                ),
            }
        )
    if not trials:
        raise ValueError("No capacity-matched stock candidate could be fitted")
    best = min(trials, key=lambda row: row["development_brier"])
    standard_error = float(best.get("development_improvement_standard_error", 0.0))
    indistinguishable = [
        row for row in trials
        if row["development_brier"] <= best["development_brier"] + standard_error
    ]
    selected = min(
        indistinguishable,
        key=lambda row: (-float(row["penalty_strength"]), row["development_brier"]),
    )
    return {
        **selected,
        "selection_rule": "purged-development-one-standard-error-ridge-preference",
        "best_development_brier": best["development_brier"],
    }, trials


def _stock_meta_oof(
    development: pd.DataFrame,
    candidate: Mapping[str, Any],
) -> np.ndarray:
    """Generate purged OOF stock-meta probabilities for fusion fitting."""

    values = np.full(len(development), np.nan, dtype=float)
    for train_index, validation_index in _purged_development_splits(development):
        train = development.iloc[train_index].reset_index(drop=True)
        validation = development.iloc[validation_index].reset_index(drop=True)
        model = _fit_stock_meta_model(train, candidate)
        values[validation_index] = model.predict(validation)
    if np.isnan(values).any():
        # A row outside the purged folds is not silently discarded.  Use a
        # full-development fit only as a deterministic completion fallback;
        # the report records the OOF coverage so this cannot be mistaken for a
        # fully OOF sample.
        model = _fit_stock_meta_model(development, candidate)
        missing = np.isnan(values)
        values[missing] = model.predict(development.iloc[np.flatnonzero(missing)])
    return values


def _capacity_matched_frames(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Return frames whose stock base is the fair stock-only meta baseline."""

    selected, trials = _select_stock_meta_configuration(development, policy)
    development_meta = development.copy()
    development_meta["stock_probability_raw"] = development_meta["stock_probability_up"]
    development_meta["stock_probability_up"] = _stock_meta_oof(development, selected)
    model = _fit_stock_meta_model(development, selected)
    calibration_meta = calibration.copy()
    benchmark_meta = benchmark.copy()
    for frame in (calibration_meta, benchmark_meta):
        frame["stock_probability_raw"] = frame["stock_probability_up"]
    calibration_meta["stock_probability_up"] = model.predict(calibration)
    benchmark_meta["stock_probability_up"] = model.predict(benchmark)
    return development_meta, calibration_meta, benchmark_meta, selected, trials


def _select_configuration(
    development: pd.DataFrame,
    features: Sequence[str],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splits = _purged_development_splits(development)
    trials: list[dict[str, Any]] = []
    for candidate in policy["anchored_candidates"]:
        for cap in policy["news_log_odds_caps"]:
            baseline_losses: list[float] = []
            challenger_losses: list[float] = []
            fold_improvements: list[float] = []
            news_rows = 0
            for train_index, validation_index in splits:
                train = development.iloc[train_index].reset_index(drop=True)
                validation = development.iloc[validation_index].reset_index(drop=True)
                if int(train["news_available"].sum()) < max(10, len(features) + 2):
                    continue
                model = _fit_model(train, features, candidate, float(cap))
                prediction = model.predict(validation)
                target = validation["target_up"].astype(int).to_numpy()
                stock = validation["stock_probability_up"].to_numpy(dtype=float)
                baseline_losses.extend(((target - stock) ** 2).tolist())
                challenger_losses.extend(((target - prediction.probability_up) ** 2).tolist())
                fold_improvements.append(
                    float(
                        np.mean((target - stock) ** 2)
                        - np.mean((target - prediction.probability_up) ** 2)
                    )
                )
                news_rows += int(prediction.eligible_news.sum())
            if not challenger_losses:
                continue
            improvement = float(np.mean(baseline_losses) - np.mean(challenger_losses))
            trials.append(
                {
                    **dict(candidate),
                    "adjustment_cap": float(cap),
                    "development_brier_improvement": improvement,
                    "development_brier": float(np.mean(challenger_losses)),
                    "fold_brier_improvements": fold_improvements,
                    "development_improvement_standard_error": (
                        float(np.std(fold_improvements, ddof=1) / math.sqrt(len(fold_improvements)))
                        if len(fold_improvements) > 1
                        else 0.0
                    ),
                    "validation_rows": len(challenger_losses),
                    "news_present_validation_rows": news_rows,
                }
            )
    if not trials:
        raise ValueError("No anchored candidate could be fitted in purged development folds")
    best = max(trials, key=lambda row: row["development_brier_improvement"])
    # Conventional one-standard-error rule: among candidates statistically
    # indistinguishable from the best purged-fold mean, prefer the stable ridge
    # family, stronger shrinkage and the smaller allowed news adjustment.
    indistinguishable = [
        row
        for row in trials
        if row["development_brier_improvement"]
        >= best["development_brier_improvement"]
        - best["development_improvement_standard_error"]
    ]
    selected = min(
        indistinguishable,
        key=lambda row: (
            0 if row["family"] == "ridge" else 1,
            -row["penalty_strength"],
            row["adjustment_cap"],
            -row["development_brier_improvement"],
        ),
    )
    selected = {
        **selected,
        "selection_rule": "purged-development-one-standard-error-ridge-preference",
        "best_development_brier_improvement": best["development_brier_improvement"],
    }
    return selected, trials


def _select_variant(
    raw: pd.DataFrame,
    horizon: int,
    policy: Mapping[str, Any],
    features: Sequence[str] = FULL_FEATURES,
    *,
    development_end: int = 2023,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    candidates: list[tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]] = []
    for variant, rows in _variants(raw, horizon):
        dates = pd.to_datetime(rows["observation_date"])
        targets = pd.to_datetime(rows["target_date"])
        development = rows.loc[
            dates.dt.year.between(2019, int(development_end))
            & targets.lt(pd.Timestamp(f"{int(development_end) + 1}-01-01"))
        ].reset_index(drop=True)
        if development.empty:
            continue
        try:
            selected, tuning = _select_configuration(development, features, policy)
        except ValueError:
            continue
        row = {**variant, **selected}
        trials.append({**row, "candidate_trials": tuning})
        candidates.append((rows, row, selected))
    if not candidates:
        raise ValueError(f"No development-eligible v3 window for {horizon}D")
    rows, selected_row, selected_model = min(
        candidates,
        key=lambda value: (
            -value[2]["development_brier_improvement"],
            0 if value[2]["family"] == "ridge" else 1,
            value[0]["news_window_days"].iloc[0] if "news_window_days" in value[0] else 0,
        ),
    )
    return rows, selected_row, selected_model, trials


def _shared_threshold(calibration: pd.DataFrame) -> float:
    return float(
        _select_threshold(
            calibration["target_up"].astype(int),
            calibration["stock_probability_up"].to_numpy(dtype=float),
        )
    )


def _development_oof_loss_difference(
    development: pd.DataFrame,
    features: Sequence[str],
    candidate: Mapping[str, Any],
) -> np.ndarray:
    """Return news-present OOF Brier improvements for readiness power checks."""

    values: list[np.ndarray] = []
    for train_index, validation_index in _purged_development_splits(development):
        train = development.iloc[train_index].reset_index(drop=True)
        validation = development.iloc[validation_index].reset_index(drop=True)
        if int(train["news_available"].sum()) < max(10, len(features) + 2):
            continue
        model = _fit_model(
            train,
            features,
            candidate,
            float(candidate["adjustment_cap"]),
        )
        prediction = model.predict(validation)
        eligible = prediction.eligible_news
        if not eligible.any():
            continue
        target = validation["target_up"].astype(int).to_numpy()[eligible]
        stock = validation["stock_probability_up"].to_numpy(dtype=float)[eligible]
        fused = prediction.probability_up[eligible]
        values.append((target - stock) ** 2 - (target - fused) ** 2)
    return np.concatenate(values) if values else np.asarray([], dtype=float)


def _coefficient_stability(
    development: pd.DataFrame,
    features: Sequence[str],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    fold_coefficients: list[dict[str, float]] = []
    intercepts: list[float] = []
    for train_index, _ in _purged_development_splits(development):
        train = development.iloc[train_index].reset_index(drop=True)
        if int(train["news_available"].sum()) < max(10, len(features) + 2):
            continue
        model = _fit_model(
            train,
            features,
            candidate,
            float(candidate["adjustment_cap"]),
        )
        provenance = model.provenance()
        fold_coefficients.append(dict(provenance["coefficients"]))
        intercepts.append(float(provenance["news_present_intercept"]))
    feature_summary: dict[str, Any] = {}
    for feature in features:
        values = np.asarray(
            [row.get(feature, 0.0) for row in fold_coefficients], dtype=float
        )
        nonzero = values[np.abs(values) > 1e-9]
        feature_summary[feature] = {
            "mean": float(values.mean()) if len(values) else None,
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "sign_agreement": (
                float(max((nonzero > 0).mean(), (nonzero < 0).mean()))
                if len(nonzero)
                else 1.0
            ),
            "fold_values": values.tolist(),
        }
    return {
        "folds": len(fold_coefficients),
        "news_present_intercept": {
            "mean": float(np.mean(intercepts)) if intercepts else None,
            "standard_deviation": float(np.std(intercepts, ddof=1))
            if len(intercepts) > 1
            else 0.0,
            "fold_values": intercepts,
        },
        "features": feature_summary,
    }


def _subgroup_metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    stock_probability: np.ndarray,
    fusion_probability: np.ndarray,
    threshold: float,
    groups: pd.Series,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in sorted(groups.fillna("Unavailable").astype(str).unique()):
        mask = groups.fillna("Unavailable").astype(str).eq(group).to_numpy()
        if not mask.any():
            continue
        if np.unique(target[mask]).size < 2:
            output[group] = {
                "rows": int(mask.sum()),
                "status": "not_evaluated_single_target_class",
            }
            continue
        stock = _metrics(target[mask], stock_probability[mask], threshold)
        fused = _metrics(target[mask], fusion_probability[mask], threshold)
        output[group] = {
            "rows": int(mask.sum()),
            "stock_only": stock,
            "full_fusion": fused,
            "brier_improvement": float(
                stock["brier_score"] - fused["brier_score"]
            ),
            "balanced_accuracy_gain": float(
                fused["balanced_accuracy"] - stock["balanced_accuracy"]
            ),
        }
    return output


def _confidence_accuracy_curve(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    confidence = np.abs(probability - threshold)
    prediction = probability >= threshold
    quantiles = np.quantile(confidence, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(quantiles[:-1], quantiles[1:])):
        mask = (confidence >= lower) & (
            confidence <= upper if index == 3 else confidence < upper
        )
        if not mask.any():
            continue
        rows.append(
            {
                "bucket": index + 1,
                "rows": int(mask.sum()),
                "minimum_margin": float(lower),
                "maximum_margin": float(upper),
                "accuracy": float((prediction[mask] == target[mask]).mean()),
            }
        )
    return rows


def _legacy_stacker_diagnostic(
    selected_rows: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate the former broad stacker as a non-promotable diagnostic.

    Its own fitting/calibration procedure is retained to make the comparison
    recognizable, but final labels use v3's shared stock threshold and no-news
    observations are restored exactly to the stock probability.
    """

    legacy = _enhance_v2_features(selected_rows.copy())
    dates = pd.to_datetime(legacy["observation_date"])
    targets = pd.to_datetime(legacy["target_date"])
    development = legacy.loc[
        dates.dt.year.between(2019, 2023)
        & targets.lt(pd.Timestamp("2024-01-01"))
    ].reset_index(drop=True)
    calibration = legacy.loc[
        dates.dt.year.eq(2024)
        & targets.lt(pd.Timestamp("2025-01-01"))
    ].reset_index(drop=True)
    benchmark = legacy.loc[dates.dt.year.eq(2025)].reset_index(drop=True)
    policy = _read_json(LEGACY_DIAGNOSTIC_POLICY)
    fitted = _fit_v2_arm(
        development,
        calibration,
        benchmark,
        (*LEGACY_STOCK_FEATURES, *LEGACY_NEWS_FEATURES),
        policy,
    )
    probability = fitted.probability.copy()
    unavailable = _number(benchmark, "news_available").le(0).to_numpy()
    stock = benchmark["stock_probability_up"].to_numpy(dtype=float)
    probability[unavailable] = stock[unavailable]
    target = benchmark["target_up"].astype(int).to_numpy()
    return {
        "metrics": _metrics(target, probability, threshold),
        "selected_configuration": fitted.selected_parameters,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "note": (
            "Legacy 17-news-feature stacker. No-news probabilities are repaired "
            "and the v3 shared threshold is used, but this arm cannot be promoted."
        ),
    }


def _assert_metric_invariants(
    metrics: Mapping[str, Any], target: np.ndarray, prediction: np.ndarray
) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    expected = np.asarray(
        [
            [int(((target == 0) & (prediction == 0)).sum()), int(((target == 0) & (prediction == 1)).sum())],
            [int(((target == 1) & (prediction == 0)).sum()), int(((target == 1) & (prediction == 1)).sum())],
        ]
    )
    if not np.array_equal(matrix, expected):
        raise AssertionError("Confusion matrix is inconsistent with the evaluated predictions")
    expected_ba = (float(metrics["down_recall"]) + float(metrics["up_recall"])) / 2.0
    if not math.isclose(float(metrics["balanced_accuracy"]), expected_ba, abs_tol=1e-12):
        raise AssertionError("Balanced accuracy must equal mean Up/Down recall")


def _metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    metrics = _metric_summary(target, probability, threshold)
    climatology = float(np.mean(target))
    climatology_brier = float(np.mean((target - climatology) ** 2))
    metrics["brier_skill"] = (
        float(1.0 - float(metrics["brier_score"]) / climatology_brier)
        if climatology_brier > 0
        else None
    )
    metrics["climatology_brier"] = climatology_brier
    prediction = np.asarray(probability) >= float(threshold)
    _assert_metric_invariants(metrics, target, prediction.astype(int))
    return metrics


def _placebo_frame(frame: pd.DataFrame, *, mode: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    date = pd.to_datetime(result["observation_date"])
    group_columns = [result["ticker"], date.dt.year, date.dt.month]
    content_columns = [*TONE_FEATURES, *EVENT_FEATURES]
    # Content-only keeps the occurrence calendar fixed and asks whether the
    # observed tone/category matters.  Complete-event shuffling moves the
    # occurrence indicator and every derived event feature together, while
    # grouping still preserves each ticker/month's news frequency.
    columns = list(
        ("news_available", *FULL_FEATURES)
        if mode == "complete_event"
        else content_columns
    )
    generator = np.random.default_rng(seed)
    for _, index in result.groupby(group_columns, sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        if len(positions) > 1:
            shuffled = generator.permutation(positions)
            result.loc[positions, columns] = result.loc[shuffled, columns].to_numpy()
    return result


def _placebo_distribution(
    benchmark: pd.DataFrame,
    model: IncrementalNewsFusion,
    actual_improvement: float,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    target = benchmark["target_up"].astype(int).to_numpy()
    stock = benchmark["stock_probability_up"].to_numpy(dtype=float)
    distributions: dict[str, list[float]] = {"content_only": [], "complete_event": []}
    for mode_index, mode in enumerate(distributions):
        for iteration in range(iterations):
            placebo = _placebo_frame(
                benchmark, mode=mode, seed=seed + mode_index * 1_000_000 + iteration
            )
            probability = model.predict(placebo).probability_up
            distributions[mode].append(
                float(brier_score_loss(target, stock) - brier_score_loss(target, probability))
            )
    summaries = {}
    for mode, values in distributions.items():
        observed = np.asarray(values, dtype=float)
        summaries[mode] = {
            "iterations": len(values),
            "actual_percentile": float((observed < actual_improvement).mean()),
            "mean": float(observed.mean()),
            "p95": float(np.quantile(observed, 0.95)),
            "distribution": values,
        }
    return summaries


def _power_analysis(loss_difference: np.ndarray, minimum_effect: float) -> dict[str, Any]:
    values = np.asarray(loss_difference, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"power": 0.0, "rows": len(values), "minimum_effect": minimum_effect}
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    power = float(norm.cdf(abs(minimum_effect) / max(standard_error, 1e-12) - norm.ppf(0.975)))
    required = int(math.ceil((norm.ppf(0.975) + norm.ppf(0.8)) ** 2 * values.var(ddof=1) / max(minimum_effect**2, 1e-12)))
    return {
        "power": power,
        "rows": len(values),
        "minimum_effect": minimum_effect,
        "standard_error": standard_error,
        "estimated_rows_for_80_percent_power": required,
    }


def _readiness(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    policy: Mapping[str, Any],
    development_loss_difference: np.ndarray,
) -> dict[str, Any]:
    gates = policy["readiness_gates"]
    news_benchmark = benchmark.loc[benchmark["news_available"].gt(0)]
    per_company = news_benchmark.groupby("ticker").size().astype(int).to_dict()
    event_ids: dict[str, set[str]] = {}
    combined = pd.concat([development, calibration], ignore_index=True)
    for ticker, rows in combined.groupby("ticker", sort=False):
        identifiers: set[str] = set()
        for value in rows.get("news_material_event_ids", pd.Series(dtype=str)):
            try:
                parsed = json.loads(value) if isinstance(value, str) else []
            except (ValueError, TypeError):
                parsed = []
            identifiers.update(str(item) for item in parsed if str(item))
        event_ids[str(ticker)] = identifiers
    power = _power_analysis(
        development_loss_difference,
        float(gates["minimum_detectable_brier_improvement"]),
    )
    required_news_rows = int(gates["minimum_news_present_benchmark_rows"])
    required_company_news = int(
        gates["minimum_news_present_per_company_horizon"]
    )
    required_company_events = int(
        gates["minimum_unique_material_events_per_company"]
    )
    supported_tickers = sorted(
        set(combined.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
        | set(benchmark.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
    )
    company_news_gaps = {
        ticker: max(0, required_company_news - int(per_company.get(ticker, 0)))
        for ticker in supported_tickers
    }
    company_event_gaps = {
        ticker: max(
            0,
            required_company_events - len(event_ids.get(ticker, set())),
        )
        for ticker in supported_tickers
    }
    checks = {
        "minimum_news_present_benchmark_rows": len(news_benchmark)
        >= required_news_rows,
        "minimum_news_present_per_company_horizon": len(per_company) >= 10
        and min(per_company.values(), default=0) >= required_company_news,
        "minimum_unique_material_events_per_company": len(event_ids) >= 10
        and min((len(value) for value in event_ids.values()), default=0)
        >= required_company_events,
        "development_power": power["power"] >= float(gates["minimum_power"]),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "news_present_benchmark_rows": int(len(news_benchmark)),
        "news_present_by_company": per_company,
        "unique_material_events_by_company": {key: len(value) for key, value in event_ids.items()},
        "requirements": {
            "news_present_benchmark_rows": required_news_rows,
            "news_present_per_company_horizon": required_company_news,
            "unique_material_events_per_company": required_company_events,
            "minimum_power": float(gates["minimum_power"]),
        },
        "evidence_gaps": {
            "news_present_benchmark_rows": max(
                0, required_news_rows - int(len(news_benchmark))
            ),
            "news_present_by_company": company_news_gaps,
            "unique_material_events_by_company": company_event_gaps,
        },
        "power_analysis": power,
    }


def _identity(frame: pd.DataFrame, dataset_path: Path, policy_hash: str) -> dict[str, Any]:
    active_stock = _read_json(ACTIVE_STOCK_REGISTRY)
    sentiment = _read_json(SENTIMENT_RUNTIME_CONFIG)
    active_variant = str(sentiment.get("active_variant") or "")
    active_config_hash = hashlib.sha256(SENTIMENT_RUNTIME_CONFIG.read_bytes()).hexdigest() if SENTIMENT_RUNTIME_CONFIG.is_file() else ""
    checks = {
        "feature_schema_v4": set(
            frame.get("feature_schema_version", pd.Series(dtype=str))
            .dropna()
            .astype(str)
        )
        == {"point-in-time-news-features-v4"},
        "scientific_nested_oof": set(frame.get("prediction_role", pd.Series(dtype=str)).dropna().astype(str)) == {"scientific_nested_model_selection_oof"},
        "stock_run_recorded": bool(
            set(
                frame.get("stock_registry_run_id", pd.Series(dtype=str))
                .dropna()
                .astype(str)
            )
        ),
        "stock_feature_schema_recorded": bool(
            set(
                frame.get("stock_feature_schema_sha256", pd.Series(dtype=str))
                .dropna()
                .astype(str)
            )
        ),
        "stock_dataset_matches_active": set(frame.get("stock_registry_dataset_sha256", pd.Series(dtype=str)).dropna().astype(str)) == {str(active_stock.get("dataset_sha256") or "")},
        "sentiment_variant_matches_active": set(frame.get("sentiment_model_variant", pd.Series(dtype=str)).dropna().astype(str)) == {active_variant},
        "sentiment_config_matches_active": set(frame.get("sentiment_runtime_config_sha256", pd.Series(dtype=str)).dropna().astype(str)) == {active_config_hash},
        "article_level_sentiment_identity_verified": set(_number(frame, "article_sentiment_identity_verified").astype(bool)) == {True},
        "article_corpus_hash_recorded": bool(set(frame.get("article_corpus_sha256", pd.Series(dtype=str)).dropna().astype(str))),
        "aggregation_policy_hash_recorded": bool(set(frame.get("news_aggregation_policy_sha256", pd.Series(dtype=str)).dropna().astype(str))),
    }
    return {
        "dataset_sha256": dataset_manifest_hash(dataset_path),
        "policy_sha256": policy_hash,
        "active_stock_run_id": active_stock.get("run_id"),
        "stock_dataset_sha256": active_stock.get("dataset_sha256"),
        "scientific_stock_oof_run_ids": sorted(
            set(
                frame.get("stock_registry_run_id", pd.Series(dtype=str))
                .dropna()
                .astype(str)
            )
        ),
        "stock_feature_schema_sha256": sorted(
            set(
                frame.get("stock_feature_schema_sha256", pd.Series(dtype=str))
                .dropna()
                .astype(str)
            )
        ),
        "sentiment_variant": active_variant,
        "sentiment_config_sha256": active_config_hash,
        "checks": checks,
        "strict_identity_passed": all(checks.values()),
    }


def _rolling_results(
    raw: pd.DataFrame,
    horizon: int,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin in policy["rolling_origins"]:
        try:
            selected_rows, selected, candidate, _ = _select_variant(
                raw, horizon, policy, development_end=int(origin["train_end"])
            )
        except ValueError:
            continue
        dates = pd.to_datetime(selected_rows["observation_date"])
        targets = pd.to_datetime(selected_rows["target_date"])
        calibration_year = int(origin["calibration"])
        evaluation_year = int(origin["evaluation"])
        development_raw = selected_rows.loc[
            dates.dt.year.between(2019, int(origin["train_end"]))
            & targets.lt(pd.Timestamp(f"{calibration_year}-01-01"))
        ].reset_index(drop=True)
        calibration_raw = selected_rows.loc[
            dates.dt.year.eq(calibration_year)
            & targets.lt(pd.Timestamp(f"{evaluation_year}-01-01"))
        ].reset_index(drop=True)
        evaluation_raw = selected_rows.loc[dates.dt.year.eq(evaluation_year)].reset_index(drop=True)
        if min(len(development_raw), len(calibration_raw), len(evaluation_raw)) == 0:
            continue
        development, calibration, evaluation, _, _ = _capacity_matched_frames(
            development_raw,
            calibration_raw,
            evaluation_raw,
            policy,
        )
        candidate, _ = _select_configuration(development, FULL_FEATURES, policy)
        model = _fit_model(development, FULL_FEATURES, candidate, candidate["adjustment_cap"])
        probability = model.predict(evaluation).probability_up
        stock = evaluation["stock_probability_up"].to_numpy(dtype=float)
        target = evaluation["target_up"].astype(int).to_numpy()
        threshold = _shared_threshold(calibration)
        rows.append(
            {
                "evaluation_year": evaluation_year,
                "selected_configuration": selected,
                "rows": len(evaluation),
                "news_present_rows": int(evaluation["news_available"].sum()),
                "brier_improvement": float(brier_score_loss(target, stock) - brier_score_loss(target, probability)),
                "balanced_accuracy_gain": float(
                    balanced_accuracy_score(target, probability >= threshold)
                    - balanced_accuracy_score(target, stock >= threshold)
                ),
            }
        )
    return rows


def _leave_one_company_out(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ticker in sorted(benchmark["ticker"].unique()):
        train = development.loc[development["ticker"].ne(ticker)].reset_index(drop=True)
        calibrate = calibration.loc[calibration["ticker"].ne(ticker)].reset_index(drop=True)
        evaluate = benchmark.loc[benchmark["ticker"].eq(ticker)].reset_index(drop=True)
        if min(len(train), len(calibrate), len(evaluate)) == 0 or int(train["news_available"].sum()) < len(FULL_FEATURES) + 2:
            continue
        model = _fit_model(train, FULL_FEATURES, candidate, candidate["adjustment_cap"])
        probability = model.predict(evaluate).probability_up
        stock = evaluate["stock_probability_up"].to_numpy(dtype=float)
        target = evaluate["target_up"].astype(int).to_numpy()
        threshold = _shared_threshold(calibrate)
        result[str(ticker)] = {
            "rows": len(evaluate),
            "news_present_rows": int(evaluate["news_available"].sum()),
            "brier_improvement": float(brier_score_loss(target, stock) - brier_score_loss(target, probability)),
            "balanced_accuracy_gain": float(
                balanced_accuracy_score(target, probability >= threshold)
                - balanced_accuracy_score(target, stock >= threshold)
            ),
        }
    return result


def _write_registry(report: Mapping[str, Any], registry_path: Path) -> dict[str, Any]:
    horizons = {}
    for key, value in report["horizons"].items():
        path = PROJECT_ROOT / str(value["artifact_path"])
        horizons[key] = {
            "accepted": bool(value["accepted"]),
            "artifact_path": value["artifact_path"],
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "selected_window_days": value["selected_window_days"],
            "selected_half_life_days": value["selected_half_life_days"],
            "decision_threshold": value["shared_decision_threshold"],
            "runtime_label": "Fusion-supported" if value["accepted"] else "Stock-led fallback",
        }
    accepted_horizons = [int(key) for key, item in horizons.items() if item["accepted"]]
    all_horizon_accepted = bool(horizons) and len(accepted_horizons) == len(horizons)
    registry = {
        "schema_version": "fusion-registry-v5",
        "status": "partially_accepted" if any(item["accepted"] for item in horizons.values()) else "research_only",
        "overall_status": report.get("overall_status", "stock_led_fallback"),
        "overall_all_horizon_accepted": all_horizon_accepted,
        "runtime_action": "enable_fusion" if all_horizon_accepted else "stock_led_fallback",
        "claim_scope": report["claim_scope"],
        "policy_sha256": report["policy_sha256"],
        "dataset_sha256": report["dataset_sha256"],
        "provenance": report["provenance"],
        # Historical surprise features require trailing state that is not yet
        # materialised by the online NewsEvidence adapter.  Keep this explicit
        # so even an accepted research horizon cannot activate by accident.
        "runtime_feature_contract_ready": False,
        "accepted_horizons": accepted_horizons,
        "horizons": horizons,
        "report_path": _project_reference(Path(str(report["_report_path"]))),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def run_fusion_v3_preflight(
    dataset_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
    report_path: Path | None = None,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run selection-aware readiness checks without touching 2025 outcomes.

    Window/model selection and power estimation use development data only.
    Benchmark labels are never read by the preflight; 2025 is used solely for
    deterministic evidence-coverage counts required by the frozen policy.
    """

    policy, policy_hash = load_policy(policy_path)
    raw = _enhance_features(load_fusion_dataset(dataset_path))
    provenance = _identity(raw, dataset_path, policy_hash)
    if not provenance["strict_identity_passed"]:
        failed = [key for key, passed in provenance["checks"].items() if not passed]
        raise ValueError(f"Fusion v3 preflight rejected stale evidence: {failed}")
    requested = tuple(int(value) for value in (horizons or policy["supported_horizons"]))
    payload: dict[str, Any] = {
        "schema_version": "fusion-v3-readiness-preflight-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": "Retrospective evidence, 2019-2025",
        "policy_sha256": policy_hash,
        "dataset_sha256": provenance["dataset_sha256"],
        "provenance": provenance,
        "horizons": {},
    }
    for horizon in requested:
        selected_rows, selected, selected_candidate, window_trials = _select_variant(
            raw, horizon, policy
        )
        dates = pd.to_datetime(selected_rows["observation_date"])
        targets = pd.to_datetime(selected_rows["target_date"])
        development = selected_rows.loc[
            dates.dt.year.between(2019, 2023)
            & targets.lt(pd.Timestamp("2024-01-01"))
        ].reset_index(drop=True)
        calibration = selected_rows.loc[
            dates.dt.year.eq(2024) & targets.lt(pd.Timestamp("2025-01-01"))
        ].reset_index(drop=True)
        benchmark = selected_rows.loc[dates.dt.year.eq(2025)].reset_index(drop=True)
        loss_difference = _development_oof_loss_difference(
            development,
            FULL_FEATURES,
            selected_candidate,
        )
        readiness = _readiness(
            development,
            calibration,
            benchmark,
            policy,
            loss_difference,
        )
        payload["horizons"][str(horizon)] = {
            "selected_window_days": float(selected.get("news_window_days", 0.0)),
            "selected_half_life_days": float(
                selected.get("news_half_life_days", 0.0)
            ),
            "selected_configuration": selected_candidate,
            "development_rows": len(development),
            "calibration_rows": len(calibration),
            "benchmark_rows": len(benchmark),
            "window_trial_count": len(window_trials),
            "readiness": readiness,
        }
    payload["passed_horizons"] = [
        int(key)
        for key, value in payload["horizons"].items()
        if value["readiness"]["passed"]
    ]
    payload["overall_ready"] = len(payload["passed_horizons"]) == len(requested)
    payload["scientific_result_sha256"] = scientific_result_sha256(payload)
    if report_path is not None:
        _atomic_write_json(report_path, payload)
    return payload


def compare_fusion_v3_reports(
    first_path: Path,
    second_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare two final runs by canonical scientific content."""

    first = _read_json(first_path)
    second = _read_json(second_path)
    first_hash = scientific_result_sha256(first)
    second_hash = scientific_result_sha256(second)
    payload = {
        "schema_version": "fusion-v3-reproducibility-v1",
        "first_report": _project_reference(first_path),
        "second_report": _project_reference(second_path),
        "first_scientific_result_sha256": first_hash,
        "second_scientific_result_sha256": second_hash,
        "passed": first_hash == second_hash,
        "dataset_sha256_matches": first.get("dataset_sha256")
        == second.get("dataset_sha256"),
        "policy_sha256_matches": first.get("policy_sha256")
        == second.get("policy_sha256"),
        "accepted_horizons_match": first.get("accepted_horizons")
        == second.get("accepted_horizons"),
    }
    if output_path is not None:
        _atomic_write_json(output_path, payload)
    return payload


def run_fusion_evaluation_v3(
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
    checkpoint_path: Path | None = None,
    resume: bool = True,
    enforce_readiness: bool = True,
) -> dict[str, Any]:
    policy, policy_hash = load_policy(policy_path)
    raw = _enhance_features(load_fusion_dataset(dataset_path))
    provenance = _identity(raw, dataset_path, policy_hash)
    if not provenance["strict_identity_passed"]:
        failed = [key for key, passed in provenance["checks"].items() if not passed]
        raise ValueError(
            "Fusion v3 rejects stale or incomplete evidence. Rebuild the historical "
            f"news feature table; failed identity checks: {failed}"
        )
    requested = tuple(int(value) for value in (horizons or policy["supported_horizons"]))
    effective_permutations = int(
        permutation_iterations
        if permutation_iterations is not None
        else policy["permutation_iterations"]
    )
    effective_bootstraps = int(
        bootstrap_iterations
        if bootstrap_iterations is not None
        else policy["bootstrap_iterations"]
    )
    preflight: dict[str, Any] | None = None
    if enforce_readiness:
        # The expensive resampling stage is fail-closed.  A diagnostic caller
        # may explicitly pass enforce_readiness=False (the CLI exposes this as
        # --allow-unready), but that run is never promotion evidence.
        preflight = run_fusion_v3_preflight(
            dataset_path,
            policy_path=policy_path,
            horizons=requested,
        )
        if not preflight.get("overall_ready"):
            failed_horizons = {
                str(key): value.get("readiness", {}).get("evidence_gaps", {})
                for key, value in (preflight.get("horizons") or {}).items()
                if not value.get("readiness", {}).get("passed")
            }
            raise FusionV3ReadinessError(
                "Fusion v3 evaluation stopped before permutations/bootstrap: "
                "evidence readiness has not passed. "
                f"Run preflight-v3 and add lawful evidence; gaps={failed_horizons}"
            )
    checkpoint_path = checkpoint_path or report_path.with_suffix(".checkpoint.json")
    run_manifest = {
        "schema_version": "fusion-v3-run-manifest-v1",
        "dataset_sha256": provenance["dataset_sha256"],
        "policy_sha256": policy_hash,
        "requested_horizons": list(requested),
        "permutation_iterations": effective_permutations,
        "bootstrap_iterations": effective_bootstraps,
        "include_rolling_origins": bool(include_rolling_origins),
    }
    report: dict[str, Any] = {
        "schema_version": "stocksense-fusion-evaluation-v3",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": "Retrospective evidence, 2019-2025",
        "periods": policy["periods"],
        "policy_sha256": policy_hash,
        "dataset_sha256": provenance["dataset_sha256"],
        "provenance": provenance,
        "arms": ["stock_only", "stock_only_meta", "occurrence_attention", "sentiment_tone", "material_event_category", "full_fusion", "legacy_17_feature_stacker", "permuted_news_placebos"],
        "run_manifest": run_manifest,
        "checkpoint_path": _project_reference(checkpoint_path),
        "horizons": {},
    }
    if preflight is not None:
        report["readiness_preflight"] = {
            "scientific_result_sha256": preflight.get("scientific_result_sha256"),
            "passed_horizons": preflight.get("passed_horizons", []),
            "overall_ready": bool(preflight.get("overall_ready")),
        }
    if resume and checkpoint_path.is_file():
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get("run_manifest") != run_manifest:
            raise ValueError(
                "Fusion v3 checkpoint does not match the requested dataset, policy "
                "or evaluation settings. Use a new checkpoint path or remove it."
            )
        prior = checkpoint.get("report")
        if isinstance(prior, Mapping):
            report["created_at"] = str(prior.get("created_at") or report["created_at"])
            report["horizons"] = dict(prior.get("horizons") or {})
    raw_p_values: dict[int, float] = {}
    for key, value in report["horizons"].items():
        if isinstance(value, Mapping):
            raw_p_values[int(key)] = float(
                (value.get("clark_west") or {}).get("one_sided_p_value", 1.0)
            )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for horizon in requested:
        completed = report["horizons"].get(str(horizon))
        if isinstance(completed, Mapping):
            reference = str(completed.get("artifact_path") or "")
            artifact = Path(reference)
            if not artifact.is_absolute():
                artifact = PROJECT_ROOT / artifact
            if artifact.is_file():
                continue
            report["horizons"].pop(str(horizon), None)
            raw_p_values.pop(horizon, None)
        selected_rows, selected, _selected_candidate, window_trials = _select_variant(raw, horizon, policy)
        dates = pd.to_datetime(selected_rows["observation_date"])
        targets = pd.to_datetime(selected_rows["target_date"])
        development_raw = selected_rows.loc[dates.dt.year.between(2019, 2023) & targets.lt(pd.Timestamp("2024-01-01"))].reset_index(drop=True)
        calibration_raw = selected_rows.loc[dates.dt.year.eq(2024) & targets.lt(pd.Timestamp("2025-01-01"))].reset_index(drop=True)
        benchmark_raw = selected_rows.loc[dates.dt.year.eq(2025)].reset_index(drop=True)
        if min(len(development_raw), len(calibration_raw), len(benchmark_raw)) == 0:
            raise ValueError(f"Horizon {horizon} lacks a required v3 evaluation period")
        (
            development,
            calibration,
            benchmark,
            stock_meta_configuration,
            stock_meta_trials,
        ) = _capacity_matched_frames(
            development_raw,
            calibration_raw,
            benchmark_raw,
            policy,
        )
        threshold = _shared_threshold(calibration)
        target = benchmark["target_up"].astype(int).to_numpy()
        stock_probability = benchmark["stock_probability_up"].to_numpy(dtype=float)
        stock_prediction = (stock_probability >= threshold).astype(int)
        stock_metrics = _metrics(target, stock_probability, threshold)
        raw_stock_probability = benchmark["stock_probability_raw"].to_numpy(dtype=float)
        raw_stock_metrics = _metrics(target, raw_stock_probability, threshold)

        arms: dict[str, Any] = {
            "stock_only": {
                "metrics": raw_stock_metrics,
                "diagnostic_only": True,
                "note": "Raw frozen scientific stock probability; retained for transparency.",
            },
            "stock_only_meta": {
                "metrics": stock_metrics,
                "selected_configuration": stock_meta_configuration,
                "development_trials": stock_meta_trials,
                "model_provenance": _fit_stock_meta_model(
                    development_raw, stock_meta_configuration
                ).provenance(),
                "capacity_matched": True,
            },
        }
        selected_models: dict[str, IncrementalNewsFusion] = {}
        for arm_name, features in ARM_FEATURES.items():
            arm_selected, arm_trials = _select_configuration(development, features, policy)
            model = _fit_model(development, features, arm_selected, arm_selected["adjustment_cap"])
            probability = model.predict(benchmark).probability_up
            arms[arm_name] = {
                "metrics": _metrics(target, probability, threshold),
                "selected_configuration": arm_selected,
                "development_trials": arm_trials,
                "model_provenance": model.provenance(),
            }
            selected_models[arm_name] = model
        arms["legacy_17_feature_stacker"] = _legacy_stacker_diagnostic(
            selected_rows,
            threshold,
        )
        full_model = selected_models["full_fusion"]
        full_prediction = full_model.predict(benchmark)
        fusion_probability = full_prediction.probability_up
        fusion_decision = (fusion_probability >= threshold).astype(int)
        if not np.array_equal(fusion_probability[~full_prediction.eligible_news], stock_probability[~full_prediction.eligible_news]):
            raise AssertionError("No-news probabilities must exactly match stock-only")
        if not np.array_equal(fusion_decision[~full_prediction.eligible_news], stock_prediction[~full_prediction.eligible_news]):
            raise AssertionError("No-news directions must exactly match stock-only")
        fusion_metrics = arms["full_fusion"]["metrics"]
        brier_improvement = float(stock_metrics["brier_score"] - fusion_metrics["brier_score"])
        ba_gain = float(fusion_metrics["balanced_accuracy"] - stock_metrics["balanced_accuracy"])
        bootstrap = _block_bootstrap(
            benchmark,
            target,
            stock_probability,
            fusion_probability,
            stock_prediction,
            fusion_decision,
            block_size=int(policy["bootstrap_blocks"][str(horizon)]),
            iterations=effective_bootstraps,
            seed=int(policy["random_seed"]) + horizon,
        )
        placebo = _placebo_distribution(
            benchmark,
            full_model,
            brier_improvement,
            iterations=effective_permutations,
            seed=int(policy["random_seed"]) + horizon * 100_000,
        )
        clark_west = _clark_west(target, stock_probability, fusion_probability, benchmark["observation_date"])
        raw_p_values[horizon] = float(clark_west["one_sided_p_value"])
        paired = _paired_details(benchmark, target, stock_prediction, fusion_decision)
        company_gains = [float(row["gain"]) for row in paired["per_company"].values()]
        rolling = _rolling_results(raw, horizon, policy) if include_rolling_origins else []
        full_selected_configuration = arms["full_fusion"]["selected_configuration"]
        loco = _leave_one_company_out(
            development,
            calibration,
            benchmark,
            full_selected_configuration,
        )
        development_loss_difference = _development_oof_loss_difference(
            development,
            FULL_FEATURES,
            full_selected_configuration,
        )
        coefficient_stability = _coefficient_stability(
            development,
            FULL_FEATURES,
            full_selected_configuration,
        )
        relationship_state = pd.Series("available_other", index=benchmark.index)
        relationship_state.loc[_number(benchmark, "news_available").le(0)] = "unavailable"
        relationship_state.loc[
            _number(benchmark, "news_available").gt(0)
            & _number(benchmark, "stock_news_agreement", 0.5).ge(0.75)
        ] = "stock_news_aligned"
        relationship_state.loc[
            _number(benchmark, "news_available").gt(0)
            & _number(benchmark, "stock_news_agreement", 0.5).le(0.25)
        ] = "stock_news_opposed"
        relationship_state.loc[
            _number(benchmark, "news_conflicting_signal").gt(0)
        ] = "conflicting_news"
        reliability_value = _number(
            benchmark, "news_event_reliability_mean"
        ).clip(0.0, 1.0)
        reliability_state = pd.Series("unavailable", index=benchmark.index)
        news_present = _number(benchmark, "news_available").gt(0)
        reliability_state.loc[news_present & reliability_value.lt(0.40)] = "low"
        reliability_state.loc[
            news_present & reliability_value.ge(0.40) & reliability_value.lt(0.70)
        ] = "medium"
        reliability_state.loc[news_present & reliability_value.ge(0.70)] = "high"
        readiness = _readiness(development, calibration, benchmark, policy, development_loss_difference)
        gates = policy["acceptance_gates"]
        checks = {
            "readiness": readiness["passed"],
            "minimum_rows": len(benchmark) >= int(gates["minimum_benchmark_rows"]),
            "all_tickers": benchmark["ticker"].nunique() >= int(gates["minimum_tickers"]),
            "brier_improves": brier_improvement > 0,
            "brier_bootstrap_lower_above_zero": bootstrap["brier_improvement"]["lower"] > 0,
            "balanced_accuracy_gain": ba_gain >= float(gates["minimum_balanced_accuracy_gain"]),
            "balanced_accuracy_lower_non_negative": bootstrap["balanced_accuracy_gain"]["lower"] >= 0,
            "macro_f1_not_materially_worse": fusion_metrics["macro_f1"] >= stock_metrics["macro_f1"] - float(gates["maximum_macro_f1_loss"]),
            "ece_not_materially_worse": fusion_metrics["ece"] <= stock_metrics["ece"] + float(gates["maximum_ece_increase"]),
            "content_placebo": placebo["content_only"]["actual_percentile"] >= float(gates["minimum_placebo_percentile"]),
            "complete_event_placebo": placebo["complete_event"]["actual_percentile"] >= float(gates["minimum_placebo_percentile"]),
            "rolling_year_consistency": sum(item["brier_improvement"] > 0 for item in rolling) >= int(gates["minimum_positive_rolling_years"]),
            "rolling_complete": include_rolling_origins and len(rolling) == len(policy["rolling_origins"]),
            "median_company_gain_non_negative": bool(company_gains) and float(np.median(company_gains)) >= 0,
            "company_non_deterioration": sum(value >= 0 for value in company_gains) >= int(gates["minimum_non_deteriorating_companies"]),
            "no_large_company_loss": bool(company_gains) and min(company_gains) >= -float(gates["maximum_company_balanced_accuracy_loss"]),
            "strict_artifact_identity": provenance["strict_identity_passed"],
        }
        artifact_path = artifact_dir / f"fusion-v3-{provenance['dataset_sha256'][:12]}-{policy_hash[:12]}-{horizon}d.joblib"
        joblib.dump(
            {
                "schema_version": "fusion-model-artifact-v3",
                "horizon_days": horizon,
                "model": full_model,
                "shared_decision_threshold": threshold,
                "selected_configuration": full_selected_configuration,
                "selected_window_days": float(selected.get("news_window_days", 0)),
                "selected_half_life_days": float(
                    selected.get("news_half_life_days", 0)
                ),
                "policy_sha256": policy_hash,
                "dataset_sha256": provenance["dataset_sha256"],
                "provenance": provenance,
            },
            artifact_path,
        )
        report["horizons"][str(horizon)] = {
            "benchmark_rows": len(benchmark),
            "news_present_rows": int(benchmark["news_available"].sum()),
            "ticker_count": int(benchmark["ticker"].nunique()),
            "primary_comparison_baseline": "stock_only_meta",
            "raw_stock_only_metrics": raw_stock_metrics,
            "capacity_matched_stock_only_metrics": stock_metrics,
            "capacity_matched_stock_configuration": stock_meta_configuration,
            "selected_window_days": float(selected.get("news_window_days", 0)),
            "selected_half_life_days": float(selected.get("news_half_life_days", 0)),
            "shared_decision_threshold": threshold,
            "selected_configuration": full_selected_configuration,
            "window_and_model_trials": window_trials,
            "arms": arms,
            "brier_improvement": brier_improvement,
            "balanced_accuracy_gain": ba_gain,
            "bootstrap": bootstrap,
            "clark_west": clark_west,
            "placebo": placebo,
            "calibration": {
                "stock_only": {"curve": _calibration_curve(target, stock_probability), **_calibration_regression(target, stock_probability)},
                "full_fusion": {"curve": _calibration_curve(target, fusion_probability), **_calibration_regression(target, fusion_probability)},
            },
            "error_transitions": paired["error_transitions"],
            "per_company": paired["per_company"],
            "rolling_origins": rolling,
            "leave_one_company_out": loco,
            "coefficient_stability": coefficient_stability,
            "subgroups": {
                "news_coverage": _subgroup_metrics(
                    benchmark,
                    target,
                    stock_probability,
                    fusion_probability,
                    threshold,
                    benchmark.get(
                        "news_coverage",
                        pd.Series("Unavailable", index=benchmark.index),
                    ),
                ),
                "agreement_conflict": _subgroup_metrics(
                    benchmark,
                    target,
                    stock_probability,
                    fusion_probability,
                    threshold,
                    relationship_state,
                ),
                "news_reliability": _subgroup_metrics(
                    benchmark,
                    target,
                    stock_probability,
                    fusion_probability,
                    threshold,
                    reliability_state,
                ),
            },
            "confidence_accuracy": {
                "stock_only": _confidence_accuracy_curve(
                    target, stock_probability, threshold
                ),
                "full_fusion": _confidence_accuracy_curve(
                    target, fusion_probability, threshold
                ),
            },
            "readiness": readiness,
            "acceptance_checks": checks,
            "artifact_path": _project_reference(artifact_path),
            "runtime_label": "Stock-led fallback",
            "accepted": False,
        }
        _atomic_write_json(
            checkpoint_path,
            {
                "schema_version": "fusion-v3-checkpoint-v1",
                "status": "in_progress",
                "run_manifest": run_manifest,
                "completed_horizons": sorted(int(key) for key in report["horizons"]),
                "report": report,
            },
        )
    adjusted = _holm_adjust(raw_p_values)
    for horizon, adjusted_p in adjusted.items():
        result = report["horizons"][str(horizon)]
        result["holm_adjusted_clark_west_p_value"] = adjusted_p
        result["acceptance_checks"]["holm_adjusted_significance"] = adjusted_p < float(policy["acceptance_gates"]["holm_alpha"])
        result["accepted"] = all(result["acceptance_checks"].values())
        result["runtime_label"] = "Fusion-supported" if result["accepted"] else "Stock-led fallback"
    report["accepted_horizons"] = [int(key) for key, value in report["horizons"].items() if value["accepted"]]
    report["overall_status"] = "horizon_specific_acceptance" if report["accepted_horizons"] else "stock_led_fallback"
    report["scientific_result_sha256"] = scientific_result_sha256(report)
    report["_report_path"] = str(report_path)
    report["registry_path"] = _project_reference(registry_path)
    # Write the registry and report from the same complete metadata object.
    # Previously the report was serialized before registry_path/registry_status
    # were added, leaving the standalone report without its registry pointer.
    registry = _write_registry(report, registry_path)
    report["registry_status"] = registry["status"]
    _atomic_write_json(
        report_path,
        {key: value for key, value in report.items() if key != "_report_path"},
    )
    report.pop("_report_path", None)
    _atomic_write_json(
        checkpoint_path,
        {
            "schema_version": "fusion-v3-checkpoint-v1",
            "status": "complete",
            "run_manifest": run_manifest,
            "completed_horizons": sorted(int(key) for key in report["horizons"]),
            "scientific_result_sha256": report["scientific_result_sha256"],
            "report_path": _project_reference(report_path),
            "registry_path": _project_reference(registry_path),
        },
    )
    return report
