"""Experimental stock/news late-fusion model contracts and training scaffold.

This module deliberately sits beside the existing stock-model pipeline.  It does
not modify or replace any stock artifact.  A fusion artifact can only be trained
when a point-in-time historical fusion feature set is supplied; runtime data
retrievals are not silently used as a training set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_SUPPORTED_HORIZONS = (1, 5, 10)
FUSION_REGISTRY_SCHEMA_VERSION = "fusion-registry-v1"
DEFAULT_FUSION_DATASET = PROJECT_ROOT / "data" / "fusion" / "fusion-features.parquet"
DEFAULT_FUSION_REGISTRY = PROJECT_ROOT / "models" / "fusion" / "fusion-registry.json"

# Keep this list stable: it is part of the persisted artifact contract and is
# intentionally limited to leakage-safe, point-in-time features.
FUSION_FEATURE_NAMES = (
    "stock_probability_up",
    "stock_threshold_margin",
    "stock_evidence_strength",
    "news_weighted_sentiment",
    "news_positive_share",
    "news_neutral_share",
    "news_negative_share",
    "news_mean_confidence",
    "news_max_confidence",
    "news_usable_article_count",
    "news_unique_source_count",
    "news_newest_age_days",
    "news_mean_age_days",
    "news_coverage_score",
    "news_available",
    "stock_news_agreement",
)
FUSION_REQUIRED_COLUMNS = ("ticker", "observation_date", "horizon_days", "target_up") + FUSION_FEATURE_NAMES
# Missing news is an explicit runtime state, not a neutral sentiment label.
# These fields are therefore imputed to zero while ``news_available`` remains
# available to the model as a separate indicator.
NEWS_FUSION_FEATURES = {
    "news_weighted_sentiment",
    "news_positive_share",
    "news_neutral_share",
    "news_negative_share",
    "news_mean_confidence",
    "news_max_confidence",
    "news_usable_article_count",
    "news_unique_source_count",
    "news_newest_age_days",
    "news_mean_age_days",
    "news_coverage_score",
    "news_available",
    "stock_news_agreement",
}
EVIDENCE_STRENGTH_SCORES = {
    "strong": 1.0,
    "moderate": 0.66,
    "limited": 0.33,
    "unavailable": 0.0,
    "experimental": 0.33,
}


class FusionDatasetNotReadyError(FileNotFoundError):
    """Raised when training is requested without a valid historical dataset."""


@dataclass(frozen=True)
class FusionReadiness:
    """Machine-readable readiness report for a point-in-time fusion dataset."""

    ready: bool
    dataset_path: str
    row_count: int = 0
    ticker_count: int = 0
    horizon_counts: dict[str, int] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return "ready" if self.ready else "; ".join(self.errors) or "dataset is not ready"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["errors"] = list(self.errors)
        value["reason"] = self.reason
        return value


@dataclass(frozen=True)
class PromotionThresholds:
    """Frozen acceptance gates for promoting a horizon-specific fusion model."""

    min_balanced_accuracy_gain: float = 0.02
    min_mcc: float = 0.0
    max_macro_f1_regression: float = 0.0
    require_brier_improvement: bool = True
    require_log_loss_improvement: bool = True
    max_ece_regression: float = 0.01
    min_up_recall: float = 0.35
    min_down_recall: float = 0.35
    min_predicted_up_rate: float = 0.20
    max_predicted_up_rate: float = 0.80
    min_positive_mcc_folds: int = 4
    max_company_balanced_accuracy_loss: float = 0.05


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "checks": dict(self.checks), "reasons": list(self.reasons)}


@dataclass
class FusionModelArtifact:
    """Serializable, horizon-aware logistic late-fusion artifact."""

    horizon_days: int
    model_version: str
    feature_names: tuple[str, ...]
    estimator: Any
    threshold: float = 0.5
    calibrator: Any | None = None
    validation_status: str = "experimental"
    dataset_manifest_sha256: str = ""
    trained_at: str = ""
    development_start: str = ""
    development_end: str = ""
    calibration_start: str = ""
    calibration_end: str = ""
    benchmark_start: str = ""
    benchmark_end: str = ""
    shadow_start: str = ""
    shadow_end: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Compatibility alias used by callers that handle stock artifacts."""

        return self.feature_names

    def __post_init__(self) -> None:
        if self.horizon_days not in FUSION_SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported fusion horizon: {self.horizon_days}")
        if tuple(self.feature_names) != tuple(FUSION_FEATURE_NAMES):
            raise ValueError("Fusion artifact feature schema does not match FUSION_FEATURE_NAMES")
        if self.validation_status not in {"accepted", "limited", "experimental", "unavailable"}:
            raise ValueError("validation_status must be accepted, limited, experimental, or unavailable")

    def to_dict(self) -> dict[str, Any]:
        """Return artifact metadata without serializing the estimator itself."""
        return {
            "horizon_days": self.horizon_days,
            "model_version": self.model_version,
            "feature_names": list(self.feature_names),
            "threshold": self.threshold,
            "validation_status": self.validation_status,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "trained_at": self.trained_at,
            "development_start": self.development_start,
            "development_end": self.development_end,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "benchmark_start": self.benchmark_start,
            "benchmark_end": self.benchmark_end,
            "shadow_start": self.shadow_start,
            "shadow_end": self.shadow_end,
            "metrics": dict(self.metrics),
            "baseline_metrics": dict(self.baseline_metrics),
        }

    def _select_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Fusion input is missing columns: {missing}")
        return prepare_fusion_features(frame, self.feature_names)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        selected = self._select_features(frame)
        raw = np.asarray(self.estimator.predict_proba(selected)[:, 1], dtype=float)
        if self.calibrator is not None:
            if hasattr(self.calibrator, "predict_proba"):
                raw = np.asarray(self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
            elif hasattr(self.calibrator, "predict"):
                raw = np.asarray(self.calibrator.predict(raw), dtype=float)
            elif callable(self.calibrator):
                raw = np.asarray(self.calibrator(raw), dtype=float)
            else:
                raise TypeError("Fusion calibrator must expose predict_proba, predict, or be callable")
        raw = np.clip(raw, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - raw, raw])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame)[:, 1] >= float(self.threshold)).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FusionModelArtifact":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected FusionModelArtifact at {path}")
        return value


@dataclass
class FusionRegistry:
    """JSON registry describing which per-horizon fusion artifacts are usable."""

    created_at: str
    dataset_manifest_sha256: str
    horizons: dict[str, dict[str, Any]]
    schema_version: str = FUSION_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FUSION_REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported fusion registry schema: {self.schema_version}")

    def entry(self, horizon_days: int) -> dict[str, Any] | None:
        return self.horizons.get(str(horizon_days))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "horizons": self.horizons,
        }

    @classmethod
    def read(cls, path: Path) -> "FusionRegistry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            created_at=str(payload["created_at"]),
            dataset_manifest_sha256=str(payload.get("dataset_manifest_sha256", "")),
            horizons=dict(payload.get("horizons", {})),
            schema_version=str(payload.get("schema_version", "")),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dataset_manifest_hash(path: Path) -> str:
    """Return a content hash used to invalidate incompatible registry entries."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fusion_dataset(path: Path) -> pd.DataFrame:
    """Load a normalized fusion feature file without coupling to one storage engine."""

    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported fusion dataset format: {suffix or '<none>'}; use CSV, JSONL, or Parquet")


def prepare_fusion_features(
    frame: pd.DataFrame,
    feature_names: Sequence[str] = FUSION_FEATURE_NAMES,
) -> pd.DataFrame:
    """Apply the deterministic missing-news contract before inference/training.

    Stock features must be present because the Stage 2 model is a late fusion
    challenger built on top of a stock prediction.  News features may be
    absent for a ticker/date; in that case zero values plus ``news_available``
    let the model learn the difference between no coverage and neutral tone.
    """
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Fusion input is missing columns: {missing}")
    selected = frame.loc[:, list(feature_names)].copy()
    for name in feature_names:
        if name == "stock_evidence_strength":
            selected[name] = selected[name].map(
                lambda value: EVIDENCE_STRENGTH_SCORES.get(str(value).strip().casefold(), value)
            )
        selected[name] = pd.to_numeric(selected[name], errors="coerce")
        if name in NEWS_FUSION_FEATURES:
            selected[name] = selected[name].fillna(0.0)
    stock_features = [name for name in feature_names if name not in NEWS_FUSION_FEATURES]
    if selected[stock_features].isna().any().any():
        missing_stock = [name for name in stock_features if selected[name].isna().any()]
        raise ValueError(f"Fusion stock features contain missing values: {missing_stock}")
    return selected.astype(float)


def check_fusion_dataset_readiness(
    dataset_path: Path = DEFAULT_FUSION_DATASET,
    *,
    supported_tickers: Sequence[str] | None = None,
    min_rows_per_horizon: int = 100,
    minimum_start_year: int = 2019,
    required_end_year: int = 2026,
) -> FusionReadiness:
    """Validate the historical, point-in-time dataset before training.

    The default dataset is intentionally absent in this repository.  Returning a
    structured report makes status pages and CLI tooling actionable; training
    converts the report into a clear ``FusionDatasetNotReadyError``.
    """

    path = Path(dataset_path)
    if not path.exists():
        return FusionReadiness(False, str(path), errors=(f"historical fusion dataset not found: {path}",))
    try:
        frame = load_fusion_dataset(path)
    except Exception as exc:  # pragma: no cover - provider/engine-specific details
        return FusionReadiness(False, str(path), errors=(f"could not read historical fusion dataset: {exc}",))

    errors: list[str] = []
    missing = [name for name in FUSION_REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        errors.append(f"dataset is missing required columns: {missing}")
    if frame.empty:
        errors.append("historical fusion dataset is empty")
    row_count = len(frame)
    tickers = frame["ticker"].dropna().astype(str).unique().tolist() if "ticker" in frame else []
    horizon_counts: dict[str, int] = {}
    if "horizon_days" in frame:
        horizon_values = pd.to_numeric(frame["horizon_days"], errors="coerce")
        if horizon_values.isna().any() or not set(horizon_values.dropna().astype(int).unique()).issubset(FUSION_SUPPORTED_HORIZONS):
            errors.append("horizon_days must contain only 1, 5, or 10")
        horizon_counts = {str(h): int((horizon_values == h).sum()) for h in FUSION_SUPPORTED_HORIZONS}
        for horizon, count in horizon_counts.items():
            if count < min_rows_per_horizon:
                errors.append(f"horizon {horizon} has {count} rows; requires at least {min_rows_per_horizon}")
    if supported_tickers:
        missing_tickers = sorted(set(supported_tickers) - set(tickers))
        if missing_tickers:
            errors.append(f"dataset is missing supported tickers: {missing_tickers}")
    if "observation_date" in frame:
        dates = pd.to_datetime(frame["observation_date"], errors="coerce")
        if dates.isna().any():
            errors.append("observation_date contains invalid values")
        else:
            if dates.min().year > minimum_start_year:
                errors.append(f"dataset starts at {dates.min().date()}, before {minimum_start_year} coverage is preferred")
            if dates.max().year < required_end_year:
                errors.append(f"dataset ends at {dates.max().date()}, requires coverage through {required_end_year}")
    if "target_up" in frame:
        target = pd.to_numeric(frame["target_up"], errors="coerce")
        if target.isna().any() or not set(target.dropna().unique()).issubset({0, 1}):
            errors.append("target_up must contain only binary 0/1 values")

    return FusionReadiness(not errors, str(path), row_count, len(tickers), horizon_counts, tuple(errors))


def _raise_if_not_ready(readiness: FusionReadiness) -> None:
    if not readiness.ready:
        raise FusionDatasetNotReadyError(
            "Fusion training is unavailable until a historical point-in-time dataset is supplied: "
            + readiness.reason
        )


def train_fusion_model(
    frame: pd.DataFrame,
    horizon_days: int,
    *,
    model_version: str = "fusion-logistic-experimental",
    threshold: float = 0.5,
    dataset_manifest_sha256: str = "",
    **logistic_kwargs: Any,
) -> FusionModelArtifact:
    """Fit one experimental logistic stacker from a validated feature frame.

    Chronological splitting, out-of-fold stock probabilities, and probability
    calibration belong in the caller's evaluation pipeline.  This low-level
    function intentionally only fits the estimator and marks it experimental.
    """

    if horizon_days not in FUSION_SUPPORTED_HORIZONS:
        raise ValueError(f"Unsupported fusion horizon: {horizon_days}")
    missing = [name for name in (*FUSION_FEATURE_NAMES, "target_up") if name not in frame.columns]
    if missing:
        raise ValueError(f"Fusion training frame is missing columns: {missing}")
    target = pd.to_numeric(frame["target_up"], errors="coerce")
    if len(frame) < 2 or target.isna().any() or target.nunique() < 2:
        raise ValueError("Fusion training requires at least two rows and both target classes")
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", **logistic_kwargs)),
        ]
    )
    estimator.fit(prepare_fusion_features(frame, FUSION_FEATURE_NAMES), target.astype(int))
    return FusionModelArtifact(
        horizon_days=horizon_days,
        model_version=model_version,
        feature_names=tuple(FUSION_FEATURE_NAMES),
        estimator=estimator,
        threshold=threshold,
        dataset_manifest_sha256=dataset_manifest_sha256,
        trained_at=_now_iso(),
    )


def train_fusion_models(
    dataset_path: Path = DEFAULT_FUSION_DATASET,
    *,
    registry_path: Path | None = DEFAULT_FUSION_REGISTRY,
    supported_tickers: Sequence[str] | None = None,
    min_rows_per_horizon: int = 100,
) -> dict[int, FusionModelArtifact]:
    """Training entry point; fail clearly when no historical dataset exists."""

    readiness = check_fusion_dataset_readiness(
        dataset_path,
        supported_tickers=supported_tickers,
        min_rows_per_horizon=min_rows_per_horizon,
    )
    _raise_if_not_ready(readiness)
    frame = load_fusion_dataset(Path(dataset_path))
    manifest_hash = dataset_manifest_hash(Path(dataset_path))
    artifacts: dict[int, FusionModelArtifact] = {}
    registry_entries: dict[str, dict[str, Any]] = {}
    for horizon in FUSION_SUPPORTED_HORIZONS:
        subset = frame.loc[frame["horizon_days"].astype(int) == horizon].copy()
        artifact = train_fusion_model(subset, horizon, dataset_manifest_sha256=manifest_hash)
        artifacts[horizon] = artifact
        if registry_path is not None:
            artifact_path = Path(registry_path).parent / f"fusion-{horizon}d-{artifact.model_version}.joblib"
            artifact.save(artifact_path)
            try:
                artifact_reference = str(artifact_path.relative_to(PROJECT_ROOT))
            except ValueError:
                artifact_reference = str(artifact_path)
            registry_entries[str(horizon)] = {
                "artifact_path": artifact_reference,
                "model_version": artifact.model_version,
                "validation_status": artifact.validation_status,
                "feature_names": list(artifact.feature_names),
                "metrics": artifact.metrics,
            }
    if registry_path is not None:
        registry = FusionRegistry(_now_iso(), manifest_hash, registry_entries)
        registry.write(Path(registry_path))
    return artifacts


# ``fit`` is a convenient alias for notebooks and future CLI wiring while the
# explicit ``train`` name remains the public pipeline entry point.
fit_fusion_model = train_fusion_model


def evaluate_promotion_gate(
    challenger_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    *,
    fold_mcc: Sequence[float] | None = None,
    bootstrap_balanced_accuracy_gain_ci: tuple[float, float] | None = None,
    per_company_balanced_accuracy_gain: Mapping[str, float] | None = None,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> PromotionDecision:
    """Apply the frozen acceptance gates without selecting thresholds implicitly."""

    def metric(values: Mapping[str, float], key: str) -> float:
        aliases = {
            "brier_score": ("brier_score", "brier"),
            "log_loss": ("log_loss", "logloss"),
            "ece": ("ece", "expected_calibration_error"),
            "up_recall": ("up_recall", "recall_up"),
            "down_recall": ("down_recall", "recall_down"),
        }.get(key, (key,))
        for candidate in aliases:
            if candidate in values:
                return float(values[candidate])
        raise ValueError(f"Promotion metrics missing required field: {key}")

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    checks["balanced_accuracy_gain"] = metric(challenger_metrics, "balanced_accuracy") - metric(baseline_metrics, "balanced_accuracy") >= thresholds.min_balanced_accuracy_gain
    checks["positive_mcc"] = metric(challenger_metrics, "mcc") > thresholds.min_mcc
    checks["macro_f1_not_worse"] = metric(challenger_metrics, "macro_f1") >= metric(baseline_metrics, "macro_f1") - thresholds.max_macro_f1_regression
    checks["brier_improved"] = (not thresholds.require_brier_improvement) or metric(challenger_metrics, "brier_score") < metric(baseline_metrics, "brier_score")
    checks["log_loss_improved"] = (not thresholds.require_log_loss_improvement) or metric(challenger_metrics, "log_loss") < metric(baseline_metrics, "log_loss")
    checks["ece_within_tolerance"] = metric(challenger_metrics, "ece") <= metric(baseline_metrics, "ece") + thresholds.max_ece_regression
    checks["up_recall"] = metric(challenger_metrics, "up_recall") >= thresholds.min_up_recall
    checks["down_recall"] = metric(challenger_metrics, "down_recall") >= thresholds.min_down_recall
    up_rate = metric(challenger_metrics, "predicted_up_rate")
    checks["prediction_mix"] = thresholds.min_predicted_up_rate <= up_rate <= thresholds.max_predicted_up_rate
    checks["positive_mcc_folds"] = fold_mcc is not None and sum(float(value) > 0 for value in fold_mcc) >= thresholds.min_positive_mcc_folds
    checks["bootstrap_gain_ci"] = bootstrap_balanced_accuracy_gain_ci is not None and float(bootstrap_balanced_accuracy_gain_ci[0]) >= 0
    checks["company_stability"] = per_company_balanced_accuracy_gain is not None and bool(per_company_balanced_accuracy_gain) and min(float(value) for value in per_company_balanced_accuracy_gain.values()) >= -thresholds.max_company_balanced_accuracy_loss
    for name, passed in checks.items():
        if not passed:
            reasons.append(f"promotion gate failed: {name}")
    return PromotionDecision(not reasons, checks, tuple(reasons))
