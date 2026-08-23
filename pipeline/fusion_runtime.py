"""Runtime activation of accepted stock/news fusion artifacts."""

from __future__ import annotations

import hashlib
import json
import joblib
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.download_financial_news_data import NewsContextResult
from pipeline.evidence_synthesis import (
    CombinedOutlookResult,
    _context,
    _cutoff_from_news,
    _relationship,
    _tone,
    build_stock_led_outlook,
)
from pipeline.fusion_model import (
    DEFAULT_FUSION_REGISTRY,
    EVIDENCE_STRENGTH_SCORES,
    FusionModelArtifact,
    FusionRegistry,
)
from pipeline.predict_stock_trend import PredictionResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# v2 retains its conservative Medium/High runtime policy.  Fusion v3 was
# explicitly trained on all target-complete rows, including Low coverage, and
# uses the deterministic reliability features to limit the news correction.
NUMERICAL_FUSION_COVERAGE = {"medium", "high"}
ACTIVE_STOCK_REGISTRY = PROJECT_ROOT / "models" / "model-registry.json"
SENTIMENT_RUNTIME_CONFIG = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"


def fusion_registry_signature(registry_path: Path = DEFAULT_FUSION_REGISTRY) -> str:
    try:
        return hashlib.sha256(Path(registry_path).read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _v4_registry_guard(payload: dict[str, Any], horizon_days: int) -> str | None:
    """Fail closed for staged v2 studies before any artifact is deserialized."""

    if payload.get("schema_version") != "fusion-registry-v4":
        return None
    if not bool(payload.get("overall_all_horizon_accepted")):
        return "fusion-evaluation-v2 did not pass the strict all-horizon acceptance policy"
    if str(payload.get("status") or "").casefold() != "accepted":
        return "fusion-evaluation-v2 registry is not marked accepted"
    entry = dict((payload.get("horizons") or {}).get(str(horizon_days)) or {})
    if not bool(entry.get("accepted")):
        return f"fusion-evaluation-v2 horizon {horizon_days} is not accepted"

    provenance = dict(payload.get("provenance") or {})
    if not bool(provenance.get("strict_identity_passed")):
        return "fusion-evaluation-v2 artifact provenance did not pass strict identity checks"
    active_stock = _read_json(ACTIVE_STOCK_REGISTRY)
    if str(active_stock.get("run_id") or "") != str(
        provenance.get("active_stock_run_id") or ""
    ):
        return "active stock run no longer matches the fusion-evaluation-v2 registry"
    if str(active_stock.get("dataset_sha256") or "") != str(
        provenance.get("stock_dataset_sha256") or ""
    ):
        return "active stock dataset no longer matches the fusion-evaluation-v2 registry"
    sentiment_hash = (
        hashlib.sha256(SENTIMENT_RUNTIME_CONFIG.read_bytes()).hexdigest()
        if SENTIMENT_RUNTIME_CONFIG.is_file()
        else ""
    )
    if sentiment_hash != str(provenance.get("sentiment_config_sha256") or ""):
        return "active sentiment configuration no longer matches the fusion-evaluation-v2 registry"
    # V2 includes rolling news-surprise features.  Loading is deliberately
    # withheld until the reviewed promotion command materialises the matching
    # runtime baseline state.  This prevents silently substituting zeros.
    return (
        "fusion-evaluation-v2 passed research gates but has not been reviewed for "
        "runtime baseline-state activation"
    )


def _v5_registry_guard(payload: dict[str, Any], horizon_days: int) -> str | None:
    """Validate horizon-specific v3 activation before deserializing artifacts."""

    if payload.get("schema_version") != "fusion-registry-v5":
        return None
    entry = dict((payload.get("horizons") or {}).get(str(horizon_days)) or {})
    override = dict(payload.get("acceptance_override") or {})
    override_enabled = bool(override.get("active")) and bool(
        override.get("operator_authorized")
    ) and int(horizon_days) in [
        int(value) for value in payload.get("runtime_enabled_horizons")
        or payload.get("accepted_horizons")
        or []
    ]
    if not bool(entry.get("accepted")) and not override_enabled:
        return f"fusion-evaluation-v3 horizon {horizon_days} is stock-led fallback"
    provenance = dict(payload.get("provenance") or {})
    if not bool(provenance.get("strict_identity_passed")):
        return "fusion-evaluation-v3 artifact identity checks did not pass"
    active_stock = _read_json(ACTIVE_STOCK_REGISTRY)
    if str(active_stock.get("run_id") or "") != str(
        provenance.get("active_stock_run_id") or ""
    ):
        return "active stock run no longer matches fusion-evaluation-v3"
    if str(active_stock.get("dataset_sha256") or "") != str(
        provenance.get("stock_dataset_sha256") or ""
    ):
        return "active stock dataset no longer matches fusion-evaluation-v3"
    sentiment_hash = (
        hashlib.sha256(SENTIMENT_RUNTIME_CONFIG.read_bytes()).hexdigest()
        if SENTIMENT_RUNTIME_CONFIG.is_file()
        else ""
    )
    if sentiment_hash != str(provenance.get("sentiment_config_sha256") or ""):
        return "active sentiment configuration no longer matches fusion-evaluation-v3"
    if not bool(payload.get("runtime_feature_contract_ready")):
        return (
            "fusion-evaluation-v3 passed research gates but its runtime surprise "
            "baseline state has not been materialised"
        )
    state_reference = str(payload.get("runtime_state_path") or "").strip()
    if not state_reference:
        return "fusion-evaluation-v3 has no runtime surprise state path"
    state_path = Path(state_reference)
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path
    if not state_path.is_file():
        return "fusion-evaluation-v3 runtime surprise state is missing"
    state = _read_json(state_path)
    if str(state.get("dataset_sha256") or "") != str(
        payload.get("dataset_sha256") or ""
    ):
        return "fusion-evaluation-v3 runtime state dataset does not match the registry"
    expected_state_hash = str(payload.get("runtime_state_sha256") or "").strip()
    if expected_state_hash and hashlib.sha256(state_path.read_bytes()).hexdigest() != expected_state_hash:
        return "fusion-evaluation-v3 runtime surprise state failed its integrity check"
    return None


class _V3FusionArtifactAdapter:
    """Adapt a staged v3 joblib payload to the runtime artifact interface.

    Fusion v3 deliberately stores the anchored ``IncrementalNewsFusion``
    estimator in a small dictionary rather than pretending it is the legacy
    ``FusionModelArtifact``.  Keeping this adapter at the runtime boundary
    prevents the v3 registry from being mis-read by the v2 loader while
    preserving the exact no-news fallback and component explanation API.
    """

    validation_status = "accepted"

    def __init__(self, payload: dict[str, Any]) -> None:
        self._model = payload["model"]
        self.horizon_days = int(payload["horizon_days"])
        self.threshold = float(payload["shared_decision_threshold"])
        self.model_version = "fusion-v3-anchored-residual"
        self.fusion_method = "anchored_incremental_news"

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = self._model.predict(frame)
        probability = np.asarray(prediction.probability_up, dtype=float)
        return np.column_stack([1.0 - probability, probability])

    def component_contributions(self, frame: pd.DataFrame) -> dict[str, float]:
        if len(frame) != 1:
            raise ValueError("Fusion component contributions require exactly one row")
        full_probability = float(self.predict_proba(frame)[0, 1])
        stock_frame = frame.copy()
        # The anchored estimator's public invariant is exact stock fallback
        # whenever news_available is zero; no hand-maintained feature list is
        # needed here and future v3 feature additions remain safe.
        stock_frame["news_available"] = 0.0
        stock_probability = float(self.predict_proba(stock_frame)[0, 1])
        return {
            "base_probability_up": 0.5,
            "stock_component_probability_up": stock_probability,
            "stock_contribution_probability_points": (stock_probability - 0.5) * 100.0,
            "news_adjustment_probability_points": (full_probability - stock_probability) * 100.0,
            "aggregate_probability_up": full_probability,
        }


@lru_cache(maxsize=12)
def _load_artifact_cached(
    horizon_days: int, registry_path_text: str, registry_signature: str
) -> tuple[Any | None, str | None]:
    del registry_signature  # included solely to invalidate the cache
    registry_path = Path(registry_path_text)
    if not registry_path.exists():
        return None, "accepted fusion registry is unavailable"
    raw_registry = _read_json(registry_path)
    v5_reason = _v5_registry_guard(raw_registry, horizon_days)
    if v5_reason is not None:
        return None, v5_reason
    if raw_registry.get("schema_version") == "fusion-registry-v5":
        entry = dict((raw_registry.get("horizons") or {}).get(str(horizon_days)) or {})
        reference = str(entry.get("artifact_path") or "").strip()
        if not reference:
            return None, "accepted Fusion v3 registry entry has no artifact path"
        artifact_path = Path(reference)
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        if not artifact_path.is_file():
            return None, f"accepted Fusion v3 artifact is missing: {artifact_path}"
        expected_hash = str(entry.get("artifact_sha256") or "").strip()
        if not expected_hash:
            return None, "accepted Fusion v3 registry entry has no artifact integrity hash"
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != expected_hash:
            return None, "accepted Fusion v3 artifact failed its integrity check"
        try:
            payload = joblib.load(artifact_path)
        except (OSError, ValueError, TypeError) as exc:
            return None, f"accepted Fusion v3 artifact could not be loaded: {exc}"
        if not isinstance(payload, dict) or payload.get("schema_version") != "fusion-model-artifact-v3":
            return None, "accepted Fusion v3 artifact has an invalid schema"
        if payload.get("model") is None:
            return None, "accepted Fusion v3 artifact has no anchored model"
        try:
            return _V3FusionArtifactAdapter(payload), None
        except (KeyError, TypeError, ValueError) as exc:
            return None, f"accepted Fusion v3 artifact is incomplete: {exc}"
    v4_reason = _v4_registry_guard(raw_registry, horizon_days)
    if v4_reason is not None:
        return None, v4_reason
    try:
        registry = FusionRegistry.read(registry_path)
        entry = registry.entry(horizon_days) or {}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"fusion registry could not be read: {exc}"
    if str(entry.get("validation_status", "")).casefold() != "accepted":
        return None, f"fusion horizon {horizon_days} has not passed the frozen acceptance benchmark"
    reference = str(entry.get("artifact_path", "")).strip()
    if not reference:
        return None, "accepted fusion registry entry has no artifact path"
    artifact_path = Path(reference)
    if not artifact_path.is_absolute():
        artifact_path = PROJECT_ROOT / artifact_path
    if not artifact_path.is_file():
        return None, f"accepted fusion artifact is missing: {artifact_path}"
    expected_hash = str(entry.get("artifact_sha256", "")).strip()
    if not expected_hash:
        return None, "accepted fusion registry entry has no artifact integrity hash"
    observed_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        return None, "accepted fusion artifact failed its integrity check"
    try:
        artifact = FusionModelArtifact.load(artifact_path)
    except (OSError, ValueError, TypeError) as exc:
        return None, f"accepted fusion artifact could not be loaded: {exc}"
    if artifact.validation_status != "accepted":
        return None, "fusion artifact is not marked accepted"
    if artifact.horizon_days != horizon_days:
        return None, "fusion artifact horizon does not match the request"
    return artifact, None


def load_accepted_fusion_artifact(
    horizon_days: int, registry_path: Path = DEFAULT_FUSION_REGISTRY
) -> tuple[Any | None, str | None]:
    path = Path(registry_path)
    return _load_artifact_cached(
        int(horizon_days), str(path.resolve()), fusion_registry_signature(path)
    )


def build_runtime_fusion_frame(
    stock: PredictionResult,
    news: NewsContextResult,
    horizon_days: int,
    *,
    runtime_state: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the point-in-time v3 feature contract from live evidence.

    The offline feature builder and this adapter use the same names and
    explicit missing-news semantics.  The optional state contains only
    trailing, already-observed baselines for the two surprise features.
    """
    context = _context(news, horizon_days)
    usable = int(context.get("usable_article_count", 0) or 0)
    positive = int(context.get("positive_count", 0) or 0)
    neutral = int(context.get("neutral_count", 0) or 0)
    negative = int(context.get("negative_count", 0) or 0)
    score = context.get("weighted_sentiment_score")
    score_value = float(score) if isinstance(score, (int, float)) else 0.0
    coverage = str(context.get("coverage_confidence") or "Unavailable")
    coverage_score = {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.25,
        "unavailable": 0.0,
    }.get(coverage.casefold(), 0.0)
    half_life = float(
        context.get("half_life_days")
        or {1: 3.5, 5: 7.0, 10: 15.0}.get(int(horizon_days), 1.0)
    )
    newest_age = float(context.get("newest_article_age_days") or 0.0)
    mean_confidence = float(context.get("mean_sentiment_confidence") or 0.0)
    mean_entity_confidence = float(context.get("mean_entity_confidence") or 1.0)
    mean_text_quality = float(context.get("mean_text_quality") or 1.0)
    recency = math.exp(-math.log(2.0) * max(newest_age, 0.0) / max(half_life, 0.25))
    event_reliability = (
        float(context.get("news_event_reliability_mean"))
        if usable and context.get("news_event_reliability_mean") is not None
        else (
            coverage_score
            * mean_confidence
            * mean_entity_confidence
            * mean_text_quality
            * recency
            if usable
            else 0.0
        )
    )
    positive_share_raw = context.get("news_positive_share")
    neutral_share_raw = context.get("news_neutral_share")
    negative_share_raw = context.get("news_negative_share")
    positive_share = (
        float(positive_share_raw)
        if positive_share_raw is not None
        else positive / usable
        if usable
        else 0.0
    )
    neutral_share = (
        float(neutral_share_raw)
        if neutral_share_raw is not None
        else neutral / usable
        if usable
        else 0.0
    )
    negative_share = (
        float(negative_share_raw)
        if negative_share_raw is not None
        else negative / usable
        if usable
        else 0.0
    )
    shares = np.asarray([negative_share, neutral_share, positive_share], dtype=float)
    dispersion = (
        float(-(shares * np.log(np.clip(shares, 1e-12, 1.0))).sum() / math.log(3.0))
        if usable
        else 0.0
    )
    event_count = int(context.get("independent_event_count", 0) or 0)
    article_count = int(context.get("candidate_article_count", usable) or usable)
    volume_log = math.log1p(event_count)
    tone_value = score_value
    baselines = runtime_state or {}
    baseline_ready = bool(baselines.get("surprise_baseline_ready", True))
    # Missing trailing history is represented by a zero baseline, matching
    # the offline rolling transform's initial ``fillna(0)`` behaviour.  Do not
    # use ``or`` here: an observed zero baseline is a valid value.
    volume_baseline_raw = baselines.get("volume_baseline_mean", 0.0)
    tone_baseline_raw = baselines.get("tone_baseline_mean", 0.0)
    volume_baseline = float(volume_baseline_raw if volume_baseline_raw is not None else 0.0)
    tone_baseline = float(tone_baseline_raw if tone_baseline_raw is not None else 0.0)
    category_names = (
        "financial_results_guidance",
        "dividends_distributions",
        "contracts_order_wins",
        "acquisitions_disposals_investments",
        "financing_capital_changes",
        "regulatory_legal_governance",
        "operational_product_strategy",
        "management_changes",
    )
    category_shares = {
        name: float(context.get(f"news_event_category_{name}_share", 0.0) or 0.0)
        for name in category_names
    }
    stock_uncertainty = 1.0 - min(
        1.0, abs(2.0 * (float(stock.probability_up) - 0.5))
    )
    stock_up = float(stock.probability_up) >= float(stock.decision_threshold)
    if score_value > 0.05:
        stock_news_agreement = float(stock_up)
    elif score_value < -0.05:
        stock_news_agreement = float(not stock_up)
    else:
        # Balanced tone is neither agreement nor conflict.
        stock_news_agreement = 0.5
    evidence_score = EVIDENCE_STRENGTH_SCORES.get(
        str(stock.evidence_strength).casefold(), 0.0
    )
    positive_impact_raw = context.get("news_positive_event_impact")
    negative_impact_raw = context.get("news_negative_event_impact")
    positive_impact = (
        float(positive_impact_raw)
        if positive_impact_raw is not None
        else positive_share * event_reliability
    )
    negative_impact = (
        float(negative_impact_raw)
        if negative_impact_raw is not None
        else negative_share * event_reliability
    )
    row = {
                "stock_probability_up": float(stock.probability_up),
                "stock_threshold_margin": float(stock.probability_up)
                - float(stock.decision_threshold),
                "stock_evidence_strength": evidence_score,
                "news_weighted_sentiment": score_value,
                "news_positive_share": positive_share,
                "news_neutral_share": neutral_share,
                "news_negative_share": negative_share,
                "news_mean_confidence": mean_confidence,
                "news_max_confidence": float(
                    context.get("max_sentiment_confidence", 0.0) or 0.0
                ),
                "news_usable_article_count": usable,
                "news_unique_source_count": int(
                    context.get("unique_source_count", 0) or 0
                ),
                "news_independent_event_count": event_count,
                "news_newest_age_days": newest_age if usable else 0.0,
                "news_mean_age_days": float(
                    context.get("mean_article_age_days", 0.0) or 0.0
                ),
                "news_coverage_score": coverage_score,
                "news_available": 1.0 if usable else 0.0,
                "stock_news_agreement": stock_news_agreement,
                # v3 anchored residual features
                "news_log_event_count": volume_log if usable else 0.0,
                "news_log_article_count": math.log1p(usable) if usable else 0.0,
                "news_volume_surprise": (volume_log - volume_baseline) if usable and baseline_ready else 0.0,
                "news_official_disclosure_share": float(
                    context.get("news_official_disclosure_share", 0.0) or 0.0
                ) if usable else 0.0,
                "news_positive_impact": positive_impact if usable else 0.0,
                "news_negative_impact": negative_impact if usable else 0.0,
                "news_sentiment_surprise": (tone_value - tone_baseline) if usable and baseline_ready else 0.0,
                "news_sentiment_dispersion": dispersion if usable else 0.0,
                "news_conflicting_signal": 1.0 if usable and positive_share > 0 and negative_share > 0 else 0.0,
                "news_source_diversity": math.log1p(int(context.get("unique_source_count", 0) or 0)) if usable else 0.0,
                "news_reliability_x_stock_uncertainty": event_reliability * stock_uncertainty,
            }
    for name, value in category_shares.items():
        row[f"news_event_category_{name}_share"] = value if usable else 0.0
    return pd.DataFrame([row])


def build_aggregate_outlook(
    stock: PredictionResult | None,
    news: NewsContextResult | None,
    *,
    ticker: str,
    company_name: str,
    horizon_days: int,
    registry_path: Path = DEFAULT_FUSION_REGISTRY,
) -> CombinedOutlookResult:
    """Use accepted numerical fusion or return the explicit stock-led fallback."""

    registry_payload = _read_json(Path(registry_path))
    acceptance_override = dict(registry_payload.get("acceptance_override") or {})
    override_active = bool(acceptance_override.get("active"))
    override_reason = str(acceptance_override.get("reason") or "").strip() or None

    fallback = build_stock_led_outlook(
        stock,
        news,
        ticker=ticker,
        company_name=company_name,
        horizon_days=horizon_days,
    )
    fallback.acceptance_override_active = override_active
    fallback.acceptance_override_reason = override_reason
    if stock is None:
        fallback.fallback_reason = "stock signal unavailable"
        return fallback
    if news is None:
        fallback.fallback_reason = "news signal unavailable"
        return fallback
    if not (
        stock.evidence_status == "accepted"
        and stock.threshold_constraints_passed
    ):
        fallback.fallback_reason = (
            "stock horizon has not passed the frozen acceptance benchmark; "
            "numerical fusion is disabled"
        )
        return fallback
    context = _context(news, horizon_days)
    coverage = str(context.get("coverage_confidence") or "Unavailable")
    is_v3_registry = registry_payload.get("schema_version") == "fusion-registry-v5"
    allowed_coverage = (
        {"low", "medium", "high"}
        if is_v3_registry
        else NUMERICAL_FUSION_COVERAGE
    )
    if coverage.casefold() not in allowed_coverage:
        fallback.fallback_reason = (
            f"news coverage is {coverage}; numerical fusion requires "
            + (
                "eligible Low, Medium or High evidence for v3"
                if is_v3_registry
                else "Medium or High coverage"
            )
        )
        return fallback
    artifact, reason = load_accepted_fusion_artifact(horizon_days, registry_path)
    if artifact is None:
        fallback.fallback_reason = reason
        return fallback

    runtime_state = None
    if is_v3_registry:
        state_reference = str(registry_payload.get("runtime_state_path") or "").strip()
        if state_reference:
            state_path = Path(state_reference)
            if not state_path.is_absolute():
                state_path = PROJECT_ROOT / state_path
            state_payload = _read_json(state_path)
            horizon_state = (state_payload.get("horizons") or {}).get(
                str(horizon_days), {}
            )
            ticker_state = (horizon_state.get("tickers") or {}).get(
                str(ticker), {}
            )
            if isinstance(ticker_state, dict):
                runtime_state = ticker_state
    frame = build_runtime_fusion_frame(
        stock, news, horizon_days, runtime_state=runtime_state
    )
    probability = float(artifact.predict_proba(frame)[0, 1])
    direction = "Up" if probability >= float(artifact.threshold) else "Down"
    score = float(context.get("weighted_sentiment_score"))
    tone = _tone(score)
    relationship = _relationship(stock.prediction_label, tone)
    contributions = artifact.component_contributions(frame)
    news_adjustment = contributions["news_adjustment_probability_points"]
    reversed_direction = direction != stock.prediction_label
    if reversed_direction:
        explanation = (
            f"Aggregate outlook; news changed the stock-only direction after a "
            f"{news_adjustment:+.1f} percentage-point adjustment."
        )
    elif abs(news_adjustment) < 0.05:
        explanation = (
            "Aggregate outlook; the available news had a negligible numerical effect."
        )
    else:
        verb = "strengthened" if (
            (direction == "Up" and news_adjustment > 0)
            or (direction == "Down" and news_adjustment < 0)
        ) else "weakened"
        explanation = (
            f"Aggregate outlook; news {verb} the stock-led direction by "
            f"{abs(news_adjustment):.1f} percentage points."
        )
    return CombinedOutlookResult(
        ticker=stock.ticker,
        company_name=stock.company_name,
        horizon_days=horizon_days,
        mode="validated_fusion",
        direction=direction,
        probability_up=probability,
        decision_threshold=float(artifact.threshold),
        probability_kind="combined",
        validation_status="accepted",
        relationship=relationship,
        news_tone=tone,
        news_coverage=coverage,
        as_of_date=stock.as_of_date,
        news_cutoff_at=_cutoff_from_news(news),
        explanation=explanation,
        stock_probability_up=float(stock.probability_up),
        stock_direction=stock.prediction_label,
        news_score=score,
        usable_article_count=int(context.get("usable_article_count", 0) or 0),
        source_count=int(context.get("unique_source_count", 0) or 0),
        component_contributions=contributions,
        fusion_model_version=artifact.model_version,
        fusion_method=artifact.fusion_method,
        acceptance_override_active=override_active,
        acceptance_override_reason=override_reason,
    )
