"""Point-in-time runtime materialisation for Fusion v4 artifacts.

The v4 evaluation uses a compact ``v4_*`` feature contract and a small
target-specific event-impact model.  This module is intentionally kept at
the runtime boundary: it converts the live stock/news response into the
sealed feature names used by the staged artifact and exposes the same
interface as the older v3 adapter.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

from pipeline.evidence_synthesis import _context


V4_FEATURES = (
    "v4_log_event_count",
    "v4_attention_surprise",
    "v4_positive_impact",
    "v4_negative_impact",
    "v4_sentiment_surprise",
    "v4_event_reliability",
    "v4_official_share",
    "v4_source_diversity",
    "v4_event_impact_logit",
)

IMPACT_INPUTS = (
    "v4_positive_impact",
    "v4_negative_impact",
    "v4_sentiment_surprise",
    "v4_log_event_count",
    "v4_attention_surprise",
    "v4_event_reliability",
    "v4_official_share",
    "v4_source_diversity",
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _clip(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(_float(value), low, high))


def build_runtime_v4_frame(
    stock: Any,
    news: Any,
    horizon_days: int,
    *,
    runtime_state: dict[str, Any] | None = None,
    event_impact_model: Any | None = None,
) -> pd.DataFrame:
    """Build one point-in-time v4 row from the live application context.

    ``runtime_state`` contains only trailing baselines computed from sessions
    strictly before the current observation.  Missing news is represented by
    zero-valued v4 features and ``news_available=0``; the anchored estimator
    then returns the exact stock probability.
    """

    context = _context(news, int(horizon_days))
    usable = int(_float(context.get("usable_article_count"), 0.0))
    available = usable > 0
    score = _float(context.get("weighted_sentiment_score"), 0.0)

    event_count = max(
        0,
        int(
            _float(
                context.get("independent_event_count"),
                context.get("candidate_article_count", usable),
            )
        ),
    )
    article_count = max(0, int(_float(context.get("candidate_article_count"), usable)))
    positive_share = _clip(context.get("news_positive_share"), 0.0, 1.0)
    negative_share = _clip(context.get("news_negative_share"), 0.0, 1.0)

    coverage = str(context.get("coverage_confidence") or "Unavailable").casefold()
    coverage_score = {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.25,
        "unavailable": 0.0,
    }.get(coverage, 0.0)
    mean_confidence = _clip(context.get("mean_sentiment_confidence"), 0.0, 1.0)
    mean_entity_confidence = _clip(context.get("mean_entity_confidence"), 0.0, 1.0)
    mean_text_quality = _clip(context.get("mean_text_quality"), 0.0, 1.0)
    half_life = _float(
        context.get("half_life_days"),
        {1: 3.5, 5: 7.0, 10: 15.0}.get(int(horizon_days), 1.0),
    )
    newest_age = max(0.0, _float(context.get("newest_article_age_days"), 0.0))
    recency = math.exp(-math.log(2.0) * newest_age / max(half_life, 0.25))
    if available and context.get("news_event_reliability_mean") is not None:
        reliability = _clip(context.get("news_event_reliability_mean"))
    else:
        reliability = (
            coverage_score
            * mean_confidence
            * mean_entity_confidence
            * mean_text_quality
            * recency
            if available
            else 0.0
        )

    positive_impact = context.get("news_positive_event_impact")
    negative_impact = context.get("news_negative_event_impact")
    positive_impact = (
        _clip(positive_impact) if positive_impact is not None else positive_share * reliability
    )
    negative_impact = (
        _clip(negative_impact) if negative_impact is not None else negative_share * reliability
    )

    state = runtime_state or {}
    volume_baseline = _float(state.get("volume_baseline_mean"), 0.0)
    tone_baseline = _float(state.get("tone_baseline_mean"), 0.0)
    volume_log = math.log1p(event_count)
    article_log = math.log1p(article_count)
    attention_surprise = volume_log - volume_baseline
    if event_count <= 0 and article_count > 0:
        attention_surprise = article_log - volume_baseline
    sentiment_surprise = score - tone_baseline

    row: dict[str, Any] = {
        "stock_probability_up": _float(getattr(stock, "probability_up", 0.5), 0.5),
        "stock_threshold_margin": _float(getattr(stock, "probability_up", 0.5), 0.5)
        - _float(getattr(stock, "decision_threshold", 0.5), 0.5),
        "stock_evidence_strength": 0.0,
        "news_available": 1.0 if available else 0.0,
        "v4_log_event_count": volume_log if available else 0.0,
        "v4_attention_surprise": attention_surprise if available else 0.0,
        "v4_positive_impact": positive_impact if available else 0.0,
        "v4_negative_impact": negative_impact if available else 0.0,
        "v4_sentiment_surprise": sentiment_surprise if available else 0.0,
        "v4_event_reliability": reliability if available else 0.0,
        "v4_official_share": _clip(
            context.get("news_official_disclosure_share"), 0.0, 1.0
        )
        if available
        else 0.0,
        "v4_source_diversity": math.log1p(
            max(0, int(_float(context.get("unique_source_count"), 0.0)))
        )
        if available
        else 0.0,
    }

    for name in IMPACT_INPUTS:
        row.setdefault(name, 0.0)
    if available and event_impact_model is not None:
        impact_frame = pd.DataFrame([row], columns=[*IMPACT_INPUTS, "news_available"])
        impact_probability = np.asarray(
            event_impact_model.predict_proba_up(impact_frame), dtype=float
        )
        row["v4_event_impact_logit"] = float(logit(np.clip(impact_probability[0], 1e-6, 1 - 1e-6)))
    else:
        row["v4_event_impact_logit"] = 0.0
    if not available:
        for feature in V4_FEATURES:
            row[feature] = 0.0
    return pd.DataFrame([row], columns=["stock_probability_up", "stock_threshold_margin", "stock_evidence_strength", "news_available", *V4_FEATURES])


class V4FusionArtifactAdapter:
    """Expose an augmented v4 joblib payload through the runtime interface."""

    validation_status = "accepted"

    def __init__(self, payload: dict[str, Any]) -> None:
        self._model = payload["model"]
        self._event_impact_model = payload["event_impact_model"]
        if not getattr(self._event_impact_model, "ready", False):
            raise ValueError("v4 event-impact model is not ready")
        self.horizon_days = int(payload["horizon_days"])
        self.threshold = float(payload["shared_decision_threshold"])
        self.selected_window_days = float(payload.get("selected_window_days", 0.0))
        self.selected_half_life_days = float(payload.get("selected_half_life_days", 0.0))
        self.model_version = "fusion-v4-compact-target-specific"
        self.fusion_method = "v4_anchored_incremental_news"

    def build_runtime_frame(
        self,
        stock: Any,
        news: Any,
        horizon_days: int,
        *,
        runtime_state: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        return build_runtime_v4_frame(
            stock,
            news,
            horizon_days,
            runtime_state=runtime_state,
            event_impact_model=self._event_impact_model,
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = self._model.predict(frame)
        probability = np.asarray(prediction.probability_up, dtype=float)
        return np.column_stack([1.0 - probability, probability])

    def component_contributions(self, frame: pd.DataFrame) -> dict[str, float]:
        if len(frame) != 1:
            raise ValueError("Fusion component contributions require exactly one row")
        full_probability = float(self.predict_proba(frame)[0, 1])
        stock_frame = frame.copy()
        stock_frame["news_available"] = 0.0
        for feature in V4_FEATURES:
            stock_frame[feature] = 0.0
        stock_probability = float(self.predict_proba(stock_frame)[0, 1])
        return {
            "base_probability_up": 0.5,
            "stock_component_probability_up": stock_probability,
            "stock_contribution_probability_points": (stock_probability - 0.5) * 100.0,
            "news_adjustment_probability_points": (full_probability - stock_probability) * 100.0,
            "aggregate_probability_up": full_probability,
        }
