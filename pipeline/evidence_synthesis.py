"""Transparent, non-probabilistic synthesis of the independent StockSense signals.

This module intentionally does not combine a stock probability with a news tone
score.  FinBERT tone is a probability about article language, not a probability
of a future stock move.  A validated late-fusion model can populate the same
response shape later; until then the result is explicitly marked stock-led.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pipeline.download_financial_news_data import NewsContextResult
from pipeline.predict_stock_trend import PredictionResult


Relationship = Literal["reinforcing", "conflicting", "balanced", "news_unavailable"]
SynthesisMode = Literal["stock_led_synthesis", "validated_fusion", "unavailable"]
ProbabilityKind = Literal["stock_model", "combined"]


@dataclass
class CombinedOutlookResult:
    """Stable response contract for the two-stage combined-outlook rollout."""

    ticker: str
    company_name: str
    horizon_days: int
    mode: SynthesisMode
    direction: str | None
    probability_up: float | None
    decision_threshold: float | None
    probability_kind: ProbabilityKind | None
    validation_status: str
    relationship: Relationship
    news_tone: str
    news_coverage: str
    as_of_date: str | None
    news_cutoff_at: str | None
    explanation: str
    stock_probability_up: float | None = None
    stock_direction: str | None = None
    news_score: float | None = None
    usable_article_count: int = 0
    source_count: int = 0
    component_contributions: dict[str, float] = field(default_factory=dict)
    fusion_model_version: str | None = None
    fusion_method: str | None = None
    fallback_reason: str | None = None
    acceptance_override_active: bool = False
    acceptance_override_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _context(news: NewsContextResult | None, horizon_days: int) -> dict[str, Any]:
    if news is None:
        return {}
    return news.contexts.get(str(horizon_days), {}) or {}


def _tone(score: float | None) -> str:
    if score is None:
        return "unavailable"
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "balanced"


def _relationship(direction: str, tone: str) -> Relationship:
    if tone in {"unavailable", "balanced"}:
        return "news_unavailable" if tone == "unavailable" else "balanced"
    direction_up = direction.casefold() == "up"
    tone_up = tone == "positive"
    return "reinforcing" if direction_up == tone_up else "conflicting"


def _cutoff_from_news(news: NewsContextResult | None) -> str | None:
    if news is None:
        return None
    metadata = news.provider_metadata or {}
    value = metadata.get("prediction_cutoff_at") or metadata.get("cutoff_at")
    if value:
        return str(value)
    if news.as_of_date:
        return f"{news.as_of_date}T17:00:00+08:00"
    return None


def build_stock_led_outlook(
    stock: PredictionResult | None,
    news: NewsContextResult | None,
    *,
    ticker: str,
    company_name: str,
    horizon_days: int,
) -> CombinedOutlookResult:
    """Build the safe Stage 1 result from the independent runtime signals."""

    if stock is None:
        return CombinedOutlookResult(
            ticker=ticker,
            company_name=company_name,
            horizon_days=horizon_days,
            mode="unavailable",
            direction=None,
            probability_up=None,
            decision_threshold=None,
            probability_kind=None,
            validation_status="unavailable",
            relationship="news_unavailable",
            news_tone="unavailable",
            news_coverage="Unavailable",
            as_of_date=None,
            news_cutoff_at=_cutoff_from_news(news),
            explanation="A stock-led outlook cannot be produced because the stock signal is unavailable.",
        )

    context = _context(news, horizon_days)
    raw_score = context.get("weighted_sentiment_score")
    score = float(raw_score) if isinstance(raw_score, (int, float)) else None
    tone = _tone(score)
    coverage = str(context.get("coverage_confidence") or "Unavailable")
    relationship = _relationship(stock.prediction_label, tone)
    coverage_note = "" if coverage.casefold() in {"high", "medium"} else " with limited coverage"

    if relationship == "reinforcing":
        explanation = f"Stock-led outlook; available news sentiment reinforces the {stock.prediction_label} direction{coverage_note}."
    elif relationship == "conflicting":
        explanation = f"Stock-led outlook; available news sentiment conflicts with the {stock.prediction_label} direction{coverage_note}."
    elif relationship == "balanced":
        explanation = "Stock-led outlook; the available news tone is balanced and does not materially support either direction."
    else:
        explanation = "Stock-led outlook; usable news sentiment is unavailable, so no news adjustment is applied."

    return CombinedOutlookResult(
        ticker=stock.ticker,
        company_name=stock.company_name,
        horizon_days=horizon_days,
        mode="stock_led_synthesis",
        direction=stock.prediction_label,
        probability_up=float(stock.probability_up),
        decision_threshold=float(stock.decision_threshold),
        probability_kind="stock_model",
        validation_status=stock.evidence_status,
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
    )


def iso_cutoff_for_session(as_of_date: str) -> str:
    """Return the Bursa Malaysia session cutoff in Asia/Kuala_Lumpur time."""

    return f"{as_of_date}T17:00:00+08:00"
