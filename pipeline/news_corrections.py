"""Analysis-scoped human corrections for entity-filtered news evidence."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Literal

from pipeline.download_financial_news_data import (
    NewsContextResult,
    add_sentiment_scores,
    annotate_contribution_horizons,
    build_news_contexts,
    load_sentiment_model,
    prediction_cutoff_at,
)
from pipeline.evidence_synthesis import build_stock_led_outlook
from pipeline.news_relevance import stable_article_id


OverrideAction = Literal["include", "exclude", "reset"]


class NewsCorrectionError(ValueError):
    """A correction cannot be applied to the requested analysis evidence."""


def _article_id(article: dict) -> str:
    value = str(article.get("article_id") or "").strip()
    return value or stable_article_id(article)


def _baseline_news(news: NewsContextResult) -> tuple[list[dict], list[dict]]:
    included = deepcopy(news.automatic_articles or news.articles)
    review = deepcopy(news.automatic_review_articles or news.review_articles)
    for article in [*included, *review]:
        article["article_id"] = _article_id(article)
    return included, review


def _recalculate_news(
    news: NewsContextResult,
    included: list[dict],
    review: list[dict],
    overrides: dict[str, str],
) -> NewsContextResult:
    # A user can restore an entity-review article, but hard-excluded rows never
    # enter either baseline list and therefore cannot be restored by the API.
    for article in included:
        article["automatic_included"] = bool(article.get("automatic_included", True))
        article["effective_included"] = True
        article["user_override"] = overrides.get(_article_id(article))
    for article in review:
        article["automatic_included"] = False
        article["effective_included"] = False
        article["user_override"] = overrides.get(_article_id(article))

    effective: list[dict] = []
    effective_review: list[dict] = []
    by_id = {_article_id(article): article for article in [*included, *review]}
    for article_id, article in by_id.items():
        action = overrides.get(article_id)
        if action == "exclude":
            article["effective_included"] = False
            effective_review.append(article)
        elif action == "include":
            if article.get("automatic_relevance") == "hard_excluded":
                raise NewsCorrectionError("This article is structurally excluded and cannot be restored.")
            article["effective_included"] = True
            effective.append(article)
        elif article.get("automatic_included", False):
            effective.append(article)
        else:
            effective_review.append(article)

    needs_scoring = [
        article for article in effective
        if article.get("sentiment_label") not in {"positive", "neutral", "negative"}
    ]
    if needs_scoring:
        add_sentiment_scores(needs_scoring, load_sentiment_model())

    as_of = date.fromisoformat(news.as_of_date)
    cutoff_at = prediction_cutoff_at(as_of)
    annotate_contribution_horizons(effective, as_of, cutoff_at)
    contexts = build_news_contexts(effective, as_of, cutoff_at)
    return NewsContextResult(
        ticker=news.ticker,
        company_name=news.company_name,
        query_text=news.query_text,
        as_of_date=news.as_of_date,
        retrieved_at=news.retrieved_at,
        raw_article_count=news.raw_article_count,
        relevant_article_count=len(effective),
        results_truncated=news.results_truncated,
        contexts=contexts,
        articles=effective,
        output_paths=news.output_paths,
        provider_mode=news.provider_mode,
        provider_article_counts={
            **dict(news.provider_article_counts or {}),
            "effective": len(effective),
        },
        provider_status=dict(news.provider_status or {}),
        provider_metadata={
            **dict(news.provider_metadata or {}),
            "user_adjusted": bool(overrides),
            "user_override_count": len(overrides),
        },
        historical_articles=list(news.historical_articles or []),
        review_articles=effective_review,
        automatic_articles=deepcopy(news.automatic_articles or included),
        automatic_review_articles=deepcopy(news.automatic_review_articles or review),
        automatic_contexts=deepcopy(news.automatic_contexts or news.contexts),
        user_overrides=dict(overrides),
    )


def apply_news_override(
    analysis,
    article_id: str,
    action: OverrideAction,
):
    """Return a copied dashboard analysis with one news correction applied."""

    if analysis.news is None:
        raise NewsCorrectionError("No news evidence is available for this analysis.")
    if action not in {"include", "exclude", "reset"}:
        raise NewsCorrectionError("Unsupported news correction action.")
    target_id = str(article_id or "").strip()
    included, review = _baseline_news(analysis.news)
    available = {_article_id(article) for article in [*included, *review]}
    if target_id not in available:
        raise NewsCorrectionError("The article is not part of this analysis.")
    overrides = dict(analysis.news.user_overrides or {})
    if action == "reset":
        overrides.pop(target_id, None)
    else:
        overrides[target_id] = action
    updated_news = _recalculate_news(analysis.news, included, review, overrides)
    updated = deepcopy(analysis)
    updated.news = updated_news
    updated.combined_outlook = build_stock_led_outlook(
        updated.stock,
        updated_news,
        ticker=updated.ticker,
        company_name=updated.company_name,
        horizon_days=updated.horizon_days,
    )
    return updated


def reset_news_overrides(analysis):
    """Restore the immutable automatic result for an analysis."""

    if analysis.news is None:
        return deepcopy(analysis)
    included, review = _baseline_news(analysis.news)
    updated = deepcopy(analysis)
    updated.news = _recalculate_news(analysis.news, included, review, {})
    updated.combined_outlook = build_stock_led_outlook(
        updated.stock,
        updated.news,
        ticker=updated.ticker,
        company_name=updated.company_name,
        horizon_days=updated.horizon_days,
    )
    return updated
