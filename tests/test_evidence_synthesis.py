from __future__ import annotations

import unittest
from datetime import date

from pipeline.download_financial_news_data import (
    NewsContextResult,
    article_age_days,
    build_news_contexts,
    build_news_context_metadata,
    prediction_cutoff_at,
)
from pipeline.evidence_synthesis import build_stock_led_outlook
from pipeline.predict_stock_trend import PredictionResult


def _stock(direction: str = "Up", probability: float = 0.62) -> PredictionResult:
    return PredictionResult(
        ticker="1155.KL",
        company_name="Malayan Banking Berhad",
        horizon_days=1,
        as_of_date="2026-07-21",
        prediction_label=direction,
        probability_up=probability,
        decision_threshold=0.5,
        model_name="test-model",
        model_version="test-version",
        refinement_round=1,
        benchmark_passed=True,
        status="accepted",
        stale_data_warning=False,
        latest_data_age_days=0,
    )


def _news(score: float | None, coverage: str = "Medium") -> NewsContextResult:
    context = {
        "horizon_days": 1,
        "window_days": 7,
        "half_life_days": 3.5,
        "usable_article_count": 3 if score is not None else 0,
        "unique_source_count": 2 if score is not None else 0,
        "weighted_sentiment_score": score,
        "coverage_confidence": coverage if score is not None else "Unavailable",
    }
    return NewsContextResult(
        ticker="1155.KL",
        company_name="Malayan Banking Berhad",
        query_text="Maybank",
        as_of_date="2026-07-21",
        retrieved_at="2026-07-21T10:00:00+00:00",
        raw_article_count=context["usable_article_count"],
        relevant_article_count=context["usable_article_count"],
        results_truncated=False,
        contexts={"1": context},
        articles=[],
        output_paths={},
        provider_metadata={**build_news_context_metadata(), "prediction_cutoff_at": "2026-07-21T17:00:00+08:00"},
    )


class EvidenceSynthesisTests(unittest.TestCase):
    def test_stock_probability_is_unchanged_and_positive_news_reinforces_up(self) -> None:
        result = build_stock_led_outlook(_stock("Up", 0.62), _news(0.55), ticker="1155.KL", company_name="Maybank", horizon_days=1)
        self.assertEqual(result.mode, "stock_led_synthesis")
        self.assertEqual(result.probability_kind, "stock_model")
        self.assertEqual(result.probability_up, 0.62)
        self.assertEqual(result.relationship, "reinforcing")
        self.assertEqual(result.news_tone, "positive")

    def test_positive_news_conflicts_with_down_stock_signal(self) -> None:
        result = build_stock_led_outlook(_stock("Down", 0.42), _news(0.2), ticker="1155.KL", company_name="Maybank", horizon_days=1)
        self.assertEqual(result.relationship, "conflicting")
        self.assertEqual(result.direction, "Down")
        self.assertEqual(result.probability_up, 0.42)

    def test_balanced_or_missing_news_never_changes_stock_direction(self) -> None:
        balanced = build_stock_led_outlook(_stock(), _news(0.02), ticker="1155.KL", company_name="Maybank", horizon_days=1)
        missing = build_stock_led_outlook(_stock(), _news(None), ticker="1155.KL", company_name="Maybank", horizon_days=1)
        self.assertEqual(balanced.relationship, "balanced")
        self.assertEqual(missing.relationship, "news_unavailable")
        self.assertEqual(missing.explanation.split(";")[0], "Stock-led outlook")
        self.assertEqual(missing.probability_up, 0.62)

    def test_bursa_session_cutoff_excludes_after_close_articles(self) -> None:
        observation = date(2026, 7, 21)
        cutoff = prediction_cutoff_at(observation)
        self.assertGreater(article_age_days("2026-07-21T10:00:00Z", observation, cutoff), 0)
        self.assertEqual(article_age_days("2026-07-21T09:00:00Z", observation, cutoff), 0.0)
        rows = [
            {"published_at": "2026-07-21T08:00:00Z", "sentiment_uncertain": False, "sentiment_score": 1, "sentiment_probability": 0.9, "sentiment_label": "positive", "source_domain": "before.example"},
            {"published_at": "2026-07-21T10:00:00Z", "sentiment_uncertain": False, "sentiment_score": -1, "sentiment_probability": 0.9, "sentiment_label": "negative", "source_domain": "after.example"},
        ]
        context = build_news_contexts(rows, observation, cutoff)["1"]
        self.assertEqual(context["candidate_article_count"], 1)
        self.assertEqual(context["negative_count"], 0)


if __name__ == "__main__":
    unittest.main()
