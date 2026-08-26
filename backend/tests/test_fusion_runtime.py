from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from pipeline.download_financial_news_data import NewsContextResult
from pipeline.fusion_runtime import build_aggregate_outlook
from pipeline.predict_stock_trend import PredictionResult


class _FakeFusionArtifact:
    threshold = 0.5
    model_version = "test-fusion"
    fusion_method = "anchored_incremental_news"

    def predict_proba(self, _frame):
        return np.asarray([[0.42, 0.58]])

    def component_contributions(self, _frame):
        return {
            "base_probability_up": 0.5,
            "stock_component_probability_up": 0.56,
            "stock_contribution_probability_points": 6.0,
            "news_adjustment_probability_points": 2.0,
            "aggregate_probability_up": 0.58,
        }


def _stock(*, probability: float = 0.56) -> PredictionResult:
    return PredictionResult(
        ticker="1155.KL",
        company_name="Malayan Banking Berhad",
        horizon_days=1,
        as_of_date="2026-08-21",
        prediction_label="Down",
        probability_up=probability,
        decision_threshold=0.511,
        model_name="test-stock",
        model_version="test",
        refinement_round=1,
        benchmark_passed=False,
        status="limited",
        stale_data_warning=False,
        latest_data_age_days=0,
        threshold_constraints_passed=False,
    )


def _news(*, articles: int = 2, score: float | None = 0.4) -> NewsContextResult:
    return NewsContextResult(
        ticker="1155.KL",
        company_name="Malayan Banking Berhad",
        query_text="Maybank",
        as_of_date="2026-08-21",
        retrieved_at="2026-08-21T12:00:00+00:00",
        raw_article_count=articles,
        relevant_article_count=articles,
        results_truncated=False,
        contexts={
            "1": {
                "weighted_sentiment_score": score,
                "coverage_confidence": "Medium" if articles else "Unavailable",
                "usable_article_count": articles,
                "unique_source_count": 2,
            }
        },
        articles=[],
        output_paths={},
    )


class FusionRuntimePolicyTests(unittest.TestCase):
    @patch("pipeline.fusion_runtime.build_runtime_fusion_frame")
    @patch("pipeline.fusion_runtime.load_accepted_fusion_artifact", return_value=(_FakeFusionArtifact(), None))
    def test_valid_stock_and_news_use_fusion_even_when_stock_is_limited(self, _load, frame):
        frame.return_value = object()
        result = build_aggregate_outlook(
            _stock(),
            _news(),
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            horizon_days=1,
            registry_path="missing-test-registry.json",
        )
        self.assertEqual(result.mode, "validated_fusion")
        self.assertEqual(result.direction, "Up")

    def test_missing_news_keeps_stock_led_fallback(self):
        result = build_aggregate_outlook(
            _stock(),
            None,
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            horizon_days=1,
            registry_path="missing-test-registry.json",
        )
        self.assertEqual(result.mode, "stock_led_synthesis")
        self.assertEqual(result.fallback_reason, "news signal unavailable")

    def test_invalid_stock_probability_keeps_stock_led_fallback(self):
        result = build_aggregate_outlook(
            _stock(probability=float("nan")),
            _news(),
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            horizon_days=1,
            registry_path="missing-test-registry.json",
        )
        self.assertEqual(result.mode, "stock_led_synthesis")
        self.assertIn("invalid", result.fallback_reason or "")


if __name__ == "__main__":
    unittest.main()
