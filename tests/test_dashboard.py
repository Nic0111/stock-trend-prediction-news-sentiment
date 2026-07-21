from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from pipeline.finbert_sentiment_model import FinBERTDependencyError
from pipeline import dashboard_service
from pipeline.download_financial_news_data import NewsContextResult, build_news_context_metadata
from pipeline.predict_stock_trend import PredictionResult
from pipeline.stock_features import FEATURE_SCHEMA_VERSION


def make_prediction(horizon_days: int = 5) -> PredictionResult:
    return PredictionResult(
        ticker="1155.KL",
        company_name="Malayan Banking Berhad",
        horizon_days=horizon_days,
        as_of_date="2026-07-09",
        prediction_label="Up",
        probability_up=0.61,
        decision_threshold=0.51,
        model_name="logistic-regression",
        model_version="test-version",
        refinement_round=2,
        benchmark_passed=False,
        status="experimental",
        stale_data_warning=False,
        latest_data_age_days=1,
        market_reference_date="2026-07-09",
        observation_lag_sessions=0,
        freshness_status="fresh",
    )


def make_news() -> NewsContextResult:
    context = {
        "horizon_days": 5,
        "window_days": 14,
        "half_life_days": 7.0,
        "candidate_article_count": 1,
        "usable_article_count": 1,
        "uncertain_article_count": 0,
        "unique_source_count": 1,
        "weighted_sentiment_score": 0.4,
        "positive_count": 1,
        "neutral_count": 0,
        "negative_count": 0,
        "coverage_confidence": "Low",
        "low_confidence_flag": True,
    }
    return NewsContextResult(
        ticker="1155.KL",
        company_name="Malayan Banking Berhad",
        query_text='"Malayan Banking Berhad"',
        as_of_date="2026-07-10",
        retrieved_at="2026-07-10T08:00:00",
        raw_article_count=1,
        relevant_article_count=1,
        results_truncated=False,
        contexts={"5": context},
        articles=[
            {
                "title": "Maybank reports steady growth",
                "description": "A test article.",
                "url": "https://example.com/maybank",
                "source_name": "Example Source",
                "published_at": "2026-07-10T07:00:00Z",
                "sentiment_label": "positive",
            }
        ],
        output_paths={},
        provider_metadata=build_news_context_metadata(),
    )


class DashboardServiceTests(unittest.TestCase):
    def test_expired_cache_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "expired.json"
            captured_at = (datetime.now(timezone.utc) - timedelta(seconds=dashboard_service.CACHE_TTL_SECONDS + 1)).isoformat()
            dashboard_service._write_cache(path, {"value": 1}, captured_at)
            self.assertIsNone(dashboard_service._read_cache(path))

    def test_legacy_news_context_cache_is_not_used(self) -> None:
        news = make_news()
        news.provider_metadata["news_context_config_version"] = 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(dashboard_service, "NEWS_CACHE_DIR", root / "news"):
                path = root / "news" / "1155-kl.json"
                dashboard_service._write_cache(path, news.to_dict(), dashboard_service._now_iso())
                loaded, captured_at = dashboard_service._load_news_fallback("1155.KL")
        self.assertIsNone(loaded)
        self.assertIsNone(captured_at)

    def test_legacy_stock_feature_cache_is_not_used(self) -> None:
        prediction = make_prediction()
        payload = prediction.to_dict()
        payload["feature_schema_version"] = "legacy-market-alignment"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(dashboard_service, "STOCK_CACHE_DIR", root / "stock"):
                path = root / "stock" / "1155-kl-5d.json"
                dashboard_service._write_cache(path, payload, dashboard_service._now_iso())
                loaded, captured_at = dashboard_service._load_stock_fallback("1155.KL", 5)
        self.assertIsNone(loaded)
        self.assertIsNone(captured_at)

    def test_current_stock_feature_cache_is_accepted(self) -> None:
        prediction = make_prediction()
        prediction.feature_schema_version = FEATURE_SCHEMA_VERSION
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(dashboard_service, "STOCK_CACHE_DIR", root / "stock"):
                path = root / "stock" / "1155-kl-5d.json"
                dashboard_service._write_cache(path, prediction.to_dict(), dashboard_service._now_iso())
                loaded, captured_at = dashboard_service._load_stock_fallback("1155.KL", 5)
        self.assertIsNotNone(loaded)
        self.assertIsNotNone(captured_at)

    def test_live_results_are_cached_and_each_signal_can_fallback_independently(self) -> None:
        stock = make_prediction()
        news = make_news()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(dashboard_service, "STOCK_CACHE_DIR", root / "stock"),
                patch.object(dashboard_service, "NEWS_CACHE_DIR", root / "news"),
                patch.object(dashboard_service, "predict_stock_trend", return_value=stock) as stock_call,
                patch.object(dashboard_service, "retrieve_news_with_marketaux_fallback", return_value=news) as news_call,
            ):
                live = dashboard_service.run_dashboard_analysis("1155.KL", 5)
                self.assertEqual(live.stock_source, "live")
                self.assertEqual(live.news_source, "live")
                self.assertTrue((root / "stock" / "1155-kl-5d.json").exists())
                self.assertTrue((root / "news" / "1155-kl.json").exists())

                stock_call.side_effect = RuntimeError("Yahoo Finance unavailable")
                fallback = dashboard_service.run_dashboard_analysis("1155.KL", 5)

            self.assertEqual(fallback.stock_source, "fallback")
            self.assertEqual(fallback.news_source, "live")
            self.assertIsNotNone(fallback.stock_warning)
            self.assertIn("saved result", fallback.stock_warning or "")
            self.assertEqual(fallback.news, news)
            self.assertEqual(stock_call.call_count, 2)
            self.assertEqual(news_call.call_count, 2)

    def test_cache_isolated_by_horizon_and_corrupt_cache_is_unavailable(self) -> None:
        stock = make_prediction(5)
        news = make_news()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(dashboard_service, "STOCK_CACHE_DIR", root / "stock"),
                patch.object(dashboard_service, "NEWS_CACHE_DIR", root / "news"),
                patch.object(dashboard_service, "predict_stock_trend", return_value=stock),
                patch.object(dashboard_service, "retrieve_news_with_marketaux_fallback", return_value=news),
            ):
                dashboard_service.run_dashboard_analysis("1155.KL", 5)
                (root / "stock" / "1155-kl-1d.json").parent.mkdir(parents=True, exist_ok=True)
                (root / "stock" / "1155-kl-1d.json").write_text("{not-json", encoding="utf-8")

                with (
                    patch.object(
                        dashboard_service,
                        "predict_stock_trend",
                        side_effect=RuntimeError("Yahoo Finance unavailable"),
                    ),
                    patch.object(
                        dashboard_service,
                        "retrieve_news_with_marketaux_fallback",
                        side_effect=RuntimeError("NewsAPI unavailable"),
                    ),
                ):
                    result = dashboard_service.run_dashboard_analysis("1155.KL", 1)

            self.assertEqual(result.stock_source, "unavailable")
            self.assertEqual(result.news_source, "fallback")
            self.assertIsNone(result.stock)
            self.assertIsNotNone(result.news)
            self.assertIn("no saved result", result.stock_warning or "")

    def test_finbert_runtime_failure_has_actionable_news_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(dashboard_service, "STOCK_CACHE_DIR", root / "stock"),
                patch.object(dashboard_service, "NEWS_CACHE_DIR", root / "news"),
                patch.object(dashboard_service, "predict_stock_trend", return_value=make_prediction(1)),
                patch.object(
                    dashboard_service,
                    "retrieve_news_with_marketaux_fallback",
                    side_effect=FinBERTDependencyError("Install torch and transformers."),
                ),
            ):
                result = dashboard_service.run_dashboard_analysis("1295.KL", 1)

        self.assertEqual(result.news_source, "unavailable")
        self.assertIn("Sentiment engine unavailable", result.news_warning or "")
        self.assertIn("Install torch", result.news_warning or "")


if __name__ == "__main__":
    unittest.main()
