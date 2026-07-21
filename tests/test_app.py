from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import (
    _article_context_note,
    _article_provider_labels,
    _comparison_option_records,
    _format_article_timestamp,
    _stock_signal_presentation,
)
from pipeline.dashboard_service import DashboardAnalysisResult
from pipeline.download_financial_news_data import NewsContextResult
from pipeline.predict_stock_trend import PredictionResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StreamlitAppSmokeTests(unittest.TestCase):
    def test_comparison_options_keep_all_tickers_visible(self) -> None:
        tickers = ["1155.KL", "1295.KL", "1023.KL", "5347.KL"]
        options = _comparison_option_records(
            tickers,
            {ticker: ticker for ticker in tickers},
            {"1295.KL", "1023.KL"},
        )
        self.assertEqual([item["value"] for item in options], tickers)
        self.assertFalse(options[0]["locked"])
        self.assertTrue(options[1]["locked"])
        self.assertTrue(options[2]["locked"])

    def test_related_article_outside_selected_window_is_marked_as_not_used(self) -> None:
        news = NewsContextResult(
            ticker="1023.KL",
            company_name="CIMB Group Holdings Berhad",
            query_text="CIMB",
            as_of_date="2026-07-10",
            retrieved_at="2026-07-10T10:00:00",
            raw_article_count=1,
            relevant_article_count=1,
            results_truncated=False,
            contexts={"5": {"window_days": 14}},
            articles=[],
            output_paths={},
        )
        analysis = DashboardAnalysisResult(
            ticker="1023.KL",
            company_name="CIMB Group Holdings Berhad",
            horizon_days=5,
            stock=None,
            stock_source="unavailable",
            stock_captured_at=None,
            stock_warning=None,
            news=news,
            news_source="live",
            news_captured_at=news.retrieved_at,
            news_warning=None,
            generated_at=news.retrieved_at,
        )
        note = _article_context_note(
            {"published_at": "2026-06-22T03:13:26Z", "sentiment_uncertain": False, "sentiment_score": 1},
            analysis,
        )
        self.assertIn("outside the selected 14-day sentiment window", note)
        self.assertIn("not included in the tone", note)

    def test_article_metadata_uses_provider_provenance_and_12_hour_time(self) -> None:
        self.assertEqual(
            _article_provider_labels(
                {"provider_sources": ["newsapi", "marketaux"]}
            ),
            "NewsAPI / Marketaux",
        )
        self.assertEqual(
            _format_article_timestamp("2026-06-25T07:41:00Z"),
            "25 Jun 2026 · 07:41 AM",
        )

    def test_experimental_horizon_still_presents_direction(self) -> None:
        result = PredictionResult(
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            horizon_days=10,
            as_of_date="2026-07-10",
            prediction_label="Up",
            probability_up=0.54,
            decision_threshold=0.53,
            model_name="logistic-regression",
            model_version="test",
            refinement_round=2,
            benchmark_passed=False,
            status="experimental",
            stale_data_warning=False,
            latest_data_age_days=1,
        )
        self.assertEqual(_stock_signal_presentation(result)[0], "positive")
        self.assertEqual(_stock_signal_presentation(result)[1], "Up")
        self.assertEqual(result.evidence_status, "experimental")
        self.assertFalse(result.directional_signal_available)
        self.assertIn("54.0% probability of upward movement", _stock_signal_presentation(result)[3])

    def test_down_signal_shows_downward_probability(self) -> None:
        result = PredictionResult(
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            horizon_days=1,
            as_of_date="2026-07-10",
            prediction_label="Down",
            probability_up=0.455,
            decision_threshold=0.46,
            model_name="hist-gradient-boosting",
            model_version="test",
            refinement_round=4,
            benchmark_passed=True,
            status="accepted",
            stale_data_warning=False,
            latest_data_age_days=1,
        )
        presentation = _stock_signal_presentation(result)
        self.assertEqual(presentation[1], "Down")
        self.assertIn("54.5% probability of downward movement", presentation[3])

    def test_app_starts_without_external_requests(self) -> None:
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
        app.run(timeout=30)

        self.assertFalse(app.exception)
        hero = next((item.value for item in app.markdown if 'class="apple-hero"' in item.value), "")
        self.assertIn("Start with one company", hero)
        self.assertEqual(len(app.metric), 3)


if __name__ == "__main__":
    unittest.main()
