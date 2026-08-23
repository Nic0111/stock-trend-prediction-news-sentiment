from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from pipeline.download_financial_news_data import NewsContextResult, build_news_contexts
from pipeline.evidence_synthesis import build_stock_led_outlook
from pipeline.news_corrections import apply_news_override, reset_news_overrides
from pipeline.predict_stock_trend import PredictionResult
from pipeline.news_relevance import stable_article_id
from pipeline.news_correction_audit import correction_audit_summary, record_news_correction


class NewsCorrectionTests(unittest.TestCase):
    def _analysis(self):
        included = {
            "title": "Maybank reports profit growth",
            "description": "Malayan Banking Berhad reported improved earnings.",
            "url": "https://example.com/maybank",
            "source_domain": "example.com",
            "published_at": "2026-07-20T08:00:00Z",
            "sentiment_label": "positive",
            "sentiment_score": 1,
            "sentiment_probability": 0.9,
            "sentiment_uncertain": False,
            "automatic_relevance": "included",
            "automatic_included": True,
        }
        review = {
            "title": "Public bank ballot discussion",
            "description": "A generic public bank system was discussed.",
            "url": "https://example.com/generic",
            "source_domain": "example.com",
            "published_at": "2026-07-20T09:00:00Z",
            "sentiment_label": "neutral",
            "sentiment_score": 0,
            "sentiment_probability": 0.8,
            "sentiment_uncertain": False,
            "automatic_relevance": "unrelated",
            "automatic_included": False,
        }
        for row in (included, review):
            row["article_id"] = stable_article_id(row)
        contexts = build_news_contexts([included], __import__("datetime").date(2026, 7, 21))
        news = NewsContextResult(
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            query_text="Maybank",
            as_of_date="2026-07-21",
            retrieved_at="2026-07-21T10:00:00+00:00",
            raw_article_count=2,
            relevant_article_count=1,
            results_truncated=False,
            contexts=contexts,
            articles=[included],
            output_paths={},
            review_articles=[review],
            automatic_articles=[included],
            automatic_review_articles=[review],
            automatic_contexts=contexts,
        )
        stock = PredictionResult(
            ticker="1155.KL", company_name="Malayan Banking Berhad", horizon_days=1,
            as_of_date="2026-07-21", prediction_label="Up", probability_up=0.6,
            decision_threshold=0.5, model_name="test", model_version="test",
            refinement_round=1, benchmark_passed=True, status="accepted",
            stale_data_warning=False, latest_data_age_days=0,
        )
        from pipeline.dashboard_service import DashboardAnalysisResult
        return DashboardAnalysisResult(
            ticker="1155.KL", company_name="Malayan Banking Berhad", horizon_days=1,
            stock=stock, stock_source="live", stock_captured_at="2026-07-21T10:00:00+00:00",
            stock_warning=None, news=news, news_source="live", news_captured_at="2026-07-21T10:00:00+00:00",
            news_warning=None, generated_at="2026-07-21T10:00:00+00:00",
            combined_outlook=build_stock_led_outlook(stock, news, ticker="1155.KL", company_name="Malayan Banking Berhad", horizon_days=1),
        )

    def test_exclude_and_reset_restore_automatic_result(self) -> None:
        analysis = self._analysis()
        article_id = analysis.news.articles[0]["article_id"]
        excluded = apply_news_override(analysis, article_id, "exclude")
        self.assertEqual(excluded.news.contexts["1"]["coverage_confidence"], "Unavailable")
        self.assertEqual(excluded.news.user_overrides[article_id], "exclude")
        restored = reset_news_overrides(excluded)
        self.assertEqual(len(restored.news.articles), 1)
        self.assertEqual(restored.news.user_overrides, {})

    def test_include_review_article_recalculates_evidence(self) -> None:
        analysis = self._analysis()
        article_id = analysis.news.review_articles[0]["article_id"]
        included = apply_news_override(analysis, article_id, "include")
        self.assertEqual(len(included.news.articles), 2)
        self.assertEqual(included.news.user_overrides[article_id], "include")

    def test_correction_audit_records_before_and_after_without_article_text(self) -> None:
        before = self._analysis()
        article_id = before.news.articles[0]["article_id"]
        after = apply_news_override(before, article_id, "exclude")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite3"
            record_news_correction(
                analysis_id="analysis-1",
                article_id=article_id,
                action="exclude",
                before=before,
                after=after,
                registry_signature="registry-test",
                db_path=path,
            )
            summary = correction_audit_summary(path)
        self.assertEqual(summary["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
