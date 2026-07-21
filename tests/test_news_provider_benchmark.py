from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.news_provider_benchmark import evaluate_news_provider_benchmark


class NewsProviderBenchmarkTests(unittest.TestCase):
    def test_evaluation_uses_only_newsapi_and_marketaux_and_waits_for_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "snapshots" / "2026-07-18" / "run-1"
            run_dir.mkdir(parents=True)
            result = {
                "ticker": "1155.KL",
                "company_name": "Malayan Banking Berhad",
                "query_text": "Maybank",
                "as_of_date": "2026-07-17",
                "retrieved_at": "2026-07-18T00:00:00Z",
                "raw_article_count": 1,
                "relevant_article_count": 1,
                "results_truncated": False,
                "contexts": {
                    "1": {
                        "usable_article_count": 1,
                        "coverage_confidence": "Low",
                    }
                },
                "articles": [
                    {
                        "title": "Maybank reports growth",
                        "description": "Earnings improved.",
                        "url": "https://example.com/story",
                        "source_name": "Example",
                        "published_at": "2026-07-17T08:00:00Z",
                        "sentiment_label": "positive",
                    }
                ],
            }
            for provider in ("newsapi", "marketaux"):
                payload = {
                    "provider": provider,
                    "ticker": "1155.KL",
                    "as_of_date": "2026-07-17",
                    "status": "ok",
                    "elapsed_seconds": 1.0,
                    "error": None,
                    "result": {**result, "provider_mode": provider},
                    "run_id": "run-1",
                    "collection_date": "2026-07-18",
                    "cutoff_lag_days": 1,
                    "sentiment_backend": "finbert",
                    "sentiment_model": "models/experimental/finbert-sentiment",
                    "provider_metadata": {},
                }
                (run_dir / f"1155-kl-{provider}.json").write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_news_provider_benchmark(root)
        self.assertEqual(set(report["metrics"]), {"newsapi", "marketaux"})
        self.assertEqual(report["decision_status"], "awaiting_user_approval")
        self.assertIsNone(report["recommendation"])


if __name__ == "__main__":
    unittest.main()

