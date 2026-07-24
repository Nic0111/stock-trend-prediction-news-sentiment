from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.historical_news_fusion_data import (
    FUSION_FEATURE_SCHEMA_VERSION,
    assess_fusion_data_readiness,
    build_point_in_time_feature_table,
    import_historical_news,
    load_normalized_historical_news,
    normalize_historical_news_frame,
)


class HistoricalNewsFusionDataTests(unittest.TestCase):
    def _news(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "query_ticker": "1155.KL",
                    "query_company": "Malayan Banking Berhad",
                    "provider": "NewsAPI",
                    "source_name": "Example News",
                    "title": "Maybank reports growth",
                    "description": "Profit rises.",
                    "url": "https://www.example.com/story?utm_source=test",
                    "published_at": "2026-07-20T04:00:00Z",
                    "sentiment_label": "positive",
                    "sentiment_probability": 0.90,
                },
                {
                    "query_ticker": "1155.KL",
                    "query_company": "Malayan Banking Berhad",
                    "provider": "Marketaux",
                    "source_name": "Second News",
                    "source_domain": "second.example",
                    "title": "Maybank faces costs",
                    "description": "Costs increase.",
                    "url": "https://second.example/story",
                    "published_at": "2026-07-19T04:00:00Z",
                    "sentiment_label": "negative",
                    "sentiment_probability": 0.80,
                },
                {
                    "query_ticker": "1155.KL",
                    "provider": "NewsAPI",
                    "source_name": "Future News",
                    "title": "Future article must be excluded",
                    "description": "Published after observation.",
                    "url": "https://future.example/story",
                    "published_at": "2026-07-22T12:00:00Z",
                    "sentiment_label": "positive",
                    "sentiment_probability": 0.99,
                },
            ]
        )

    def test_normalization_maps_provider_fields_and_deduplicates(self) -> None:
        source = self._news().iloc[[0, 0]].copy()
        normalized = normalize_historical_news_frame(source)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.loc[0, "ticker"], "1155.KL")
        self.assertEqual(normalized.loc[0, "source_domain"], "example.com")
        self.assertEqual(normalized.loc[0, "sentiment_score"], 1)

    def test_import_writes_manifest_and_partitioned_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "news.csv"
            self._news().to_csv(source_path, index=False)
            result = import_historical_news(source_path, output_dir=root / "fusion")
            self.assertEqual(result.rows, 3)
            self.assertTrue(result.manifest_path.exists())
            loaded = load_normalized_historical_news(root / "fusion")
            self.assertEqual(len(loaded), 3)

    def test_point_in_time_features_exclude_future_news(self) -> None:
        stock = pd.DataFrame(
            {
                "Date": ["2026-07-21"],
                "Ticker": ["1155.KL"],
                "Target_1D": [1],
                "Target_Date_1D": ["2026-07-22"],
            }
        )
        news = normalize_historical_news_frame(self._news())
        features = build_point_in_time_feature_table(stock, news, horizons=[1])
        row = features.iloc[0]
        self.assertEqual(row["candidate_article_count"], 2)
        self.assertEqual(row["usable_article_count"], 2)
        self.assertTrue(row["news_available"])
        self.assertEqual(row["feature_schema_version"] if "feature_schema_version" in row else FUSION_FEATURE_SCHEMA_VERSION, FUSION_FEATURE_SCHEMA_VERSION)

    def test_stock_evidence_labels_are_encoded_for_fusion_features(self) -> None:
        stock = pd.DataFrame(
            {
                "Date": ["2026-07-21"],
                "Ticker": ["1155.KL"],
                "Target_1D": [1],
                "Target_Date_1D": ["2026-07-22"],
            }
        )
        probabilities = pd.DataFrame(
            {
                "Date": ["2026-07-21"],
                "Ticker": ["1155.KL"],
                "horizon": [1],
                "probability_up": [0.62],
                "threshold_margin": [0.12],
                "evidence_strength": ["Strong"],
            }
        )
        news = normalize_historical_news_frame(self._news())
        features = build_point_in_time_feature_table(stock, news, stock_probability_frame=probabilities, horizons=[1])
        self.assertEqual(features.iloc[0]["stock_evidence_strength"], 1.0)

    def test_readiness_requires_source_diversity_and_reports_missing_tickers(self) -> None:
        news = normalize_historical_news_frame(self._news())
        report = assess_fusion_data_readiness(
            news,
            required_tickers=["1155.KL", "1295.KL"],
            min_news_present_observations=1,
            min_source_domains=2,
        )
        self.assertFalse(report["ready"])
        self.assertEqual(report["missing_tickers"], ["1295.KL"])
        self.assertTrue(report["per_ticker"]["1155.KL"]["passes_coverage"])


if __name__ == "__main__":
    unittest.main()
