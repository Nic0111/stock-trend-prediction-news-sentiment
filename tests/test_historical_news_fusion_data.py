from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.historical_news_fusion_data import (
    FUSION_FEATURE_SCHEMA_VERSION,
    assess_fusion_data_readiness,
    build_point_in_time_feature_table,
    import_historical_news,
    load_normalized_historical_news,
    normalize_historical_news_frame,
    score_historical_news_with_production_finbert,
    canonical_material_event_category,
)


class HistoricalNewsFusionDataTests(unittest.TestCase):
    def test_legacy_event_categories_map_conservatively_to_v3_taxonomy(self) -> None:
        self.assertEqual(
            canonical_material_event_category("earnings", "Quarterly results"),
            "financial_results_guidance",
        )
        self.assertEqual(
            canonical_material_event_category(
                "corporate_action", "Company declares interim dividend"
            ),
            "dividends_distributions",
        )
        self.assertEqual(
            canonical_material_event_category("other", "General market update"),
            "unclassified",
        )

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

    def test_normalization_accepts_mixed_timestamp_formats_without_dropping_rows(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "ticker": "1023.KL",
                    "published_at": "2019-01-24 04:00:00+00:00",
                    "title": "Quarterly profit rises",
                    "url": "https://example.test/old",
                },
                {
                    "ticker": "8869.KL",
                    "published_at": "2019-01-24T04:00:00+00:00",
                    "title": "Quarterly revenue rises",
                    "url": "https://example.test/new",
                },
            ]
        )
        normalized = normalize_historical_news_frame(source)
        self.assertEqual(len(normalized), 2)
        self.assertTrue(normalized["published_at"].notna().all())

    def test_normalization_maps_recovery_package_url_and_article_text(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "published_at": "2023-01-01T00:00:00Z",
                    "title": "Maybank reports growth",
                    "source_url": "https://Example.com/story?utm_source=archive",
                    "article_text": "Recovered publisher article body.",
                }
            ]
        )

        normalized = normalize_historical_news_frame(source)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.loc[0, "url"], "https://example.com/story")
        self.assertEqual(normalized.loc[0, "source_domain"], "example.com")
        self.assertEqual(
            normalized.loc[0, "content_snippet"],
            "Recovered publisher article body.",
        )

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

    def test_after_close_article_is_not_available_to_same_session(self) -> None:
        stock = pd.DataFrame(
            {
                "Date": ["2026-07-21", "2026-07-22"],
                "Ticker": ["1155.KL", "1155.KL"],
                "Target_1D": [1, 0],
                "Target_Date_1D": ["2026-07-22", "2026-07-23"],
            }
        )
        news = normalize_historical_news_frame(
            pd.DataFrame(
                [
                    {
                        "query_ticker": "1155.KL",
                        "provider": "NewsAPI",
                        "source_name": "After Close News",
                        "title": "Maybank issues an evening update",
                        "description": "The update was published after Bursa closed.",
                        "url": "https://after-close.example/story",
                        # 10:00 UTC is 18:00 MYT, one hour after the 17:00 cutoff.
                        "published_at": "2026-07-21T10:00:00Z",
                        "sentiment_label": "neutral",
                        "sentiment_probability": 0.90,
                    }
                ]
            )
        )

        features = build_point_in_time_feature_table(stock, news, horizons=[1])
        same_session = features.loc[features["Date"].eq("2026-07-21")].iloc[0]
        next_session = features.loc[features["Date"].eq("2026-07-22")].iloc[0]
        self.assertEqual(same_session["candidate_article_count"], 0)
        self.assertEqual(next_session["candidate_article_count"], 1)

    def test_current_session_window_includes_news_since_previous_bursa_close(self) -> None:
        stock = pd.DataFrame(
            {
                "Date": ["2026-07-21"],
                "Ticker": ["1155.KL"],
                "Target_1D": [1],
                "Target_Date_1D": ["2026-07-22"],
            }
        )
        news = normalize_historical_news_frame(
            pd.DataFrame(
                [
                    {
                        "query_ticker": "1155.KL",
                        "provider": "NewsAPI",
                        "source_name": "Session News",
                        "title": "Maybank announces a same-session update",
                        "description": "Published before the current Bursa close.",
                        "url": "https://session.example/story",
                        "published_at": "2026-07-21T04:00:00Z",
                        "sentiment_label": "positive",
                        "sentiment_probability": 0.90,
                    }
                ]
            )
        )

        features = build_point_in_time_feature_table(
            stock,
            news,
            horizons=[1],
            context_windows={1: 0},
            context_half_lives={1: 0.25},
        )
        self.assertEqual(features.iloc[0]["candidate_article_count"], 1)

    def test_syndicated_event_uses_strongest_representation_once(self) -> None:
        stock = pd.DataFrame(
            {
                "Date": ["2026-07-21"],
                "Ticker": ["1155.KL"],
                "Target_1D": [1],
                "Target_Date_1D": ["2026-07-22"],
            }
        )
        common = {
            "query_ticker": "1155.KL",
            "provider": "NewsAPI",
            "title": "Maybank reports quarterly profit growth",
            "description": "One financial event syndicated by two sources.",
            "published_at": "2026-07-21T04:00:00Z",
        }
        news = normalize_historical_news_frame(
            pd.DataFrame(
                [
                    {
                        **common,
                        "source_name": "Primary",
                        "url": "https://primary.example/story",
                        "sentiment_label": "positive",
                        "sentiment_probability": 0.90,
                    },
                    {
                        **common,
                        "source_name": "Syndicated",
                        "url": "https://syndicated.example/story",
                        "sentiment_label": "negative",
                        "sentiment_probability": 0.50,
                    },
                ]
            )
        )

        row = build_point_in_time_feature_table(
            stock,
            news,
            horizons=[1],
            context_windows={1: 3},
            context_half_lives={1: 1.5},
        ).iloc[0]
        self.assertEqual(row["news_independent_event_count"], 1)
        self.assertEqual(row["news_positive_share"], 1.0)
        self.assertEqual(row["news_negative_share"], 0.0)
        self.assertGreater(row["news_positive_event_impact"], 0.0)

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

    @patch("pipeline.historical_news_fusion_data.load_sentiment_model")
    def test_historical_scoring_uses_production_probabilities_not_reference_label(
        self, load_model
    ) -> None:
        class Model:
            classes_ = np.asarray(["negative", "neutral", "positive"])

            def predict_proba(self, texts):
                return np.tile(np.asarray([[0.8, 0.1, 0.1]]), (len(texts), 1))

        load_model.return_value = Model()
        source = pd.DataFrame(
            [
                {
                    "article_id": "relevant",
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "published_at": "2023-01-01T00:00:00Z",
                    "title": "Maybank result",
                    "description": "Profit rose.",
                    "url": "https://example.com/relevant",
                    "entity_relevant": True,
                    "sentiment_label": "positive",
                    "label_source": "llm_assisted_label",
                    "event_category": "earnings",
                },
                {
                    "article_id": "irrelevant",
                    "ticker": "1155.KL",
                    "published_at": "2023-01-02T00:00:00Z",
                    "title": "Unrelated result",
                    "description": "Not about the company.",
                    "url": "https://example.com/irrelevant",
                    "entity_relevant": False,
                    "sentiment_label": "neutral",
                },
            ]
        )

        result = score_historical_news_with_production_finbert(source)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["sentiment_label"], "negative")
        self.assertEqual(result.iloc[0]["reference_sentiment_label"], "positive")
        self.assertEqual(result.iloc[0]["reference_label_source"], "llm_assisted_label")
        self.assertEqual(
            result.iloc[0]["sentiment_model_role"],
            "production_finbert_historical_score",
        )
        self.assertEqual(
            result.iloc[0]["event_category"], "financial_results_guidance"
        )

    @patch("pipeline.historical_news_fusion_data.load_sentiment_model")
    def test_scoring_unlabelled_official_rows_does_not_claim_reviewed_labels(
        self, load_model
    ) -> None:
        class Model:
            classes_ = np.asarray(["negative", "neutral", "positive"])

            def predict_proba(self, texts):
                return np.tile(np.asarray([[0.2, 0.6, 0.2]]), (len(texts), 1))

        load_model.return_value = Model()
        source = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "published_at": "2023-01-01T00:00:00Z",
                    "title": "Quarterly financial results",
                    "description": "Issuer announcement.",
                    "url": "https://issuer.example/announcement",
                    "source_name": "issuer.example",
                    "source_kind": "issuer_ir",
                }
            ]
        )
        result = score_historical_news_with_production_finbert(source)
        self.assertEqual(result.iloc[0]["label_validation_status"], "unlabelled_reference")
        self.assertFalse(bool(result.iloc[0]["llm_assisted_label"]))


if __name__ == "__main__":
    unittest.main()
