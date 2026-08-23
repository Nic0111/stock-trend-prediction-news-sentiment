from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from pipeline.bursa_news_dataset import (
    build_annotation_template,
    build_year_quality_report,
    materialize_external_single_weak_dataset,
    materialize_llm_assisted_dataset,
    resolve_independent_annotations,
    split_for_timestamp,
    validate_bursa_news_dataset,
)


class BursaNewsDatasetTests(unittest.TestCase):
    def test_chronological_split_is_locked_by_year(self) -> None:
        self.assertEqual(split_for_timestamp("2023-12-31T10:00:00Z"), "training")
        self.assertEqual(split_for_timestamp("2024-01-01T00:00:00Z"), "validation")
        self.assertEqual(split_for_timestamp("2025-07-01T00:00:00Z"), "test")
        self.assertEqual(split_for_timestamp("2026-01-01T00:00:00Z"), "")

    @patch(
        "pipeline.bursa_news_dataset.load_stock_metadata",
        return_value={"1155.KL": "Malayan Banking Berhad"},
    )
    def test_template_deduplicates_and_builds_runtime_model_text(self, _metadata) -> None:
        source = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "provider": "GDELT/CommonCrawl",
                    "source_name": "Example",
                    "source_domain": "example.com",
                    "published_at": "2024-07-01T02:00:00Z",
                    "title": "Maybank profit rises",
                    "description": "Quarterly profit improved.",
                    "url": "https://example.com/a?utm_source=x",
                    "automatic_relevance": "included",
                },
                {
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "provider": "GDELT/CommonCrawl",
                    "source_name": "Example",
                    "source_domain": "example.com",
                    "published_at": "2024-07-01T02:00:00Z",
                    "title": "Maybank profit rises",
                    "description": "Quarterly profit improved.",
                    "url": "https://example.com/a?utm_source=y",
                    "automatic_relevance": "included",
                },
            ]
        )
        result = build_annotation_template(source)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["split"], "validation")
        self.assertEqual(
            result.iloc[0]["model_text"],
            "Maybank profit rises. Quarterly profit improved.",
        )

    @patch(
        "pipeline.bursa_news_dataset.load_stock_metadata",
        return_value={"1155.KL": "Malayan Banking Berhad"},
    )
    def test_template_preserves_distinct_recovery_package_rows(self, _metadata) -> None:
        source = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "published_at": "2023-01-01T00:00:00Z",
                    "title": "Maybank result one",
                    "source_url": "https://one.example/story",
                    "article_text": "First recovered article body.",
                },
                {
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "published_at": "2023-01-02T00:00:00Z",
                    "title": "Maybank result two",
                    "source_url": "https://two.example/story",
                    "article_text": "Second recovered article body.",
                },
            ]
        )

        result = build_annotation_template(source)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            set(result["url"]),
            {"https://one.example/story", "https://two.example/story"},
        )
        self.assertTrue(
            result["model_text"].str.contains("recovered article body").all()
        )

    def test_annotation_disagreement_requires_adjudication(self) -> None:
        frame = pd.DataFrame(
            {
                "annotator_1_label": ["positive", "positive"],
                "annotator_2_label": ["positive", "negative"],
                "adjudicated_label": ["", "neutral"],
                "sentiment_label": ["", ""],
            }
        )
        result = resolve_independent_annotations(frame)
        self.assertEqual(result.loc[0, "sentiment_label"], "positive")
        self.assertEqual(result.loc[0, "annotation_agreement"], "agreed")
        self.assertEqual(result.loc[1, "sentiment_label"], "neutral")
        self.assertEqual(result.loc[1, "annotation_agreement"], "adjudicated")

    def test_materialize_external_labels_rewrites_to_llm_assisted_label(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "article_id": "article-a",
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "published_at": "2024-07-01T02:00:00Z",
                    "title": "Maybank profit rises",
                    "article_text": "Maybank reported stronger quarterly profit.",
                    "source_name": "Example",
                    "source_domain": "example.com",
                    "url": "https://example.com/a",
                    "content_sha256": "content-a",
                    "automatic_entity_relevant": True,
                    "entity_relevant": True,
                    "sentiment_label": "positive",
                    "event_category": "earnings",
                    "language": "en",
                    "labelled_at": "2026-08-01T00:00:00Z",
                    "label_source": "external_single_weak",
                }
            ]
        )

        result = materialize_llm_assisted_dataset(source)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["label_source"], "llm_assisted_label")
        self.assertEqual(
            result.iloc[0]["annotation_agreement"], "llm_assisted_label"
        )
        self.assertEqual(result.iloc[0]["label_source_original"], "external_single_weak")
        self.assertTrue(result.iloc[0]["llm_assisted_label"])
        self.assertEqual(result.iloc[0]["split"], "validation")
        self.assertTrue(str(result.iloc[0]["event_id"]))
        self.assertIn("stronger quarterly profit", result.iloc[0]["model_text"])

        legacy_alias = materialize_external_single_weak_dataset(source)
        self.assertEqual(legacy_alias.iloc[0]["label_source"], "llm_assisted_label")

    def test_llm_assisted_label_policy_accepts_migrated_rows(self) -> None:
        rows = []
        for index, (published_at, split, sentiment) in enumerate(
            (
                ("2023-06-01T00:00:00Z", "training", "negative"),
                ("2024-06-01T00:00:00Z", "validation", "neutral"),
                ("2025-06-01T00:00:00Z", "test", "positive"),
            )
        ):
            rows.append(
                {
                    "article_id": f"article-{index}",
                    "ticker": "1155.KL",
                    "published_at": published_at,
                    "model_text": f"article text {index}",
                    "language": "en",
                    "automatic_entity_relevant": True,
                    "entity_relevant": True,
                    "sentiment_label": sentiment,
                    "event_id": f"event-{index}",
                    "split": split,
                    "content_sha256": f"content-{index}",
                    "label_source": "llm_assisted_label",
                }
            )
        policy = {
            "policy_version": "test-llm-assisted-label",
            "labeling": {
                "mode": "llm_assisted_label",
                "require_llm_assisted_label": True,
                "require_independent_double_labels": False,
                "enforce_entity_filter_metrics": False,
            },
            "required_columns": list(rows[0]),
            "minimum_rows": 3,
            "minimum_split_rows": {
                "training": 1,
                "validation": 1,
                "test": 1,
            },
            "minimum_rows_per_ticker": 3,
            "minimum_test_rows_per_ticker": 1,
            "required_tickers": ["1155.KL"],
            "minimum_double_label_fraction": {},
            "year_quality_gates": {},
        }

        report = validate_bursa_news_dataset(pd.DataFrame(rows), policy=policy)

        self.assertTrue(report.ready, report.reason)
        self.assertIsNone(report.inter_annotator_kappa)
        self.assertTrue(any("LLM-assisted" in item for item in report.warnings))

    def test_flexible_single_label_policy_reports_counts_without_frozen_size_gates(self) -> None:
        rows = []
        for index, (published_at, split, sentiment) in enumerate(
            (
                ("2023-06-01T00:00:00Z", "training", "negative"),
                ("2024-06-01T00:00:00Z", "validation", "neutral"),
                ("2025-06-01T00:00:00Z", "test", "positive"),
            )
        ):
            rows.append(
                {
                    "article_id": f"single-{index}",
                    "ticker": "1155.KL",
                    "published_at": published_at,
                    "model_text": f"article text {index}",
                    "language": "en",
                    "automatic_entity_relevant": True,
                    "entity_relevant": True,
                    "sentiment_label": sentiment,
                    "event_id": f"single-event-{index}",
                    "split": split,
                    "content_sha256": f"single-content-{index}",
                }
            )
        policy = {
            "policy_version": "test-flexible-single-label",
            "labeling": {
                "mode": "single_label",
                "require_independent_double_labels": False,
                "enforce_entity_filter_metrics": False,
            },
            "readiness_gates": {
                "enforce_dataset_size": False,
                "require_nonempty_splits": True,
                "require_all_tickers": True,
                "enforce_independent_agreement": False,
            },
            "required_columns": list(rows[0]),
            "minimum_rows": 3000,
            "minimum_split_rows": {
                "training": 2000,
                "validation": 500,
                "test": 500,
            },
            "minimum_rows_per_ticker": 300,
            "minimum_test_rows_per_ticker": 30,
            "required_tickers": ["1155.KL"],
            "split_periods": {
                "training": ["2019-01-01", "2023-12-31"],
                "validation": ["2024-01-01", "2024-12-31"],
                "test": ["2025-01-01", "2025-12-31"],
            },
            "year_quality_gates": {},
        }

        report = validate_bursa_news_dataset(pd.DataFrame(rows), policy=policy)

        self.assertTrue(report.ready, report.reason)
        self.assertEqual(report.labelled_relevant_rows, 3)
        self.assertEqual(report.split_counts, {"training": 1, "validation": 1, "test": 1})
        self.assertIsNone(report.inter_annotator_kappa)
        self.assertTrue(any("volume readiness gates are disabled" in item for item in report.warnings))
        self.assertTrue(any("double-labelling" in item for item in report.warnings))

    def test_readiness_fails_when_event_crosses_temporal_splits(self) -> None:
        policy = {
            "policy_version": "test",
            "required_columns": [
                "article_id",
                "ticker",
                "published_at",
                "model_text",
                "automatic_entity_relevant",
                "entity_relevant",
                "sentiment_label",
                "event_id",
                "annotator_1_label",
                "annotator_2_label",
                "adjudicated_label",
                "split",
                "content_sha256",
            ],
            "minimum_rows": 2,
            "minimum_split_rows": {"training": 1, "validation": 1},
            "minimum_rows_per_ticker": 2,
            "minimum_test_rows_per_ticker": 0,
            "required_tickers": ["1155.KL"],
            "minimum_double_label_fraction": {
                "training": 1.0,
                "validation": 1.0,
            },
            "minimum_inter_annotator_kappa": -1.0,
            "year_quality_gates": {
                "minimum_entity_precision": 0.0,
                "minimum_entity_recall": 0.0,
            },
        }
        frame = pd.DataFrame(
            [
                {
                    "article_id": "a",
                    "ticker": "1155.KL",
                    "published_at": "2023-12-31T00:00:00Z",
                    "model_text": "A",
                    "automatic_entity_relevant": True,
                    "entity_relevant": True,
                    "sentiment_label": "positive",
                    "event_id": "same-event",
                    "annotator_1_label": "positive",
                    "annotator_2_label": "positive",
                    "adjudicated_label": "",
                    "split": "training",
                    "content_sha256": "one",
                },
                {
                    "article_id": "b",
                    "ticker": "1155.KL",
                    "published_at": "2024-01-01T00:00:00Z",
                    "model_text": "B",
                    "automatic_entity_relevant": True,
                    "entity_relevant": True,
                    "sentiment_label": "negative",
                    "event_id": "same-event",
                    "annotator_1_label": "negative",
                    "annotator_2_label": "negative",
                    "adjudicated_label": "",
                    "split": "validation",
                    "content_sha256": "two",
                },
            ]
        )
        report = validate_bursa_news_dataset(frame, policy=policy)
        self.assertFalse(report.ready)
        self.assertIn("event clusters cross", report.reason)

    def test_year_report_requires_recovery_precision_and_company_coverage(self) -> None:
        policy = {
            "required_tickers": ["1155.KL"],
            "year_quality_gates": {
                "minimum_text_recovery_rate": 0.60,
                "minimum_entity_precision": 0.90,
                "minimum_entity_recall": 0.75,
                "minimum_timestamp_completeness": 0.95,
                "minimum_source_domains_per_ticker": 2,
                "minimum_accepted_articles": 2,
            },
        }
        indexed = pd.DataFrame([{"id": 1}, {"id": 2}])
        accepted = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "source_domain": "one.example",
                    "published_at": "2024-01-01",
                },
                {
                    "ticker": "1155.KL",
                    "source_domain": "two.example",
                    "published_at": "2024-01-02",
                },
            ]
        )
        reviewed = pd.DataFrame(
            [
                {"automatic_included": True, "actually_related": True},
                {"automatic_included": False, "actually_related": False},
            ]
        )
        report = build_year_quality_report(
            2024, indexed, indexed.copy(), accepted, reviewed, policy=policy
        )
        self.assertTrue(report["ready"])


if __name__ == "__main__":
    unittest.main()
