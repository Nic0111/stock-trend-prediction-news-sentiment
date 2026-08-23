from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.news_evidence_readiness import check_news_evidence_readiness


class NewsEvidenceReadinessTests(unittest.TestCase):
    def test_missing_malaysian_benchmark_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "policy_version": "test",
                        "benchmark_path": "data/does-not-exist.csv",
                        "minimum_rows": 1,
                        "minimum_rows_per_ticker": 1,
                        "required_tickers": ["1155.KL"],
                        "required_columns": ["ticker", "split"],
                        "required_splits": ["development", "test"],
                    }
                ),
                encoding="utf-8",
            )
            report = check_news_evidence_readiness(policy)
            self.assertFalse(report.ready)
            self.assertIn("not found", report.reason)

    def test_extended_bursa_quality_errors_are_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text("ticker,split\n1155.KL,test\n", encoding="utf-8")
            quality = root / "quality.json"
            quality.write_text(
                json.dumps(
                    {
                        "policy_version": "quality-test",
                        "required_columns": ["ticker", "split", "model_text"],
                    }
                ),
                encoding="utf-8",
            )
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "policy_version": "test",
                        "benchmark_path": str(dataset),
                        "minimum_rows": 1,
                        "minimum_rows_per_ticker": 1,
                        "required_tickers": ["1155.KL"],
                        "required_columns": ["ticker", "split"],
                        "required_splits": ["test"],
                        "dataset_quality_policy_path": str(quality),
                    }
                ),
                encoding="utf-8",
            )
            report = check_news_evidence_readiness(policy)
            self.assertFalse(report.ready)
            self.assertIn("model_text", report.reason)

    def test_disabled_volume_gates_keep_basic_ticker_and_split_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.csv"
            dataset.write_text(
                "ticker,split\n1155.KL,development\n1155.KL,test\n",
                encoding="utf-8",
            )
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "policy_version": "flexible-volume-test",
                        "benchmark_path": str(dataset),
                        "minimum_rows": 3000,
                        "minimum_rows_per_ticker": 300,
                        "readiness_gates": {
                            "enforce_dataset_size": False,
                            "require_all_tickers": True,
                        },
                        "required_tickers": ["1155.KL"],
                        "required_columns": ["ticker", "split"],
                        "required_splits": ["development", "test"],
                    }
                ),
                encoding="utf-8",
            )
            report = check_news_evidence_readiness(policy)
            self.assertTrue(report.ready, report.reason)
            self.assertEqual(report.row_count, 2)


if __name__ == "__main__":
    unittest.main()
