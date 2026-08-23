from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.sentiment_remediation import (
    audit_sentiment_coverage,
    remediation_status,
    repair_codex_provenance,
)


def _row(ticker: str, label: str, *, source: str = "llm_assisted_label") -> dict:
    return {
        "article_id": f"{ticker}-{label}",
        "ticker": ticker,
        "language": "en",
        "entity_relevant": True,
        "sentiment_label": label,
        "split": "training",
        "label_source": source,
        "label_source_original": "codex_single_review" if source == "llm_assisted_label" else source,
        "external_labelled_by": "codex" if source == "llm_assisted_label" else "",
        "validation_status": "llm_single_pass" if source == "llm_assisted_label" else source,
        "llm_assisted_label": source == "llm_assisted_label",
    }


class SentimentRemediationTests(unittest.TestCase):
    def test_audit_prioritises_sparse_and_weak_companies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            rows = [_row("1155.KL", "positive"), _row("1155.KL", "neutral")]
            rows.extend(
                {
                    **_row("1023.KL", ("negative", "neutral", "positive")[index % 3], source="llm_assisted_label"),
                    "article_id": f"1023-{index}",
                }
                for index in range(150)
            )
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = audit_sentiment_coverage(dataset, benchmark_report=root / "missing.json")
            self.assertIn("1155.KL", report["priority_tickers"])
            self.assertNotIn("1023.KL", report["priority_tickers"])
            self.assertEqual(report["companies"]["1155.KL"]["eligible_rows"], 2)

    def test_audit_does_not_treat_moderate_neutral_f1_as_a_collection_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            rows = []
            for index in range(120):
                label = ("negative", "neutral", "positive")[index % 3]
                rows.append({**_row("1023.KL", label), "article_id": f"1023-{index}"})
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            benchmark = root / "benchmark.json"
            benchmark.write_text(
                json.dumps(
                    {
                        "final_2025": {
                            "per_company": {
                                "1023.KL": {
                                    "rows": 30,
                                    "ensemble": {"macro_f1": 0.65, "neutral_f1": 0.55},
                                    "phrasebank": {"macro_f1": 0.62},
                                    "macro_f1_delta": 0.03,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = audit_sentiment_coverage(dataset, benchmark_report=benchmark)
            self.assertNotIn("1023.KL", report["priority_tickers"])

    def test_provenance_repair_preserves_labels_and_writes_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            row = _row("1155.KL", "negative")
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = repair_codex_provenance(dataset, backup_dir=root / "backups")
            repaired = json.loads(dataset.read_text(encoding="utf-8"))
            self.assertEqual(repaired["sentiment_label"], "negative")
            self.assertEqual(repaired["label_source"], "llm_assisted_label")
            self.assertTrue(repaired["llm_assisted_label"])
            self.assertTrue(Path(result["backup_path"]).exists())
            self.assertFalse(result["labels_changed"])

    def test_status_is_read_only_for_unknown_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = remediation_status(run_id="missing", run_root=temporary)
            self.assertEqual(result["status"], "not_started")


if __name__ == "__main__":
    unittest.main()
