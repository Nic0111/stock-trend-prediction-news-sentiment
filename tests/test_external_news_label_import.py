from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.external_news_label_import import (
    ExternalNewsImportError,
    import_external_news_labels,
)


def _hash(article_id: str) -> str:
    return hashlib.sha256(article_id.encode("utf-8")).hexdigest()


def _article(article_id: str, year: int) -> dict[str, object]:
    return {
        "article_id": article_id,
        "ticker": "1155.KL",
        "company_name": "Malayan Banking Berhad",
        "published_at": f"{year}-06-01T08:00:00Z",
        "title": f"Maybank result {article_id}",
        "source_domain": "example.test",
        "source_url": f"https://example.test/{article_id}",
        "model_text": "The company reported stronger earnings.",
        "content_sha256": _hash(article_id),
    }


def _single(article: dict[str, object], *, sentiment: str = "positive") -> dict[str, object]:
    return {
        "article_id": article["article_id"],
        "content_sha256": article["content_sha256"],
        "entity_relevant": True,
        "sentiment": sentiment,
        "event_category": "earnings",
        "language": "en",
        "confidence": 0.9,
        "evidence": ["stronger earnings"],
        "rationale": "The article describes company earnings.",
        "label_status": "labelled",
        "labelled_by": "codex-reviewer",
        "labelled_at": "2026-07-31T00:00:00Z",
    }


class ExternalNewsLabelImportTests(unittest.TestCase):
    def _write_snapshot(self, root: Path, rows: list[dict[str, object]]) -> Path:
        path = root / "external-label-input.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def _write_labels(self, root: Path, rows: list[dict[str, object]]) -> Path:
        path = root / "returned.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_imports_single_and_consensus_without_touching_production(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_article("a2018", 2018), _article("a2019", 2019), _article("a2024", 2024), _article("a2025", 2025)]
            snapshot_path = self._write_snapshot(root, rows)
            single = _single(rows[1])
            # Prefix records are easier to consume from automated dual-label
            # artifacts than top-level fields, so keep only the prefix fields.
            consensus = {
                "article_id": rows[2]["article_id"],
                "content_sha256": rows[2]["content_sha256"],
                "labeler_a_entity_relevant": True,
                "labeler_a_sentiment": "positive",
                "labeler_a_event_category": "earnings",
                "labeler_a_language": "en",
                "labeler_a_confidence": 0.9,
                "labeler_a_model": "model-a",
                "labeler_a_label_status": "labelled",
                "labeler_b_entity_relevant": True,
                "labeler_b_sentiment": "positive",
                "labeler_b_event_category": "earnings",
                "labeler_b_language": "en",
                "labeler_b_confidence": 0.8,
                "labeler_b_model": "model-b",
                "labeler_b_label_status": "labelled",
                "consensus_status": "accepted",
            }
            review = _single(rows[3])
            review["label_status"] = "needs_review"
            labels_path = self._write_labels(root, [single, consensus, review])
            output = root / "experimental.jsonl"
            report = import_external_news_labels(
                snapshot_path,
                labels_path,
                output_dataset=output,
            )
            imported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            queue = [json.loads(line) for line in (root / "experimental-review-queue.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["accepted_rows"], 2)
            self.assertEqual(report["review_rows"], 1)
            self.assertEqual({row["split"] for row in imported}, {"training", "validation"})
            self.assertEqual({row["label_source"] for row in imported}, {"llm_assisted_label"})
            self.assertEqual(report["coverage"]["accepted_by_sentiment"]["positive"], 2)
            self.assertEqual(report["coverage"]["accepted_by_year"]["2024"], 1)
            single_row = next(
                row
                for row in imported
                if row["label_source_original"] == "external_single_weak"
            )
            self.assertEqual(single_row["labeler_a_model"], "")
            self.assertEqual(single_row["external_labelled_by"], "codex-reviewer")
            self.assertTrue(single_row["llm_assisted_label"])
            self.assertEqual(queue[0]["review_reason"], "needs_review")
            self.assertFalse((root / "bursa-news-llm-pseudolabelled.jsonl").exists())

    def test_backfill_is_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = _article("old", 2018)
            current = _article("current", 2019)
            snapshot = self._write_snapshot(root, [old, current])
            labels = self._write_labels(root, [_single(old), _single(current)])
            normal = import_external_news_labels(snapshot, labels, output_dataset=root / "normal.jsonl")
            self.assertEqual(normal["excluded_backfill_rows"], 1)
            self.assertEqual(normal["accepted_rows"], 1)
            backfill = import_external_news_labels(snapshot, labels, output_dataset=root / "backfill.jsonl", allow_backfill=True)
            self.assertEqual(backfill["accepted_rows"], 2)
            imported = [json.loads(line) for line in (root / "backfill.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["split"] == "training" for row in imported))

    def test_rejects_missing_duplicate_stale_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _article("first", 2024)
            second = _article("second", 2025)
            snapshot = self._write_snapshot(root, [first, second])
            with self.assertRaisesRegex(ExternalNewsImportError, "missing labels"):
                import_external_news_labels(snapshot, self._write_labels(root, [_single(first)]), output_dataset=root / "out.jsonl")
            duplicate = _single(first)
            with self.assertRaisesRegex(ExternalNewsImportError, "duplicate article_id"):
                import_external_news_labels(snapshot, self._write_labels(root, [duplicate, dict(duplicate)]), output_dataset=root / "out.jsonl")
            stale = _single(first)
            stale["article_id"] = "stale"
            stale["content_sha256"] = _hash("stale")
            with self.assertRaisesRegex(ExternalNewsImportError, "stale/unknown"):
                import_external_news_labels(snapshot, self._write_labels(root, [stale, _single(second)]), output_dataset=root / "out.jsonl")
            bad_hash = _single(first)
            bad_hash["content_sha256"] = _hash("wrong")
            with self.assertRaisesRegex(ExternalNewsImportError, "content_sha256 mismatch"):
                import_external_news_labels(snapshot, self._write_labels(root, [bad_hash, _single(second)]), output_dataset=root / "out.jsonl")

    def test_reviewer_csv_is_supported_and_disagreement_is_queued(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _article("first", 2024)
            second = _article("second", 2025)
            snapshot = self._write_snapshot(root, [first, second])
            returned = root / "review.csv"
            with returned.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "article_id",
                        "content_sha256",
                        "annotator_1_entity_relevant",
                        "annotator_1_sentiment",
                        "annotator_1_labelled_by",
                        "annotator_2_entity_relevant",
                        "annotator_2_sentiment",
                        "annotator_2_labelled_by",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "article_id": first["article_id"],
                        "content_sha256": first["content_sha256"],
                        "annotator_1_entity_relevant": "true",
                        "annotator_1_sentiment": "positive",
                        "annotator_1_labelled_by": "a",
                        "annotator_2_entity_relevant": "true",
                        "annotator_2_sentiment": "negative",
                        "annotator_2_labelled_by": "b",
                    }
                )
                writer.writerow(
                    {
                        "article_id": second["article_id"],
                        "content_sha256": second["content_sha256"],
                        "annotator_1_entity_relevant": "true",
                        "annotator_1_sentiment": "neutral",
                        "annotator_1_labelled_by": "a",
                        "annotator_2_entity_relevant": "true",
                        "annotator_2_sentiment": "neutral",
                        "annotator_2_labelled_by": "b",
                    }
                )
            report = import_external_news_labels(snapshot, returned, output_dataset=root / "out.jsonl")
            self.assertEqual(report["accepted_rows"], 1)
            self.assertEqual(report["review_rows"], 1)
            queue = [json.loads(line) for line in (root / "out-review-queue.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(queue[0]["review_reason"], "label_disagreement")

    def test_compact_reviewer_columns_can_label_sentiment_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            article = _article("compact", 2025)
            snapshot = self._write_snapshot(root, [article])
            returned = root / "compact.csv"
            with returned.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["article_id", "content_sha256", "label_a", "label_b"])
                writer.writeheader()
                writer.writerow(
                    {
                        "article_id": article["article_id"],
                        "content_sha256": article["content_sha256"],
                        "label_a": "positive",
                        "label_b": "positive",
                    }
                )
            result = import_external_news_labels(snapshot, returned, output_dataset=root / "compact.jsonl")
            self.assertEqual(result["consensus_rows"], 1)

    def test_label_text_hash_is_locked_when_snapshot_provides_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            article = _article("label-text", 2024)
            article["label_text_sha256"] = _hash("bounded-view")
            snapshot = self._write_snapshot(root, [article])
            label = _single(article)
            label["label_text_sha256"] = article["label_text_sha256"]
            returned = self._write_labels(root, [label])
            result = import_external_news_labels(snapshot, returned, output_dataset=root / "out.jsonl")
            self.assertEqual(result["accepted_rows"], 1)
            label["label_text_sha256"] = _hash("changed-view")
            self._write_labels(root, [label])
            with self.assertRaisesRegex(ExternalNewsImportError, "label_text_sha256 mismatch"):
                import_external_news_labels(snapshot, returned, output_dataset=root / "bad.jsonl")


if __name__ == "__main__":
    unittest.main()
