from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.external_news_label_export import export_external_label_package


def _row(
    *,
    ticker: str = "1155.KL",
    url: str = "https://example.test/maybank?utm_source=test",
    source: str = "publisher",
    quality: str = "",
) -> dict[str, object]:
    body = " ".join(
        ["Maybank reported improved earnings and stronger financial results."]
        * 35
    )
    return {
        "ticker": ticker,
        "company_name": "Malayan Banking Berhad",
        "published_at": "2023-06-01T08:00:00Z",
        "title": "Maybank reports improved earnings",
        "url": url,
        "content_snippet": body,
        "text_source": source,
        "is_pseudo_document": source == "gdelt_webngrams",
        "publisher_full_text_available": source == "publisher",
        "recovery_status": "recovered",
        "reconstruction_quality_tier": quality,
        "reconstruction_quality_score": 0.8 if quality == "high" else 0.4,
        "unique_token_count": 220 if quality == "high" else 90,
    }


class ExternalNewsLabelExportTests(unittest.TestCase):
    def test_export_deduplicates_and_keeps_high_quality_pseudo_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovery = root / "2023" / "recovery" / "raw"
            recovery.mkdir(parents=True)
            rows = [
                _row(),
                _row(url="https://example.test/maybank"),
                _row(
                    ticker="1023.KL",
                    url="https://example.test/cimb",
                    source="gdelt_webngrams",
                    quality="high",
                ),
                _row(
                    ticker="1295.KL",
                    url="https://example.test/public-bank",
                    source="gdelt_webngrams",
                    quality="medium",
                ),
            ]
            path = recovery / "hybrid-recovered.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            result = export_external_label_package(
                root,
                output_dir=root / "labels",
                years=(2023,),
                batch_size=1,
            )
            exported = [
                json.loads(line)
                for line in (root / "labels" / "external-label-input.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        self.assertEqual(result["deduplicated_label_rows"], 2)
        self.assertEqual(result["batch_count"], 2)
        self.assertEqual(
            {row["text_source"] for row in exported},
            {"publisher", "gdelt_webngrams"},
        )

    def test_export_never_calls_a_labelling_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = export_external_label_package(
                Path(directory),
                output_dir=Path(directory) / "labels",
                years=(2015, 2025),
            )
        self.assertEqual(result["llm_calls"], 0)
        self.assertFalse(result["production_changes"])

    def test_label_text_is_bounded_while_article_text_and_manifest_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovery = root / "2023" / "recovery" / "raw"
            recovery.mkdir(parents=True)
            source = _row()
            recovery.joinpath("hybrid-recovered.jsonl").write_text(
                json.dumps(source) + "\n", encoding="utf-8"
            )
            result = export_external_label_package(
                root,
                output_dir=root / "labels",
                years=(2023,),
            )
            exported = json.loads(
                (root / "labels" / "external-label-input.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            manifest = json.loads(
                (root / "labels" / "external-label-snapshot-manifest.json")
                .read_text(encoding="utf-8")
            )
            snapshot_hash = hashlib.sha256(
                (root / "labels" / "external-label-input.jsonl").read_bytes()
            ).hexdigest()

        self.assertLessEqual(len(exported["label_text"].split()), 128)
        self.assertEqual(exported["article_text"], source["content_snippet"])
        self.assertEqual(
            exported["label_text_sha256"],
            hashlib.sha256(exported["label_text"].encode("utf-8")).hexdigest(),
        )
        self.assertEqual(result["snapshot_sha256"], manifest["snapshot_sha256"])
        self.assertEqual(result["snapshot_sha256"], snapshot_hash)

    def test_batches_respect_row_and_label_content_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovery = root / "2023" / "recovery" / "raw"
            recovery.mkdir(parents=True)
            rows = [
                _row(
                    ticker=f"{1000 + index}.KL",
                    url=f"https://example.test/article-{index}",
                )
                for index in range(30)
            ]
            recovery.joinpath("hybrid-recovered.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            result = export_external_label_package(
                root,
                output_dir=root / "labels",
                years=(2023,),
                batch_size=100,
            )
            manifest = json.loads(
                (root / "labels" / "external-label-snapshot-manifest.json")
                .read_text(encoding="utf-8")
            )

        self.assertEqual(result["batch_size"], 25)
        self.assertEqual(result["batch_count"], 2)
        self.assertTrue(all(value <= 25 for value in result["batch_row_counts"].values()))
        self.assertTrue(all(value <= 60000 for value in result["batch_label_chars"].values()))
        self.assertEqual(manifest["batch_limits"]["max_rows"], 25)

    def test_chat_batches_are_compact_and_keep_full_text_in_snapshot_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recovery = root / "2023" / "recovery" / "raw"
            recovery.mkdir(parents=True)
            source = _row()
            source["content_snippet"] = " ".join(["Long article context."] * 400)
            recovery.joinpath("hybrid-recovered.jsonl").write_text(
                json.dumps(source) + "\n", encoding="utf-8"
            )
            output = root / "labels"
            export_external_label_package(root, output_dir=output, years=(2023,))
            batch_path = output / "batches" / "batch-0001.jsonl"
            batch_row = json.loads(batch_path.read_text(encoding="utf-8").splitlines()[0])
            snapshot_row = json.loads(
                (output / "external-label-input.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            batch_bytes = batch_path.stat().st_size

        self.assertNotIn("article_text", batch_row)
        self.assertNotIn("model_text", batch_row)
        self.assertEqual(snapshot_row["article_text"], source["content_snippet"])
        self.assertLessEqual(batch_bytes, 60000)


if __name__ == "__main__":
    unittest.main()
