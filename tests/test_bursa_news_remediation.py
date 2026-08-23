import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.bursa_news_remediation import audit_bursa_news_labels, export_label_expansion_queue


class BursaNewsRemediationTests(unittest.TestCase):
    def test_audit_and_queue_do_not_mutate_master(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "master.jsonl"
            pd.DataFrame(
                [
                    {"article_id": "a", "ticker": "1023.KL", "published_at": "2024-01-01", "entity_relevant": True, "sentiment_label": "positive", "model_text": "x"},
                    {"article_id": "b", "ticker": "1023.KL", "published_at": "2024-01-02", "entity_relevant": True, "sentiment_label": "neutral", "model_text": "y"},
                ]
            ).to_json(master, orient="records", lines=True)
            pack = root / "pack.csv"
            pd.DataFrame(
                [
                    {"article_id": "a", "ticker": "1023.KL", "sentiment": "positive", "article_text": "duplicate"},
                    {"article_id": "c", "ticker": "1155.KL", "sentiment": "", "article_text": "candidate"},
                ]
            ).to_csv(pack, index=False)
            report = audit_bursa_news_labels(master, review_packs=(pack,), output_dir=root / "audit")
            self.assertEqual(report["class_counts"]["neutral"], 1)
            result = export_label_expansion_queue((pack,), master_path=master, output_path=root / "queue.csv")
            self.assertEqual(result["rows"], 1)
            self.assertEqual(len(pd.read_json(master, lines=True)), 2)
            self.assertTrue(json.loads((root / "audit" / "label-quality-audit.json").read_text())["master_rows"] == 2)
