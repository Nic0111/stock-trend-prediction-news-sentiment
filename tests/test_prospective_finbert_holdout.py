from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.prospective_finbert_holdout import (
    ProspectiveHoldoutError,
    bind_prospective_candidate,
    import_prospective_labels,
    prepare_prospective_holdout,
    prospective_holdout_status,
)
from pipeline.sentiment_text import TEXT_MODE_TARGET_AWARE, build_sentiment_text


class ProspectiveFinBertHoldoutTests(unittest.TestCase):
    def _policy(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "required_tickers": ["1155.KL", "1295.KL"],
                    "sentiment_ensemble": {
                        "prospective_holdout": {
                            "year": 2026,
                            "target_candidates_per_ticker": 5,
                            "shadow_fraction": 0.2,
                            "minimum_final_rows": 4,
                            "minimum_final_rows_per_ticker": 1,
                            "minimum_final_class_rows": 1,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_prepare_then_import_keeps_2026_data_outside_master(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_path = root / "policy.json"
            self._policy(policy_path)
            master_path = root / "master.csv"
            pd.DataFrame(
                [
                    {
                        "ticker": "1155.KL",
                        "published_at": "2025-12-31T00:00:00Z",
                        "title": "Maybank old result",
                        "description": "Existing canonical article.",
                        "url": "https://example.test/old",
                    }
                ]
            ).to_csv(master_path, index=False)
            candidates_path = root / "candidates.csv"
            pd.DataFrame(
                [
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "published_at": "2026-01-10T00:00:00Z",
                        "title": "Maybank earnings improve",
                        "description": "Maybank reported higher quarterly profit.",
                        "url": "https://example.test/maybank-1",
                        "language": "en",
                    },
                    {
                        "ticker": "1295.KL",
                        "company_name": "Public Bank Berhad",
                        "published_at": "2026-01-11T00:00:00Z",
                        "title": "Public Bank lending grows",
                        "description": "Public Bank announced higher lending income.",
                        "url": "https://example.test/public-bank-1",
                        "language": "en",
                    },
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "published_at": "2026-01-12T00:00:00Z",
                        "title": "Maybank old result",
                        "description": "Existing canonical article.",
                        "url": "https://example.test/old",
                        "language": "en",
                    },
                ]
            ).to_csv(candidates_path, index=False)
            output_dir = root / "holdout"

            prepared = prepare_prospective_holdout(
                candidates_path,
                output_dir=output_dir,
                policy_path=policy_path,
                master_dataset_path=master_path,
            )
            self.assertTrue(prepared["prepared"])
            cohort = pd.read_csv(output_dir / "candidates.csv")
            self.assertEqual(len(cohort), 2)
            self.assertTrue(cohort["published_at"].str.startswith("2026").all())
            self.assertFalse(master_path.with_suffix(".jsonl").exists())
            label_template = pd.read_csv(output_dir / "llm-assisted-label-template.csv")
            self.assertIn("recovery_status", label_template.columns)
            self.assertIn("label_evidence_scope", label_template.columns)
            self.assertIn("automatic_entity_status", label_template.columns)

            labels = cohort[["article_id"]].copy()
            labels["llm_assisted_entity_relevant"] = True
            labels["llm_assisted_sentiment_label"] = ["positive", "neutral"]
            labels["label_status"] = "llm_assisted_label"
            labels_path = root / "labels.csv"
            labels.to_csv(labels_path, index=False)
            imported = import_prospective_labels(labels_path, output_dir=output_dir)
            self.assertTrue(imported["imported"])
            self.assertEqual(prospective_holdout_status(output_dir=output_dir)["status"], "labels_sealed")
            self.assertTrue((output_dir / "llm-assisted-labelled.csv").exists())

    def test_import_rejects_relevant_rows_without_llm_assisted_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_path = root / "policy.json"
            self._policy(policy_path)
            master_path = root / "master.csv"
            pd.DataFrame(columns=["ticker", "published_at", "title", "description", "url"]).to_csv(master_path, index=False)
            candidates_path = root / "candidates.csv"
            pd.DataFrame(
                [
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "published_at": "2026-01-10T00:00:00Z",
                        "title": "Maybank profit rises",
                        "description": "Maybank improved profit.",
                        "url": "https://example.test/maybank",
                        "language": "en",
                    }
                ]
            ).to_csv(candidates_path, index=False)
            output_dir = root / "holdout"
            prepare_prospective_holdout(
                candidates_path,
                output_dir=output_dir,
                policy_path=policy_path,
                master_dataset_path=master_path,
            )
            cohort = pd.read_csv(output_dir / "candidates.csv")
            labels_path = root / "labels.csv"
            pd.DataFrame(
                {
                    "article_id": cohort["article_id"],
                    "llm_assisted_entity_relevant": [True],
                    "llm_assisted_sentiment_label": ["positive"],
                    "label_status": [""],
                }
            ).to_csv(labels_path, index=False)
            with self.assertRaises(ProspectiveHoldoutError):
                import_prospective_labels(labels_path, output_dir=output_dir)

    def test_prepare_accepts_numeric_true_entity_flags_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_path = root / "policy.json"
            self._policy(policy_path)
            master_path = root / "master.csv"
            pd.DataFrame(columns=["ticker", "published_at", "title", "description", "url"]).to_csv(master_path, index=False)
            candidates_path = root / "candidates.jsonl"
            candidates_path.write_text(
                json.dumps(
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "published_at": "2026-01-10T00:00:00Z",
                        "title": "Maybank profit rises",
                        "description": "Maybank reported stronger earnings.",
                        "url": "https://example.test/maybank-numeric-flag",
                        "language": "en",
                        "automatic_entity_relevant": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            prepare_prospective_holdout(
                candidates_path,
                output_dir=root / "holdout",
                policy_path=policy_path,
                master_dataset_path=master_path,
            )
            cohort = pd.read_csv(root / "holdout" / "candidates.csv")
            self.assertEqual(len(cohort), 1)

    def test_target_aware_text_avoids_serialized_nan(self) -> None:
        text = build_sentiment_text(
            {
                "company_name": "Malayan Banking Berhad",
                "ticker": "1155.KL",
                "title": "Maybank profit rises",
                "description": pd.NA,
                "content_snippet": "Maybank announced a stronger quarterly result.",
            },
            mode=TEXT_MODE_TARGET_AWARE,
        )
        self.assertIn("Maybank", text)
        self.assertNotIn("nan", text.casefold())

    def test_bind_rejects_candidate_that_failed_development_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "holdout"
            output.mkdir()
            labelled = output / "llm-assisted-labelled.csv"
            pd.DataFrame({"article_id": ["article-1"]}).to_csv(labelled, index=False)
            import hashlib

            labelled_hash = hashlib.sha256(labelled.read_bytes()).hexdigest()
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "labels_sealed",
                        "labelled_file": str(labelled),
                        "labelled_sha256": labelled_hash,
                    }
                ),
                encoding="utf-8",
            )
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "gates": {"checks": {"validation_macro_f1_uplift": False}},
                        "runtime_candidate": {"backend": "weighted_finbert_ensemble"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ProspectiveHoldoutError):
                bind_prospective_candidate(report, output_dir=output)


if __name__ == "__main__":
    unittest.main()
