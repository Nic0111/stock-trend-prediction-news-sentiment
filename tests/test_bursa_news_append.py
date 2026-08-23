from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from pipeline.bursa_news_append import (
    BursaNewsAppendError,
    _filter_append_rows,
    _balance_targets,
    _route_balanced_rows,
    _sha256_file,
    commit_bursa_news_append,
    prepare_bursa_news_append,
    recover_bursa_news_append,
    triage_bursa_news_append,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _base_row(article_id: str = "existing") -> dict[str, object]:
    return {
        "article_id": article_id,
        "ticker": "1155.KL",
        "company_name": "Malayan Banking Berhad",
        "published_at": "2021-02-01T08:00:00+00:00",
        "url": f"https://news.example/{article_id}",
        "title": "Maybank reports earnings",
        "description": "Earnings increased.",
        "model_text": "Maybank reports earnings. Earnings increased.",
        "content_sha256": hashlib.sha256(article_id.encode()).hexdigest(),
        "entity_relevant": True,
        "sentiment_label": "positive",
        "split": "training",
        "event_id": f"event-{article_id}",
        "language": "en",
    }


class BursaNewsAppendTests(unittest.TestCase):
    def test_balance_targets_require_split_sum(self) -> None:
        self.assertEqual(
            _balance_targets(
                target_per_class_per_ticker=40,
                training_target=28,
                validation_target=6,
                test_target=6,
            ),
            {"training": 28, "validation": 6, "test": 6},
        )
        with self.assertRaises(ValueError):
            _balance_targets(
                target_per_class_per_ticker=40,
                training_target=30,
                validation_target=6,
                test_target=6,
            )

    def test_prepare_excludes_existing_url_and_selects_only_deficit_ticker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            _write_jsonl(dataset, [_base_row()])
            exports = root / "exports" / "2021" / "exports"
            _write_jsonl(
                exports / "2021-01.jsonl",
                [
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "url": "https://news.example/existing?utm_source=test",
                        "title": "Maybank reports earnings",
                        "published_at": "2021-02-01T08:00:00Z",
                    },
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "url": "https://news.example/new",
                        "title": "Maybank launches a new service",
                        "published_at": "2021-03-01T08:00:00Z",
                    },
                ],
            )
            policy = {"required_tickers": ["1155.KL"]}
            with patch("pipeline.bursa_news_append.load_policy", return_value=policy):
                report = prepare_bursa_news_append(
                    dataset_path=dataset,
                    candidate_root=root / "exports",
                    work_dir=root / "work",
                    years=[2021],
                    target_relevant_per_ticker=2,
                    recovery_multiplier=1,
                )
            self.assertEqual(report["candidate_rejections"]["url"], 1)
            self.assertEqual(report["selected_rows"], 1)
            selected = [
                json.loads(line)
                for line in Path(report["candidate_files"]["2021"]["path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(selected[0]["url"], "https://news.example/new")

    def test_recovery_uses_prepared_year_and_keeps_baseline_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            _write_jsonl(dataset, [_base_row()])
            work = root / "work"
            candidate = work / "candidates" / "2021" / "candidates.jsonl"
            _write_jsonl(candidate, [_base_row("candidate")])
            manifest = {
                "base_dataset": {
                    "dataset_path": str(dataset),
                    "dataset_sha256": _sha256_file(dataset),
                },
                "candidate_files": {"2021": {"path": str(candidate)}},
            }
            (work / "append-manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (work / "append-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            calls: list[dict[str, object]] = []

            def fake_runner(**kwargs):
                calls.append(kwargs)
                return {"completed": True, "recovery_attempted": True}

            report = recover_bursa_news_append(
                work_dir=work, max_runtime_minutes=1, recovery_runner=fake_runner
            )
            self.assertEqual(report["summaries"][0]["year"], 2021)
            self.assertEqual(calls[0]["paired_per_ticker"], 0)
            self.assertEqual(dataset.read_text(encoding="utf-8"), json.dumps(_base_row()) + "\n")

    def test_prepare_reuses_existing_workspace_instead_of_overwriting_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            _write_jsonl(dataset, [_base_row()])
            work = root / "work"
            manifest = {
                "status": "prepared",
                "base_dataset": {
                    "dataset_path": str(dataset),
                    "dataset_sha256": _sha256_file(dataset),
                },
                "selected_rows": 9,
            }
            work.mkdir()
            (work / "append-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = prepare_bursa_news_append(
                dataset_path=dataset,
                candidate_root=root / "missing-exports",
                work_dir=work,
                years=[2021],
            )
            self.assertTrue(report["resumed_existing_workspace"])
            self.assertEqual(report["selected_rows"], 9)

    def test_balanced_prepare_writes_class_split_matrix_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            _write_jsonl(dataset, [_base_row()])
            exports = root / "exports" / "2021" / "exports"
            candidates = []
            for index, split_date in enumerate(
                ("2021-03-01T08:00:00Z", "2024-03-01T08:00:00Z", "2025-03-01T08:00:00Z")
            ):
                candidates.append(
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "url": f"https://news.example/balance-{index}",
                        "title": f"Maybank balance candidate {index}",
                        "published_at": split_date,
                    }
                )
            # The loader is year-scoped; use a second export directory for the
            # validation/test examples while keeping the fixture tiny.
            _write_jsonl(exports / "2021-01.jsonl", candidates[:1])
            _write_jsonl(root / "exports" / "2024" / "exports" / "2024-01.jsonl", candidates[1:2])
            _write_jsonl(root / "exports" / "2025" / "exports" / "2025-01.jsonl", candidates[2:])
            with patch(
                "pipeline.bursa_news_append.load_policy",
                return_value={"required_tickers": ["1155.KL"]},
            ):
                report = prepare_bursa_news_append(
                    dataset_path=dataset,
                    candidate_root=root / "exports",
                    work_dir=root / "work",
                    years=[2021, 2024, 2025],
                    target_per_class_per_ticker=4,
                    training_target_per_class=2,
                    validation_target_per_class=1,
                    test_target_per_class=1,
                    recovery_multiplier=1,
                    max_candidates_per_cycle=20,
                )
            self.assertEqual(report["selection_mode"], "class_split_balanced")
            self.assertEqual(report["split_targets_per_class"]["training"], 2)
            self.assertIn("1155.KL", report["targets"])
            self.assertEqual(report["production_changes"], False)

    def test_triage_queue_is_explicitly_non_labeling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            _write_jsonl(dataset, [_base_row()])
            work = root / "work"
            recovery = work / "recovery-root" / "2021" / "recovery" / "raw"
            _write_jsonl(
                recovery / "hybrid-recovered.jsonl",
                [
                    {
                        "article_id": "triage-1",
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "published_at": "2021-03-01T08:00:00Z",
                        "url": "https://news.example/triage-1",
                        "title": "Maybank opens a new service",
                        "article_text": "Maybank reported a new service and stable earnings. " * 30,
                        "text_source": "publisher",
                    }
                ],
            )
            manifest = {
                "base_dataset": {
                    "dataset_path": str(dataset),
                    "dataset_sha256": _sha256_file(dataset),
                },
                "years": [2021],
                "split_targets_per_class": {"training": 2, "validation": 1, "test": 1},
            }
            work.mkdir(parents=True, exist_ok=True)
            (work / "append-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "pipeline.bursa_news_append.load_policy",
                return_value={"required_tickers": ["1155.KL"]},
            ), patch(
                "pipeline.finbert_sentiment_model.load_finbert_sentiment_model",
                side_effect=RuntimeError("test fixture has no model"),
            ):
                report = triage_bursa_news_append(work_dir=work)
            self.assertEqual(report["queue_rows"], 1)
            self.assertFalse(report["model_available"])
            queued = json.loads(
                (work / "triage" / "label-queue.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertTrue(queued["triage_model_predictions_are_not_labels"])
            self.assertNotIn("sentiment_label", queued)

    def test_balanced_commit_routing_keeps_surplus_and_irrelevant_in_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = pd.DataFrame([_base_row()])
            positive = dict(_base_row("new-positive"))
            positive.update(
                {
                    "url": "https://news.example/new-positive",
                    "title": "Maybank reports stable growth",
                    "content_sha256": hashlib.sha256(b"new-positive").hexdigest(),
                }
            )
            irrelevant = dict(positive)
            irrelevant.update(
                {
                    "article_id": "irrelevant",
                    "entity_relevant": False,
                    "sentiment_label": "",
                    "url": "https://news.example/irrelevant",
                    "content_sha256": hashlib.sha256(b"irrelevant").hexdigest(),
                }
            )
            non_english = dict(positive)
            non_english.update(
                {
                    "article_id": "non-english",
                    "url": "https://news.example/non-english",
                    "language": "ms",
                    "content_sha256": hashlib.sha256(b"non-english").hexdigest(),
                }
            )
            manifest = {
                "selection_mode": "class_split_balanced",
                "cycle_index": 1,
                "split_targets_per_class": {"training": 2, "validation": 1, "test": 1},
            }
            with patch(
                "pipeline.bursa_news_append.load_policy",
                return_value={"required_tickers": ["1155.KL"]},
            ):
                accepted, routed, reserve_paths = _route_balanced_rows(
                    [positive, irrelevant, non_english],
                    baseline,
                    manifest=manifest,
                    output=root,
                )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(routed["canonical"], 1)
            self.assertEqual(routed["entity_negative"], 1)
            self.assertEqual(routed["non_english_relevant"], 1)
            self.assertTrue(Path(reserve_paths["entity_negative"]).exists())
            self.assertTrue(Path(reserve_paths["non_english_relevant"]).exists())

    def test_final_deduplication_preserves_existing_row(self) -> None:
        base = pd.DataFrame([_base_row()])
        duplicate = dict(_base_row("new-id"))
        duplicate["url"] = "https://news.example/existing"
        duplicate["content_sha256"] = hashlib.sha256(b"new-id").hexdigest()
        unique = dict(_base_row("unique"))
        unique.update(
            {
                "url": "https://news.example/unique",
                "title": "Maybank opens a branch abroad",
                "content_sha256": hashlib.sha256(b"unique").hexdigest(),
            }
        )
        accepted, rejected = _filter_append_rows([duplicate, unique], base)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["article_id"], "unique")
        self.assertIn("deduplicated_url", rejected)

    def test_commit_aborts_if_baseline_changes_after_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            _write_jsonl(dataset, [_base_row()])
            work = root / "work"
            work.mkdir()
            manifest = {
                "base_dataset": {
                    "dataset_path": str(dataset),
                    "dataset_sha256": "not-the-current-hash",
                }
            }
            (work / "append-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(BursaNewsAppendError):
                commit_bursa_news_append(work_dir=work)

    def test_commit_preserves_existing_raw_lines_and_appends_only_validated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            original = _base_row()
            _write_jsonl(dataset, [original])
            work = root / "work"
            (work / "staging").mkdir(parents=True)
            new = dict(_base_row("new"))
            new.update(
                {
                    "url": "https://news.example/new",
                    "title": "Maybank completes a regional deal",
                    "content_sha256": hashlib.sha256(b"new").hexdigest(),
                    "event_id": "",
                }
            )
            _write_jsonl(work / "staging" / "codex-reviewed.jsonl", [new])
            manifest = {
                "base_dataset": {
                    "dataset_path": str(dataset),
                    "dataset_sha256": _sha256_file(dataset),
                }
            }
            (work / "append-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            ready = SimpleNamespace(ready=True, errors=(), to_dict=lambda: {"ready": True})
            with patch("pipeline.bursa_news_append.validate_bursa_news_dataset", return_value=ready), patch(
                "pipeline.bursa_news_append.load_policy", return_value={}
            ):
                report = commit_bursa_news_append(work_dir=work)
            rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["rows_appended"], 1)
            self.assertEqual(rows[0], original)
            self.assertEqual(rows[1]["article_id"], "new")


if __name__ == "__main__":
    unittest.main()
