from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd

from pipeline.bursa_news_dataset import build_annotation_template
from pipeline.gdelt_ngram_pilot import (
    DownloadBudgetExceeded,
    GDELTWebNGramClient,
    PublisherClient,
    ReconstructionResult,
    _checkpoint_recovered_payloads,
    _merge_recovered_rows,
    refresh_cumulative_recovered_output,
    merge_overlapping_windows,
    reconstruct_webngrams,
    run_gdelt_ngram_pilot,
)


def _synthetic_ngram_records(ticker: str = "1155.KL") -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for decile in range(0, 100, 10):
        prefix = ["Maybank", "reported"] if decile == 0 else ["The", "company"]
        tokens = (
            prefix
            + [f"term{decile}_{index}" for index in range(24)]
            + ["and", "the", "financial", "outlook", "improved", "."]
        )
        for center in range(7, len(tokens) - 7):
            records.append(
                {
                    "date": "2023-01-01T00:00:00Z",
                    "lang": "eng",
                    "pos": decile,
                    "pre": " ".join(tokens[center - 7 : center]),
                    "ngram": tokens[center],
                    "post": " ".join(tokens[center + 1 : center + 8]),
                    "url": "https://example.test/story",
                }
            )
    return records


class GDELTNGramPilotTests(unittest.TestCase):
    def test_merge_recovered_rows_preserves_existing_text_when_retry_is_unavailable(
        self,
    ) -> None:
        existing = {
            "ticker": "1155.KL",
            "url": "https://example.test/story?utm_source=test",
            "published_at": "2023-01-01T00:00:00Z",
            "content_snippet": "Maybank reported stronger earnings and loan growth.",
            "text_source": "publisher",
            "recovery_status": "recovered",
        }
        unavailable_retry = {
            "ticker": "1155.KL",
            "url": "https://example.test/story",
            "published_at": "2023-01-01T00:00:00Z",
            "content_snippet": "",
            "text_source": "",
            "recovery_status": "unavailable",
        }
        new_recovery = {
            "ticker": "1295.KL",
            "url": "https://example.test/public-bank",
            "published_at": "2023-01-02T00:00:00Z",
            "content_snippet": "Public Bank announced a new customer programme.",
            "text_source": "publisher",
            "recovery_status": "recovered",
        }

        merged, preserved = _merge_recovered_rows(
            [existing], [unavailable_retry, new_recovery]
        )

        self.assertEqual(preserved, 1)
        self.assertEqual(len(merged), 2)
        by_ticker = {row["ticker"]: row for row in merged}
        self.assertEqual(by_ticker["1155.KL"]["text_source"], "publisher")
        self.assertIn("stronger earnings", by_ticker["1155.KL"]["content_snippet"])

    def test_checkpoint_payloads_prefer_publisher_then_high_quality_ngram(self) -> None:
        publisher = {
            "payload": {
                "ticker": "1155.KL",
                "url": "https://example.test/publisher",
                "content_snippet": "Publisher recovery.",
                "text_source": "publisher",
            }
        }
        high_ngram = {
            "payload": {
                "ticker": "1295.KL",
                "url": "https://example.test/ngram",
                "content_snippet": "High-quality reconstructed context.",
                "text_source": "gdelt_webngrams",
                "reconstruction_quality_tier": "high",
            }
        }
        archive = {
            "payload": {
                "ticker": "1295.KL",
                "url": "https://example.test/ngram",
                "content_snippet": "Archive recovery.",
                "text_source": "wayback",
            }
        }

        rows = _checkpoint_recovered_payloads(
            {"publisher": publisher},
            {"ngram": high_ngram},
            {"ngram": archive},
        )

        by_ticker = {row["ticker"]: row for row in rows}
        self.assertEqual(by_ticker["1155.KL"]["text_source"], "publisher")
        self.assertEqual(by_ticker["1295.KL"]["text_source"], "gdelt_webngrams")

    def test_refresh_cumulative_output_rehydrates_checkpoint_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "recovery"
            raw = root / "raw"
            raw.mkdir(parents=True)
            payload = {
                "ticker": "1155.KL",
                "url": "https://example.test/recovered",
                "published_at": "2023-01-01T00:00:00Z",
                "content_snippet": "Maybank reported a recovered article body.",
                "text_source": "publisher",
                "recovery_status": "recovered",
            }
            (raw / "publisher-state.jsonl").write_text(
                json.dumps(
                    {
                        "key": "checkpoint-key",
                        "status": "recovered",
                        "payload": payload,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = refresh_cumulative_recovered_output(root)
            rows = pd.read_json(raw / "hybrid-recovered.jsonl", lines=True)

        self.assertEqual(summary["checkpoint_recovered_rows"], 1)
        self.assertEqual(summary["cumulative_recovered_rows"], 1)
        self.assertEqual(rows.iloc[0]["text_source"], "publisher")

    def test_pre_2020_candidate_only_export_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            pd.DataFrame(
                [
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "url": "https://example.test/2019-story",
                        "published_at": "2019-05-02T00:00:00Z",
                    }
                ]
            ).to_json(candidate / "2019-05.jsonl", orient="records", lines=True)
            result = run_gdelt_ngram_pilot(
                year=2019,
                candidate_dir=candidate,
                output_dir=root / "output",
                sample_per_ticker=100,
                paired_per_ticker=0,
                candidate_only=True,
                run_labels=False,
            )
        self.assertTrue(result["completed"])
        self.assertEqual(result["year"], 2019)

    def test_overlap_reconstruction_is_deterministic(self) -> None:
        windows = [
            "one two three four five six seven eight".split(),
            "four five six seven eight nine ten eleven".split(),
            "seven eight nine ten eleven twelve thirteen fourteen".split(),
        ]
        chains, attachment = merge_overlapping_windows(windows)
        self.assertEqual(
            chains[0],
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split(),
        )
        self.assertEqual(attachment, 1.0)
        self.assertEqual(
            merge_overlapping_windows(windows),
            merge_overlapping_windows(windows),
        )

    def test_quality_score_accepts_coherent_company_document(self) -> None:
        result = reconstruct_webngrams(
            _synthetic_ngram_records(),
            ticker="1155.KL",
            file_timestamps=["20230101000000"],
        )
        self.assertEqual(result.quality_tier, "high")
        self.assertGreaterEqual(result.quality_score, 0.75)
        self.assertEqual(len(result.deciles_present), 10)
        self.assertTrue(result.entity_present)
        self.assertGreaterEqual(result.unique_token_count, 180)
        self.assertIn("Maybank", result.text)

    def test_quality_score_rejects_document_without_target_entity(self) -> None:
        records = _synthetic_ngram_records()
        for row in records:
            row["pre"] = str(row["pre"]).replace("Maybank", "AnotherCompany")
            row["ngram"] = str(row["ngram"]).replace("Maybank", "AnotherCompany")
            row["post"] = str(row["post"]).replace("Maybank", "AnotherCompany")
        result = reconstruct_webngrams(records, ticker="1155.KL")
        self.assertEqual(result.quality_tier, "low")
        self.assertFalse(result.entity_present)

    def test_ngram_client_matches_only_the_exact_normalised_url(self) -> None:
        records = _synthetic_ngram_records()
        records.append(
            {
                **records[0],
                "url": "https://example.test/different-story",
                "ngram": "wrong",
            }
        )
        payload = gzip.compress(
            "\n".join(json.dumps(row) for row in records).encode("utf-8")
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("20230101000000.webngrams.json.gz"):
                return httpx.Response(200, content=payload)
            return httpx.Response(404)

        with tempfile.TemporaryDirectory() as directory:
            with GDELTWebNGramClient(
                Path(directory),
                max_download_bytes=10 * 1024 * 1024,
                request_interval_seconds=0,
                transport=httpx.MockTransport(handler),
            ) as client:
                result = client.recover(
                    {
                        "ticker": "1155.KL",
                        "url": "https://example.test/story?utm_source=test",
                        "published_at": "2023-01-01T00:00:00Z",
                    }
                )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotIn("wrong", result.text)
        self.assertEqual(result.file_timestamps, ("20230101000000",))

    def test_ngram_client_enforces_download_ceiling(self) -> None:
        payload = gzip.compress(b'{"lang":"eng"}\n')

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=payload)

        with tempfile.TemporaryDirectory() as directory:
            with GDELTWebNGramClient(
                Path(directory),
                max_download_bytes=1,
                request_interval_seconds=0,
                transport=httpx.MockTransport(handler),
            ) as client:
                with self.assertRaises(DownloadBudgetExceeded):
                    client.recover(
                        {
                            "ticker": "1155.KL",
                            "url": "https://example.test/story",
                            "published_at": "2023-01-01T00:00:00Z",
                        }
                    )

    def test_ngram_client_stops_after_exact_non_english_url(self) -> None:
        payload = gzip.compress(
            json.dumps(
                {
                    "date": "2023-01-01T00:00:00Z",
                    "lang": "id",
                    "pos": 10,
                    "pre": "CIMB melaporkan",
                    "ngram": "hasil",
                    "post": "keuangan perusahaan",
                    "url": "https://example.test/story",
                }
            ).encode("utf-8")
        )
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.path)
            if request.url.path.endswith("20230101000000.webngrams.json.gz"):
                return httpx.Response(200, content=payload)
            return httpx.Response(404)

        with tempfile.TemporaryDirectory() as directory:
            with GDELTWebNGramClient(
                Path(directory),
                max_download_bytes=1024 * 1024,
                request_interval_seconds=0,
                transport=httpx.MockTransport(handler),
            ) as client:
                result = client.recover(
                    {
                        "ticker": "1023.KL",
                        "url": "https://example.test/story",
                        "published_at": "2023-01-01T00:00:00Z",
                    }
                )
                reason = client.last_failure_reason
        self.assertIsNone(result)
        self.assertEqual(reason, "ngram_non_english")
        self.assertEqual(len(requests), 1)

    def test_publisher_recovery_respects_robots_and_validates_text(self) -> None:
        body = " ".join(
            ["Maybank", "reported", "that", "the", "financial", "result", "improved"]
            + ["and the company will continue its banking strategy"] * 30
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nAllow: /\n")
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=f"<html><head><title>Maybank result</title></head><body><article>{body}</article></body></html>",
            )

        with PublisherClient(
            request_interval_seconds=0,
            transport=httpx.MockTransport(handler),
        ) as client:
            payload, reason = client.recover(
                {
                    "ticker": "1155.KL",
                    "url": "https://example.test/story",
                    "published_at": "2023-01-01T00:00:00Z",
                }
            )
        self.assertEqual(reason, "")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["text_source"], "publisher")
        self.assertTrue(payload["publisher_full_text_available"])

    def test_publisher_caches_failed_robots_origin(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            raise httpx.ConnectError("robots endpoint unavailable", request=request)

        item = {
            "ticker": "1155.KL",
            "url": "https://example.test/story",
            "published_at": "2023-01-01T00:00:00Z",
        }
        with PublisherClient(
            request_interval_seconds=0,
            transport=httpx.MockTransport(handler),
        ) as client:
            first_payload, first_reason = client.recover(item)
            second_payload, second_reason = client.recover(
                {**item, "url": "https://example.test/another-story"}
            )

        self.assertIsNone(first_payload)
        self.assertIsNone(second_payload)
        self.assertEqual(first_reason, "robots_request_failed")
        self.assertEqual(second_reason, "robots_request_failed")
        self.assertEqual(requests, ["https://example.test/robots.txt"])

    @patch("pipeline.gdelt_ngram_pilot._score_finbert_pairs")
    @patch("pipeline.gdelt_ngram_pilot.WaybackClient")
    @patch("pipeline.gdelt_ngram_pilot.CommonCrawlClient")
    @patch("pipeline.gdelt_ngram_pilot.GDELTWebNGramClient")
    @patch("pipeline.gdelt_ngram_pilot.PublisherClient")
    def test_hybrid_orchestrator_uses_wayback_after_other_failures(
        self,
        publisher_class,
        ngram_class,
        common_crawl_class,
        wayback_class,
        finbert_score,
    ) -> None:
        publisher = publisher_class.return_value.__enter__.return_value
        publisher.recover.return_value = (None, "publisher_unavailable")
        ngram = ngram_class.return_value.__enter__.return_value
        ngram.downloaded_bytes = 0
        ngram.recover.return_value = ReconstructionResult(
            text="Maybank partial context",
            quality_score=0.2,
            quality_tier="low",
            deciles_present=(0,),
            unique_token_count=3,
            overlap_attachment_rate=0.0,
            entity_present=True,
            sentence_boundary_coverage=0.0,
            record_count=1,
            file_timestamps=("20230101000000",),
        )
        common_crawl = common_crawl_class.return_value
        common_crawl.collections.return_value = []
        common_crawl.recover.return_value = None
        wayback = wayback_class.return_value.__enter__.return_value
        wayback.recover.return_value = (
            {
                "ticker": "1155.KL",
                "url": "https://example.test/story",
                "published_at": "2023-01-01T00:00:00Z",
                "title": "Maybank result",
                "description": "Maybank result improved.",
                "content_snippet": "Maybank result improved.",
                "text_source": "wayback",
                "recovery_status": "recovered",
            },
            "",
        )
        finbert_score.return_value = (
            pd.DataFrame(),
            {"available": False, "reason": "no pairs"},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidates"
            candidate.mkdir()
            pd.DataFrame(
                [
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "url": "https://example.test/story",
                        "published_at": "2023-01-01T00:00:00Z",
                        "page_title": "Maybank result",
                    }
                ]
            ).to_json(candidate / "2023-01.jsonl", orient="records", lines=True)
            output = root / "output"
            report = run_gdelt_ngram_pilot(
                candidate_dir=candidate,
                sample_per_ticker=1,
                paired_per_ticker=1,
                output_dir=output,
                max_runtime_minutes=1,
                max_download_gib=0.01,
                request_interval_seconds=0,
                run_labels=False,
            )
            recovered = pd.read_json(
                output / "raw" / "hybrid-recovered.jsonl", lines=True
            )
        self.assertTrue(report["completed"])
        self.assertEqual(recovered.iloc[0]["text_source"], "wayback")
        publisher.recover.assert_called_once()
        ngram.recover.assert_called_once()
        common_crawl.recover.assert_called_once()
        wayback.recover.assert_called_once()

    @patch("pipeline.gdelt_ngram_pilot._score_finbert_pairs")
    @patch("pipeline.gdelt_ngram_pilot.GDELTWebNGramClient")
    @patch("pipeline.gdelt_ngram_pilot.PublisherClient")
    def test_hybrid_orchestrator_resumes_without_repeating_completed_requests(
        self, publisher_class, ngram_class, finbert_score
    ) -> None:
        publisher = publisher_class.return_value.__enter__.return_value
        publisher.recover.return_value = (
            {
                "ticker": "1155.KL",
                "url": "https://example.test/story",
                "published_at": "2023-01-01T00:00:00Z",
                "title": "Maybank result",
                "description": "Maybank result improved.",
                "content_snippet": "Maybank result improved.",
                "text_source": "publisher",
                "recovery_status": "recovered",
                "publisher_full_text_available": True,
            },
            "",
        )
        ngram = ngram_class.return_value.__enter__.return_value
        ngram.downloaded_bytes = 0
        ngram.recover.return_value = ReconstructionResult(
            text="Maybank result improved.",
            quality_score=0.9,
            quality_tier="high",
            deciles_present=(0, 10, 20, 30, 40, 50),
            unique_token_count=200,
            overlap_attachment_rate=1.0,
            entity_present=True,
            sentence_boundary_coverage=1.0,
            record_count=200,
            file_timestamps=("20230101000000",),
        )
        finbert_score.return_value = (
            pd.DataFrame(),
            {"available": False, "reason": "test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidates"
            candidate.mkdir()
            pd.DataFrame(
                [
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "url": "https://example.test/story",
                        "published_at": "2023-01-01T00:00:00Z",
                        "page_title": "Maybank result",
                    }
                ]
            ).to_json(candidate / "2023-01.jsonl", orient="records", lines=True)
            kwargs = {
                "candidate_dir": candidate,
                "sample_per_ticker": 1,
                "paired_per_ticker": 1,
                "output_dir": root / "output",
                "max_runtime_minutes": 1,
                "max_download_gib": 0.01,
                "request_interval_seconds": 0,
                "run_labels": False,
                "skip_archives": True,
            }
            run_gdelt_ngram_pilot(**kwargs)
            run_gdelt_ngram_pilot(**kwargs)
        self.assertEqual(publisher.recover.call_count, 1)
        self.assertEqual(ngram.recover.call_count, 1)

    @patch("pipeline.gdelt_ngram_pilot._score_finbert_pairs")
    @patch("pipeline.gdelt_ngram_pilot.PublisherClient")
    def test_hybrid_orchestrator_retries_transient_publisher_failure(
        self, publisher_class, finbert_score
    ) -> None:
        publisher = publisher_class.return_value.__enter__.return_value
        publisher.recover.side_effect = [
            (None, "robots_request_failed"),
            (
                {
                    "ticker": "1155.KL",
                    "url": "https://example.test/story",
                    "published_at": "2019-01-01T00:00:00Z",
                    "title": "Maybank result",
                    "description": "Maybank result improved.",
                    "content_snippet": "Maybank result improved.",
                    "text_source": "publisher",
                    "recovery_status": "recovered",
                    "publisher_full_text_available": True,
                },
                "",
            ),
        ]
        finbert_score.return_value = (
            pd.DataFrame(),
            {"available": False, "reason": "test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidates"
            candidate.mkdir()
            pd.DataFrame(
                [
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "url": "https://example.test/story",
                        "published_at": "2019-01-01T00:00:00Z",
                        "page_title": "Maybank result",
                    }
                ]
            ).to_json(candidate / "2019-01.jsonl", orient="records", lines=True)
            kwargs = {
                "year": 2019,
                "candidate_dir": candidate,
                "sample_per_ticker": 1,
                "paired_per_ticker": 0,
                "output_dir": root / "output",
                "max_runtime_minutes": 1,
                "request_interval_seconds": 0,
                "run_labels": False,
                "skip_archives": True,
            }
            run_gdelt_ngram_pilot(**kwargs)
            run_gdelt_ngram_pilot(**kwargs)
            checkpoint = [
                json.loads(line)
                for line in (
                    root / "output" / "raw" / "publisher-state.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(publisher.recover.call_count, 2)
        self.assertEqual(checkpoint[-1]["status"], "recovered")
        self.assertEqual(checkpoint[-1]["attempt_count"], 2)

    @patch("pipeline.gdelt_ngram_pilot.GDELTWebNGramClient")
    @patch("pipeline.gdelt_ngram_pilot.PublisherClient")
    def test_candidate_only_exports_all_rows_without_recovery(
        self, publisher_class, ngram_class
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidates"
            candidate.mkdir()
            pd.DataFrame(
                [
                    {
                        "ticker": "1155.KL",
                        "company_name": "Malayan Banking Berhad",
                        "url": "https://example.test/story-one",
                        "published_at": "2023-01-01T00:00:00Z",
                        "page_title": "Maybank result",
                    },
                    {
                        "ticker": "1295.KL",
                        "company_name": "Public Bank Berhad",
                        "url": "https://example.test/story-two",
                        "published_at": "2023-02-01T00:00:00Z",
                        "page_title": "Public Bank result",
                    },
                ]
            ).to_json(candidate / "2023.jsonl", orient="records", lines=True)
            output = root / "output"
            report = run_gdelt_ngram_pilot(
                candidate_dir=candidate,
                output_dir=output,
                candidate_only=True,
            )
            exported = pd.read_json(
                output / "raw" / "candidate-corpus.jsonl", lines=True
            )

        self.assertTrue(report["completed"])
        self.assertEqual(report["mode"], "candidate_only")
        self.assertEqual(report["candidate_count"], 2)
        self.assertFalse(report["acceptance_gates_evaluated"])
        self.assertEqual(len(exported), 2)
        publisher_class.assert_not_called()
        ngram_class.assert_not_called()

    @patch(
        "pipeline.bursa_news_dataset.load_stock_metadata",
        return_value={"1155.KL": "Malayan Banking Berhad"},
    )
    def test_recovery_provenance_survives_annotation_template(
        self, _metadata
    ) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "company_name": "Malayan Banking Berhad",
                    "provider": "GDELT",
                    "source_name": "example",
                    "source_domain": "example.test",
                    "published_at": "2023-01-01T00:00:00Z",
                    "title": "Maybank earnings",
                    "description": "Maybank reported improved earnings.",
                    "content_snippet": "Maybank reported improved earnings.",
                    "url": "https://example.test/story",
                    "automatic_relevance": "included",
                    "text_source": "gdelt_webngrams",
                    "is_pseudo_document": True,
                    "reconstruction_quality_tier": "high",
                    "reconstruction_quality_score": 0.9,
                }
            ]
        )
        prepared = build_annotation_template(frame)
        self.assertEqual(prepared.iloc[0]["text_source"], "gdelt_webngrams")
        self.assertTrue(bool(prepared.iloc[0]["is_pseudo_document"]))
        self.assertEqual(prepared.iloc[0]["reconstruction_quality_tier"], "high")


if __name__ == "__main__":
    unittest.main()
