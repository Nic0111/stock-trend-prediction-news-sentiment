from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.gdelt_historical_news import (
    CommonCrawlClient,
    build_entity_approval_queue,
    apply_entity_approvals,
    build_gdelt_bigquery_sql,
    build_gdelt_pilot_report,
    evaluate_indexed_candidate_gate,
    month_date_ranges,
    parse_common_crawl_record,
    recover_common_crawl_content,
    collect_external_label_candidate_years,
    run_authenticated_gdelt_queries,
    select_pilot_candidates,
    load_gdelt_export,
    write_gdelt_query_bundle,
)


class GDELTHistoricalNewsTests(unittest.TestCase):
    def test_external_collection_reuses_2023_and_dry_runs_other_years(self) -> None:
        calls: list[dict[str, object]] = []

        def runner(project_id, **kwargs):
            calls.append({"project_id": project_id, **kwargs})
            return {
                "planned_bytes": 10 * 1024**3,
                "project_month_bytes_before_run": 5 * 1024**3,
                "completed_shards": 0,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reused = root / "reused-2023"
            reused.mkdir()
            records = []
            for month in range(1, 13):
                shard = f"2023-{month:02d}"
                (reused / f"{shard}.jsonl").write_text(
                    json.dumps(
                        {
                            "ticker": "1155.KL",
                            "company_name": "Malayan Banking Berhad",
                            "url": f"https://example.test/{shard}",
                            "published_at": f"{shard}-01T00:00:00Z",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                records.append(
                    {
                        "shard": shard,
                        "status": "completed",
                        "rows_exported": 1,
                        "bytes_processed": 1024,
                    }
                )
            (reused / "query-state.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n",
                encoding="utf-8",
            )
            result = collect_external_label_candidate_years(
                "test-project",
                output_dir=root / "collection",
                years=(2022, 2023),
                reusable_2023_export=reused,
                runner=runner,
            )
        self.assertTrue(result["ready_for_execution"])
        self.assertEqual([item["year"] for item in result["reused_exports"]], [2023])
        self.assertEqual(len(calls), 1)
        self.assertTrue(bool(calls[0]["dry_run_only"]))

    def test_external_collection_never_executes_when_aggregate_exceeds_cap(self) -> None:
        calls: list[bool] = []

        def runner(_project_id, **kwargs):
            calls.append(bool(kwargs["dry_run_only"]))
            return {
                "planned_bytes": 500 * 1024**3,
                "project_month_bytes_before_run": 0,
                "completed_shards": 0,
            }

        with tempfile.TemporaryDirectory() as directory:
            result = collect_external_label_candidate_years(
                "test-project",
                output_dir=Path(directory),
                years=(2021, 2022),
                reusable_2023_export=None,
                monthly_budget_gib=900,
                execute=True,
                runner=runner,
            )
        self.assertFalse(result["ready_for_execution"])
        self.assertEqual(calls, [True, True])

    def test_entity_approval_queue_only_contains_borderline_matches(self) -> None:
        excluded = pd.DataFrame(
            [
                {
                    "article_id": "a1",
                    "ticker": "1155.KL",
                    "automatic_relevance": "review",
                    "relevance_score": 0.5,
                    "requires_user_approval": True,
                },
                {
                    "article_id": "a2",
                    "ticker": "1155.KL",
                    "automatic_relevance": "unrelated",
                    "relevance_score": 0.1,
                    "requires_user_approval": False,
                },
            ]
        )
        queue = build_entity_approval_queue(excluded)
        self.assertEqual(queue["article_id"].tolist(), ["a1"])
        self.assertEqual(queue["approval_status"].tolist(), ["pending"])

    def test_entity_approval_moves_only_borderline_rows(self) -> None:
        accepted = pd.DataFrame([{"ticker": "1155.KL", "url": "https://a"}])
        excluded = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "url": "https://review",
                    "automatic_relevance": "review",
                },
                {
                    "ticker": "1155.KL",
                    "url": "https://hard",
                    "automatic_relevance": "hard_excluded",
                },
            ]
        )
        approvals = pd.DataFrame(
            [{"ticker": "1155.KL", "url": "https://review", "approval_status": "approved"}]
        )
        updated, remaining = apply_entity_approvals(accepted, excluded, approvals)
        self.assertIn("https://review", updated["url"].tolist())
        self.assertIn("https://hard", remaining["url"].tolist())

    def test_authenticated_runner_streams_jsonl_and_resumes(self) -> None:
        class FakeRow:
            def items(self):
                return [("ticker", "1155.KL"), ("published_at", "2023-01-01T00:00:00+00:00")]

        class FakeJob:
            job_id = "job-1"
            total_bytes_processed = 1024

            def result(self):
                return [FakeRow()]

        class FakeClient:
            def __init__(self):
                self.calls = 0

            def query(self, sql, job_config=None, location=None):
                self.calls += 1
                return FakeJob()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_dir = root / "queries"
            query_dir.mkdir()
            (query_dir / "2023-01.sql").write_text("SELECT 1", encoding="utf-8")
            result = run_authenticated_gdelt_queries(
                "project-e479489e-567d-4f30-bc6",
                query_dir=query_dir,
                export_dir=root / "exports",
                maximum_bytes_billed_gib=1,
                monthly_budget_gib=1,
                client=FakeClient(),
            )
            self.assertEqual(result["completed_shards"], 1)
            self.assertTrue((root / "exports" / "2023-01.jsonl").exists())
            self.assertIn('"ticker": "1155.KL"', (root / "exports" / "2023-01.jsonl").read_text(encoding="utf-8"))

    @patch(
        "pipeline.gdelt_historical_news.load_stock_metadata",
        return_value={"1155.KL": "Malayan Banking Berhad"},
    )
    def test_query_uses_organizations_but_not_gdelt_tone(self, _metadata) -> None:
        sql = build_gdelt_bigquery_sql(tickers=["1155.KL"])
        self.assertIn("V2Organizations", sql)
        self.assertIn("DocumentIdentifier", sql)
        self.assertIn("page_title", sql)
        self.assertIn("match_channel", sql)
        self.assertIn("title_distinctive", sql)
        self.assertIn("organization_ambiguous_anchored", sql)
        self.assertNotIn("V2Tone", sql)
        self.assertIn("'1155.KL' AS ticker", sql)

    def test_indexed_candidate_gate_requires_every_ticker(self) -> None:
        rows = []
        for ticker in ("1155.KL", "1295.KL"):
            for month in ("2023-01", "2023-02"):
                rows.append(
                    {
                        "ticker": ticker,
                        "url": f"https://{ticker}-{month}.example/story",
                        "source_domain": f"{month}.example",
                        "published_at": f"{month}-05T00:00:00Z",
                        "match_channel": "title_distinctive",
                    }
                )
        gate = evaluate_indexed_candidate_gate(
            pd.DataFrame(rows),
            minimum_urls_per_ticker=2,
            minimum_domains_per_ticker=2,
            minimum_months_per_ticker=2,
            required_tickers=["1155.KL", "1295.KL"],
        )
        self.assertTrue(gate["ready_for_recovery"])
        failed = evaluate_indexed_candidate_gate(
            pd.DataFrame(rows),
            minimum_urls_per_ticker=2,
            minimum_domains_per_ticker=2,
            minimum_months_per_ticker=2,
            required_tickers=["1155.KL", "1295.KL", "5347.KL"],
        )
        self.assertFalse(failed["ready_for_recovery"])
        self.assertFalse(failed["companies"]["5347.KL"]["passed"])

    def test_indexed_candidate_gate_supports_declared_ticker_override(self) -> None:
        rows = [
            {
                "ticker": "8869.KL",
                "url": f"https://press-metal-{index}.example/story",
                "source_domain": f"source-{index % 2}.example",
                "published_at": f"2023-0{(index % 6) + 1}-05T00:00:00Z",
                "match_channel": "title_distinctive",
            }
            for index in range(30)
        ]
        gate = evaluate_indexed_candidate_gate(
            pd.DataFrame(rows),
            minimum_urls_per_ticker=50,
            minimum_urls_by_ticker={"8869.KL": 30},
            minimum_domains_per_ticker=2,
            minimum_months_per_ticker=6,
            required_tickers=["8869.KL"],
        )
        self.assertTrue(gate["ready_for_recovery"])
        self.assertEqual(
            gate["companies"]["8869.KL"]["required_distinct_urls"],
            30,
        )

    def test_common_crawl_record_extracts_title_and_readable_text(self) -> None:
        warc = (
            b"WARC/1.0\r\nContent-Type: application/http; msgtype=response\r\n\r\n"
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
            b"<html><head><title>Maybank earnings</title></head>"
            b"<body><p>Profit increased strongly.</p><script>ignore()</script></body></html>"
        )
        title, body = parse_common_crawl_record(gzip.compress(warc))
        self.assertEqual(title, "Maybank earnings")
        self.assertIn("Profit increased strongly.", body)
        self.assertNotIn("ignore()", body)

    def test_candidate_indexes_are_limited_to_publication_and_next_year(self) -> None:
        collections = [
            {"id": "CC-MAIN-2024-10"},
            {"id": "CC-MAIN-2025-05"},
            {"id": "CC-MAIN-2023-50"},
        ]
        self.assertEqual(
            CommonCrawlClient.candidate_indexes(collections, 2024),
            ["CC-MAIN-2024-10", "CC-MAIN-2025-05"],
        )

    def test_monthly_ranges_keep_boundaries_and_candidate_sampling_is_balanced(self) -> None:
        self.assertEqual(
            month_date_ranges("2023-01-15", "2023-03-02"),
            [("2023-01-15", "2023-01-31"), ("2023-02-01", "2023-02-28"), ("2023-03-01", "2023-03-02")],
        )
        rows = []
        for ticker in ("1155.KL", "1295.KL"):
            for month in ("2023-01", "2023-02"):
                for domain in ("one.example", "two.example"):
                    rows.append(
                        {
                            "ticker": ticker,
                            "published_at": f"{month}-10T12:00:00Z",
                            "source_domain": domain,
                            "url": f"https://{domain}/{ticker}/{month}",
                        }
                    )
        sampled = select_pilot_candidates(pd.DataFrame(rows), per_ticker=3)
        self.assertEqual(sampled.groupby("ticker").size().to_dict(), {"1155.KL": 3, "1295.KL": 3})
        self.assertEqual(
            sampled["ticker"].tolist(),
            ["1155.KL", "1295.KL", "1155.KL", "1295.KL", "1155.KL", "1295.KL"],
        )
        self.assertEqual(
            sampled.sort_values(["ticker", "url"]).reset_index(drop=True).to_json(
                date_format="iso"
            ),
            select_pilot_candidates(pd.DataFrame(rows), per_ticker=3)
            .sort_values(["ticker", "url"])
            .reset_index(drop=True)
            .to_json(date_format="iso"),
        )

    @patch(
        "pipeline.gdelt_historical_news.load_stock_metadata",
        return_value={"1155.KL": "Malayan Banking Berhad"},
    )
    def test_query_bundle_refuses_to_cross_free_monthly_budget(self, _metadata) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_gdelt_query_bundle(
                    Path(directory),
                    start_date="2023-01-01",
                    end_date="2024-12-31",
                    max_shard_gib=75,
                    monthly_budget_gib=900,
                )

    def test_export_directory_combines_monthly_files_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "ticker": "1155.KL",
                "company_name": "Malayan Banking Berhad",
                "url": "https://example.test/story",
                "published_at": "2023-01-02T00:00:00Z",
            }
            pd.DataFrame([row]).to_csv(root / "2023-01.csv", index=False)
            pd.DataFrame([row]).to_csv(root / "2023-02.csv", index=False)
            result = load_gdelt_export(root)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["source_domain"], "example.test")

    def test_rate_limit_is_checkpointed_before_recovery_stops(self) -> None:
        class RateLimitedClient:
            def collections(self):
                return []

            def recover(self, *args, **kwargs):
                from pipeline.gdelt_historical_news import CommonCrawlRateLimitError

                raise CommonCrawlRateLimitError("throttled")

        frame = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "url": "https://example.test/story",
                    "published_at": "2023-01-02T00:00:00Z",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.jsonl"
            with self.assertRaises(RuntimeError):
                recover_common_crawl_content(
                    frame, client=RateLimitedClient(), checkpoint_path=checkpoint
                )
            self.assertIn('"status": "rate_limited"', checkpoint.read_text(encoding="utf-8"))

    def test_recovery_runtime_guard_pauses_before_network_work(self) -> None:
        class IdleClient:
            def collections(self):
                return []

            def recover(self, *args, **kwargs):
                raise AssertionError("runtime guard did not stop the job")

        frame = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "url": "https://example.test/story",
                    "published_at": "2023-01-02T00:00:00Z",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            heartbeat = Path(directory) / "heartbeat.json"
            result = recover_common_crawl_content(
                frame,
                client=IdleClient(),
                max_runtime_minutes=0,
                heartbeat_path=heartbeat,
            )
            self.assertTrue(result.empty)
            payload = __import__("json").loads(heartbeat.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "paused")
            self.assertEqual(payload["stop_reason"], "maximum runtime reached")

    @patch(
        "pipeline.gdelt_historical_news.load_stock_metadata",
        return_value={"1155.KL": "Malayan Banking Berhad"},
    )
    def test_pilot_cannot_pass_without_manual_precision_review(self, _metadata) -> None:
        indexed = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "url": "https://one.example/a",
                    "source_domain": "one.example",
                    "published_at": "2023-01-01T00:00:00Z",
                    "match_channel": "organization_exact",
                }
            ]
        )
        accepted = pd.DataFrame(
            [
                {"ticker": "1155.KL", "source_domain": "one.example"},
                {"ticker": "1155.KL", "source_domain": "two.example"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = build_gdelt_pilot_report(
                indexed,
                indexed.copy(),
                accepted,
                output_dir=Path(directory),
            )
            self.assertFalse(result.ready_for_full_collection)
            self.assertTrue(result.report_path.exists())

    @patch(
        "pipeline.gdelt_historical_news.load_stock_metadata",
        return_value={"1155.KL": "Malayan Banking Berhad"},
    )
    def test_pilot_passes_only_with_precision_and_horizon_coverage(self, _metadata) -> None:
        indexed = pd.DataFrame(
            [
                {"ticker": "1155.KL", "published_at": "2023-01-01T00:00:00Z"},
                {
                    "ticker": "1155.KL",
                    "url": "https://two.example/b",
                    "source_domain": "two.example",
                    "published_at": "2023-01-02T00:00:00Z",
                    "match_channel": "title_distinctive",
                },
            ]
        )
        indexed.loc[0, "url"] = "https://one.example/a"
        indexed.loc[0, "source_domain"] = "one.example"
        indexed.loc[0, "match_channel"] = "organization_exact"
        accepted = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "source_domain": "one.example",
                    "published_at": "2023-01-01T00:00:00Z",
                },
                {
                    "ticker": "1155.KL",
                    "source_domain": "two.example",
                    "published_at": "2023-01-02T00:00:00Z",
                },
            ]
        )
        review = pd.DataFrame(
            [
                {"automatic_included": "true", "actually_related": "true"},
                {"automatic_included": "false", "actually_related": "false"},
            ]
        )
        features = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "horizon_days": horizon,
                    "news_coverage_score": 0.5,
                }
                for horizon in (1, 5, 10)
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = build_gdelt_pilot_report(
                indexed,
                indexed.copy(),
                accepted,
                output_dir=Path(directory),
                manually_reviewed=review,
                feature_frame=features,
                pilot_policy={
                    "minimum_indexed_urls_per_ticker": 2,
                    "minimum_indexed_domains_per_ticker": 2,
                    "minimum_indexed_months_per_ticker": 1,
                    "minimum_text_recovery_rate": 1.0,
                    "minimum_entity_precision": 1.0,
                    "minimum_entity_recall": 1.0,
                    "minimum_timestamp_completeness": 1.0,
                    "minimum_source_domains_per_ticker": 2,
                    "minimum_accepted_articles": 2,
                    "minimum_accepted_per_ticker": 2,
                },
            )
        self.assertTrue(result.ready_for_full_collection)

    @patch(
        "pipeline.gdelt_historical_news.load_policy",
        return_value={
            "year_quality_gates": {},
            "labeling": {
                "minimum_parse_success_rate": 1.0,
                "minimum_consensus_coverage": 1.0,
                "minimum_entity_kappa": 0.9,
                "minimum_sentiment_kappa": 0.9,
            },
        },
    )
    @patch(
        "pipeline.gdelt_historical_news.load_stock_metadata",
        return_value={"1155.KL": "Malayan Banking Berhad"},
    )
    def test_pilot_accepts_dual_llm_consensus_reference(
        self, _metadata, _policy
    ) -> None:
        indexed = pd.DataFrame(
            [
                {
                    "ticker": "1155.KL",
                    "url": f"https://{domain}/{index}",
                    "source_domain": domain,
                    "published_at": f"2023-01-0{index + 1}T00:00:00Z",
                    "match_channel": "organization_exact",
                }
                for index, domain in enumerate(
                    ("one.example", "two.example", "one.example", "two.example")
                )
            ]
        )
        labels = indexed.copy()
        labels["labelling_completed"] = True
        labels["consensus_status"] = "accepted"
        labels["label_source"] = "llm_consensus"
        labels["labeler_a_entity_relevant"] = [True, True, False, False]
        labels["labeler_b_entity_relevant"] = [True, True, False, False]
        labels["entity_relevant"] = [True, True, False, False]
        labels["automatic_entity_relevant"] = [True, True, False, False]
        labels["labeler_a_sentiment"] = ["positive", "negative", "", ""]
        labels["labeler_b_sentiment"] = ["positive", "negative", "", ""]
        labels["sentiment_label"] = ["positive", "negative", "", ""]
        with tempfile.TemporaryDirectory() as directory:
            result = build_gdelt_pilot_report(
                indexed,
                indexed.copy(),
                indexed.iloc[:2].copy(),
                output_dir=Path(directory),
                llm_labelled=labels,
                pilot_policy={
                    "minimum_indexed_urls_per_ticker": 4,
                    "minimum_indexed_domains_per_ticker": 2,
                    "minimum_indexed_months_per_ticker": 1,
                    "minimum_text_recovery_rate": 1.0,
                    "minimum_entity_precision": 1.0,
                    "minimum_entity_recall": 1.0,
                    "minimum_timestamp_completeness": 1.0,
                    "minimum_source_domains_per_ticker": 2,
                    "minimum_accepted_articles": 2,
                    "minimum_accepted_per_ticker": 2,
                },
            )
            report = __import__("json").loads(
                result.report_path.read_text(encoding="utf-8")
            )
        self.assertTrue(result.ready_for_full_collection)
        self.assertEqual(
            report["entity_reference_type"], "dual_llm_consensus_pseudolabel"
        )


if __name__ == "__main__":
    unittest.main()
