from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.main import app
from pipeline.dashboard_service import DashboardAnalysisResult
from pipeline.openrouter_client import OpenRouterAnswer
from pipeline.predict_stock_trend import PredictionResult


class StockSenseApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_live_and_ready_health_contracts_are_json(self) -> None:
        live = self.client.get("/api/v1/health/live")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json(), {"status": "ok"})

        ready = self.client.get("/api/v1/health/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertIn(ready.json()["status"], {"ready", "not_ready"})
        self.assertIn(
            ready.json()["market_calendar"]["precision"],
            {"bursa_exchange_calendar", "weekday_fallback"},
        )

    def test_companies_preserves_ten_company_metadata_without_network_calls(self) -> None:
        response = self.client.get("/api/v1/companies")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 10)
        self.assertEqual(payload[0]["ticker"], "1155.KL")
        self.assertIn("primary_logo_url", payload[0])

    def test_prediction_rejects_invalid_horizon_before_pipeline_call(self) -> None:
        with patch("backend.app.main.predict_stock_trend") as predict:
            response = self.client.get("/api/v1/predictions/1155.KL?horizon=3")
        self.assertEqual(response.status_code, 422)
        predict.assert_not_called()
        self.assertEqual(response.json()["error"]["code"], "INVALID_HORIZON")

    def test_comparison_rejects_duplicate_tickers(self) -> None:
        response = self.client.post(
            "/api/v1/comparisons",
            json={"tickers": ["1155.KL", "1155.KL"], "horizon": 1},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "DUPLICATE_TICKER")

    def test_system_status_does_not_return_secret_values(self) -> None:
        response = self.client.get("/api/v1/system/status")
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertNotIn("api_key", body.casefold())
        self.assertIn("supported_horizons", response.json())

    def test_performance_diagnostics_expose_only_bounded_cache_metrics(self) -> None:
        response = self.client.get("/api/v1/system/performance")
        self.assertEqual(response.status_code, 200)
        caches = response.json()["caches"]
        self.assertIn("raw_market", caches)
        self.assertGreater(caches["raw_market"]["max_entries"], 0)
        self.assertNotIn("value", response.text.casefold())

    def test_model_diagnostics_expose_recorded_stock_and_sentiment_evidence(self) -> None:
        response = self.client.get("/api/v1/system/model-diagnostics")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            [row["horizon_days"] for row in payload["stock"]["horizons"]],
            [1, 5, 10],
        )
        self.assertIn("models", payload["sentiment"])
        self.assertNotIn("artifact_hashes", response.text)
        self.assertNotIn("model_directory", response.text)

    def test_model_performance_exposes_presentation_evidence_without_paths(self) -> None:
        response = self.client.get("/api/v1/system/model-performance")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            [row["horizon_days"] for row in payload["stock"]["horizons"]],
            [1, 5, 10],
        )
        self.assertIn("candidate_families", payload["stock"]["horizons"][0])
        self.assertIn("per_company", payload["sentiment"])
        self.assertNotIn("artifact_hashes", response.text)
        self.assertNotIn("model_directory", response.text)
        self.assertNotIn("report_path", response.text)

    def test_model_performance_follows_active_fusion_registry(self) -> None:
        response = self.client.get("/api/v1/system/model-performance")
        self.assertEqual(response.status_code, 200)
        fusion = response.json()["fusion_ablation"]
        self.assertEqual(fusion["schema_version"], "stocksense-fusion-evaluation-v4")
        self.assertTrue(fusion["runtime_enabled"])
        self.assertEqual(fusion["accepted_horizons"], [1, 5, 10])
        self.assertEqual(
            [fusion["horizons"][str(days)]["runtime_label"] for days in (1, 5, 10)],
            ["Fusion-supported", "Fusion-supported", "Fusion-supported"],
        )
        # Runtime enablement must not erase the report's scientific gate result.
        self.assertEqual(fusion["scientific_accepted_horizons"], [])
        self.assertTrue(
            all(
                fusion["horizons"][str(days)]["scientific_accepted"] is False
                for days in (1, 5, 10)
            )
        )

    def test_staged_model_performance_rejects_superseded_fusion_ablation(self) -> None:
        response = self.client.get(
            "/api/v1/system/model-performance?run_id=20260812T203852522157"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["fusion_ablation"], {})

    def test_analysis_contract_exposes_evidence_properties(self) -> None:
        result = DashboardAnalysisResult(
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            horizon_days=1,
            stock=PredictionResult(
                ticker="1155.KL",
                company_name="Malayan Banking Berhad",
                horizon_days=1,
                as_of_date="2026-07-20",
                prediction_label="Up",
                probability_up=0.62,
                decision_threshold=0.5,
                model_name="test",
                model_version="test",
                refinement_round=1,
                benchmark_passed=True,
                status="accepted",
                stale_data_warning=False,
                latest_data_age_days=0,
            ),
            stock_source="live",
            stock_captured_at="2026-07-20T00:00:00+00:00",
            stock_warning=None,
            news=None,
            news_source="unavailable",
            news_captured_at=None,
            news_warning="No news",
            generated_at="2026-07-20T00:00:00+00:00",
        )
        with (
            patch("backend.app.main._prediction", return_value=result.stock),
            patch("backend.app.main.run_dashboard_analysis", return_value=result),
        ):
            response = self.client.post("/api/v1/analyses", json={"ticker": "1155.KL", "horizon": 1})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["stock"]["directional_signal_available"])
        self.assertTrue(payload["stock"]["prediction_available"])
        self.assertEqual(payload["stock"]["evidence_status"], "accepted")

    def test_aggregate_tutor_rejects_expired_or_missing_analysis(self) -> None:
        with patch("backend.app.main.complete_chat") as complete_chat:
            response = self.client.post(
                "/api/v1/chat",
                json={
                    "ticker": "1155.KL",
                    "horizon": 1,
                    "analysis_id": "missing-analysis",
                    "intent": "aggregate_explanation",
                    "messages": [{"role": "user", "content": "Explain this result."}],
                },
            )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "ANALYSIS_EXPIRED")
        complete_chat.assert_not_called()

    def test_aggregate_tutor_uses_the_matching_live_analysis(self) -> None:
        analysis_id = "aggregate-tutor-test"
        result = DashboardAnalysisResult(
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            horizon_days=1,
            stock=PredictionResult(
                ticker="1155.KL",
                company_name="Malayan Banking Berhad",
                horizon_days=1,
                as_of_date="2026-07-20",
                prediction_label="Up",
                probability_up=0.62,
                decision_threshold=0.5,
                model_name="test",
                model_version="test",
                refinement_round=1,
                benchmark_passed=True,
                status="accepted",
                stale_data_warning=False,
                latest_data_age_days=0,
            ),
            stock_source="live",
            stock_captured_at="2026-07-20T00:00:00+00:00",
            stock_warning=None,
            news=None,
            news_source="unavailable",
            news_captured_at=None,
            news_warning="No news",
            generated_at="2026-07-20T00:00:00+00:00",
        )
        api_main.analysis_cache.set(analysis_id, result, 60)
        api_main.analysis_signature_cache.set(analysis_id, api_main._analysis_registry_signature(), 60)
        with patch(
            "backend.app.main.complete_chat",
            return_value=OpenRouterAnswer(
                content="The stock model is Up with a 62% probability of upward movement [MODEL-1].",
                model="test-model",
            ),
        ) as complete_chat:
            response = self.client.post(
                "/api/v1/chat",
                json={
                    "ticker": "1155.KL",
                    "horizon": 1,
                    "analysis_id": analysis_id,
                    "intent": "aggregate_explanation",
                    "messages": [{"role": "user", "content": "Explain the aggregate result."}],
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("62%", response.json()["direct_answer"])
        self.assertEqual(response.json()["evidence_refs"][0]["id"], "model-1")
        self.assertNotIn("[MODEL-1]", response.json()["content"])
        system_prompt = complete_chat.call_args.args[0][0]["content"]
        self.assertIn("Explain only the displayed Stock Outlook result", system_prompt)
        self.assertIn("what the probability means", system_prompt)
        self.assertIn("investment-learning terms", system_prompt)


if __name__ == "__main__":
    unittest.main()
