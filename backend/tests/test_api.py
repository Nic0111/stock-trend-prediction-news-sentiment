from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app
from pipeline.dashboard_service import DashboardAnalysisResult
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
        with patch("backend.app.main.run_dashboard_analysis", return_value=result):
            response = self.client.post("/api/v1/analyses", json={"ticker": "1155.KL", "horizon": 1})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["stock"]["directional_signal_available"])
        self.assertTrue(payload["stock"]["prediction_available"])
        self.assertEqual(payload["stock"]["evidence_status"], "accepted")


if __name__ == "__main__":
    unittest.main()
