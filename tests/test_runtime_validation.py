from __future__ import annotations

import unittest

from pipeline.model_artifacts import CalibratedStockModel
from pipeline.runtime_validation import sentiment_integrity, validate_stock_artifact


class RuntimeValidationTests(unittest.TestCase):
    @staticmethod
    def artifact() -> CalibratedStockModel:
        return CalibratedStockModel(
            estimator=object(),
            calibrator=object(),
            threshold=0.51,
            input_columns=["Return_1D", "RSI_14"],
            numeric_feature_columns=["Return_1D", "RSI_14"],
            refinement_round=2,
            model_name="test-model",
            horizon_days=5,
            threshold_constraints_passed=True,
        )

    @staticmethod
    def entry() -> dict:
        return {
            "decision_threshold": 0.51,
            "input_columns": ["Return_1D", "RSI_14"],
            "numeric_feature_columns": ["Return_1D", "RSI_14"],
            "refinement_round": 2,
            "model_name": "test-model",
            "threshold_constraints_passed": True,
        }

    def test_matching_artifact_contract_is_accepted(self) -> None:
        self.assertEqual(
            validate_stock_artifact(self.artifact(), self.entry(), horizon_days=5),
            [],
        )

    def test_wrong_horizon_and_feature_order_are_rejected(self) -> None:
        artifact = self.artifact()
        artifact.horizon_days = 10
        artifact.input_columns = ["RSI_14", "Return_1D"]
        errors = validate_stock_artifact(artifact, self.entry(), horizon_days=5)
        self.assertIn("horizon_days does not match the registry", errors)
        self.assertIn("input_columns do not match the registry", errors)

    def test_sentiment_integrity_follows_persisted_active_ensemble(self) -> None:
        report = sentiment_integrity()
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["variant"],
            "bursa_phrasebank_defensible_soft_vote_sentiment-remediation-20260809-v1",
        )
        self.assertEqual(report["backend"], "weighted_finbert_ensemble")
        self.assertAlmostEqual(float(report["weights"]["bursa"]), 0.7)
        self.assertAlmostEqual(float(report["weights"]["phrasebank"]), 0.3)


if __name__ == "__main__":
    unittest.main()
