from __future__ import annotations

import unittest

from pipeline.model_artifacts import CalibratedStockModel
from pipeline.runtime_validation import validate_stock_artifact


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


if __name__ == "__main__":
    unittest.main()
