from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fusion_model import (
    FUSION_FEATURE_NAMES,
    FusionDatasetNotReadyError,
    FusionModelArtifact,
    check_fusion_dataset_readiness,
    evaluate_promotion_gate,
    train_fusion_model,
    train_fusion_models,
)


def _frame(rows: int = 12) -> pd.DataFrame:
    values = {name: np.linspace(0.1, 0.9, rows) for name in FUSION_FEATURE_NAMES}
    values.update(
        {
            "ticker": ["1155.KL"] * rows,
            "observation_date": pd.date_range("2019-01-01", periods=rows, freq="D"),
            "horizon_days": [1] * rows,
            "target_up": [0, 1] * (rows // 2),
        }
    )
    return pd.DataFrame(values)


class FusionModelTests(unittest.TestCase):
    def test_readiness_reports_missing_historical_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.parquet"
            report = check_fusion_dataset_readiness(path)
            self.assertFalse(report.ready)
            self.assertIn("not found", report.reason)
            with self.assertRaises(FusionDatasetNotReadyError):
                train_fusion_models(path)

    def test_training_creates_horizon_aware_experimental_artifact(self) -> None:
        artifact = train_fusion_model(_frame(), 1)
        self.assertIsInstance(artifact, FusionModelArtifact)
        self.assertEqual(artifact.horizon_days, 1)
        self.assertEqual(artifact.validation_status, "experimental")
        probabilities = artifact.predict_proba(_frame().iloc[:2])
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_missing_news_features_are_imputed_but_stock_features_are_required(self) -> None:
        frame = _frame()
        for name in FUSION_FEATURE_NAMES:
            if name.startswith("news_") or name == "stock_news_agreement":
                frame[name] = np.nan
        frame["news_available"] = 0
        artifact = train_fusion_model(frame, 1)
        probabilities = artifact.predict_proba(frame.iloc[:2])
        self.assertEqual(probabilities.shape, (2, 2))

    def test_promotion_gate_accepts_only_when_all_frozen_checks_pass(self) -> None:
        baseline = {
            "balanced_accuracy": 0.50,
            "mcc": 0.02,
            "macro_f1": 0.48,
            "brier_score": 0.25,
            "log_loss": 0.69,
            "ece": 0.04,
        }
        challenger = {
            **baseline,
            "balanced_accuracy": 0.53,
            "mcc": 0.10,
            "macro_f1": 0.49,
            "brier_score": 0.23,
            "log_loss": 0.65,
            "ece": 0.04,
            "up_recall": 0.50,
            "down_recall": 0.42,
            "predicted_up_rate": 0.55,
        }
        decision = evaluate_promotion_gate(
            challenger,
            baseline,
            fold_mcc=[0.01, 0.02, -0.01, 0.03, 0.01],
            bootstrap_balanced_accuracy_gain_ci=(0.001, 0.05),
            per_company_balanced_accuracy_gain={"1155.KL": 0.01, "1295.KL": 0.00},
        )
        self.assertTrue(decision.accepted)

    def test_promotion_gate_explains_failed_metric(self) -> None:
        baseline = {"balanced_accuracy": 0.50, "mcc": 0.02, "macro_f1": 0.48, "brier_score": 0.25, "log_loss": 0.69, "ece": 0.04}
        challenger = {**baseline, "balanced_accuracy": 0.505, "mcc": -0.01, "macro_f1": 0.40, "brier_score": 0.26, "log_loss": 0.70, "ece": 0.10, "up_recall": 0.2, "down_recall": 0.2, "predicted_up_rate": 0.95}
        decision = evaluate_promotion_gate(challenger, baseline)
        self.assertFalse(decision.accepted)
        self.assertIn("promotion gate failed: balanced_accuracy_gain", decision.reasons)


if __name__ == "__main__":
    unittest.main()
