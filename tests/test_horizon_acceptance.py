import unittest

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score

from pipeline.horizon_acceptance import (
    POLICY,
    date_block_bootstrap_gain,
    evaluate_acceptance,
    weaker_status,
)


class HorizonAcceptanceTests(unittest.TestCase):
    def test_policy_has_horizon_specific_dependence_settings(self):
        self.assertEqual(POLICY["horizons"]["1"]["bootstrap_block_sessions"], 5)
        self.assertEqual(POLICY["horizons"]["5"]["bootstrap_block_sessions"], 5)
        self.assertEqual(POLICY["horizons"]["10"]["bootstrap_block_sessions"], 10)

    def test_block_bootstrap_is_deterministic_and_preserves_positive_gain(self):
        dates = np.repeat(np.arange(240), 2)
        y = np.tile([0, 1], 240)
        predictions = y.copy()
        baseline = 1 - y
        first = date_block_bootstrap_gain(
            y, predictions, baseline, dates, block_size=5, repetitions=150, seed=42
        )
        second = date_block_bootstrap_gain(
            y, predictions, baseline, dates, block_size=5, repetitions=150, seed=42
        )
        self.assertEqual(first, second)
        self.assertGreater(first["lower"], 0)

    def test_clear_signal_is_accepted(self):
        dates = np.arange(240)
        y = np.arange(240) % 2
        predictions = y.copy()
        baseline = 1 - y
        probability = np.where(y == 1, 0.9, 0.1)
        metrics = {
            "balanced_accuracy": balanced_accuracy_score(y, predictions),
            "mcc": matthews_corrcoef(y, predictions),
            "macro_f1": f1_score(y, predictions, average="macro"),
            "recall": recall_score(y, predictions),
            "specificity": recall_score(y, predictions, pos_label=0),
            "predicted_up_rate": float(predictions.mean()),
        }
        baseline_metrics = {
            "balanced_accuracy": balanced_accuracy_score(y, baseline),
            "macro_f1": f1_score(y, baseline, average="macro"),
        }
        result = evaluate_acceptance(
            horizon=1,
            y_true=y,
            predictions=predictions,
            probability_up=probability,
            baseline_predictions=baseline,
            dates=dates,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            oof_metrics={"threshold_constraints_passed": True},
            cv_positive_mcc_folds=5,
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["failed_checks"], [])

    def test_weaker_status_composition_never_upgrades_evidence(self):
        self.assertEqual(weaker_status("accepted", "limited"), "limited")
        self.assertEqual(weaker_status("limited", "experimental"), "experimental")
        self.assertEqual(weaker_status("accepted", "accepted"), "accepted")


if __name__ == "__main__":
    unittest.main()
