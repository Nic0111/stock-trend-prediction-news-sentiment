from __future__ import annotations

import unittest

from pipeline.benchmark_finbert_ensemble import _gate_report


class FinBERTEnsembleBenchmarkGateTests(unittest.TestCase):
    @staticmethod
    def _selected() -> dict[str, float]:
        return {
            "macro_f1": 0.80,
            "balanced_accuracy": 0.80,
            "ece": 0.05,
            "negative_f1": 0.70,
            "neutral_f1": 0.70,
            "positive_f1": 0.70,
        }

    def test_independent_holdout_is_required_even_when_other_metrics_pass(self) -> None:
        policy = {
            "sentiment_ensemble": {
                "promotion_gates": {
                    "minimum_validation_macro_f1_uplift": 0.02,
                    "minimum_validation_balanced_accuracy_uplift": 0.02,
                    "minimum_macro_f1": 0.70,
                    "minimum_balanced_accuracy": 0.70,
                    "maximum_ece": 0.10,
                    "minimum_class_f1": 0.60,
                    "minimum_bootstrap_ci_lower": 0.0,
                    "maximum_phrasebank_macro_f1_degradation": 0.02,
                    "maximum_seed_macro_f1_std": 0.02,
                    "maximum_company_macro_f1_loss": 0.05,
                    "minimum_holdout_rows": 200,
                    "minimum_holdout_rows_per_ticker": 15,
                    "require_independent_holdout": True,
                }
            }
        }
        baseline = dict(self._selected())
        baseline["macro_f1"] = 0.75
        baseline["balanced_accuracy"] = 0.75
        result = _gate_report(
            self._selected(),
            baseline,
            self._selected(),
            baseline,
            {"ci95_lower": 0.0},
            policy=policy,
            independent_holdout=False,
            component_benchmark={
                "seed_macro_f1_std": 0.01,
                "phrasebank_macro_f1_degradation": 0.01,
            },
            company_check=True,
        )
        self.assertFalse(result["checks"]["independent_holdout"])
        self.assertFalse(result["eligible_for_promotion"])
        result = _gate_report(
            self._selected(),
            baseline,
            self._selected(),
            baseline,
            {"ci95_lower": 0.0},
            policy=policy,
            independent_holdout=True,
            component_benchmark={
                "seed_macro_f1_std": 0.01,
                "phrasebank_macro_f1_degradation": 0.01,
            },
            company_check=True,
        )
        self.assertTrue(result["eligible_for_promotion"])


if __name__ == "__main__":
    unittest.main()
