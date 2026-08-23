from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.train_stock_models import (
    OFFICIAL_FEATURE_ROUNDS,
    OFFICIAL_MODEL_UNIVERSE,
    SklearnCompatibleCatBoostClassifier,
    available_model_names,
    build_estimator,
    catboost_compatibility_report,
    search_plan_for_profile,
    get_parameter_space,
    official_development_checks,
    validate_official_horizon_for_promotion,
)


class StockModelSelectionTests(unittest.TestCase):
    def test_official_profile_locks_one_round_and_common_universe(self):
        plan = search_plan_for_profile("official", available_model_names())
        for horizon, rounds in plan.items():
            self.assertEqual(list(rounds), [OFFICIAL_FEATURE_ROUNDS[horizon]])
            self.assertEqual(rounds[OFFICIAL_FEATURE_ROUNDS[horizon]], OFFICIAL_MODEL_UNIVERSE)

    def test_official_profile_uses_a_bounded_search_not_quick_defaults(self):
        space, budget = get_parameter_space("elastic-net-logistic", "official")
        self.assertGreater(budget, 1)
        self.assertGreater(len(space["classifier__C"]), 1)

    def test_official_development_checks_enforce_seed_and_fold_stability(self):
        policy = {
            "minimum_positive_mcc_seed_fraction": 0.8,
            "minimum_positive_mcc_fold_fraction": 0.8,
        }
        checks = official_development_checks(
            {"positive_mcc_seed_fraction": 0.79, "cv_positive_mcc_folds": 4},
            policy,
        )
        self.assertFalse(checks["positive_mcc_seed_fraction"])
        self.assertTrue(checks["positive_mcc_fold_fraction"])

    def test_promotion_counts_unique_shadow_dates_not_pooled_rows(self):
        policy = {
            "minimum_matured_shadow_sessions": 100,
            "minimum_paired_gain_lower_bound": 0.0,
        }
        entry = {
            "observed_acceptance": {
                "status": "accepted",
                "bootstrap": {"lower": 0.01},
            },
            "shadow_acceptance": {
                "status": "accepted",
                "rows": 1000,
                "unique_dates": 10,
            },
            "official_selection": {
                "development_checks": {
                    "positive_mcc_seed_fraction": True,
                    "positive_mcc_fold_fraction": True,
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "10 unique matured shadow sessions"):
            validate_official_horizon_for_promotion("5", entry, policy)

    def test_gate_override_requires_written_reason(self):
        from pipeline.train_stock_models import promote_stock_models

        with self.assertRaisesRegex(ValueError, "requires a written reason"):
            promote_stock_models(
                "missing-run",
                approve=True,
                override_gates=True,
            )

    def test_catboost_adapter_has_sklearn_tags_and_fits_when_installed(self):
        report = catboost_compatibility_report()
        if not report["installed"]:
            self.skipTest("CatBoost is optional in this environment")
        self.assertTrue(report["compatible"])
        estimator = build_estimator("catboost", 2, ["feature"])
        classifier = estimator.named_steps["classifier"]
        self.assertIsInstance(classifier, SklearnCompatibleCatBoostClassifier)
        frame = pd.DataFrame(
            {
                "Ticker": ["A.KL", "B.KL"] * 5,
                "feature": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8],
            }
        )
        labels = np.asarray([0, 1] * 5)
        estimator.set_params(classifier__iterations=5, classifier__depth=3)
        estimator.fit(frame, labels)
        self.assertEqual(estimator.predict_proba(frame).shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
