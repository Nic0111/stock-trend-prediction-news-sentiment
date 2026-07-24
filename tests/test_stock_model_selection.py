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
)


class StockModelSelectionTests(unittest.TestCase):
    def test_official_profile_locks_one_round_and_common_universe(self):
        plan = search_plan_for_profile("official", available_model_names())
        for horizon, rounds in plan.items():
            self.assertEqual(list(rounds), [OFFICIAL_FEATURE_ROUNDS[horizon]])
            self.assertEqual(rounds[OFFICIAL_FEATURE_ROUNDS[horizon]], OFFICIAL_MODEL_UNIVERSE)

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
