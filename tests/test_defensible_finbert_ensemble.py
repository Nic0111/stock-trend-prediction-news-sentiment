from __future__ import annotations

import unittest

from pipeline.defensible_finbert_ensemble import (
    _final_gate_report,
    normalize_training_recipe_grid,
    normalize_phrasebank_replay_shares,
    select_replay_candidate,
)


class DefensibleEnsembleProtocolTests(unittest.TestCase):
    @staticmethod
    def _candidate(replay: float, macro_f1: float) -> dict[str, float]:
        return {
            "phrasebank_replay_share": replay,
            "bursa_training_share": 1.0 - replay,
            "macro_f1": macro_f1,
            "balanced_accuracy": 0.72,
            "log_loss": 0.50,
            "ece": 0.04,
            "seed_macro_f1_std": 0.01,
        }

    @staticmethod
    def _metrics(value: float = 0.75) -> dict[str, float]:
        return {
            "macro_f1": value,
            "balanced_accuracy": value,
            "ece": 0.04,
            "negative_f1": 0.70,
            "neutral_f1": 0.65,
            "positive_f1": 0.80,
        }

    def test_replay_grid_is_canonical_and_allows_zero_replay(self) -> None:
        self.assertEqual(
            normalize_phrasebank_replay_shares([0.30, 0.0, 0.10, 0.10]),
            (0.0, 0.1, 0.3),
        )
        with self.assertRaises(ValueError):
            normalize_phrasebank_replay_shares([1.0])

    def test_training_recipe_grid_rejects_duplicate_identifiers(self) -> None:
        recipes = normalize_training_recipe_grid(
            [
                {"id": "baseline", "text_mode": "headline_lead", "sampling_mode": "shuffle", "loss_mode": "inverse_frequency_ce"},
                {"id": "balanced", "text_mode": "target_aware", "sampling_mode": "ticker_class_balanced", "loss_mode": "effective_number_focal"},
            ]
        )
        self.assertEqual([item["id"] for item in recipes], ["baseline", "balanced"])
        with self.assertRaises(ValueError):
            normalize_training_recipe_grid(
                [
                    {"id": "duplicate"},
                    {"id": "duplicate"},
                ]
            )

    def test_replay_selection_uses_development_quality_before_tie_breakers(self) -> None:
        selected = select_replay_candidate(
            [self._candidate(0.0, 0.68), self._candidate(0.2, 0.70)]
        )
        self.assertEqual(selected["phrasebank_replay_share"], 0.2)
        self.assertIn("development_macro_f1_desc", selected["selection_order"])

    def test_component_non_degeneracy_blocks_a_near_bursa_only_weight(self) -> None:
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
                    "minimum_component_weight": 0.20,
                    "maximum_component_weight": 0.80,
                    "minimum_matching_fold_count": 3,
                    "require_retrospective_test_disclosure": True,
                    "allow_retrospective_manual_promotion": True,
                }
            }
        }
        company = {
            "1155.KL": {"rows": 21, "macro_f1_delta": 0.03},
            "1295.KL": {"rows": 21, "macro_f1_delta": 0.02},
        }
        common = dict(
            policy=policy,
            final_metrics=self._metrics(),
            final_phrasebank_metrics=self._metrics(0.70),
            validation_metrics=self._metrics(),
            validation_phrasebank_metrics=self._metrics(0.70),
            bootstrap={"ci95_lower": 0.01},
            per_company=company,
            weight_stability={"matching_fold_count": 3},
            seed_macro_f1_std=0.01,
            phrasebank_macro_f1_degradation=0.01,
            test_rows=210,
        )
        rejected = _final_gate_report(bursa_weight=0.99, **common)
        self.assertFalse(rejected["checks"]["component_non_degeneracy"])
        self.assertFalse(rejected["eligible_for_promotion"])
        accepted = _final_gate_report(bursa_weight=0.50, **common)
        self.assertTrue(accepted["checks"]["component_non_degeneracy"])
        self.assertTrue(accepted["eligible_for_promotion"])


if __name__ == "__main__":
    unittest.main()
