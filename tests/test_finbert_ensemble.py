from __future__ import annotations

import unittest

import numpy as np

from pipeline.download_financial_news_data import add_sentiment_scores
from pipeline.finbert_sentiment_model import AveragedFinBERTEnsemble, WeightedFinBERTEnsemble


class _FakeSentimentModel:
    def __init__(self, probabilities: list[list[float]], classes=None) -> None:
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray(classes or ["negative", "neutral", "positive"], dtype=object)
        self.artifact_signature = "fake"

    def predict_proba(self, texts):
        return self._probabilities[: len(texts)]


class FinBERTEnsembleTests(unittest.TestCase):
    def test_weighted_probabilities_preserve_contract(self) -> None:
        bursa = _FakeSentimentModel([[0.10, 0.20, 0.70]])
        phrasebank = _FakeSentimentModel([[0.30, 0.40, 0.30]])
        ensemble = WeightedFinBERTEnsemble(
            {"bursa": bursa, "phrasebank": phrasebank},
            {"bursa": 0.45, "phrasebank": 0.55},
            variant_id="test",
        )
        probabilities = ensemble.predict_proba(["article"])
        self.assertTrue(np.allclose(probabilities, [[0.21, 0.31, 0.48]]))
        self.assertEqual(ensemble.predict(["article"]).tolist(), ["positive"])

    def test_article_scoring_keeps_component_provenance(self) -> None:
        bursa = _FakeSentimentModel([[0.10, 0.20, 0.70]])
        phrasebank = _FakeSentimentModel([[0.30, 0.40, 0.30]])
        ensemble = WeightedFinBERTEnsemble(
            {"bursa": bursa, "phrasebank": phrasebank},
            {"bursa": 0.45, "phrasebank": 0.55},
            variant_id="test",
        )
        rows = [{"title": "Strong earnings", "description": ""}]
        add_sentiment_scores(rows, ensemble)
        self.assertEqual(rows[0]["sentiment_model_variant"], "test")
        self.assertAlmostEqual(rows[0]["bursa_probability_positive"], 0.70)
        self.assertAlmostEqual(rows[0]["phrasebank_probability_neutral"], 0.40)
        self.assertEqual(rows[0]["sentiment_label"], "positive")

    def test_ensemble_temperature_is_applied_after_probability_blend(self) -> None:
        bursa = _FakeSentimentModel([[0.02, 0.18, 0.80]])
        phrasebank = _FakeSentimentModel([[0.30, 0.40, 0.30]])
        calibrated = WeightedFinBERTEnsemble(
            {"bursa": bursa, "phrasebank": phrasebank},
            {"bursa": 0.5, "phrasebank": 0.5},
            variant_id="temperature-test",
            ensemble_temperature=2.0,
        )
        probabilities = calibrated.predict_proba(["article"])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=7)
        self.assertLess(float(probabilities[0, 2]), 0.5)

    def test_class_order_mismatch_is_rejected(self) -> None:
        bursa = _FakeSentimentModel([[0.10, 0.20, 0.70]])
        reversed_model = _FakeSentimentModel(
            [[0.30, 0.40, 0.30]], classes=["positive", "neutral", "negative"]
        )
        with self.assertRaises(ValueError):
            WeightedFinBERTEnsemble(
                {"bursa": bursa, "phrasebank": reversed_model},
                {"bursa": 0.45, "phrasebank": 0.55},
                variant_id="test",
            )

    def test_seed_average_is_a_normalized_runtime_component(self) -> None:
        first = _FakeSentimentModel([[0.10, 0.20, 0.70]])
        second = _FakeSentimentModel([[0.30, 0.50, 0.20]])
        averaged = AveragedFinBERTEnsemble(
            {"42": first, "52": second}, variant_id="bursa-seed-average"
        )
        self.assertTrue(np.allclose(averaged.predict_proba(["article"]), [[0.20, 0.35, 0.45]]))
        self.assertEqual(averaged.predict(["article"]).tolist(), ["positive"])


if __name__ == "__main__":
    unittest.main()
