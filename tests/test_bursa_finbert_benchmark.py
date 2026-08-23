from __future__ import annotations

import unittest

import pandas as pd

from pipeline.benchmark_bursa_finbert import _mixed_training_frame
from pipeline.sentiment_text import (
    TEXT_MODE_TARGET_AWARE,
    apply_sentiment_text_mode,
)


class BursaFinBERTBenchmarkTests(unittest.TestCase):
    def test_replay_mix_is_deterministic_and_bursa_led(self) -> None:
        bursa = pd.DataFrame(
            {
                "model_text": [f"bursa-{index}" for index in range(70)],
                "sentiment_label": ["neutral"] * 70,
            }
        )
        phrasebank = pd.DataFrame(
            {
                "model_text": [f"phrase-{index}" for index in range(100)],
                "sentiment_label": ["positive"] * 100,
            }
        )
        first = _mixed_training_frame(bursa, phrasebank, seed=42)
        second = _mixed_training_frame(bursa, phrasebank, seed=42)
        self.assertEqual(first["model_text"].tolist(), second["model_text"].tolist())
        self.assertEqual(int(first["domain"].eq("bursa").sum()), 70)
        self.assertEqual(int(first["domain"].eq("phrasebank").sum()), 30)

    def test_replay_share_is_explicit_and_validated(self) -> None:
        bursa = pd.DataFrame(
            {"model_text": [f"bursa-{index}" for index in range(60)], "sentiment_label": ["neutral"] * 60}
        )
        phrasebank = pd.DataFrame(
            {"model_text": [f"phrase-{index}" for index in range(100)], "sentiment_label": ["positive"] * 100}
        )
        mixed = _mixed_training_frame(
            bursa, phrasebank, seed=42, bursa_replay_share=0.60
        )
        self.assertEqual(int(mixed["domain"].eq("bursa").sum()), 60)
        self.assertEqual(int(mixed["domain"].eq("phrasebank").sum()), 40)
        with self.assertRaises(ValueError):
            _mixed_training_frame(bursa, phrasebank, seed=42, bursa_replay_share=0.0)

    def test_target_aware_representation_is_shared_by_bursa_and_phrasebank_rows(self) -> None:
        bursa = pd.DataFrame(
            {
                "ticker": ["1155.KL"],
                "company_name": ["Malayan Banking Berhad"],
                "title": ["Maybank profit rises"],
                "description": ["Maybank reported higher quarterly profit."],
                "model_text": ["Maybank profit rises. Maybank reported higher quarterly profit."],
                "sentiment_label": ["positive"],
            }
        )
        phrasebank = pd.DataFrame(
            {"model_text": ["Company earnings increased."], "sentiment_label": ["positive"]}
        )
        transformed = apply_sentiment_text_mode(bursa, mode=TEXT_MODE_TARGET_AWARE)
        self.assertIn("Target company: Malayan Banking Berhad (1155.KL)", transformed.iloc[0]["model_text"])

        mixed = _mixed_training_frame(
            bursa,
            phrasebank,
            seed=42,
            bursa_replay_share=0.5,
            text_mode=TEXT_MODE_TARGET_AWARE,
        )
        phrasebank_text = mixed.loc[mixed["domain"].eq("phrasebank"), "model_text"].tolist()
        self.assertEqual(phrasebank_text, ["Company earnings increased."])


if __name__ == "__main__":
    unittest.main()
