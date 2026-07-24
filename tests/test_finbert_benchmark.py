import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.benchmark_finbert_sentiment import (
    _resolve_device,
    build_frozen_split,
    expected_calibration_error,
    fit_temperature,
    metrics_from_probabilities,
)


class FinBERTBenchmarkTests(unittest.TestCase):
    def test_cuda_request_fails_instead_of_silently_falling_back(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def device(name: str) -> str:
                return name

        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            _resolve_device(FakeTorch, "cuda")

    def test_temperature_is_finite_and_probabilities_are_valid(self) -> None:
        logits = np.asarray([[3.0, 0.0, -1.0], [0.2, 1.4, 0.0], [-1.0, 0.0, 2.0]])
        labels = np.asarray([0, 1, 2])
        temperature = fit_temperature(logits, labels)
        self.assertGreater(temperature, 0.0)
        probabilities = np.exp(logits / temperature)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_metrics_include_per_class_and_calibration_fields(self) -> None:
        probabilities = np.asarray(
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
        )
        labels = np.asarray([0, 1, 2])
        metrics = metrics_from_probabilities("test", probabilities, labels)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertIn("positive_f1", metrics)
        self.assertIn("ece", metrics)

    def test_split_manifest_is_stratified_and_disjoint(self) -> None:
        rows = []
        for label, word in [("negative", "loss"), ("neutral", "steady"), ("positive", "gain")]:
            for index in range(20):
                rows.append({"Sentiment": label, "Sentence": f"{word} company result {index}"})
        # The split builder intentionally persists the manifest used by the
        # offline benchmark.  Redirect that write so a unit test can never
        # replace the real 4,838-row project manifest with this tiny fixture.
        with TemporaryDirectory() as directory, patch(
            "pipeline.benchmark_finbert_sentiment.SPLIT_FILE",
            Path(directory) / "sentiment-split-manifest.csv",
        ):
            development, test, manifest = build_frozen_split(pd.DataFrame(rows))
        self.assertEqual(len(development) + len(test), len(rows))
        self.assertEqual(manifest["sentence_hash"].nunique(), len(rows))
        self.assertEqual(manifest.groupby("sentence_hash")["split"].nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
