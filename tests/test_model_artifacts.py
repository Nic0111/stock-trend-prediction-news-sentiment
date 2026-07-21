from __future__ import annotations

import warnings
import unittest

import numpy as np
import pandas as pd

from pipeline.model_artifacts import CalibratedStockModel


class FakePreprocessor:
    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([[1.0, 2.0] for _ in range(len(frame))])


class FakeClassifier:
    feature_names_in_ = np.asarray(["Column_0", "Column_1"])

    def __init__(self) -> None:
        self.last_input: pd.DataFrame | None = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.last_input = frame
        return np.tile(np.asarray([[0.25, 0.75]]), (len(frame), 1))


class FakePipeline:
    def __init__(self, classifier: FakeClassifier) -> None:
        self.classifier = classifier
        self.named_steps = {
            "preprocess": FakePreprocessor(),
            "classifier": classifier,
        }


class IdentityCalibrator:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray([[0.25, 0.75]]), (len(values), 1))


class ModelArtifactTests(unittest.TestCase):
    def test_transformed_lightgbm_input_preserves_fitted_names_without_warning(self) -> None:
        classifier = FakeClassifier()
        artifact = CalibratedStockModel(
            estimator=FakePipeline(classifier),
            calibrator=IdentityCalibrator(),
            threshold=0.5,
            input_columns=["Ticker"],
            numeric_feature_columns=[],
            refinement_round=2,
            model_name="lightgbm",
            horizon_days=5,
        )
        selected = pd.DataFrame({"Ticker": ["1295.KL"]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probabilities = artifact._predict_estimator_proba(selected)

        self.assertEqual(probabilities.tolist(), [0.75])
        self.assertEqual(len(caught), 0)
        self.assertIsInstance(classifier.last_input, pd.DataFrame)
        self.assertEqual(classifier.last_input.columns.tolist(), ["Column_0", "Column_1"])


if __name__ == "__main__":
    unittest.main()
