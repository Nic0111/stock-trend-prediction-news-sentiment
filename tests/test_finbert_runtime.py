import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.finbert_sentiment_model import (
    FinBERTDependencyError,
    LABEL_ORDER,
    _softmax,
)
from pipeline import download_financial_news_data


class FinBERTRuntimeTests(unittest.TestCase):
    def test_probability_adapter_preserves_class_order(self) -> None:
        probabilities = _softmax(np.asarray([[4.0, 1.0, -1.0], [-1.0, 1.0, 4.0]]))
        self.assertEqual(LABEL_ORDER.tolist(), ["negative", "neutral", "positive"])
        self.assertEqual(probabilities.argmax(axis=1).tolist(), [0, 2])
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_temperature_scaling_preserves_probability_contract(self) -> None:
        logits = np.asarray([[4.0, 1.0, -1.0]])
        sharp = _softmax(logits, temperature=1.0)
        calibrated = _softmax(logits, temperature=1.52)
        self.assertEqual(int(sharp.argmax(axis=1)[0]), int(calibrated.argmax(axis=1)[0]))
        self.assertLess(float(calibrated.max()), float(sharp.max()))
        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1.0))

    def test_runtime_rejects_non_finbert_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "sentiment-runtime-config.json"
            config_path.write_text(json.dumps({"backend": "tfidf"}), encoding="utf-8")
            with patch.object(download_financial_news_data, "SENTIMENT_RUNTIME_CONFIG_FILE", config_path):
                with self.assertRaisesRegex(ValueError, "must be FinBERT"):
                    download_financial_news_data.load_sentiment_model()

    def test_newsapi_is_not_called_when_sentiment_runtime_is_unavailable(self) -> None:
        with (
            patch.object(
                download_financial_news_data,
                "load_sentiment_model",
                side_effect=FinBERTDependencyError("missing runtime"),
            ),
            patch.object(download_financial_news_data, "fetch_newsapi_response") as fetch,
        ):
            with self.assertRaises(FinBERTDependencyError):
                download_financial_news_data.retrieve_financial_news("1295.KL")
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
