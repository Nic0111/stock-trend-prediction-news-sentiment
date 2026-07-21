"""Runtime adapter for the evaluated StockSense FinBERT sentiment model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINBERT_MODEL_DIR = PROJECT_ROOT / "models" / "experimental" / "finbert-sentiment"
LABEL_ORDER = np.asarray(["negative", "neutral", "positive"], dtype=object)
MAX_LENGTH = 128
INFERENCE_BATCH_SIZE = 8


class FinBERTRuntimeError(RuntimeError):
    """Base error for failures in the shared FinBERT runtime."""


class FinBERTDependencyError(FinBERTRuntimeError):
    """Raised when the Python packages needed by FinBERT are unavailable."""


class FinBERTArtifactError(FinBERTRuntimeError):
    """Raised when the deployed local FinBERT artifact cannot be loaded."""


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-6)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


class FinBERTSentimentModel:
    """Expose the same prediction contract as the previous sklearn pipeline."""

    classes_ = LABEL_ORDER

    def __init__(
        self,
        model: object,
        tokenizer: object,
        torch: object,
        device: object,
        temperature: float = 1.0,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._device = device
        self.temperature = float(temperature)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        values = [str(text or "") for text in texts]
        if not values:
            return np.empty((0, len(LABEL_ORDER)), dtype=float)
        probabilities = []
        self._model.eval()
        with self._torch.no_grad():
            for start in range(0, len(values), INFERENCE_BATCH_SIZE):
                batch_texts = values[start : start + INFERENCE_BATCH_SIZE]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits.detach().float().cpu().numpy()
                probabilities.append(_softmax(logits, self.temperature))
        return np.concatenate(probabilities, axis=0)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)]


@lru_cache(maxsize=1)
def load_finbert_sentiment_model(
    model_dir: str = str(FINBERT_MODEL_DIR), temperature: float = 1.0
) -> FinBERTSentimentModel:
    model_path = Path(model_dir)
    required_files = [model_path / "config.json", model_path / "model.safetensors"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FinBERTArtifactError(
            "Deployed FinBERT sentiment model is incomplete. Missing: " + ", ".join(missing)
        )
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise FinBERTDependencyError(
            "FinBERT runtime dependencies are missing. Install torch, transformers, "
            "and accelerate in the active project environment."
        ) from exc

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(device)
    except ImportError as exc:
        raise FinBERTDependencyError(
            "FinBERT runtime dependencies are missing. Install torch, transformers, "
            "and accelerate in the active project environment."
        ) from exc
    except (OSError, ValueError) as exc:
        raise FinBERTArtifactError(
            f"FinBERT model files could not be loaded from {model_path}."
        ) from exc
    model.eval()
    return FinBERTSentimentModel(model, tokenizer, torch, device, temperature=temperature)
