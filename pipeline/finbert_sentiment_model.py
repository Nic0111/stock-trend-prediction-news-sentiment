"""Runtime adapter for the evaluated StockSense FinBERT sentiment model."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

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

    def __init__(
        self,
        model: object,
        tokenizer: object,
        torch: object,
        device: object,
        temperature: float = 1.0,
        class_order: Sequence[str] = tuple(LABEL_ORDER.tolist()),
        max_length: int = MAX_LENGTH,
        artifact_signature: str = "",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._device = device
        self.temperature = float(temperature)
        self.classes_ = np.asarray(list(class_order), dtype=object)
        self.max_length = int(max_length)
        self.artifact_signature = str(artifact_signature)
        self.variant_id = "finbert_phrasebank_v1"
        self.sentiment_model_variant = self.variant_id
        self.component_weights = {"phrasebank": 1.0}

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        values = [str(text or "") for text in texts]
        if not values:
            return np.empty((0, len(self.classes_)), dtype=float)
        probabilities = []
        self._model.eval()
        with self._torch.no_grad():
            for start in range(0, len(values), INFERENCE_BATCH_SIZE):
                batch_texts = values[start : start + INFERENCE_BATCH_SIZE]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits.detach().float().cpu().numpy()
                probabilities.append(_softmax(logits, self.temperature))
        return np.concatenate(probabilities, axis=0)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)]


class WeightedFinBERTEnsemble:
    """Blend calibrated FinBERT probabilities without changing the model API.

    The ensemble deliberately combines probabilities rather than hard labels.  This
    preserves uncertainty from both checkpoints and makes the blend auditable at
    article level.  Every component must expose the same ordered class contract.
    """

    def __init__(
        self,
        components: Mapping[str, FinBERTSentimentModel],
        weights: Mapping[str, float],
        *,
        variant_id: str,
        ensemble_temperature: float = 1.0,
    ) -> None:
        if not components:
            raise ValueError("At least one FinBERT component is required")
        if set(components) != set(weights):
            raise ValueError("Ensemble component names and weights must match")
        first_name = next(iter(components))
        first_classes = [str(value).casefold() for value in components[first_name].classes_]
        if set(first_classes) != {"negative", "neutral", "positive"}:
            raise ValueError("FinBERT components must expose negative/neutral/positive classes")
        normalized_weights = {str(name): float(value) for name, value in weights.items()}
        if any(value < 0.0 for value in normalized_weights.values()):
            raise ValueError("Ensemble weights cannot be negative")
        weight_total = sum(normalized_weights.values())
        if weight_total <= 0.0:
            raise ValueError("Ensemble weights must have a positive total")
        normalized_weights = {
            name: value / weight_total for name, value in normalized_weights.items()
        }
        for name, component in components.items():
            classes = [str(value).casefold() for value in component.classes_]
            if classes != first_classes:
                raise ValueError(
                    f"FinBERT component {name!r} has an incompatible class order"
                )
        self.component_models = dict(components)
        self.component_weights = normalized_weights
        self.ensemble_temperature = float(ensemble_temperature)
        if not np.isfinite(self.ensemble_temperature) or self.ensemble_temperature <= 0.0:
            raise ValueError("ensemble_temperature must be a finite positive value")
        self.classes_ = np.asarray(first_classes, dtype=object)
        self.variant_id = str(variant_id)
        self.sentiment_model_variant = self.variant_id
        self.artifact_signature = hashlib.sha256(
            json.dumps(
                {
                    "variant": self.variant_id,
                    "weights": self.component_weights,
                    "ensemble_temperature": self.ensemble_temperature,
                    "components": {
                        name: getattr(model, "artifact_signature", "")
                        for name, model in self.component_models.items()
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def predict_component_proba(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(model.predict_proba(texts), dtype=float)
            for name, model in self.component_models.items()
        }

    def blend_component_proba(
        self, component_probabilities: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        if not component_probabilities:
            return np.empty((0, len(self.classes_)), dtype=float)
        blended = np.zeros_like(next(iter(component_probabilities.values())), dtype=float)
        for name, probabilities in component_probabilities.items():
            if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes_):
                raise ValueError(f"FinBERT component {name!r} returned an invalid probability shape")
            blended += self.component_weights[name] * probabilities
        totals = blended.sum(axis=1, keepdims=True)
        blended = blended / np.maximum(totals, 1e-12)
        if self.ensemble_temperature != 1.0:
            blended = _softmax(np.log(np.clip(blended, 1e-12, 1.0)), self.ensemble_temperature)
        return blended

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        component_probabilities = self.predict_component_proba(texts)
        return self.blend_component_proba(component_probabilities)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)]


class AveragedFinBERTEnsemble:
    """Average several compatible FinBERT checkpoints as one component.

    This is intentionally separate from :class:`WeightedFinBERTEnsemble`:
    seed averaging reduces stochastic variation within one Bursa-domain model,
    while weighted blending combines the Bursa-domain component with the
    PhraseBank component.  The class preserves the normal ``predict_proba``
    contract so it can be used by existing article-scoring code.
    """

    def __init__(
        self,
        components: Mapping[str, FinBERTSentimentModel],
        *,
        variant_id: str,
    ) -> None:
        if len(components) < 2:
            raise ValueError("AveragedFinBERTEnsemble requires at least two seed components")
        first_name = next(iter(components))
        first_classes = [str(value).casefold() for value in components[first_name].classes_]
        if set(first_classes) != {"negative", "neutral", "positive"}:
            raise ValueError("FinBERT seed components must expose negative/neutral/positive classes")
        for name, component in components.items():
            classes = [str(value).casefold() for value in component.classes_]
            if classes != first_classes:
                raise ValueError(
                    f"FinBERT seed component {name!r} has an incompatible class order"
                )
        self.component_models = dict(components)
        self.component_weights = {
            str(name): 1.0 / len(self.component_models) for name in self.component_models
        }
        self.classes_ = np.asarray(first_classes, dtype=object)
        self.variant_id = str(variant_id)
        self.sentiment_model_variant = self.variant_id
        self.artifact_signature = hashlib.sha256(
            json.dumps(
                {
                    "variant": self.variant_id,
                    "aggregation": "uniform_seed_average",
                    "components": {
                        name: getattr(model, "artifact_signature", "")
                        for name, model in self.component_models.items()
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def predict_component_proba(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(model.predict_proba(texts), dtype=float)
            for name, model in self.component_models.items()
        }

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        components = self.predict_component_proba(texts)
        if not components:
            return np.empty((0, len(self.classes_)), dtype=float)
        values = np.mean(list(components.values()), axis=0)
        return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        probabilities = self.predict_proba(texts)
        return self.classes_[probabilities.argmax(axis=1)]


def finbert_directory_signature(model_dir: str | Path = FINBERT_MODEL_DIR) -> str:
    model_path = Path(model_dir)
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "metadata.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "model.safetensors",
    ):
        path = model_path / name
        digest.update(name.encode("utf-8"))
        if not path.exists():
            digest.update(b"missing")
            continue
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _load_finbert_sentiment_model_cached(
    model_dir: str,
    temperature: float,
    artifact_signature: str,
) -> FinBERTSentimentModel:
    model_path = Path(model_dir)
    required_files = [model_path / "config.json", model_path / "model.safetensors"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FinBERTArtifactError(
            "Deployed FinBERT sentiment model is incomplete. Missing: " + ", ".join(missing)
        )
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
        id2label = config.get("id2label") or {}
        class_order = [
            str(id2label.get(str(index), id2label.get(index, ""))).casefold()
            for index in range(int(config.get("num_labels", len(id2label) or 3)))
        ]
        metadata_order = [str(value).casefold() for value in metadata.get("class_order", [])]
        max_length = int(metadata.get("max_length", 0))
    except (OSError, ValueError, TypeError) as exc:
        raise FinBERTArtifactError("FinBERT metadata is missing or invalid.") from exc
    if (
        class_order != metadata_order
        or set(class_order) != {"negative", "neutral", "positive"}
        or max_length < 1
    ):
        raise FinBERTArtifactError(
            "FinBERT label order or maximum sequence length is incompatible."
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
    return FinBERTSentimentModel(
        model,
        tokenizer,
        torch,
        device,
        temperature=temperature,
        class_order=class_order,
        max_length=max_length,
        artifact_signature=artifact_signature,
    )


def load_finbert_sentiment_model(
    model_dir: str = str(FINBERT_MODEL_DIR), temperature: float = 1.0
) -> FinBERTSentimentModel:
    signature = finbert_directory_signature(model_dir)
    return _load_finbert_sentiment_model_cached(
        str(Path(model_dir).resolve()), float(temperature), signature
    )


def clear_finbert_runtime_cache() -> None:
    _load_finbert_sentiment_model_cached.cache_clear()
