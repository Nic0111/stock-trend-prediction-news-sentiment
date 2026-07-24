"""Leakage-safe FinBERT challenger benchmark for StockSense.

This module deliberately does not replace the deployed TF-IDF sentiment model.
It compares the existing baseline with a finance-pretrained FinBERT model and
an optional calibrated probability ensemble, then writes a recommendation for
manual review.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from pipeline.train_financial_news_sentiment_model import (
    LABEL_ORDER,
    RANDOM_STATE,
    build_sentiment_pipeline,
    load_sentiment_training_data,
    sentiment_parameter_space,
)
from pipeline.runtime_provenance import SENTIMENT_RUNTIME_CONFIG_FILE, sentiment_artifact_hashes


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "finbert-benchmark"
MODELS_DIR = PROJECT_ROOT / "models" / "experimental" / "finbert-sentiment"
BASELINE_METADATA_FILE = PROJECT_ROOT / "models" / "financial-news-sentiment-model-metadata.json"
SPLIT_FILE = RESULTS_DIR / "sentiment-split-manifest.csv"
COMPARISON_FILE = RESULTS_DIR / "sentiment-model-comparison.csv"
REPORT_FILE = RESULTS_DIR / "sentiment-model-comparison.md"
RECOMMENDATION_FILE = RESULTS_DIR / "sentiment-model-recommendation.json"
FINBERT_METADATA_FILE = MODELS_DIR / "metadata.json"

# This is a finance-pretrained base model, not the ProsusAI checkpoint that is
# already fine-tuned on Financial PhraseBank.
FINBERT_BASE_MODEL = "yiyanghkust/finbert-pretrain"
MAX_LENGTH = 128
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _sentence_hash(sentence: str) -> str:
    return hashlib.sha256(sentence.strip().encode("utf-8")).hexdigest()


def build_frozen_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and persist the exact split used by every challenger."""
    development, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["Sentiment"],
        random_state=RANDOM_STATE,
    )
    development = development.reset_index(drop=True)
    test = test.reset_index(drop=True)
    manifest = pd.concat(
        [
            development.assign(split="development"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    manifest.insert(0, "row_id", range(len(manifest)))
    manifest["sentence_hash"] = manifest["Sentence"].map(_sentence_hash)
    if manifest["sentence_hash"].duplicated().any():
        raise ValueError("The frozen sentiment split contains duplicate sentence hashes.")
    split_counts = manifest.groupby("sentence_hash")["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError("A sentence appears in more than one sentiment split.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest[["row_id", "sentence_hash", "Sentiment", "split"]].to_csv(
        SPLIT_FILE, index=False
    )
    return development, test, manifest


def _require_transformer_dependencies() -> Tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "FinBERT dependencies are missing. Install the project requirements, "
            "including torch, transformers and accelerate, then retry."
        ) from exc
    return torch, DataLoader, Dataset, (AutoModelForSequenceClassification, AutoTokenizer)


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the official FinBERT run, but no CUDA device is available.")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


def _stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponent = np.exp(scaled)
    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit scalar temperature on development out-of-fold logits only."""
    candidates = np.linspace(0.50, 5.00, 451)
    losses = [log_loss(labels, _stable_softmax(logits, value), labels=list(range(3))) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    total = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(0.0, 1.0, bins + 1)[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            total += float(mask.mean()) * abs(float(confidence[mask].mean()) - float((predictions[mask] == labels[mask]).mean()))
    return float(total)


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def metrics_from_probabilities(name: str, probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(3)), zero_division=0
    )
    return {
        "model": name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=list(range(3)))),
        "brier_score": multiclass_brier(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels),
        "negative_precision": float(precision[0]),
        "negative_recall": float(recall[0]),
        "negative_f1": float(f1[0]),
        "neutral_precision": float(precision[1]),
        "neutral_recall": float(recall[1]),
        "neutral_f1": float(f1[1]),
        "positive_precision": float(precision[2]),
        "positive_recall": float(recall[2]),
        "positive_f1": float(f1[2]),
    }


def _bootstrap_macro_f1_delta(
    first: np.ndarray, second: np.ndarray, labels: np.ndarray, iterations: int = 2000
) -> Dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    deltas = np.empty(iterations, dtype=float)
    indices = np.arange(len(labels))
    for index in range(iterations):
        sample = rng.choice(indices, size=len(indices), replace=True)
        deltas[index] = f1_score(labels[sample], first[sample], average="macro", zero_division=0) - f1_score(
            labels[sample], second[sample], average="macro", zero_division=0
        )
    return {
        "delta_macro_f1": float(deltas.mean()),
        "ci95_lower": float(np.percentile(deltas, 2.5)),
        "ci95_upper": float(np.percentile(deltas, 97.5)),
    }


def _run_tfidf_benchmark(
    development: pd.DataFrame,
    test: pd.DataFrame,
    profile: str,
) -> Dict[str, Any]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # Reuse the parameters already selected by the existing baseline training
    # pipeline. Re-running the full randomized search here would add several
    # hours without changing the baseline being compared.
    if BASELINE_METADATA_FILE.exists():
        metadata = json.loads(BASELINE_METADATA_FILE.read_text(encoding="utf-8"))
        best_parameters = metadata.get("best_parameters", {})
    else:
        best_parameters = {
            "tfidf__ngram_range": (1, 2),
            "tfidf__min_df": 2,
            "tfidf__max_features": 10_000,
            "tfidf__sublinear_tf": False,
            "classifier__C": 10.0,
        }
    best_parameters = {
        key: tuple(value) if key == "tfidf__ngram_range" else value
        for key, value in best_parameters.items()
    }
    final_model = build_sentiment_pipeline().set_params(**best_parameters)
    final_model.fit(development["Sentence"], development["Sentiment"])
    class_order = [str(value).lower() for value in final_model.classes_]
    class_to_index = {label: index for index, label in enumerate(class_order)}
    labels = development["Sentiment"].map(class_to_index).to_numpy(dtype=int)
    oof_logits = np.zeros((len(development), 3), dtype=float)
    for train_indices, validation_indices in cv.split(development["Sentence"], development["Sentiment"]):
        fold_model = build_sentiment_pipeline().set_params(**best_parameters)
        fold_model.fit(
            development.iloc[train_indices]["Sentence"],
            development.iloc[train_indices]["Sentiment"],
        )
        oof_logits[validation_indices] = fold_model.decision_function(
            development.iloc[validation_indices]["Sentence"]
        )
    test_logits = final_model.decision_function(test["Sentence"])
    return {
        "name": "tfidf_logistic_regression",
        "oof_logits": oof_logits,
        "test_logits": np.asarray(test_logits, dtype=float),
        "best_parameters": best_parameters,
        "cv_macro_f1": float(f1_score(labels, oof_logits.argmax(axis=1), average="macro", zero_division=0)),
        "class_order": class_order,
    }


class _TokenTensorDataset:
    """Small torch Dataset wrapper kept local to avoid a hard torch import."""

    def __init__(self, encoded: Dict[str, Any], labels: np.ndarray, torch: Any) -> None:
        self.encoded = encoded
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        return item


def _autocast(torch: Any, device: Any, enabled: bool) -> Any:
    if not enabled:
        from contextlib import nullcontext

        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _make_finbert_model(model_class: Any, torch: Any) -> Any:
    model = model_class.from_pretrained(
        FINBERT_BASE_MODEL,
        num_labels=3,
        id2label={index: label.upper() for index, label in enumerate(LABEL_ORDER)},
        label2id={label.upper(): index for index, label in enumerate(LABEL_ORDER)},
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _evaluate_finbert(model: Any, loader: Any, torch: Any, device: Any, amp_enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(torch, device, amp_enabled):
                outputs = model(**batch)
            logits.append(outputs.logits.detach().float().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _train_finbert(
    model: Any,
    train_loader: Any,
    epochs: int,
    torch: Any,
    device: Any,
    amp_enabled: bool,
    learning_rate: float = 2e-5,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    accumulation = 8
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast(torch, device, amp_enabled):
                loss = model(**batch).loss / accumulation
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            is_last = step == len(train_loader) - 1
            if (step + 1) % accumulation == 0 or is_last:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.train()


def _finbert_fold(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    validation_texts: Sequence[str],
    validation_labels: np.ndarray,
    tokenizer: Any,
    model_class: Any,
    torch: Any,
    DataLoader: Any,
    profile: str,
    device: Any,
) -> Tuple[np.ndarray, int, float]:
    amp_enabled = device.type == "cuda"
    batch_size = 4 if profile == "full" else 2
    epochs = 4 if profile == "full" else 1
    train_encoded = tokenizer(
        list(train_texts), padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    validation_encoded = tokenizer(
        list(validation_texts), padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    train_dataset = _TokenTensorDataset(train_encoded, train_labels, torch)
    validation_dataset = _TokenTensorDataset(validation_encoded, validation_labels, torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    model = _make_finbert_model(model_class, torch).to(device)
    best_logits = None
    best_score = -np.inf
    best_epoch = 1
    patience = 1
    stale = 0
    for epoch in range(1, epochs + 1):
        _train_finbert(model, train_loader, 1, torch, device, amp_enabled)
        validation_logits, validation_targets = _evaluate_finbert(model, validation_loader, torch, device, amp_enabled)
        score = f1_score(validation_targets, validation_logits.argmax(axis=1), average="macro", zero_division=0)
        if score > best_score + 1e-6:
            best_score = float(score)
            best_logits = validation_logits.copy()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    del model, train_loader, validation_loader, train_dataset, validation_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if best_logits is None:
        raise RuntimeError("FinBERT did not produce validation logits.")
    return best_logits, best_epoch, float(best_score)


def _run_finbert_benchmark(
    development: pd.DataFrame,
    test: pd.DataFrame,
    profile: str,
    device_name: str = "auto",
    model_output_dir: Path = MODELS_DIR,
    run_id: str | None = None,
) -> Dict[str, Any]:
    torch, DataLoader, Dataset, model_classes = _require_transformer_dependencies()
    model_class, tokenizer_class = model_classes
    _set_seed(RANDOM_STATE, torch)
    device = _resolve_device(torch, device_name)
    tokenizer = tokenizer_class.from_pretrained(FINBERT_BASE_MODEL, use_fast=True)
    development_labels = development["Sentiment"].map(LABEL_TO_ID).to_numpy(dtype=int)
    test_labels = test["Sentiment"].map(LABEL_TO_ID).to_numpy(dtype=int)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if profile != "full":
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
    oof_logits = np.zeros((len(development), 3), dtype=float)
    fold_epochs: List[int] = []
    fold_scores: List[float] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        folds.split(development["Sentence"], development["Sentiment"]), start=1
    ):
        print(f"FinBERT fold {fold_index}/{folds.n_splits}...")
        logits, best_epoch, score = _finbert_fold(
            development.iloc[train_indices]["Sentence"].tolist(),
            development_labels[train_indices],
            development.iloc[validation_indices]["Sentence"].tolist(),
            development_labels[validation_indices],
            tokenizer,
            model_class,
            torch,
            DataLoader,
            profile,
            device,
        )
        oof_logits[validation_indices] = logits
        fold_epochs.append(best_epoch)
        fold_scores.append(score)

    final_epochs = max(1, int(round(float(np.median(fold_epochs)))))
    amp_enabled = device.type == "cuda"
    batch_size = 4 if profile == "full" else 2
    full_encoded = tokenizer(
        development["Sentence"].tolist(), padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    test_encoded = tokenizer(
        test["Sentence"].tolist(), padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    full_dataset = _TokenTensorDataset(full_encoded, development_labels, torch)
    test_dataset = _TokenTensorDataset(test_encoded, test_labels, torch)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    model = _make_finbert_model(model_class, torch).to(device)
    _train_finbert(model, full_loader, final_epochs, torch, device, amp_enabled)
    test_logits, _ = _evaluate_finbert(model, test_loader, torch, device, amp_enabled)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    _write_json(
        model_output_dir / "metadata.json",
        {
            "base_model": FINBERT_BASE_MODEL,
            "max_length": MAX_LENGTH,
            "folds": folds.n_splits,
            "fold_best_epochs": fold_epochs,
            "fold_macro_f1": fold_scores,
            "final_epochs": final_epochs,
            "class_order": LABEL_ORDER,
            "dataset_split": str(SPLIT_FILE.relative_to(PROJECT_ROOT)),
            "device": str(device),
            "run_id": run_id,
        },
    )
    del model, full_loader, test_loader, full_dataset, test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": "finbert",
        "oof_logits": oof_logits,
        "test_logits": test_logits,
        "fold_best_epochs": fold_epochs,
        "fold_macro_f1": fold_scores,
        "final_epochs": final_epochs,
        "device": str(device),
        "model_output_dir": str(model_output_dir.relative_to(PROJECT_ROOT)),
        "run_id": run_id,
    }


def _select_ensemble_weight(
    tfidf_probabilities: np.ndarray, finbert_probabilities: np.ndarray, labels: np.ndarray
) -> Tuple[float, np.ndarray]:
    best: Tuple[float, float, float, np.ndarray] | None = None
    for weight in np.linspace(0.0, 1.0, 21):
        probabilities = weight * tfidf_probabilities + (1.0 - weight) * finbert_probabilities
        predictions = probabilities.argmax(axis=1)
        candidate = (
            float(f1_score(labels, predictions, average="macro", zero_division=0)),
            float(balanced_accuracy_score(labels, predictions)),
            -float(log_loss(labels, probabilities, labels=list(range(3)))),
            probabilities,
        )
        if best is None or candidate[:3] > best[:3]:
            best = (candidate[0], candidate[1], candidate[2], probabilities)
            selected_weight = float(weight)
    if best is None:
        raise RuntimeError("Unable to select an ensemble weight.")
    return selected_weight, best[3]


def write_existing_report() -> None:
    """Regenerate the human-readable report from completed benchmark artifacts."""
    comparison = pd.read_csv(COMPARISON_FILE)
    recommendation = json.loads(RECOMMENDATION_FILE.read_text(encoding="utf-8"))
    report_lines = [
        "# StockSense sentiment model challenger benchmark",
        "",
        f"- Profile: `{recommendation.get('profile', 'unknown')}`",
        f"- Frozen test rows: {recommendation.get('test_rows', 'unknown')}",
        f"- FinBERT base: `{recommendation.get('finbert_base_model', FINBERT_BASE_MODEL)}`",
        f"- Ensemble weights: TF-IDF `{float(recommendation.get('ensemble_tfidf_weight', 0.0)):.2f}`, FinBERT `{float(recommendation.get('ensemble_finbert_weight', 0.0)):.2f}`",
        f"- Recommendation: **{recommendation.get('recommendation', 'unknown')}** (manual approval required)",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Decision evidence",
        "",
        f"- FinBERT vs TF-IDF macro-F1 bootstrap interval: {recommendation['finbert_vs_tfidf_bootstrap']['ci95_lower']:.4f} to {recommendation['finbert_vs_tfidf_bootstrap']['ci95_upper']:.4f}.",
        f"- Ensemble vs best standalone macro-F1 bootstrap interval: {recommendation['ensemble_vs_best_standalone_bootstrap']['ci95_lower']:.4f} to {recommendation['ensemble_vs_best_standalone_bootstrap']['ci95_upper']:.4f}.",
        "- The deployed TF-IDF artifact was not overwritten.",
        "- PhraseBank-only evaluation does not establish Malaysian Bursa news generalisation.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")


def _metrics_table(
    tfidf: Dict[str, Any], finbert: Dict[str, Any], labels: np.ndarray
) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
    tfidf_temperature = fit_temperature(tfidf["oof_logits"], labels)
    finbert_temperature = fit_temperature(finbert["oof_logits"], labels)
    tfidf_oof_probabilities = _stable_softmax(tfidf["oof_logits"], tfidf_temperature)
    finbert_oof_probabilities = _stable_softmax(finbert["oof_logits"], finbert_temperature)
    tfidf_test_probabilities = _stable_softmax(tfidf["test_logits"], tfidf_temperature)
    finbert_test_probabilities = _stable_softmax(finbert["test_logits"], finbert_temperature)
    weight, _ = _select_ensemble_weight(tfidf_oof_probabilities, finbert_oof_probabilities, labels)
    ensemble_test_probabilities = weight * tfidf_test_probabilities + (1.0 - weight) * finbert_test_probabilities
    rows = [
        metrics_from_probabilities("tfidf_logistic_regression", tfidf_test_probabilities, _TEST_LABELS),
        metrics_from_probabilities("finbert", finbert_test_probabilities, _TEST_LABELS),
        metrics_from_probabilities("calibrated_ensemble", ensemble_test_probabilities, _TEST_LABELS),
    ]
    for row in rows:
        row["temperature"] = tfidf_temperature if row["model"].startswith("tfidf") else finbert_temperature if row["model"] == "finbert" else None
        row["tfidf_weight"] = weight if row["model"] == "calibrated_ensemble" else None
        row["finbert_weight"] = 1.0 - weight if row["model"] == "calibrated_ensemble" else None
    return rows, weight, ensemble_test_probabilities


_TEST_LABELS = np.empty(0, dtype=int)


def run_finbert_benchmark(
    profile: str = "full",
    device: str = "auto",
    promote: bool = False,
) -> Dict[str, Any]:
    global _TEST_LABELS
    if profile not in {"full", "quick"}:
        raise ValueError("profile must be 'full' or 'quick'")
    data = load_sentiment_training_data()
    development, test, manifest = build_frozen_split(data)
    _TEST_LABELS = test["Sentiment"].map(LABEL_TO_ID).to_numpy(dtype=int)
    print(f"Frozen sentiment split: development={len(development)}, test={len(test)}")
    started = time.perf_counter()
    tfidf = _run_tfidf_benchmark(development, test, profile)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    staging_dir = PROJECT_ROOT / "models" / "experimental" / f"finbert-sentiment-staging-{run_id}"
    finbert = _run_finbert_benchmark(development, test, profile, device, staging_dir, run_id)
    test_labels = _TEST_LABELS
    oof_labels = development["Sentiment"].map(LABEL_TO_ID).to_numpy(dtype=int)
    rows, ensemble_weight, ensemble_probabilities = _metrics_table(tfidf, finbert, oof_labels)
    comparison = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_FILE, index=False)

    predictions = {
        row["model"]: (
            _stable_softmax(tfidf["test_logits"], fit_temperature(tfidf["oof_logits"], oof_labels)).argmax(axis=1)
            if row["model"].startswith("tfidf")
            else _stable_softmax(finbert["test_logits"], fit_temperature(finbert["oof_logits"], oof_labels)).argmax(axis=1)
            if row["model"] == "finbert"
            else ensemble_probabilities.argmax(axis=1)
        )
        for row in rows
    }
    tfidf_row = next(row for row in rows if row["model"] == "tfidf_logistic_regression")
    finbert_row = next(row for row in rows if row["model"] == "finbert")
    ensemble_row = next(row for row in rows if row["model"] == "calibrated_ensemble")
    finbert_delta = _bootstrap_macro_f1_delta(predictions["finbert"], predictions["tfidf_logistic_regression"], test_labels, 2000)
    ensemble_baseline_name = "finbert" if finbert_row["macro_f1"] >= tfidf_row["macro_f1"] else "tfidf_logistic_regression"
    ensemble_delta = _bootstrap_macro_f1_delta(predictions["calibrated_ensemble"], predictions[ensemble_baseline_name], test_labels, 2000)
    finbert_recall_delta = min(
        finbert_row["negative_recall"] - tfidf_row["negative_recall"],
        finbert_row["neutral_recall"] - tfidf_row["neutral_recall"],
        finbert_row["positive_recall"] - tfidf_row["positive_recall"],
    )
    ensemble_recall_delta = min(
        ensemble_row["negative_recall"] - (finbert_row if ensemble_baseline_name == "finbert" else tfidf_row)["negative_recall"],
        ensemble_row["neutral_recall"] - (finbert_row if ensemble_baseline_name == "finbert" else tfidf_row)["neutral_recall"],
        ensemble_row["positive_recall"] - (finbert_row if ensemble_baseline_name == "finbert" else tfidf_row)["positive_recall"],
    )
    finbert_eligible = bool(
        finbert_delta["ci95_lower"] > 0.0
        and finbert_row["macro_f1"] - tfidf_row["macro_f1"] >= 0.02
        and finbert_row["balanced_accuracy"] >= tfidf_row["balanced_accuracy"] - 0.005
        and finbert_recall_delta >= -0.03
        and finbert_row["log_loss"] <= tfidf_row["log_loss"] + 0.01
    )
    ensemble_eligible = bool(
        0.10 <= ensemble_weight <= 0.90
        and ensemble_delta["ci95_lower"] > 0.0
        and ensemble_row["macro_f1"] - max(tfidf_row["macro_f1"], finbert_row["macro_f1"]) >= 0.01
        and ensemble_row["balanced_accuracy"] >= max(tfidf_row["balanced_accuracy"], finbert_row["balanced_accuracy"]) - 0.005
        and ensemble_recall_delta >= -0.03
        and ensemble_row["log_loss"] <= min(tfidf_row["log_loss"], finbert_row["log_loss"]) + 0.01
    )
    if ensemble_eligible:
        recommendation = "use_ensemble"
    elif finbert_eligible:
        recommendation = "replace_with_finbert"
    else:
        recommendation = "keep_tfidf"
    recommendation_payload = {
        "recommendation": recommendation,
        "requires_user_approval": True,
        "deployment_changed": False,
        "profile": profile,
        "finbert_base_model": FINBERT_BASE_MODEL,
        "ensemble_tfidf_weight": ensemble_weight,
        "ensemble_finbert_weight": 1.0 - ensemble_weight,
        "finbert_eligible": finbert_eligible,
        "ensemble_eligible": ensemble_eligible,
        "finbert_vs_tfidf_bootstrap": finbert_delta,
        "ensemble_vs_best_standalone_bootstrap": ensemble_delta,
        "test_rows": len(test),
        "elapsed_seconds": time.perf_counter() - started,
        "limitation": "Evaluation uses Financial PhraseBank only; Bursa-specific generalisation is not validated.",
        "device_requested": device,
        "device_used": finbert.get("device"),
        "staging_model_directory": finbert.get("model_output_dir"),
        "promotion_requested": promote,
    }
    if promote and finbert_eligible:
        backup_dir = PROJECT_ROOT / "models" / "experimental" / f"finbert-sentiment-backup-{run_id}"
        if MODELS_DIR.exists():
            shutil.copytree(MODELS_DIR, backup_dir)
        if MODELS_DIR.exists():
            shutil.rmtree(MODELS_DIR)
        shutil.copytree(staging_dir, MODELS_DIR)
        runtime_config = {}
        if SENTIMENT_RUNTIME_CONFIG_FILE.exists():
            runtime_config = json.loads(SENTIMENT_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
        runtime_config.update(
            {
                "backend": "finbert",
                "model_directory": str(MODELS_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "training_run_id": run_id,
                "training_device": finbert.get("device"),
                "evaluation_profile": profile,
                "artifact_hashes": sentiment_artifact_hashes(),
            }
        )
        _write_json(SENTIMENT_RUNTIME_CONFIG_FILE, runtime_config)
        recommendation_payload["promoted"] = True
        recommendation_payload["backup_model_directory"] = str(backup_dir.relative_to(PROJECT_ROOT))
    else:
        recommendation_payload["promoted"] = False
    _write_json(RECOMMENDATION_FILE, recommendation_payload)
    report_lines = [
        "# StockSense sentiment model challenger benchmark",
        "",
        f"- Profile: `{profile}`",
        f"- Frozen split: {len(development):,} development / {len(test):,} test rows",
        f"- FinBERT base: `{FINBERT_BASE_MODEL}`",
        f"- Ensemble weights: TF-IDF `{ensemble_weight:.2f}`, FinBERT `{1.0 - ensemble_weight:.2f}`",
        f"- Recommendation: **{recommendation}** (manual approval required)",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Decision evidence",
        "",
        f"- FinBERT vs TF-IDF macro-F1 bootstrap interval: {finbert_delta['ci95_lower']:.4f} to {finbert_delta['ci95_upper']:.4f}.",
        f"- Ensemble vs {ensemble_baseline_name} macro-F1 bootstrap interval: {ensemble_delta['ci95_lower']:.4f} to {ensemble_delta['ci95_upper']:.4f}.",
        "- The deployed TF-IDF artifact was not overwritten.",
        "- PhraseBank-only evaluation does not establish Malaysian Bursa news generalisation.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Benchmark recommendation: {recommendation} (manual approval required)")
    print(f"Comparison written to {COMPARISON_FILE}")
    return recommendation_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FinBERT against the StockSense TF-IDF sentiment baseline")
    parser.add_argument("--profile", choices=["full", "quick"], default="full")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--promote", action="store_true", help="Promote FinBERT only when its frozen gates pass")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        write_existing_report()
    else:
        run_finbert_benchmark(args.profile, device=args.device, promote=args.promote)


if __name__ == "__main__":
    main()
