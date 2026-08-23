"""Chronological Bursa-specific FinBERT challenger and guarded promotion.

The challenger starts from the deployed Financial PhraseBank FinBERT artifact,
mixes LLM-assisted-label Bursa training rows with PhraseBank replay examples, tunes
only against 2024 validation evidence, and evaluates once on the frozen 2025
Bursa test. Training never changes the runtime model. Promotion is a separate
hash-verified command that requires explicit approval.
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
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from pipeline.benchmark_finbert_sentiment import (
    LABEL_TO_ID,
    MAX_LENGTH,
    _TokenTensorDataset,
    _bootstrap_macro_f1_delta,
    _evaluate_finbert,
    _require_transformer_dependencies,
    _resolve_device,
    _set_seed,
    _stable_softmax,
    expected_calibration_error,
    fit_temperature,
    metrics_from_probabilities,
)
from pipeline.bursa_news_dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_POLICY_PATH,
    SENTIMENT_LABELS,
    load_policy,
    read_table,
    validate_bursa_news_dataset,
)
from pipeline.finbert_sentiment_model import FINBERT_MODEL_DIR, load_finbert_sentiment_model
from pipeline.runtime_provenance import (
    SENTIMENT_RUNTIME_CONFIG_FILE,
    sentiment_artifact_hashes,
    sha256_file,
)
from pipeline.sentiment_text import (
    SUPPORTED_TEXT_MODES,
    TEXT_MODE_DEFAULT,
    apply_sentiment_text_mode,
)
from pipeline.train_financial_news_sentiment_model import (
    RANDOM_STATE,
    load_sentiment_training_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "bursa-finbert-benchmark"
STAGING_ROOT = PROJECT_ROOT / "models" / "experimental"
SEEDS = (42, 52, 62)
PRIMARY_SEED = 42
DEFAULT_BURSA_REPLAY_SHARE = 0.70
DEFAULT_LEARNING_RATE = 2e-5
FULL_PROFILE_EPOCHS = 4
QUICK_PROFILE_EPOCHS = 1
SAMPLING_MODE_SHUFFLE = "shuffle"
SAMPLING_MODE_TICKER_CLASS_BALANCED = "ticker_class_balanced"
SAMPLING_MODE_TICKER_CLASS_NEUTRAL_BALANCED = "ticker_class_neutral_balanced"
SUPPORTED_SAMPLING_MODES = (
    SAMPLING_MODE_SHUFFLE,
    SAMPLING_MODE_TICKER_CLASS_BALANCED,
    SAMPLING_MODE_TICKER_CLASS_NEUTRAL_BALANCED,
)
LOSS_MODE_INVERSE_FREQUENCY_CE = "inverse_frequency_ce"
LOSS_MODE_EFFECTIVE_NUMBER_FOCAL = "effective_number_focal"
LOSS_MODE_LOGIT_ADJUSTED_CE = "logit_adjusted_ce"
SUPPORTED_LOSS_MODES = (
    LOSS_MODE_INVERSE_FREQUENCY_CE,
    LOSS_MODE_EFFECTIVE_NUMBER_FOCAL,
    LOSS_MODE_LOGIT_ADJUSTED_CE,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _hash_directory(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)).replace("\\", "/"): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _runtime_temperature() -> float:
    try:
        payload = json.loads(SENTIMENT_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
        return float(payload.get("temperature", 1.0))
    except (OSError, ValueError, TypeError):
        return 1.0


def _load_bursa_splits(
    dataset_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
    text_mode: str = TEXT_MODE_DEFAULT,
) -> dict[str, pd.DataFrame]:
    frame = read_table(dataset_path)
    policy = load_policy(policy_path)
    readiness = validate_bursa_news_dataset(
        frame,
        policy=policy,
        dataset_path=str(dataset_path),
    )
    if not readiness.ready:
        raise ValueError(
            "Bursa FinBERT dataset is not ready: " + readiness.reason
        )
    frame = frame.copy()
    frame["split"] = frame["split"].astype(str).str.casefold().str.strip()
    frame["sentiment_label"] = (
        frame["sentiment_label"].astype(str).str.casefold().str.strip()
    )
    frame["entity_relevant"] = frame["entity_relevant"].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().casefold() in {"1", "true", "yes", "y"}
    )
    frame = frame.loc[
        frame["entity_relevant"]
        & frame["sentiment_label"].isin(SENTIMENT_LABELS)
        & frame["model_text"].astype(str).str.strip().ne("")
    ].copy()
    frame = apply_sentiment_text_mode(frame, mode=text_mode)
    frame = frame.loc[frame["model_text"].astype(str).str.strip().ne("")].copy()
    return {
        split: frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        for split in ("training", "validation", "test")
    }


def _load_phrasebank_replay() -> tuple[pd.DataFrame, pd.DataFrame]:
    phrasebank = load_sentiment_training_data().rename(
        columns={"Sentence": "model_text", "Sentiment": "sentiment_label"}
    )
    development, test = train_test_split(
        phrasebank,
        test_size=0.20,
        stratify=phrasebank["sentiment_label"],
        random_state=RANDOM_STATE,
    )
    return development.reset_index(drop=True), test.reset_index(drop=True)


def _mixed_training_frame(
    bursa_training: pd.DataFrame,
    phrasebank_development: pd.DataFrame,
    *,
    seed: int,
    bursa_replay_share: float = DEFAULT_BURSA_REPLAY_SHARE,
    text_mode: str = TEXT_MODE_DEFAULT,
) -> pd.DataFrame:
    if not 0.0 < float(bursa_replay_share) <= 1.0:
        raise ValueError("bursa_replay_share must be in (0, 1]")
    replay_rows = int(
        round(
            len(bursa_training)
            * (1.0 - float(bursa_replay_share))
            / float(bursa_replay_share)
        )
    )
    if text_mode not in SUPPORTED_TEXT_MODES:
        raise ValueError(f"Unsupported sentiment text mode: {text_mode!r}")
    replay = phrasebank_development.sample(
        n=replay_rows,
        replace=replay_rows > len(phrasebank_development),
        random_state=seed,
    )[["model_text", "sentiment_label"]].assign(
        domain="phrasebank",
        # PhraseBank has no selected Bursa entity. Leaving these empty keeps
        # replay text identical to frozen PhraseBank evaluation text even when
        # a Bursa challenger uses target-aware article representation.
        ticker="",
        company_name="",
        title="",
        description="",
    )
    bursa = bursa_training.copy().assign(domain="bursa")
    for column in ("ticker", "company_name", "title", "description"):
        if column not in bursa:
            bursa[column] = ""
    mixed = (
        pd.concat([bursa, replay], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return apply_sentiment_text_mode(mixed, mode=text_mode)


def _class_weights(labels: np.ndarray, torch: Any, device: Any) -> Any:
    counts = np.bincount(labels, minlength=3).astype(float)
    weights = len(labels) / np.maximum(counts, 1.0) / len(counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _effective_number_class_weights(
    labels: np.ndarray,
    torch: Any,
    device: Any,
    *,
    beta: float = 0.999,
) -> Any:
    """Class-balanced weights from the effective-number formulation.

    This is deliberately an experimental challenger to the existing inverse
    frequency weighting.  It is selected only on chronological development
    folds, never on a sealed acceptance holdout.
    """

    counts = np.bincount(labels, minlength=3).astype(float)
    effective = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), 1e-12)
    effective = effective / np.maximum(effective.mean(), 1e-12)
    return torch.tensor(effective, dtype=torch.float32, device=device)


def _ticker_class_sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    """Balance Bursa ticker/class cells while preserving domain replay share."""

    labels = frame["sentiment_label"].astype(str).str.casefold()
    domains = frame.get("domain", pd.Series("bursa", index=frame.index)).astype(str)
    tickers = frame.get("ticker", pd.Series("", index=frame.index)).astype(str)
    cells = domains + "|" + tickers + "|" + labels
    domain_share = domains.value_counts(normalize=True).to_dict()
    cells_per_domain = cells.groupby(domains).nunique().to_dict()
    cell_counts = cells.value_counts().to_dict()
    values = []
    for domain, cell in zip(domains, cells):
        values.append(
            float(domain_share[str(domain)])
            / max(float(cells_per_domain[str(domain)]), 1.0)
            / max(float(cell_counts[str(cell)]), 1.0)
        )
    return np.asarray(values, dtype=float)


def _ticker_class_neutral_sampling_weights(
    frame: pd.DataFrame,
    *,
    neutral_multiplier: float = 1.35,
) -> np.ndarray:
    """Balance ticker/class cells while giving Bursa neutral examples a small lift.

    The multiplier is deliberately modest.  It compensates for the observed
    neutral shortage without manufacturing labels or allowing the replay
    domain to dominate.  PhraseBank rows retain ordinary ticker/class balance.
    """

    if float(neutral_multiplier) <= 0.0:
        raise ValueError("neutral_multiplier must be positive")
    values = _ticker_class_sampling_weights(frame)
    labels = frame["sentiment_label"].astype(str).str.casefold().to_numpy()
    domains = frame.get("domain", pd.Series("bursa", index=frame.index)).astype(str).to_numpy()
    values = values.copy()
    values[(labels == "neutral") & (domains == "bursa")] *= float(neutral_multiplier)
    return values


def _train_weighted_epoch(
    model: Any,
    loader: Any,
    *,
    torch: Any,
    device: Any,
    class_weights: Any,
    optimizer: Any,
    amp_enabled: bool,
    loss_mode: str = LOSS_MODE_INVERSE_FREQUENCY_CE,
    logit_adjustment: Any | None = None,
) -> None:
    if loss_mode not in SUPPORTED_LOSS_MODES:
        raise ValueError(f"Unsupported loss mode: {loss_mode!r}")
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    accumulation = 8
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        context = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if amp_enabled
            else __import__("contextlib").nullcontext()
        )
        with context:
            logits = model(**inputs).logits
            if loss_mode == LOSS_MODE_INVERSE_FREQUENCY_CE:
                loss = torch.nn.functional.cross_entropy(
                    logits, labels, weight=class_weights
                )
            elif loss_mode == LOSS_MODE_LOGIT_ADJUSTED_CE:
                adjusted_logits = logits + logit_adjustment
                loss = torch.nn.functional.cross_entropy(adjusted_logits, labels)
            else:
                per_row = torch.nn.functional.cross_entropy(
                    logits, labels, weight=class_weights, reduction="none"
                )
                # Gamma=2 focuses the challenger on minority mistakes without
                # changing any labels or final-holdout decision threshold.
                focal_weight = torch.pow(1.0 - torch.exp(-per_row), 2.0)
                loss = (focal_weight * per_row).mean()
            loss = loss / accumulation
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        is_last = step == len(loader) - 1
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


def _encode_loader(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    torch: Any,
    DataLoader: Any,
    batch_size: int,
    shuffle: bool,
    sampling_mode: str = SAMPLING_MODE_SHUFFLE,
) -> Any:
    if sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode!r}")
    labels = frame["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    encoded = tokenizer(
        frame["model_text"].astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    dataset = _TokenTensorDataset(encoded, labels, torch)
    sampler = None
    if sampling_mode in {
        SAMPLING_MODE_TICKER_CLASS_BALANCED,
        SAMPLING_MODE_TICKER_CLASS_NEUTRAL_BALANCED,
    }:
        weights = (
            _ticker_class_neutral_sampling_weights(frame)
            if sampling_mode == SAMPLING_MODE_TICKER_CLASS_NEUTRAL_BALANCED
            else _ticker_class_sampling_weights(frame)
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=0,
    )


def _train_seed(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    phrasebank_test: pd.DataFrame,
    *,
    seed: int,
    profile: str,
    device_name: str,
    output_dir: Path,
    bursa_replay_share: float,
    learning_rate: float,
    max_epochs: int | None,
    sampling_mode: str = SAMPLING_MODE_SHUFFLE,
    loss_mode: str = LOSS_MODE_INVERSE_FREQUENCY_CE,
) -> dict[str, Any]:
    torch, DataLoader, _, model_classes = _require_transformer_dependencies()
    model_class, tokenizer_class = model_classes
    _set_seed(seed, torch)
    device = _resolve_device(torch, device_name)
    tokenizer = tokenizer_class.from_pretrained(
        FINBERT_MODEL_DIR, local_files_only=True, use_fast=True
    )
    model = model_class.from_pretrained(
        FINBERT_MODEL_DIR, local_files_only=True
    ).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    batch_size = 4 if profile == "full" else 2
    epochs = (
        int(max_epochs)
        if max_epochs is not None
        else (FULL_PROFILE_EPOCHS if profile == "full" else QUICK_PROFILE_EPOCHS)
    )
    if epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    training_loader = _encode_loader(
        training,
        tokenizer,
        torch=torch,
        DataLoader=DataLoader,
        batch_size=batch_size,
        shuffle=True,
        sampling_mode=sampling_mode,
    )
    validation_loader = _encode_loader(
        validation,
        tokenizer,
        torch=torch,
        DataLoader=DataLoader,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    test_loader = _encode_loader(
        test,
        tokenizer,
        torch=torch,
        DataLoader=DataLoader,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    phrasebank_loader = _encode_loader(
        phrasebank_test,
        tokenizer,
        torch=torch,
        DataLoader=DataLoader,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    labels = training["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    weights = (
        _class_weights(labels, torch, device)
        if loss_mode in {LOSS_MODE_INVERSE_FREQUENCY_CE, LOSS_MODE_LOGIT_ADJUSTED_CE}
        else _effective_number_class_weights(labels, torch, device)
    )
    # Balanced-softmax style logit adjustment.  It changes the training loss
    # only; probabilities are still calibrated on the chronological
    # validation partition.  The small fixed tau prevents an aggressive
    # minority-class prior from overwhelming company-specific evidence.
    class_prior = np.bincount(labels, minlength=3).astype(float)
    class_prior = class_prior / max(float(class_prior.sum()), 1.0)
    logit_adjustment = torch.tensor(
        0.5 * np.log(np.maximum(class_prior, 1e-12)),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=0.01
    )
    amp_enabled = device.type == "cuda"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -np.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        _train_weighted_epoch(
            model,
            training_loader,
            torch=torch,
            device=device,
            class_weights=weights,
            optimizer=optimizer,
            amp_enabled=amp_enabled,
            loss_mode=loss_mode,
            logit_adjustment=logit_adjustment,
        )
        validation_logits, validation_labels = _evaluate_finbert(
            model, validation_loader, torch, device, amp_enabled
        )
        score = float(
            f1_score(
                validation_labels,
                validation_logits.argmax(axis=1),
                average="macro",
                zero_division=0,
            )
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            stale = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            stale += 1
            if stale >= 1:
                break
    model = model_class.from_pretrained(output_dir, local_files_only=True).to(device)
    validation_logits, validation_labels = _evaluate_finbert(
        model, validation_loader, torch, device, amp_enabled
    )
    test_logits, test_labels = _evaluate_finbert(
        model, test_loader, torch, device, amp_enabled
    )
    phrasebank_logits, phrasebank_labels = _evaluate_finbert(
        model, phrasebank_loader, torch, device, amp_enabled
    )
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_probabilities = _stable_softmax(validation_logits, temperature)
    test_probabilities = _stable_softmax(test_logits, temperature)
    phrasebank_probabilities = _stable_softmax(phrasebank_logits, temperature)
    metadata = {
        "schema_version": "bursa-finbert-staged-v1",
        "base_model": str(FINBERT_MODEL_DIR.relative_to(PROJECT_ROOT)),
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "temperature": temperature,
        "training_rows": len(training),
        "bursa_replay_share": float(bursa_replay_share),
        "learning_rate": float(learning_rate),
        "max_epochs": epochs,
        "sampling_mode": sampling_mode,
        "loss_mode": loss_mode,
        "device": str(device),
        "max_length": MAX_LENGTH,
        "class_order": list(SENTIMENT_LABELS),
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    result = {
        **metadata,
        "validation_probabilities": validation_probabilities,
        "validation_labels": validation_labels,
        "test_probabilities": test_probabilities,
        "test_labels": test_labels,
        "phrasebank_probabilities": phrasebank_probabilities,
        "phrasebank_labels": phrasebank_labels,
    }
    del model, training_loader, validation_loader, test_loader, phrasebank_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _production_probabilities(texts: Sequence[str]) -> np.ndarray:
    model = load_finbert_sentiment_model(
        str(FINBERT_MODEL_DIR), temperature=_runtime_temperature()
    )
    return model.predict_proba(list(texts))


def run_bursa_finbert_benchmark(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    profile: str = "full",
    device: str = "cuda",
    policy_path: Path = DEFAULT_POLICY_PATH,
    bursa_replay_share: float = DEFAULT_BURSA_REPLAY_SHARE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int | None = None,
    seeds: Sequence[int] = SEEDS,
    primary_seed: int = PRIMARY_SEED,
    text_mode: str = TEXT_MODE_DEFAULT,
    sampling_mode: str = SAMPLING_MODE_SHUFFLE,
    loss_mode: str = LOSS_MODE_INVERSE_FREQUENCY_CE,
) -> dict[str, Any]:
    if profile not in {"full", "quick"}:
        raise ValueError("profile must be full or quick")
    if not 0.0 < float(bursa_replay_share) <= 1.0:
        raise ValueError("bursa_replay_share must be in (0, 1]")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if text_mode not in SUPPORTED_TEXT_MODES:
        raise ValueError(f"Unsupported sentiment text mode: {text_mode!r}")
    if sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode!r}")
    if loss_mode not in SUPPORTED_LOSS_MODES:
        raise ValueError(f"Unsupported loss mode: {loss_mode!r}")
    selected_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not selected_seeds:
        raise ValueError("at least one seed is required")
    if int(primary_seed) not in selected_seeds:
        raise ValueError("primary_seed must be one of seeds")
    dataset_policy = load_policy(Path(policy_path))
    experimental_only = bool(dataset_policy.get("experimental_only", False))
    splits = _load_bursa_splits(
        Path(dataset_path), policy_path=Path(policy_path), text_mode=text_mode
    )
    phrasebank_development, phrasebank_test = _load_phrasebank_replay()
    phrasebank_development = apply_sentiment_text_mode(
        phrasebank_development, mode=text_mode
    )
    phrasebank_test = apply_sentiment_text_mode(phrasebank_test, mode=text_mode)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    staging = STAGING_ROOT / f"bursa-finbert-staging-{run_id}"
    test_labels = splits["test"]["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    phrasebank_labels = (
        phrasebank_test["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    )
    production_bursa_probabilities = _production_probabilities(
        splits["test"]["model_text"].tolist()
    )
    production_phrasebank_probabilities = _production_probabilities(
        phrasebank_test["model_text"].tolist()
    )
    production_bursa = metrics_from_probabilities(
        "production_phrasebank_finbert",
        production_bursa_probabilities,
        test_labels,
    )
    production_phrasebank = metrics_from_probabilities(
        "production_phrasebank_finbert",
        production_phrasebank_probabilities,
        phrasebank_labels,
    )
    seed_results: list[dict[str, Any]] = []
    for seed in selected_seeds:
        mixed = _mixed_training_frame(
            splits["training"],
            phrasebank_development,
            seed=seed,
            bursa_replay_share=float(bursa_replay_share),
            text_mode=text_mode,
        )
        print(f"Bursa FinBERT seed {seed}...")
        result = _train_seed(
            mixed,
            splits["validation"],
            splits["test"],
            phrasebank_test,
            seed=seed,
            profile=profile,
            device_name=device,
            output_dir=staging / f"seed-{seed}",
            bursa_replay_share=float(bursa_replay_share),
            learning_rate=float(learning_rate),
            max_epochs=max_epochs,
            sampling_mode=sampling_mode,
            loss_mode=loss_mode,
        )
        result["bursa_validation_metrics"] = metrics_from_probabilities(
            f"bursa_finbert_seed_{seed}_validation",
            result["validation_probabilities"],
            result["validation_labels"],
        )
        result["bursa_metrics"] = metrics_from_probabilities(
            f"bursa_finbert_seed_{seed}",
            result["test_probabilities"],
            result["test_labels"],
        )
        result["phrasebank_metrics"] = metrics_from_probabilities(
            f"bursa_finbert_seed_{seed}",
            result["phrasebank_probabilities"],
            result["phrasebank_labels"],
        )
        seed_results.append(result)
    primary = next(item for item in seed_results if item["seed"] == int(primary_seed))
    primary_predictions = primary["test_probabilities"].argmax(axis=1)
    production_predictions = production_bursa_probabilities.argmax(axis=1)
    bootstrap = _bootstrap_macro_f1_delta(
        primary_predictions, production_predictions, test_labels, 2000
    )
    seed_macro_f1 = [
        float(item["bursa_metrics"]["macro_f1"]) for item in seed_results
    ]
    gates = dataset_policy["sentiment_promotion_gates"]
    primary_bursa = primary["bursa_metrics"]
    primary_validation = primary["bursa_validation_metrics"]
    primary_phrasebank = primary["phrasebank_metrics"]
    uplift = float(primary_bursa["macro_f1"] - production_bursa["macro_f1"])
    phrasebank_degradation = float(
        production_phrasebank["macro_f1"] - primary_phrasebank["macro_f1"]
    )
    seed_std = float(np.std(seed_macro_f1))
    checks = {
        "production_label_policy": not experimental_only
        and bool(dataset_policy.get("production_promotion_allowed", True)),
        "macro_f1": primary_bursa["macro_f1"]
        >= float(gates["minimum_macro_f1"]),
        "balanced_accuracy": primary_bursa["balanced_accuracy"]
        >= float(gates["minimum_balanced_accuracy"]),
        "calibration": expected_calibration_error(
            primary["test_probabilities"], test_labels
        )
        <= float(gates["maximum_ece"]),
        "uplift": uplift >= float(gates["minimum_macro_f1_uplift"]),
        "bootstrap": bootstrap["ci95_lower"]
        >= float(gates["minimum_bootstrap_ci_lower"]),
        "phrasebank_non_inferiority": phrasebank_degradation
        <= float(gates["maximum_phrasebank_macro_f1_degradation"]),
        "seed_stability": seed_std
        <= float(gates["maximum_seed_macro_f1_std"]),
    }
    primary_dir = staging / f"seed-{PRIMARY_SEED}"
    report = {
        "schema_version": "bursa-finbert-benchmark-v2",
        "run_id": run_id,
        "profile": profile,
        "device_requested": device,
        "dataset_path": str(Path(dataset_path)),
        "dataset_policy_path": str(Path(policy_path)),
        "dataset_policy_version": str(dataset_policy.get("policy_version", "unknown")),
        "experimental_only": experimental_only,
        "splits": {key: len(value) for key, value in splits.items()},
        "production_bursa_metrics": production_bursa,
        "production_phrasebank_metrics": production_phrasebank,
        "primary_seed": int(primary_seed),
        "seeds": list(selected_seeds),
        "training_recipe": {
            "bursa_replay_share": float(bursa_replay_share),
            "phrasebank_replay_share": float(1.0 - float(bursa_replay_share)),
            "learning_rate": float(learning_rate),
            "text_mode": text_mode,
            "sampling_mode": sampling_mode,
            "loss_mode": loss_mode,
            "max_epochs": int(max_epochs)
            if max_epochs is not None
            else (FULL_PROFILE_EPOCHS if profile == "full" else QUICK_PROFILE_EPOCHS),
            "selection_rule": "fixed primary seed; best epoch selected on chronological validation macro-F1",
            "test_use": "frozen confirmatory report; do not select training settings from this split",
        },
        "primary_validation_metrics": primary_validation,
        "primary_bursa_metrics": primary_bursa,
        "primary_phrasebank_metrics": primary_phrasebank,
        "bursa_macro_f1_uplift": uplift,
        "phrasebank_macro_f1_degradation": phrasebank_degradation,
        "bootstrap_macro_f1_delta": bootstrap,
        "seed_macro_f1": seed_macro_f1,
        "seed_macro_f1_std": seed_std,
        "checks": checks,
        "eligible_for_promotion": all(checks.values()),
        "staged_model_directory": str(primary_dir.relative_to(PROJECT_ROOT)),
        "staged_artifact_hashes": _hash_directory(primary_dir),
        "deployment_changed": False,
    }
    for item in seed_results:
        for field in (
            "validation_probabilities",
            "validation_labels",
            "test_probabilities",
            "test_labels",
            "phrasebank_probabilities",
            "phrasebank_labels",
        ):
            item.pop(field, None)
    report["seed_results"] = seed_results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json(RESULTS_DIR / f"{run_id}.json", report)
    _atomic_json(RESULTS_DIR / "latest.json", report)
    print(json.dumps(report, indent=2, default=str))
    return report


def promote_bursa_finbert(run_id: str, *, approve: bool = False) -> dict[str, Any]:
    if not approve:
        raise ValueError("Promotion requires --approve after manual review")
    report_path = RESULTS_DIR / f"{run_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("eligible_for_promotion"):
        raise ValueError("Staged Bursa FinBERT did not pass every promotion gate")
    staged = PROJECT_ROOT / str(report["staged_model_directory"])
    expected = report.get("staged_artifact_hashes", {})
    actual = _hash_directory(staged)
    if actual != expected:
        raise ValueError("Staged Bursa FinBERT artifact failed its integrity check")
    backup = STAGING_ROOT / f"finbert-sentiment-backup-before-bursa-{run_id}"
    candidate = STAGING_ROOT / f"finbert-sentiment-promote-{run_id}"
    if backup.exists() or candidate.exists():
        raise FileExistsError(
            "Promotion workspace already exists; review it before retrying"
        )
    shutil.copytree(staged, candidate)
    moved_production = False
    try:
        if FINBERT_MODEL_DIR.exists():
            FINBERT_MODEL_DIR.replace(backup)
            moved_production = True
        candidate.replace(FINBERT_MODEL_DIR)
        runtime = (
            json.loads(SENTIMENT_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
            if SENTIMENT_RUNTIME_CONFIG_FILE.exists()
            else {}
        )
        runtime.update(
            {
                "backend": "finbert",
                "model_directory": str(
                    FINBERT_MODEL_DIR.relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
                "training_run_id": run_id,
                "evaluation_scope": "2019-2025 LLM-assisted-label Bursa news plus unchanged Financial PhraseBank test",
                "bursa_generalisation_validated": True,
                "temperature": float(
                    json.loads(
                        (FINBERT_MODEL_DIR / "metadata.json").read_text(
                            encoding="utf-8"
                        )
                    )["temperature"]
                ),
                "artifact_hashes": sentiment_artifact_hashes(),
            }
        )
        _atomic_json(SENTIMENT_RUNTIME_CONFIG_FILE, runtime)
    except Exception:
        if FINBERT_MODEL_DIR.exists():
            failed = STAGING_ROOT / f"finbert-sentiment-failed-promotion-{run_id}"
            FINBERT_MODEL_DIR.replace(failed)
        if moved_production and backup.exists():
            backup.replace(FINBERT_MODEL_DIR)
        raise
    result = {
        "promoted": True,
        "run_id": run_id,
        "model_directory": str(FINBERT_MODEL_DIR),
        "backup_directory": str(backup),
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train or promote the chronological Bursa FinBERT challenger."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    train.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    train.add_argument("--profile", choices=["full", "quick"], default="full")
    train.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    train.add_argument("--bursa-replay-share", type=float, default=DEFAULT_BURSA_REPLAY_SHARE)
    train.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train.add_argument("--max-epochs", type=int, default=None)
    train.add_argument("--text-mode", choices=SUPPORTED_TEXT_MODES, default=TEXT_MODE_DEFAULT)
    train.add_argument("--sampling-mode", choices=SUPPORTED_SAMPLING_MODES, default=SAMPLING_MODE_SHUFFLE)
    train.add_argument("--loss-mode", choices=SUPPORTED_LOSS_MODES, default=LOSS_MODE_INVERSE_FREQUENCY_CE)
    train.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        help="Repeat to use a fixed, explicitly recorded seed set (default: 42, 52, 62).",
    )
    train.add_argument("--primary-seed", type=int, default=PRIMARY_SEED)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--run-id", required=True)
    promote.add_argument("--approve", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "train":
        run_bursa_finbert_benchmark(
            Path(args.dataset),
            profile=args.profile,
            device=args.device,
            policy_path=Path(args.policy),
            bursa_replay_share=args.bursa_replay_share,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            text_mode=args.text_mode,
            sampling_mode=args.sampling_mode,
            loss_mode=args.loss_mode,
            seeds=tuple(args.seeds) if args.seeds else SEEDS,
            primary_seed=args.primary_seed,
        )
        return 0
    if args.command == "promote":
        promote_bursa_finbert(args.run_id, approve=args.approve)
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
