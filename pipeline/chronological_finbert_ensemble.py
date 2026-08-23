"""Leakage-safe, 2015--2025 retrospective FinBERT ensemble benchmark.

This module intentionally keeps the production sentiment configuration untouched.
It trains three Bursa-domain checkpoints for each expanding chronological fold,
averages their calibrated probabilities, and learns a conservative probability
blend against the frozen PhraseBank model.  Since every available year ends in
2025, the resulting evidence is explicitly retrospective rather than an
independent production-promotion test.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from pipeline.benchmark_bursa_finbert import (
    DEFAULT_BURSA_REPLAY_SHARE,
    DEFAULT_LEARNING_RATE,
    FULL_PROFILE_EPOCHS,
    LOSS_MODE_INVERSE_FREQUENCY_CE,
    QUICK_PROFILE_EPOCHS,
    SAMPLING_MODE_SHUFFLE,
    SUPPORTED_LOSS_MODES,
    SUPPORTED_SAMPLING_MODES,
    _load_phrasebank_replay,
    _mixed_training_frame,
    _train_seed,
)
from pipeline.sentiment_text import (
    SUPPORTED_TEXT_MODES,
    TEXT_MODE_DEFAULT,
    apply_sentiment_text_mode,
)
from pipeline.benchmark_finbert_sentiment import (
    LABEL_TO_ID,
    _stable_softmax,
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
from pipeline.finbert_sentiment_model import (
    clear_finbert_runtime_cache,
    load_finbert_sentiment_model,
)
from pipeline.runtime_provenance import sentiment_artifact_hashes_for_directory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "finbert-ensemble-chronological"
STAGING_ROOT = PROJECT_ROOT / "models" / "experimental" / "finbert-ensemble-chronological"
DEFAULT_PHRASEBANK_MODEL = PROJECT_ROOT / "models" / "experimental" / "finbert-sentiment"
DEFAULT_RUN_ID = "bursa-phrasebank-2015-2025-v1"
DEFAULT_SEEDS = (42, 52, 62)
CALIBRATION_MODE_NONE = "none"
CALIBRATION_MODE_SCALAR_TEMPERATURE = "scalar_temperature"
SUPPORTED_CALIBRATION_MODES = (
    CALIBRATION_MODE_NONE,
    CALIBRATION_MODE_SCALAR_TEMPERATURE,
)
DATE_START = pd.Timestamp("2015-01-01", tz="UTC")
DATE_END_EXCLUSIVE = pd.Timestamp("2026-01-01", tz="UTC")


@dataclass(frozen=True)
class ChronologicalFold:
    """One expanding, forward-only model-selection evaluation fold."""

    name: str
    training_end_year: int
    calibration_year: int
    evaluation_year: int


DEFAULT_FOLDS = (
    ChronologicalFold("fold-1", 2019, 2020, 2021),
    ChronologicalFold("fold-2", 2020, 2021, 2022),
    ChronologicalFold("fold-3", 2021, 2022, 2023),
    ChronologicalFold("fold-4", 2022, 2023, 2024),
    ChronologicalFold("fold-5", 2023, 2024, 2025),
)


class ChronologicalEnsembleError(RuntimeError):
    """Raised when the retrospective experiment cannot preserve its protocol."""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    os.replace(temporary, path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _prepare_dataset(dataset_path: Path, policy_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only eligible 2015--2025, entity-relevant English labelled news."""

    policy = load_policy(policy_path)
    raw = read_table(dataset_path)
    readiness = validate_bursa_news_dataset(raw, policy=policy, dataset_path=str(dataset_path))
    if not readiness.ready:
        raise ChronologicalEnsembleError(
            "Bursa news dataset is not ready for chronological benchmarking: "
            + readiness.reason
        )

    frame = raw.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame["sentiment_label"] = frame["sentiment_label"].astype(str).str.strip().str.casefold()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["entity_relevant"] = frame["entity_relevant"].map(_as_bool)
    frame["language"] = frame["language"].astype(str).str.strip().str.casefold()
    frame = frame.loc[
        frame["published_at"].notna()
        & frame["published_at"].ge(DATE_START)
        & frame["published_at"].lt(DATE_END_EXCLUSIVE)
        & frame["entity_relevant"]
        & frame["language"].eq("en")
        & frame["sentiment_label"].isin(SENTIMENT_LABELS)
        & frame["model_text"].astype(str).str.strip().ne("")
    ].copy()
    if frame.empty:
        raise ChronologicalEnsembleError("No eligible 2015--2025 labelled Bursa news rows remain")

    frame["event_id"] = frame.get("event_id", "").fillna("").astype(str).str.strip()
    fallback = (
        frame["ticker"].astype(str)
        + "|content:"
        + frame.get("content_sha256", pd.Series("", index=frame.index)).fillna("").astype(str)
    )
    frame["_event_group"] = np.where(frame["event_id"].ne(""), frame["ticker"] + "|event:" + frame["event_id"], fallback)
    frame["_content_group"] = (
        frame["ticker"].astype(str)
        + "|content:"
        + frame.get("content_sha256", pd.Series("", index=frame.index)).fillna("").astype(str)
    )
    frame = frame.sort_values(["published_at", "article_id"], kind="stable")
    before_deduplication = len(frame)
    # Keep the latest observation of duplicate content.  That cannot leak the
    # same recovered article backwards across a future evaluation boundary.
    frame = frame.drop_duplicates("_content_group", keep="last").reset_index(drop=True)
    diagnostics = {
        "eligible_rows": int(len(frame)),
        "dropped_duplicate_content_rows": int(before_deduplication - len(frame)),
        "year_counts": {
            str(year): int(count)
            for year, count in frame.groupby(frame["published_at"].dt.year).size().items()
        },
        "class_counts": {str(label): int(count) for label, count in frame["sentiment_label"].value_counts().items()},
    }
    return frame, diagnostics


def _drop_cross_partition_events(
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Drop old copies of an event, keeping it only in its latest partition."""

    pieces = []
    for name, frame in (("training", training), ("calibration", calibration), ("evaluation", evaluation)):
        item = frame.copy()
        item["_partition"] = name
        pieces.append(item)
    combined = pd.concat(pieces, ignore_index=True)
    rank = {"training": 0, "calibration": 1, "evaluation": 2}
    latest_partition = combined.groupby("_event_group")["_partition"].transform(
        lambda values: max(values, key=lambda value: rank[value])
    )
    retained = combined.loc[combined["_partition"].eq(latest_partition)].copy()
    dropped = int(len(combined) - len(retained))
    outputs = []
    for name in ("training", "calibration", "evaluation"):
        outputs.append(
            retained.loc[retained["_partition"].eq(name)].drop(columns="_partition").reset_index(drop=True)
        )
    return outputs[0], outputs[1], outputs[2], dropped


def build_chronological_fold(
    frame: pd.DataFrame,
    fold: ChronologicalFold,
    *,
    embargo_days: int = 7,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build a date-only split with an event purge and seven-day embargo."""

    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    calibration_start = pd.Timestamp(f"{fold.calibration_year}-01-01", tz="UTC")
    evaluation_start = pd.Timestamp(f"{fold.evaluation_year}-01-01", tz="UTC")
    evaluation_end = pd.Timestamp(f"{fold.evaluation_year + 1}-01-01", tz="UTC")
    embargo = pd.Timedelta(days=int(embargo_days))
    training = frame.loc[frame["published_at"].lt(calibration_start - embargo)].copy()
    calibration = frame.loc[
        frame["published_at"].ge(calibration_start + embargo)
        & frame["published_at"].lt(evaluation_start - embargo)
    ].copy()
    evaluation = frame.loc[
        frame["published_at"].ge(evaluation_start + embargo)
        & frame["published_at"].lt(evaluation_end)
    ].copy()
    training, calibration, evaluation, purged_events = _drop_cross_partition_events(
        training, calibration, evaluation
    )
    partitions = {"training": training, "calibration": calibration, "evaluation": evaluation}
    for name, partition in partitions.items():
        missing = set(SENTIMENT_LABELS) - set(partition["sentiment_label"])
        if partition.empty or missing:
            raise ChronologicalEnsembleError(
                f"{fold.name} {name} partition cannot support a three-class benchmark; "
                f"rows={len(partition)}, missing_labels={sorted(missing)}"
            )
    return partitions, {
        "fold": fold.name,
        "training_end_year": fold.training_end_year,
        "calibration_year": fold.calibration_year,
        "evaluation_year": fold.evaluation_year,
        "embargo_days": int(embargo_days),
        "event_rows_purged": purged_events,
        "partition_rows": {name: int(len(partition)) for name, partition in partitions.items()},
        "partition_years": {
            name: sorted({int(value) for value in partition["published_at"].dt.year})
            for name, partition in partitions.items()
        },
    }


def _probabilities(model: object, frame: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict_proba(frame["model_text"].astype(str).tolist()), dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ChronologicalEnsembleError("A component returned incompatible class probabilities")
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def _scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale an already-softmaxed probability vector safely."""

    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0))
    return _stable_softmax(logits, float(temperature))


def _blend(bursa: np.ndarray, phrasebank: np.ndarray, bursa_weight: float) -> np.ndarray:
    if not 0.0 <= float(bursa_weight) <= 1.0:
        raise ValueError("bursa_weight must be between 0 and 1")
    values = float(bursa_weight) * bursa + (1.0 - float(bursa_weight)) * phrasebank
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def _block_ids(frame: pd.DataFrame) -> np.ndarray:
    fallback = (
        frame["ticker"].astype(str)
        + "|month:"
        + frame["published_at"].dt.to_period("M").astype(str)
    )
    event_groups = frame.get("_event_group")
    if event_groups is None:
        event_ids = frame.get("event_id", pd.Series("", index=frame.index)).fillna("").astype(str)
        event_groups = pd.Series(
            np.where(
                event_ids.str.strip().ne(""),
                frame["ticker"].astype(str) + "|event:" + event_ids,
                fallback,
            ),
            index=frame.index,
        )
    return np.where(
        event_groups.astype(str).str.strip().ne(""), event_groups.astype(str), fallback
    )


def _group_positions(block_ids: Sequence[str]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, block_id in enumerate(block_ids):
        groups.setdefault(str(block_id), []).append(index)
    return [np.asarray(indices, dtype=int) for _, indices in sorted(groups.items())]


def _bootstrap_log_loss_standard_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    block_ids: Sequence[str],
    *,
    iterations: int,
    random_state: int = 42,
) -> float:
    groups = _group_positions(block_ids)
    if len(groups) < 2 or iterations <= 1:
        return 0.0
    rng = np.random.default_rng(int(random_state))
    values = np.empty(int(iterations), dtype=float)
    for iteration in range(int(iterations)):
        choices = rng.integers(0, len(groups), size=len(groups))
        positions = np.concatenate([groups[index] for index in choices])
        values[iteration] = log_loss(labels[positions], probabilities[positions], labels=[0, 1, 2])
    return float(np.std(values, ddof=1))


def _bootstrap_macro_f1_delta_by_block(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    block_ids: Sequence[str],
    *,
    iterations: int,
    random_state: int = 42,
) -> dict[str, float]:
    groups = _group_positions(block_ids)
    if not groups:
        return {"delta_macro_f1": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}
    rng = np.random.default_rng(int(random_state))
    deltas = np.empty(int(iterations), dtype=float)
    for iteration in range(int(iterations)):
        choices = rng.integers(0, len(groups), size=len(groups))
        positions = np.concatenate([groups[index] for index in choices])
        deltas[iteration] = f1_score(
            labels[positions], candidate[positions], average="macro", zero_division=0
        ) - f1_score(labels[positions], baseline[positions], average="macro", zero_division=0)
    return {
        "delta_macro_f1": float(deltas.mean()),
        "ci95_lower": float(np.percentile(deltas, 2.5)),
        "ci95_upper": float(np.percentile(deltas, 97.5)),
    }


def _weight_values(start: float, end: float, step: float) -> list[float]:
    count = int(round((float(end) - float(start)) / float(step)))
    return [round(float(start) + index * float(step), 6) for index in range(count + 1)]


def select_stable_weight(
    bursa_probabilities: np.ndarray,
    phrasebank_probabilities: np.ndarray,
    labels: np.ndarray,
    block_ids: Sequence[str],
    *,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    """Select a rounded blend from a one-standard-error log-loss plateau."""

    def score(weights: Iterable[float]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for weight in weights:
            probabilities = _blend(bursa_probabilities, phrasebank_probabilities, weight)
            metrics = metrics_from_probabilities("ensemble", probabilities, labels)
            rows.append(
                {
                    "bursa_weight": float(weight),
                    "phrasebank_weight": float(1.0 - weight),
                    **metrics,
                }
            )
        return rows

    def rank(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["log_loss"]),
            float(row["brier_score"]),
            float(row["ece"]),
            abs(float(row["bursa_weight"]) - 0.50),
        )

    coarse_grid = score(_weight_values(0.0, 1.0, 0.05))
    coarse_best = min(coarse_grid, key=rank)
    lower = max(0.0, float(coarse_best["bursa_weight"]) - 0.05)
    upper = min(1.0, float(coarse_best["bursa_weight"]) + 0.05)
    fine_grid = score(_weight_values(lower, upper, 0.01))
    fine_best = min(fine_grid, key=rank)
    best_probability = _blend(
        bursa_probabilities, phrasebank_probabilities, float(fine_best["bursa_weight"])
    )
    standard_error = _bootstrap_log_loss_standard_error(
        best_probability,
        labels,
        block_ids,
        iterations=int(bootstrap_iterations),
    )
    plateau_cutoff = float(fine_best["log_loss"]) + float(standard_error)
    plateau = [row for row in fine_grid if float(row["log_loss"]) <= plateau_cutoff + 1e-12]
    stable_coarse = [row for row in coarse_grid if float(row["log_loss"]) <= plateau_cutoff + 1e-12]
    if not stable_coarse:
        stable_coarse = [coarse_best]
    selected = min(
        stable_coarse,
        key=lambda row: (abs(float(row["bursa_weight"]) - 0.50), float(row["log_loss"])),
    )
    return {
        "selection_objective": "multiclass_log_loss",
        "secondary_metrics": ["brier_score", "ece"],
        "coarse_grid": coarse_grid,
        "fine_grid": fine_grid,
        "unrounded_best_weight": float(fine_best["bursa_weight"]),
        "one_standard_error": float(standard_error),
        "one_standard_error_log_loss_cutoff": float(plateau_cutoff),
        "fine_plateau_weights": [float(row["bursa_weight"]) for row in plateau],
        "coarse_plateau_weights": [float(row["bursa_weight"]) for row in stable_coarse],
        "selected_bursa_weight": float(selected["bursa_weight"]),
        "selected_phrasebank_weight": float(selected["phrasebank_weight"]),
        "selected_metrics": selected,
    }


def _seed_cache_paths(fold_dir: Path, seed: int) -> tuple[Path, Path]:
    seed_dir = fold_dir / f"seed-{seed}"
    return seed_dir / "chronological-result.npz", seed_dir / "chronological-result.json"


def _read_seed_cache(fold_dir: Path, seed: int, fingerprint: str) -> dict[str, Any] | None:
    data_path, metadata_path = _seed_cache_paths(fold_dir, seed)
    if not data_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            return None
        with np.load(data_path, allow_pickle=False) as values:
            result = {
                "validation_probabilities": values["validation_probabilities"],
                "validation_labels": values["validation_labels"],
                "test_probabilities": values["test_probabilities"],
                "test_labels": values["test_labels"],
                "metadata": metadata.get("metadata", {}),
            }
            # Newer checkpoints retain the frozen PhraseBank probabilities so
            # a final ensemble report can measure general-language retention
            # without retraining or re-running the Bursa seed.  Older cached
            # checkpoints remain readable and simply omit these optional keys.
            if "phrasebank_probabilities" in values.files:
                result["phrasebank_probabilities"] = values["phrasebank_probabilities"]
            if "phrasebank_labels" in values.files:
                result["phrasebank_labels"] = values["phrasebank_labels"]
        model_dir = fold_dir / f"seed-{seed}"
        if not (model_dir / "config.json").exists() or not (model_dir / "model.safetensors").exists():
            return None
        return result
    except (OSError, ValueError, KeyError):
        return None


def _write_seed_cache(fold_dir: Path, seed: int, fingerprint: str, result: Mapping[str, Any]) -> None:
    data_path, metadata_path = _seed_cache_paths(fold_dir, seed)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            validation_probabilities=np.asarray(result["validation_probabilities"], dtype=float),
            validation_labels=np.asarray(result["validation_labels"], dtype=int),
            test_probabilities=np.asarray(result["test_probabilities"], dtype=float),
            test_labels=np.asarray(result["test_labels"], dtype=int),
            **(
                {
                    "phrasebank_probabilities": np.asarray(
                        result["phrasebank_probabilities"], dtype=float
                    ),
                    "phrasebank_labels": np.asarray(
                        result["phrasebank_labels"], dtype=int
                    ),
                }
                if "phrasebank_probabilities" in result
                and "phrasebank_labels" in result
                else {}
            ),
        )
    os.replace(temporary, data_path)
    metadata = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "validation_probabilities",
            "validation_labels",
            "test_probabilities",
            "test_labels",
            "phrasebank_probabilities",
            "phrasebank_labels",
        }
    }
    _atomic_json(metadata_path, {"fingerprint": fingerprint, "metadata": metadata})


def _fold_fingerprint(
    fold: ChronologicalFold,
    partitions: Mapping[str, pd.DataFrame],
    *,
    seeds: Sequence[int],
    replay_share: float,
    learning_rate: float,
    max_epochs: int,
    embargo_days: int,
    text_mode: str,
    sampling_mode: str,
    loss_mode: str,
    calibration_mode: str,
) -> str:
    payload = {
        "fold": fold.__dict__,
        "article_ids": {
            name: partition["article_id"].astype(str).tolist()
            for name, partition in partitions.items()
        },
        "seeds": [int(value) for value in seeds],
        "replay_share": float(replay_share),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "embargo_days": int(embargo_days),
        "text_mode": text_mode,
        "sampling_mode": sampling_mode,
        "loss_mode": loss_mode,
        "calibration_mode": calibration_mode,
    }
    return __import__("hashlib").sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _run_or_resume_seed(
    partitions: Mapping[str, pd.DataFrame],
    *,
    fold_dir: Path,
    fold_fingerprint: str,
    seed: int,
    phrasebank_development: pd.DataFrame,
    phrasebank_test: pd.DataFrame,
    profile: str,
    device: str,
    replay_share: float,
    learning_rate: float,
    max_epochs: int,
    text_mode: str,
    sampling_mode: str,
    loss_mode: str,
) -> dict[str, Any]:
    cached = _read_seed_cache(fold_dir, seed, fold_fingerprint)
    if cached is not None:
        return cached
    mixed = _mixed_training_frame(
        partitions["training"],
        phrasebank_development,
        seed=int(seed),
        bursa_replay_share=float(replay_share),
        text_mode=text_mode,
    )
    result = _train_seed(
        mixed,
        partitions["calibration"],
        partitions["evaluation"],
        phrasebank_test,
        seed=int(seed),
        profile=profile,
        device_name=device,
        output_dir=fold_dir / f"seed-{seed}",
        bursa_replay_share=float(replay_share),
        learning_rate=float(learning_rate),
        max_epochs=int(max_epochs),
        sampling_mode=sampling_mode,
        loss_mode=loss_mode,
    )
    _write_seed_cache(fold_dir, seed, fold_fingerprint, result)
    return result


def _phrasebank_fold_probabilities(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    model_directory: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    # Temperature is re-estimated only from the fold's calibration year.  The
    # model itself remains frozen; log probabilities are sufficient logits up
    # to a row-wise additive constant.
    model = load_finbert_sentiment_model(str(model_directory), temperature=1.0)
    calibration_raw = _probabilities(model, calibration)
    evaluation_raw = _probabilities(model, evaluation)
    labels = calibration["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    temperature = fit_temperature(np.log(np.clip(calibration_raw, 1e-12, 1.0)), labels)
    calibration_scaled = _scale_probabilities(calibration_raw, temperature)
    evaluation_scaled = _scale_probabilities(evaluation_raw, temperature)
    del model
    clear_finbert_runtime_cache()
    gc.collect()
    return calibration_scaled, evaluation_scaled, float(temperature)


def _write_fold_outputs(
    fold_dir: Path,
    frame: pd.DataFrame,
    bursa: np.ndarray,
    phrasebank: np.ndarray,
    summary: Mapping[str, Any],
    ensemble: np.ndarray | None = None,
) -> None:
    output = frame[["article_id", "ticker", "published_at", "event_id", "sentiment_label"]].copy()
    for index, label in enumerate(("negative", "neutral", "positive")):
        output[f"bursa_probability_{label}"] = bursa[:, index]
        output[f"phrasebank_probability_{label}"] = phrasebank[:, index]
        if ensemble is not None:
            output[f"ensemble_probability_{label}"] = ensemble[:, index]
    output.to_csv(fold_dir / "evaluation-oof.csv", index=False)
    _atomic_json(fold_dir / "fold-summary.json", dict(summary))


def _read_fold_outputs(fold_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    output = fold_dir / "evaluation-oof.csv"
    summary = fold_dir / "fold-summary.json"
    if not output.exists() or not summary.exists():
        return None
    try:
        frame = pd.read_csv(output)
        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if frame["published_at"].isna().any():
            return None
        return frame, payload
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _fold_summary_from_oof(frame: pd.DataFrame, summary: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bursa = frame[[f"bursa_probability_{name}" for name in ("negative", "neutral", "positive")]].to_numpy(dtype=float)
    phrasebank = frame[
        [f"phrasebank_probability_{name}" for name in ("negative", "neutral", "positive")]
    ].to_numpy(dtype=float)
    labels = frame["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    if np.isnan(bursa).any() or np.isnan(phrasebank).any() or np.isnan(labels).any():
        raise ChronologicalEnsembleError(f"Invalid cached OOF values for {summary.get('fold', 'unknown fold')}")
    return bursa, phrasebank, labels


def _company_report(
    frame: pd.DataFrame,
    ensemble: np.ndarray,
    phrasebank: np.ndarray,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for ticker, positions in frame.groupby(frame["ticker"].astype(str).str.upper()).groups.items():
        indices = np.asarray(list(positions), dtype=int)
        labels = frame.iloc[indices]["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
        ensemble_metrics = metrics_from_probabilities("ensemble", ensemble[indices], labels)
        phrasebank_metrics = metrics_from_probabilities("phrasebank", phrasebank[indices], labels)
        details[str(ticker)] = {
            "rows": int(len(indices)),
            "ensemble": ensemble_metrics,
            "phrasebank": phrasebank_metrics,
            "macro_f1_delta": float(ensemble_metrics["macro_f1"] - phrasebank_metrics["macro_f1"]),
            "log_loss_delta": float(ensemble_metrics["log_loss"] - phrasebank_metrics["log_loss"]),
        }
    return details


def _fold_weight_coverage(selected_weight: float, fold_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches: list[str] = []
    for summary in fold_summaries:
        plateau = ((summary.get("selection") or {}).get("coarse_plateau_weights") or [])
        if any(abs(float(value) - float(selected_weight)) < 1e-9 for value in plateau):
            matches.append(str(summary.get("fold", "unknown")))
    return {
        "selected_weight": float(selected_weight),
        "matching_folds": matches,
        "matching_fold_count": len(matches),
        "fold_count": len(fold_summaries),
        "proportion": float(len(matches) / len(fold_summaries)) if fold_summaries else 0.0,
    }


def _retrospective_checks(
    policy: Mapping[str, Any],
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, float],
    company: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> dict[str, Any]:
    gates = ((policy.get("sentiment_ensemble") or {}).get("promotion_gates") or {})
    maximum_company_loss = float(gates.get("maximum_company_macro_f1_loss", 0.05))
    checked_companies = [
        item for item in company.values() if int(item.get("rows", 0)) >= int(gates.get("minimum_holdout_rows_per_ticker", 15))
    ]
    return {
        "macro_f1": float(metrics["macro_f1"]) >= float(gates.get("minimum_macro_f1", 0.70)),
        "balanced_accuracy": float(metrics["balanced_accuracy"])
        >= float(gates.get("minimum_balanced_accuracy", 0.70)),
        "calibration": float(metrics["ece"]) <= float(gates.get("maximum_ece", 0.10)),
        "minimum_class_f1": min(
            float(metrics["negative_f1"]), float(metrics["neutral_f1"]), float(metrics["positive_f1"])
        ) >= float(gates.get("minimum_class_f1", 0.60)),
        "bootstrap_macro_f1_noninferiority": float(bootstrap["ci95_lower"])
        >= float(gates.get("minimum_bootstrap_ci_lower", 0.0)),
        "per_company_noninferiority": all(
            float(item["macro_f1_delta"]) >= -maximum_company_loss for item in checked_companies
        ),
        "stable_weight_in_majority_of_folds": float(stability["proportion"]) >= 0.60,
        # This is deliberately always false under the 2015--2025-only protocol.
        # It ensures the report cannot be used to promote a runtime model as if
        # it had an untouched future evaluation set.
        "independent_holdout": False,
    }


def run_chronological_finbert_ensemble(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
    phrasebank_model_directory: Path = DEFAULT_PHRASEBANK_MODEL,
    run_id: str = DEFAULT_RUN_ID,
    profile: str = "full",
    device: str = "cuda",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    bursa_replay_share: float = DEFAULT_BURSA_REPLAY_SHARE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_epochs: int | None = None,
    embargo_days: int = 7,
    bootstrap_iterations: int = 2000,
    max_runtime_minutes: float = 45.0,
    selected_folds: Sequence[str] | None = None,
    dry_run: bool = False,
    text_mode: str = TEXT_MODE_DEFAULT,
    sampling_mode: str = SAMPLING_MODE_SHUFFLE,
    loss_mode: str = LOSS_MODE_INVERSE_FREQUENCY_CE,
    calibration_mode: str = CALIBRATION_MODE_SCALAR_TEMPERATURE,
) -> dict[str, Any]:
    """Run or resume a chronological 2015--2025 ensemble benchmark.

    Completed seed caches are never retrained unless their split fingerprint or
    training recipe changes.  A safe pause happens between seeds and folds.
    """

    if profile not in {"full", "quick"}:
        raise ValueError("profile must be full or quick")
    if not 0.0 < float(bursa_replay_share) <= 1.0:
        raise ValueError("bursa_replay_share must be in (0, 1]")
    if float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if int(bootstrap_iterations) < 2:
        raise ValueError("bootstrap_iterations must be at least 2")
    if float(max_runtime_minutes) <= 0.0:
        raise ValueError("max_runtime_minutes must be positive")
    if text_mode not in SUPPORTED_TEXT_MODES:
        raise ValueError(f"Unsupported sentiment text mode: {text_mode!r}")
    if sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode!r}")
    if loss_mode not in SUPPORTED_LOSS_MODES:
        raise ValueError(f"Unsupported loss mode: {loss_mode!r}")
    if calibration_mode not in SUPPORTED_CALIBRATION_MODES:
        raise ValueError(f"Unsupported calibration mode: {calibration_mode!r}")
    selected_seed_values = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not selected_seed_values:
        raise ValueError("at least one Bursa seed is required")
    selected = set(selected_folds or ())
    folds = tuple(fold for fold in DEFAULT_FOLDS if not selected or fold.name in selected)
    if not folds:
        raise ValueError("selected_folds did not match any configured chronological fold")

    dataset_path = Path(dataset_path)
    policy_path = Path(policy_path)
    phrasebank_model_directory = Path(phrasebank_model_directory)
    frame, dataset_diagnostics = _prepare_dataset(dataset_path, policy_path)
    frame = apply_sentiment_text_mode(frame, mode=text_mode)
    frame = frame.loc[frame["model_text"].astype(str).str.strip().ne("")].reset_index(drop=True)
    prepared_folds: list[tuple[ChronologicalFold, dict[str, pd.DataFrame], dict[str, Any]]] = []
    for fold in folds:
        partitions, diagnostics = build_chronological_fold(frame, fold, embargo_days=embargo_days)
        prepared_folds.append((fold, partitions, diagnostics))
    manifest = {
        "schema_version": "finbert-ensemble-chronological-v1",
        "run_id": str(run_id),
        "date_range": {"start": DATE_START.isoformat(), "end_exclusive": DATE_END_EXCLUSIVE.isoformat()},
        "dataset_path": str(dataset_path),
        "policy_path": str(policy_path),
        "phrasebank_model_directory": str(phrasebank_model_directory),
        "profile": profile,
        "device": device,
        "seeds": list(selected_seed_values),
        "bursa_replay_share": float(bursa_replay_share),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs)
        if max_epochs is not None
        else (FULL_PROFILE_EPOCHS if profile == "full" else QUICK_PROFILE_EPOCHS),
        "embargo_days": int(embargo_days),
        "text_mode": text_mode,
        "sampling_mode": sampling_mode,
        "loss_mode": loss_mode,
        "calibration_mode": calibration_mode,
        "folds": [diagnostics for _, _, diagnostics in prepared_folds],
        "dataset_diagnostics": dataset_diagnostics,
        "evaluation_status": "retrospective_2015_2025_only",
        "production_changed": False,
    }
    if dry_run:
        return {"status": "dry_run", **manifest}

    report_dir = RESULTS_ROOT / str(run_id)
    model_dir = STAGING_ROOT / str(run_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_dir / "manifest.json", manifest)
    deadline = time.monotonic() + float(max_runtime_minutes) * 60.0
    phrasebank_development, phrasebank_test = _load_phrasebank_replay()
    phrasebank_development = apply_sentiment_text_mode(
        phrasebank_development, mode=text_mode
    )
    phrasebank_test = apply_sentiment_text_mode(phrasebank_test, mode=text_mode)
    fold_summaries: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold, partitions, diagnostics in prepared_folds:
        fold_report_dir = report_dir / fold.name
        fold_model_dir = model_dir / fold.name
        cached_fold = _read_fold_outputs(fold_report_dir)
        if cached_fold is not None:
            cached_frame, cached_summary = cached_fold
            expected_fingerprint = _fold_fingerprint(
                fold,
                partitions,
                seeds=selected_seed_values,
                replay_share=bursa_replay_share,
                learning_rate=learning_rate,
                max_epochs=int(manifest["max_epochs"]),
                embargo_days=embargo_days,
                text_mode=text_mode,
                sampling_mode=sampling_mode,
                loss_mode=loss_mode,
                calibration_mode=calibration_mode,
            )
            if cached_summary.get("fingerprint") == expected_fingerprint:
                fold_summaries.append(cached_summary)
                oof_frames.append(cached_frame)
                continue
        fingerprint = _fold_fingerprint(
            fold,
            partitions,
            seeds=selected_seed_values,
            replay_share=bursa_replay_share,
            learning_rate=learning_rate,
            max_epochs=int(manifest["max_epochs"]),
            embargo_days=embargo_days,
            text_mode=text_mode,
            sampling_mode=sampling_mode,
            loss_mode=loss_mode,
            calibration_mode=calibration_mode,
        )
        seed_results: list[dict[str, Any]] = []
        for seed in selected_seed_values:
            if time.monotonic() >= deadline:
                checkpoint = {
                    **manifest,
                    "status": "paused",
                    "next_fold": fold.name,
                    "next_seed": int(seed),
                    "completed_folds": [item["fold"] for item in fold_summaries],
                    "safe_to_resume": True,
                }
                _atomic_json(report_dir / "checkpoint.json", checkpoint)
                return checkpoint
            print(f"{fold.name}: Bursa FinBERT seed {seed}...")
            result = _run_or_resume_seed(
                partitions,
                fold_dir=fold_model_dir,
                fold_fingerprint=fingerprint,
                seed=seed,
                phrasebank_development=phrasebank_development,
                phrasebank_test=phrasebank_test,
                profile=profile,
                device=device,
                replay_share=bursa_replay_share,
                learning_rate=learning_rate,
                max_epochs=int(manifest["max_epochs"]),
                text_mode=text_mode,
                sampling_mode=sampling_mode,
                loss_mode=loss_mode,
            )
            seed_results.append(result)

        if time.monotonic() >= deadline:
            checkpoint = {
                **manifest,
                "status": "paused",
                "next_fold": fold.name,
                "next_seed": None,
                "completed_folds": [item["fold"] for item in fold_summaries],
                "safe_to_resume": True,
            }
            _atomic_json(report_dir / "checkpoint.json", checkpoint)
            return checkpoint

        bursa_calibration = np.mean(
            [np.asarray(item["validation_probabilities"], dtype=float) for item in seed_results], axis=0
        )
        bursa_evaluation = np.mean(
            [np.asarray(item["test_probabilities"], dtype=float) for item in seed_results], axis=0
        )
        phrasebank_calibration, phrasebank_evaluation, phrasebank_temperature = _phrasebank_fold_probabilities(
            partitions["calibration"],
            partitions["evaluation"],
            model_directory=phrasebank_model_directory,
        )
        calibration_labels = partitions["calibration"]["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
        evaluation_labels = partitions["evaluation"]["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
        selection = select_stable_weight(
            bursa_calibration,
            phrasebank_calibration,
            calibration_labels,
            _block_ids(partitions["calibration"]),
            bootstrap_iterations=int(bootstrap_iterations),
        )
        bursa_weight = float(selection["selected_bursa_weight"])
        ensemble_calibration_raw = _blend(
            bursa_calibration, phrasebank_calibration, bursa_weight
        )
        if calibration_mode == CALIBRATION_MODE_SCALAR_TEMPERATURE:
            ensemble_temperature = fit_temperature(
                np.log(np.clip(ensemble_calibration_raw, 1e-12, 1.0)),
                calibration_labels,
            )
        else:
            ensemble_temperature = 1.0
        ensemble_calibration = _scale_probabilities(
            ensemble_calibration_raw, ensemble_temperature
        )
        ensemble_evaluation_raw = _blend(
            bursa_evaluation, phrasebank_evaluation, bursa_weight
        )
        ensemble_evaluation = _scale_probabilities(
            ensemble_evaluation_raw, ensemble_temperature
        )
        fold_summary = {
            "fold": fold.name,
            "fingerprint": fingerprint,
            "split": diagnostics,
            "bursa_seed_average": {
                "seeds": list(selected_seed_values),
                # Freshly trained seeds return metadata fields at the result
                # root, while checkpoint-resumed seeds wrap those fields in
                # ``metadata``.  Normalize both shapes so a resumed run and
                # an uninterrupted run produce the same fold report.
                "seed_temperatures": [
                    float(
                        item.get(
                            "temperature",
                            (item.get("metadata") or {}).get("temperature", 1.0),
                        )
                    )
                    for item in seed_results
                ],
                "model_directories": [str((fold_model_dir / f"seed-{seed}").relative_to(PROJECT_ROOT)).replace("\\", "/") for seed in selected_seed_values],
            },
            "seed_evaluation_metrics": [
                {
                    "seed": int(seed),
                    **metrics_from_probabilities(
                        f"bursa_seed_{seed}_evaluation",
                        np.asarray(item["test_probabilities"], dtype=float),
                        evaluation_labels,
                    ),
                }
                for seed, item in zip(selected_seed_values, seed_results)
            ],
            "phrasebank_temperature": phrasebank_temperature,
            "ensemble_temperature": float(ensemble_temperature),
            "selection": selection,
            "calibration": {
                "bursa_seed_average": metrics_from_probabilities(
                    "bursa_seed_average_calibration", bursa_calibration, calibration_labels
                ),
                "phrasebank": metrics_from_probabilities(
                    "phrasebank_calibration", phrasebank_calibration, calibration_labels
                ),
                    "ensemble": metrics_from_probabilities(
                        "ensemble_calibrated_calibration",
                        ensemble_calibration,
                        calibration_labels,
                    ),
            },
            "evaluation": {
                "bursa_seed_average": metrics_from_probabilities("bursa_seed_average", bursa_evaluation, evaluation_labels),
                "phrasebank": metrics_from_probabilities("phrasebank", phrasebank_evaluation, evaluation_labels),
                "ensemble": metrics_from_probabilities("ensemble", ensemble_evaluation, evaluation_labels),
            },
        }
        fold_summary["seed_macro_f1_std"] = float(
            np.std(
                [
                    float(item["macro_f1"])
                    for item in fold_summary["seed_evaluation_metrics"]
                ]
            )
        )
        fold_report_dir.mkdir(parents=True, exist_ok=True)
        _write_fold_outputs(
            fold_report_dir,
            partitions["evaluation"],
            bursa_evaluation,
            phrasebank_evaluation,
            fold_summary,
            ensemble=ensemble_evaluation,
        )
        fold_summaries.append(fold_summary)
        oof = partitions["evaluation"][["article_id", "ticker", "published_at", "event_id", "sentiment_label", "_event_group"]].copy()
        for index, label in enumerate(("negative", "neutral", "positive")):
            oof[f"bursa_probability_{label}"] = bursa_evaluation[:, index]
            oof[f"phrasebank_probability_{label}"] = phrasebank_evaluation[:, index]
            oof[f"ensemble_probability_{label}"] = ensemble_evaluation[:, index]
        oof_frames.append(oof)

    if (report_dir / "checkpoint.json").exists():
        (report_dir / "checkpoint.json").unlink()
    oof_frame = pd.concat(oof_frames, ignore_index=True).sort_values("published_at", kind="stable").reset_index(drop=True)
    bursa_oof = oof_frame[[f"bursa_probability_{name}" for name in ("negative", "neutral", "positive")]].to_numpy(dtype=float)
    phrasebank_oof = oof_frame[[f"phrasebank_probability_{name}" for name in ("negative", "neutral", "positive")]].to_numpy(dtype=float)
    labels_oof = oof_frame["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    final_selection = select_stable_weight(
        bursa_oof,
        phrasebank_oof,
        labels_oof,
        _block_ids(oof_frame),
        bootstrap_iterations=int(bootstrap_iterations),
    )
    final_weight = float(final_selection["selected_bursa_weight"])
    if all(
        f"ensemble_probability_{name}" in oof_frame.columns
        for name in ("negative", "neutral", "positive")
    ):
        ensemble_oof = oof_frame[
            [f"ensemble_probability_{name}" for name in ("negative", "neutral", "positive")]
        ].to_numpy(dtype=float)
    else:
        ensemble_oof = _blend(bursa_oof, phrasebank_oof, final_weight)
    candidate_metrics = metrics_from_probabilities("selected_chronological_ensemble", ensemble_oof, labels_oof)
    phrasebank_metrics = metrics_from_probabilities("phrasebank", phrasebank_oof, labels_oof)
    bursa_metrics = metrics_from_probabilities("bursa_seed_average", bursa_oof, labels_oof)
    bootstrap = _bootstrap_macro_f1_delta_by_block(
        ensemble_oof.argmax(axis=1),
        phrasebank_oof.argmax(axis=1),
        labels_oof,
        _block_ids(oof_frame),
        iterations=int(bootstrap_iterations),
    )
    company = _company_report(oof_frame, ensemble_oof, phrasebank_oof)
    stability = _fold_weight_coverage(final_weight, fold_summaries)
    checks = _retrospective_checks(policy=load_policy(policy_path), metrics=candidate_metrics, bootstrap=bootstrap, company=company, stability=stability)
    oof_frame.to_csv(report_dir / "pooled-oof.csv", index=False)
    report = {
        **manifest,
        "status": "complete",
        "fold_summaries": fold_summaries,
        "pooled_oof_rows": int(len(oof_frame)),
        "final_selection": final_selection,
        "pooled_oof_metrics": {
            "phrasebank": phrasebank_metrics,
            "bursa_seed_average": bursa_metrics,
            "ensemble": candidate_metrics,
        },
        "bootstrap_macro_f1_delta_vs_phrasebank": bootstrap,
        "per_company": company,
        "weight_stability": stability,
        "checks": checks,
        "eligible_for_production_promotion": False,
        "promotion_reason": (
            "The 2015--2025-only protocol has no untouched independent holdout. "
            "This candidate remains retrospective and experimental."
        ),
        "runtime_candidate": None,
        "active_runtime_changed": False,
        "phrasebank_artifact_hashes": sentiment_artifact_hashes_for_directory(phrasebank_model_directory),
    }
    _atomic_json(report_dir / "report.json", report)
    _atomic_json(RESULTS_ROOT / "latest.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run/resume the 2015--2025 chronological Bursa/PhraseBank ensemble benchmark."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--phrasebank-model-directory", default=str(DEFAULT_PHRASEBANK_MODEL))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--profile", choices=["full", "quick"], default="full")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--bursa-replay-share", type=float, default=DEFAULT_BURSA_REPLAY_SHARE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--embargo-days", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--max-runtime-minutes", type=float, default=45.0)
    parser.add_argument("--text-mode", choices=SUPPORTED_TEXT_MODES, default=TEXT_MODE_DEFAULT)
    parser.add_argument("--sampling-mode", choices=SUPPORTED_SAMPLING_MODES, default=SAMPLING_MODE_SHUFFLE)
    parser.add_argument("--loss-mode", choices=SUPPORTED_LOSS_MODES, default=LOSS_MODE_INVERSE_FREQUENCY_CE)
    parser.add_argument(
        "--calibration-mode",
        choices=SUPPORTED_CALIBRATION_MODES,
        default=CALIBRATION_MODE_SCALAR_TEMPERATURE,
    )
    parser.add_argument("--fold", dest="folds", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_chronological_finbert_ensemble(
        Path(args.dataset),
        policy_path=Path(args.policy),
        phrasebank_model_directory=Path(args.phrasebank_model_directory),
        run_id=args.run_id,
        profile=args.profile,
        device=args.device,
        seeds=tuple(args.seeds) if args.seeds else DEFAULT_SEEDS,
        bursa_replay_share=args.bursa_replay_share,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        embargo_days=args.embargo_days,
        bootstrap_iterations=args.bootstrap_iterations,
        max_runtime_minutes=args.max_runtime_minutes,
        selected_folds=args.folds,
        dry_run=args.dry_run,
        text_mode=args.text_mode,
        sampling_mode=args.sampling_mode,
        loss_mode=args.loss_mode,
        calibration_mode=args.calibration_mode,
    )
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
