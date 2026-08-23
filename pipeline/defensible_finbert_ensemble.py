"""Build a reproducible PhraseBank-anchored Bursa FinBERT ensemble challenger.

This module is deliberately separate from the runtime sentiment path.  It
chooses the Bursa training mixture from 2015--2024 chronological evidence,
locks the selected recipe, and evaluates it once on the project's existing
2025 retrospective test.  Promotion is always a separate explicit action.

The 2025 data has been inspected by earlier experiments.  The generated
report therefore records it as a *retrospective final test*, never an
independent future holdout.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.benchmark_bursa_finbert import _load_phrasebank_replay
from pipeline.benchmark_finbert_sentiment import LABEL_TO_ID, metrics_from_probabilities
from pipeline.bursa_news_dataset import DEFAULT_DATASET_PATH, DEFAULT_POLICY_PATH, load_policy
from pipeline.chronological_finbert_ensemble import (
    CALIBRATION_MODE_SCALAR_TEMPERATURE,
    SUPPORTED_CALIBRATION_MODES,
    DEFAULT_FOLDS,
    DEFAULT_PHRASEBANK_MODEL,
    DEFAULT_SEEDS,
    RESULTS_ROOT as CHRONOLOGICAL_RESULTS_ROOT,
    STAGING_ROOT as CHRONOLOGICAL_STAGING_ROOT,
    _blend,
    _block_ids,
    _bootstrap_macro_f1_delta_by_block,
    _company_report,
    _fold_weight_coverage,
    _scale_probabilities,
    _prepare_dataset,
    run_chronological_finbert_ensemble,
)
from pipeline.benchmark_bursa_finbert import (
    LOSS_MODE_INVERSE_FREQUENCY_CE,
    SAMPLING_MODE_SHUFFLE,
    SUPPORTED_LOSS_MODES,
    SUPPORTED_SAMPLING_MODES,
)
from pipeline.sentiment_text import SUPPORTED_TEXT_MODES, TEXT_MODE_DEFAULT
from pipeline.finbert_sentiment_model import clear_finbert_runtime_cache, load_finbert_sentiment_model
from pipeline.runtime_provenance import sentiment_artifact_hashes_for_directory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "finbert-ensemble-defensible"
DEFAULT_RUN_ID = "bursa-phrasebank-defensible-2015-2025-v1"
DEFAULT_PHRASEBANK_REPLAY_SHARES = (0.0, 0.10, 0.20, 0.30)
DEVELOPMENT_FOLDS = tuple(fold.name for fold in DEFAULT_FOLDS if fold.evaluation_year <= 2024)
FINAL_FOLD = next(fold for fold in DEFAULT_FOLDS if fold.evaluation_year == 2025)
DEFAULT_MIN_FREE_GIB = 10.0


class DefensibleEnsembleError(RuntimeError):
    """Raised when the fixed ensemble protocol cannot be preserved."""


def _free_space_gib(path: Path = PROJECT_ROOT) -> float:
    """Return free space for the filesystem containing ``path``."""

    usage = shutil.disk_usage(Path(path).resolve())
    return float(usage.free) / float(1024**3)


def _remove_development_model_artifacts(child_run_id: str) -> dict[str, Any]:
    """Hash and remove one completed development child staging directory.

    This is deliberately restricted to the chronological experiment staging
    root.  Reports, fold probabilities and selection metrics remain available
    for audit/resume; only the large model checkpoints are removed.
    """

    staging_root = CHRONOLOGICAL_STAGING_ROOT.resolve()
    child_dir = (CHRONOLOGICAL_STAGING_ROOT / str(child_run_id)).resolve()
    if staging_root not in child_dir.parents:
        raise DefensibleEnsembleError(
            f"Refusing to remove a development artifact outside {staging_root}: {child_dir}"
        )
    if not child_dir.exists():
        return {
            "mode": "sequential",
            "status": "already_removed",
            "child_run_id": str(child_run_id),
            "model_directory": _runtime_path(child_dir),
            "artifact_hashes": {},
            "removed_bytes": 0,
        }
    files = [path for path in child_dir.rglob("*") if path.is_file()]
    hashes = sentiment_artifact_hashes_for_directory(child_dir)
    removed_bytes = int(sum(path.stat().st_size for path in files))
    shutil.rmtree(child_dir)
    return {
        "mode": "sequential",
        "status": "removed_after_evaluation",
        "child_run_id": str(child_run_id),
        "model_directory": _runtime_path(child_dir),
        "artifact_hashes": hashes,
        "removed_bytes": removed_bytes,
    }


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
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DefensibleEnsembleError(f"Unable to read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DefensibleEnsembleError(f"JSON object required: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _replay_key(phrasebank_replay_share: float) -> str:
    return f"phrasebank-replay-{int(round(float(phrasebank_replay_share) * 100)):02d}"


def normalize_phrasebank_replay_shares(values: Sequence[float] | None) -> tuple[float, ...]:
    """Validate and canonicalise the phrasebank portion of Bursa fine-tuning."""

    raw = DEFAULT_PHRASEBANK_REPLAY_SHARES if values is None else tuple(values)
    if not raw:
        raise ValueError("at least one PhraseBank replay share is required")
    normalized = tuple(sorted({round(float(value), 6) for value in raw}))
    if any(value < 0.0 or value >= 1.0 for value in normalized):
        raise ValueError("PhraseBank replay shares must be in [0.0, 1.0)")
    return normalized


def normalize_training_recipe_grid(
    values: Sequence[Mapping[str, Any]] | None,
    *,
    text_mode: str = TEXT_MODE_DEFAULT,
    sampling_mode: str = SAMPLING_MODE_SHUFFLE,
    loss_mode: str = LOSS_MODE_INVERSE_FREQUENCY_CE,
    calibration_mode: str = CALIBRATION_MODE_SCALAR_TEMPERATURE,
) -> tuple[dict[str, str], ...]:
    """Validate a development-only recipe grid.

    A grid is optional.  Keeping the default as one requested recipe avoids
    unexpectedly multiplying a full GPU run, while the explicit grid makes it
    possible to compare text, sampling and loss choices without inspecting the
    2025 retrospective test until one recipe has already been selected.
    """

    raw: Sequence[Mapping[str, Any]] = values or (
        {
            "id": "requested",
            "text_mode": text_mode,
            "sampling_mode": sampling_mode,
            "loss_mode": loss_mode,
            "calibration_mode": calibration_mode,
        },
    )
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("Each training recipe must be a JSON object")
        recipe_id = str(item.get("id") or f"recipe-{index + 1}").strip()
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", recipe_id):
            raise ValueError("Training recipe id may contain only letters, numbers, '_' and '-'")
        if recipe_id in seen:
            raise ValueError(f"Duplicate training recipe id: {recipe_id}")
        selected_text_mode = str(item.get("text_mode") or text_mode)
        selected_sampling_mode = str(item.get("sampling_mode") or sampling_mode)
        selected_loss_mode = str(item.get("loss_mode") or loss_mode)
        selected_calibration_mode = str(item.get("calibration_mode") or calibration_mode)
        if selected_text_mode not in SUPPORTED_TEXT_MODES:
            raise ValueError(f"Unsupported sentiment text mode: {selected_text_mode!r}")
        if selected_sampling_mode not in SUPPORTED_SAMPLING_MODES:
            raise ValueError(f"Unsupported sampling mode: {selected_sampling_mode!r}")
        if selected_loss_mode not in SUPPORTED_LOSS_MODES:
            raise ValueError(f"Unsupported loss mode: {selected_loss_mode!r}")
        if selected_calibration_mode not in SUPPORTED_CALIBRATION_MODES:
            raise ValueError(f"Unsupported calibration mode: {selected_calibration_mode!r}")
        seen.add(recipe_id)
        result.append(
            {
                "id": recipe_id,
                "text_mode": selected_text_mode,
                "sampling_mode": selected_sampling_mode,
                "loss_mode": selected_loss_mode,
                "calibration_mode": selected_calibration_mode,
            }
        )
    return tuple(result)


def load_training_recipe_grid(path: Path | str) -> tuple[dict[str, str], ...]:
    """Load an explicit recipe grid from a checked-in JSON array."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read training recipe grid: {source}") from exc
    if not isinstance(raw, list):
        raise ValueError("Training recipe grid must be a JSON array")
    return normalize_training_recipe_grid(raw)


def build_evidence_manifest(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
    phrasebank_model_directory: Path = DEFAULT_PHRASEBANK_MODEL,
) -> dict[str, Any]:
    """Fingerprint the exact local evidence allowed into one experiment."""

    dataset_path = Path(dataset_path)
    policy_path = Path(policy_path)
    phrasebank_model_directory = Path(phrasebank_model_directory)
    frame, diagnostics = _prepare_dataset(dataset_path, policy_path)
    fields = [
        "article_id",
        "ticker",
        "published_at",
        "content_sha256",
        "event_id",
        "sentiment_label",
    ]
    canonical = frame[fields].copy()
    canonical["published_at"] = canonical["published_at"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    canonical = canonical.fillna("").astype(str).sort_values(fields, kind="stable")
    evidence_fingerprint = hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "defensible-finbert-evidence-manifest-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": _runtime_path(dataset_path),
        "policy_path": _runtime_path(policy_path),
        "phrasebank_model_directory": _runtime_path(phrasebank_model_directory),
        "dataset_file_sha256": _sha256_file(dataset_path),
        "policy_file_sha256": _sha256_file(policy_path),
        "phrasebank_artifact_hashes": sentiment_artifact_hashes_for_directory(
            phrasebank_model_directory
        ),
        "eligible_evidence_sha256": evidence_fingerprint,
        "eligible_rows": int(len(frame)),
        "eligible_article_ids": int(frame["article_id"].astype(str).nunique()),
        "dataset_diagnostics": diagnostics,
        "date_protocol": {
            "development_through": 2024,
            "retrospective_final_test_year": 2025,
            "final_test_previously_inspected": True,
            "test_tuning_prohibited_after_manifest": True,
        },
    }


def _assert_manifest_matches(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    keys = (
        "dataset_file_sha256",
        "policy_file_sha256",
        "phrasebank_artifact_hashes",
        "eligible_evidence_sha256",
        "eligible_rows",
    )
    changed = [key for key in keys if expected.get(key) != observed.get(key)]
    if changed:
        raise DefensibleEnsembleError(
            "The frozen evidence changed during the experiment: " + ", ".join(changed)
        )


def select_replay_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose a Bursa training mixture using development evidence only.

    When a policy is supplied, candidates that satisfy the development
    versions of the promotion gates are preferred.  If none satisfy every
    gate, the candidate with the smallest aggregate gate deficit is selected;
    this keeps the run resumable while making the remaining weakness explicit
    instead of silently selecting on log loss alone.
    """

    if not candidates:
        raise DefensibleEnsembleError("No completed replay candidates are available")
    required = ("macro_f1", "balanced_accuracy", "log_loss", "ece", "seed_macro_f1_std")
    for candidate in candidates:
        missing = [key for key in required if key not in candidate]
        if missing:
            raise DefensibleEnsembleError(
                "Replay candidate has incomplete development metrics: " + ", ".join(missing)
            )
    prepared = [dict(candidate) for candidate in candidates]
    if policy is not None:
        gates = dict(
            ((policy.get("sentiment_ensemble") or {}).get("promotion_gates") or {})
        )
        minimum_class_f1 = float(gates.get("minimum_class_f1", 0.60))
        maximum_ece = float(gates.get("maximum_ece", 0.10))
        maximum_company_loss = float(gates.get("maximum_company_macro_f1_loss", 0.05))
        maximum_phrasebank_degradation = float(
            gates.get("maximum_phrasebank_macro_f1_degradation", 0.02)
        )
        for item in prepared:
            failures: list[str] = []
            deficit = 0.0
            minimum_observed = float(item.get("minimum_class_f1", 0.0))
            if minimum_observed < minimum_class_f1:
                failures.append("minimum_class_f1")
                deficit += minimum_class_f1 - minimum_observed
            observed_ece = float(item.get("ece", 1.0))
            if observed_ece > maximum_ece:
                failures.append("ece")
                deficit += observed_ece - maximum_ece
            company_delta = float(item.get("worst_company_macro_f1_delta", -1.0))
            if company_delta < -maximum_company_loss:
                failures.append("per_company_noninferiority")
                deficit += (-maximum_company_loss) - company_delta
            phrasebank_delta = float(item.get("phrasebank_macro_f1_delta", -1.0))
            if phrasebank_delta < -maximum_phrasebank_degradation:
                failures.append("phrasebank_generalisation_noninferiority")
                deficit += (-maximum_phrasebank_degradation) - phrasebank_delta
            item["development_gate_failures"] = failures
            item["development_gate_deficit"] = float(deficit)
            item["development_gate_eligible"] = not failures
        selected = sorted(
            prepared,
            key=lambda item: (
                not bool(item.get("development_gate_eligible", False)),
                int(len(item.get("development_gate_failures") or [])),
                float(item.get("development_gate_deficit", 0.0)),
                -float(item["macro_f1"]),
                -float(item["balanced_accuracy"]),
                float(item["log_loss"]),
                float(item["ece"]),
                float(item["seed_macro_f1_std"]),
                abs(float(item["phrasebank_replay_share"]) - 0.20),
            ),
        )[0]
        selected["selection_order"] = [
            "development_gate_eligibility",
            "development_gate_failure_count_asc",
            "development_gate_deficit_asc",
            "development_macro_f1_desc",
            "development_balanced_accuracy_desc",
            "development_log_loss_asc",
            "development_ece_asc",
            "development_seed_stability_asc",
            "closest_to_20_percent_phrasebank_replay_tiebreak",
        ]
    else:
        selected = sorted(
            prepared,
            key=lambda item: (
                -float(item["macro_f1"]),
                -float(item["balanced_accuracy"]),
                float(item["log_loss"]),
                float(item["ece"]),
                float(item["seed_macro_f1_std"]),
                abs(float(item["phrasebank_replay_share"]) - 0.20),
            ),
        )[0]
        selected["selection_order"] = [
            "development_macro_f1_desc",
            "development_balanced_accuracy_desc",
            "development_log_loss_asc",
            "development_ece_asc",
            "development_seed_stability_asc",
            "closest_to_20_percent_phrasebank_replay_tiebreak",
        ]
    return selected


def _development_summary(
    report: Mapping[str, Any], *, phrasebank_replay_share: float, child_run_id: str
) -> dict[str, Any]:
    pooled = report.get("pooled_oof_metrics") or {}
    metrics = pooled.get("ensemble") or {}
    phrasebank_metrics = pooled.get("phrasebank") or {}
    fold_summaries = report.get("fold_summaries") or []
    seed_values = [
        float(summary.get("seed_macro_f1_std"))
        for summary in fold_summaries
        if summary.get("seed_macro_f1_std") is not None
    ]
    if not metrics or len(seed_values) != len(fold_summaries):
        raise DefensibleEnsembleError(
            f"Development run {child_run_id} has incomplete fold metrics; use a fresh run ID."
        )
    company_report = report.get("per_company") or {}
    eligible_company = [
        item for item in company_report.values() if int(item.get("rows", 0)) >= 15
    ]
    worst_company_delta = min(
        (float(item.get("macro_f1_delta", -1.0)) for item in eligible_company),
        default=-1.0,
    )
    minimum_class_f1 = min(
        float(metrics.get("negative_f1", 0.0)),
        float(metrics.get("neutral_f1", 0.0)),
        float(metrics.get("positive_f1", 0.0)),
    )
    return {
        "phrasebank_replay_share": float(phrasebank_replay_share),
        "bursa_training_share": float(1.0 - phrasebank_replay_share),
        "child_run_id": child_run_id,
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "log_loss": float(metrics["log_loss"]),
        "ece": float(metrics["ece"]),
        "negative_f1": float(metrics.get("negative_f1", 0.0)),
        "neutral_f1": float(metrics.get("neutral_f1", 0.0)),
        "positive_f1": float(metrics.get("positive_f1", 0.0)),
        "minimum_class_f1": minimum_class_f1,
        "phrasebank_macro_f1": float(phrasebank_metrics.get("macro_f1", 0.0)),
        "phrasebank_macro_f1_delta": float(
            metrics.get("macro_f1", 0.0) - phrasebank_metrics.get("macro_f1", 0.0)
        ),
        "worst_company_macro_f1_delta": worst_company_delta,
        "seed_macro_f1_std": float(np.mean(seed_values)),
        "fold_seed_macro_f1_std": seed_values,
        "selected_bursa_weight": float(
            ((report.get("final_selection") or {}).get("selected_bursa_weight"))
        ),
        "selected_phrasebank_weight": float(
            ((report.get("final_selection") or {}).get("selected_phrasebank_weight"))
        ),
    }


def _final_seed_probabilities(
    run_id: str,
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    values: list[np.ndarray] = []
    phrasebank_values: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    labels: np.ndarray | None = None
    for seed in seeds:
        path = CHRONOLOGICAL_STAGING_ROOT / run_id / FINAL_FOLD.name / f"seed-{seed}" / "chronological-result.npz"
        if not path.exists():
            raise DefensibleEnsembleError(f"Missing final seed checkpoint: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "phrasebank_probabilities" not in data.files or "phrasebank_labels" not in data.files:
                raise DefensibleEnsembleError(
                    "Final seed cache lacks PhraseBank retention probabilities. "
                    "Run the selected final fold under a new defensible run ID."
                )
            current_labels = np.asarray(data["phrasebank_labels"], dtype=int)
            if labels is None:
                labels = current_labels
            elif not np.array_equal(labels, current_labels):
                raise DefensibleEnsembleError("Final seed PhraseBank labels differ across seeds")
            phrasebank_values.append(np.asarray(data["phrasebank_probabilities"], dtype=float))
            evaluation = np.asarray(data["test_probabilities"], dtype=float)
            values.append(evaluation)
            details.append(
                {
                    "seed": int(seed),
                    "model_directory": _runtime_path(path.parent),
                    "artifact_hashes": sentiment_artifact_hashes_for_directory(path.parent),
                }
            )
    if labels is None:
        raise DefensibleEnsembleError("No final Bursa seed probabilities were recovered")
    return np.mean(values, axis=0), np.mean(phrasebank_values, axis=0), details


def _phrasebank_frozen_probabilities(
    phrasebank_model_directory: Path,
    phrasebank_temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, frozen_test = _load_phrasebank_replay()
    labels = frozen_test["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    model = load_finbert_sentiment_model(
        str(phrasebank_model_directory), temperature=float(phrasebank_temperature)
    )
    try:
        probabilities = np.asarray(
            model.predict_proba(frozen_test["model_text"].astype(str).tolist()), dtype=float
        )
    finally:
        del model
        clear_finbert_runtime_cache()
        gc.collect()
    return probabilities, labels


def _final_gate_report(
    *,
    policy: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    final_phrasebank_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    validation_phrasebank_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    per_company: Mapping[str, Any],
    weight_stability: Mapping[str, Any],
    bursa_weight: float,
    seed_macro_f1_std: float,
    phrasebank_macro_f1_degradation: float,
    test_rows: int,
    independent_holdout_ready: bool | None = None,
) -> dict[str, Any]:
    gates = dict(((policy.get("sentiment_ensemble") or {}).get("promotion_gates") or {}))
    minimum_company_rows = int(gates.get("minimum_holdout_rows_per_ticker", 15))
    evaluated_companies = [
        item for item in per_company.values() if int(item.get("rows", 0)) >= minimum_company_rows
    ]
    component_minimum = float(gates.get("minimum_component_weight", 0.20))
    component_maximum = float(gates.get("maximum_component_weight", 0.80))
    minimum_matching_folds = int(gates.get("minimum_matching_fold_count", 3))
    max_company_loss = float(gates.get("maximum_company_macro_f1_loss", 0.05))
    checks = {
        "validation_macro_f1_uplift": float(validation_metrics["macro_f1"])
        >= float(validation_phrasebank_metrics["macro_f1"])
        + float(gates.get("minimum_validation_macro_f1_uplift", 0.02)),
        "validation_balanced_accuracy_uplift": float(validation_metrics["balanced_accuracy"])
        >= float(validation_phrasebank_metrics["balanced_accuracy"])
        + float(gates.get("minimum_validation_balanced_accuracy_uplift", 0.02)),
        "final_macro_f1": float(final_metrics["macro_f1"])
        >= float(gates.get("minimum_macro_f1", 0.70)),
        "final_balanced_accuracy": float(final_metrics["balanced_accuracy"])
        >= float(gates.get("minimum_balanced_accuracy", 0.70)),
        "final_ece": float(final_metrics["ece"]) <= float(gates.get("maximum_ece", 0.10)),
        "minimum_class_f1": min(
            float(final_metrics["negative_f1"]),
            float(final_metrics["neutral_f1"]),
            float(final_metrics["positive_f1"]),
        )
        >= float(gates.get("minimum_class_f1", 0.60)),
        "bootstrap_macro_f1_noninferiority": float(bootstrap["ci95_lower"])
        >= float(gates.get("minimum_bootstrap_ci_lower", 0.0)),
        "phrasebank_generalisation_noninferiority": phrasebank_macro_f1_degradation
        <= float(gates.get("maximum_phrasebank_macro_f1_degradation", 0.02)),
        "bursa_seed_stability": seed_macro_f1_std
        <= float(gates.get("maximum_seed_macro_f1_std", 0.02)),
        "per_company_noninferiority": bool(evaluated_companies)
        and len(evaluated_companies) == len(per_company)
        and all(
            float(item["macro_f1_delta"]) >= -max_company_loss
            for item in evaluated_companies
        ),
        "test_row_coverage": int(test_rows) >= int(gates.get("minimum_holdout_rows", 200)),
        "component_non_degeneracy": component_minimum <= float(bursa_weight) <= component_maximum,
        "weight_stable_across_development_folds": int(weight_stability["matching_fold_count"])
        >= minimum_matching_folds,
        "retrospective_test_disclosed": bool(
            gates.get("require_retrospective_test_disclosure", True)
        ),
        "manual_retrospective_promotion_allowed": bool(
            gates.get("allow_retrospective_manual_promotion", False)
        ),
    }
    if bool(gates.get("require_prospective_holdout", False)):
        checks["independent_prospective_holdout"] = bool(independent_holdout_ready)
    return {
        "checks": checks,
        "eligible_for_promotion": bool(all(checks.values())),
        "thresholds": gates,
        "validation_macro_f1_uplift": float(
            validation_metrics["macro_f1"] - validation_phrasebank_metrics["macro_f1"]
        ),
        "validation_balanced_accuracy_uplift": float(
            validation_metrics["balanced_accuracy"]
            - validation_phrasebank_metrics["balanced_accuracy"]
        ),
        "final_macro_f1_uplift": float(
            final_metrics["macro_f1"] - final_phrasebank_metrics["macro_f1"]
        ),
        "phrasebank_macro_f1_degradation": float(phrasebank_macro_f1_degradation),
    }


def _prospective_holdout_status() -> dict[str, Any]:
    """Discover an independently evaluated prospective holdout, if any."""

    root = PROJECT_ROOT / "results" / "finbert-ensemble-prospective"
    reports: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.glob("*/prospective-evaluation-report.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            reports.append(
                {
                    "path": str(path),
                    "eligible_for_promotion": bool(
                        (payload.get("gates") or {}).get("eligible_for_promotion", False)
                    ),
                    "status": payload.get("status", "evaluated"),
                    "final_holdout_year": payload.get("final_holdout_year"),
                }
            )
    return {
        "required": True,
        "evaluated_reports": reports,
        "ready": bool(reports and any(item["eligible_for_promotion"] for item in reports)),
        "next_action": (
            "Prepare, LLM-assist label, bind and evaluate a sealed prospective 2026 cohort."
            if not reports
            else "Review the prospective holdout report and resolve any failed gates."
        ),
    }


def _final_report(
    *,
    parent_run_id: str,
    final_child_run_id: str,
    selected: Mapping[str, Any],
    final_result: Mapping[str, Any],
    evidence_manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    phrasebank_model_directory: Path,
    seeds: Sequence[int],
    text_mode: str = TEXT_MODE_DEFAULT,
) -> dict[str, Any]:
    summaries = final_result.get("fold_summaries") or []
    if len(summaries) != 1 or summaries[0].get("fold") != FINAL_FOLD.name:
        raise DefensibleEnsembleError("Final run did not produce exactly the locked 2025 fold")
    summary = summaries[0]
    selection = summary.get("selection") or {}
    calibration = summary.get("calibration") or {}
    if not calibration or not selection:
        raise DefensibleEnsembleError("Final run lacks calibration evidence; use a fresh run ID")
    bursa_weight = float(selection["selected_bursa_weight"])
    phrasebank_weight = float(selection["selected_phrasebank_weight"])
    oof_path = CHRONOLOGICAL_RESULTS_ROOT / final_child_run_id / FINAL_FOLD.name / "evaluation-oof.csv"
    final_frame = pd.read_csv(oof_path)
    final_frame["published_at"] = pd.to_datetime(final_frame["published_at"], utc=True, errors="coerce")
    if final_frame["published_at"].isna().any():
        raise DefensibleEnsembleError("Final OOF output contains invalid publication dates")
    labels = final_frame["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    bursa = final_frame[
        [f"bursa_probability_{label}" for label in ("negative", "neutral", "positive")]
    ].to_numpy(dtype=float)
    phrasebank = final_frame[
        [f"phrasebank_probability_{label}" for label in ("negative", "neutral", "positive")]
    ].to_numpy(dtype=float)
    if all(
        f"ensemble_probability_{label}" in final_frame.columns
        for label in ("negative", "neutral", "positive")
    ):
        ensemble = final_frame[
            [f"ensemble_probability_{label}" for label in ("negative", "neutral", "positive")]
        ].to_numpy(dtype=float)
    else:
        # Older cached children did not persist the calibrated blend.  Keep
        # the report backward compatible, but fresh remediation runs always
        # take the persisted, calibration-window-fitted probabilities above.
        ensemble = _blend(bursa, phrasebank, bursa_weight)
    ensemble_temperature = float(summary.get("ensemble_temperature", 1.0))
    final_metrics = metrics_from_probabilities("defensible_ensemble_2025", ensemble, labels)
    final_phrasebank_metrics = metrics_from_probabilities("phrasebank_2025", phrasebank, labels)
    final_bursa_metrics = metrics_from_probabilities("bursa_seed_average_2025", bursa, labels)
    bootstrap = _bootstrap_macro_f1_delta_by_block(
        ensemble.argmax(axis=1),
        phrasebank.argmax(axis=1),
        labels,
        _block_ids(final_frame),
        iterations=2000,
    )
    per_company = _company_report(final_frame, ensemble, phrasebank)
    development_report_path = RESULTS_ROOT / parent_run_id / "development-report.json"
    development_report = json.loads(development_report_path.read_text(encoding="utf-8"))
    weight_stability = _fold_weight_coverage(
        bursa_weight, development_report.get("selected_development_report", {}).get("fold_summaries", [])
    )
    _, frozen_bursa, seed_models = _final_seed_probabilities(final_child_run_id, seeds)
    frozen_phrasebank, frozen_labels = _phrasebank_frozen_probabilities(
        phrasebank_model_directory, float(summary["phrasebank_temperature"])
    )
    if frozen_bursa.shape != frozen_phrasebank.shape or len(frozen_labels) != len(frozen_bursa):
        raise DefensibleEnsembleError("Frozen PhraseBank retention probabilities are incompatible")
    frozen_ensemble = _scale_probabilities(
        _blend(frozen_bursa, frozen_phrasebank, bursa_weight), ensemble_temperature
    )
    frozen_ensemble_metrics = metrics_from_probabilities(
        "defensible_ensemble_phrasebank_frozen", frozen_ensemble, frozen_labels
    )
    frozen_phrasebank_metrics = metrics_from_probabilities(
        "phrasebank_frozen", frozen_phrasebank, frozen_labels
    )
    phrasebank_degradation = float(
        frozen_phrasebank_metrics["macro_f1"] - frozen_ensemble_metrics["macro_f1"]
    )
    prospective_holdout = _prospective_holdout_status()
    seed_macro_f1 = []
    for seed in seeds:
        path = CHRONOLOGICAL_STAGING_ROOT / final_child_run_id / FINAL_FOLD.name / f"seed-{seed}" / "chronological-result.npz"
        with np.load(path, allow_pickle=False) as data:
            seed_macro_f1.append(
                float(
                    metrics_from_probabilities(
                        f"bursa_seed_{seed}_2025",
                        np.asarray(data["test_probabilities"], dtype=float),
                        labels,
                    )["macro_f1"]
                )
            )
    seed_std = float(np.std(seed_macro_f1))
    gates = _final_gate_report(
        policy=policy,
        final_metrics=final_metrics,
        final_phrasebank_metrics=final_phrasebank_metrics,
        validation_metrics=calibration["ensemble"],
        validation_phrasebank_metrics=calibration["phrasebank"],
        bootstrap=bootstrap,
        per_company=per_company,
        weight_stability=weight_stability,
        bursa_weight=bursa_weight,
        seed_macro_f1_std=seed_std,
        phrasebank_macro_f1_degradation=phrasebank_degradation,
        test_rows=len(final_frame),
        independent_holdout_ready=bool(prospective_holdout["ready"]),
    )
    bursa_component = {
        "seed_models": [
            {
                **model,
                "temperature": float(temperature),
            }
            for model, temperature in zip(
                seed_models, (summary.get("bursa_seed_average") or {}).get("seed_temperatures") or []
            )
        ]
    }
    if len(bursa_component["seed_models"]) != len(seeds):
        raise DefensibleEnsembleError("Final seed temperatures are incomplete")
    phrasebank_component = {
        "model_directory": _runtime_path(phrasebank_model_directory),
        "temperature": float(summary["phrasebank_temperature"]),
        "artifact_hashes": sentiment_artifact_hashes_for_directory(phrasebank_model_directory),
    }
    variant_id = f"bursa_phrasebank_defensible_soft_vote_{parent_run_id}"
    report = {
        "schema_version": "defensible-finbert-ensemble-v1",
        "run_id": parent_run_id,
        "variant_id": variant_id,
        "status": "complete",
        "evidence_manifest": dict(evidence_manifest),
        "selected_training_recipe": dict(selected),
        "final_child_run_id": final_child_run_id,
        "selection_scope": "2015-2024 chronological development folds only",
        "final_evaluation_scope": "2025 retrospective final test; previously inspected before protocol freeze",
        "promotion_protocol": {
            "kind": "retrospective_locked_2025",
            "manual_promotion_only": True,
            "retrospective_test_disclosure": True,
            "independent_holdout": bool(prospective_holdout["ready"]),
            "independent_holdout_status": prospective_holdout,
            "test_tuning_prohibited_after_manifest": True,
        },
        "calibration_2024": {
            "bursa_seed_average": calibration["bursa_seed_average"],
            "phrasebank": calibration["phrasebank"],
            "ensemble": calibration["ensemble"],
        },
        "selection": dict(selection),
        "final_2025": {
            "rows": int(len(final_frame)),
            "bursa_seed_average": final_bursa_metrics,
            "phrasebank": final_phrasebank_metrics,
            "ensemble": final_metrics,
            "per_company": per_company,
            "bootstrap_macro_f1_delta_vs_phrasebank": bootstrap,
            "seed_macro_f1": seed_macro_f1,
            "seed_macro_f1_std": seed_std,
        },
        "phrasebank_retention": {
            "scope": "Financial PhraseBank frozen test",
            "phrasebank": frozen_phrasebank_metrics,
            "ensemble": frozen_ensemble_metrics,
            "macro_f1_degradation": phrasebank_degradation,
        },
        "weight_stability": weight_stability,
        "independent_prospective_holdout": prospective_holdout,
        "gates": gates,
        "artifact_hashes": {
            "bursa": {
                str(item["seed"]): dict(item["artifact_hashes"])
                for item in bursa_component["seed_models"]
            },
            "phrasebank": dict(phrasebank_component["artifact_hashes"]),
        },
        "runtime_candidate": {
            "backend": "weighted_finbert_ensemble",
            "weights": {"bursa": bursa_weight, "phrasebank": phrasebank_weight},
            "ensemble_temperature": ensemble_temperature,
            "selection_scope": "2015-2024 chronological folds; 2025 is a locked retrospective test",
            "promotion_status": "manual_retrospective_candidate_not_active",
            "benchmark_run_id": parent_run_id,
            "text_mode": text_mode,
            "components": {"bursa": bursa_component, "phrasebank": phrasebank_component},
        },
        "production_changed": False,
        "opt_in_environment_variable": "STOCKSENSE_SENTIMENT_VARIANT",
    }
    return report


def run_defensible_finbert_ensemble(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
    phrasebank_model_directory: Path = DEFAULT_PHRASEBANK_MODEL,
    run_id: str = DEFAULT_RUN_ID,
    phrasebank_replay_shares: Sequence[float] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    profile: str = "full",
    device: str = "cuda",
    learning_rate: float = 2e-5,
    max_epochs: int | None = None,
    embargo_days: int = 7,
    bootstrap_iterations: int = 2000,
    max_runtime_minutes: float | None = 45.0,
    dry_run: bool = False,
    text_mode: str = TEXT_MODE_DEFAULT,
    sampling_mode: str = SAMPLING_MODE_SHUFFLE,
    loss_mode: str = LOSS_MODE_INVERSE_FREQUENCY_CE,
    calibration_mode: str = CALIBRATION_MODE_SCALAR_TEMPERATURE,
    training_recipe_grid: Sequence[Mapping[str, Any]] | None = None,
    storage_mode: str = "all",
    min_free_gib: float = DEFAULT_MIN_FREE_GIB,
) -> dict[str, Any]:
    """Run/resume the locked development-first ensemble protocol."""

    replay_shares = normalize_phrasebank_replay_shares(phrasebank_replay_shares)
    selected_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if len(selected_seeds) < 3:
        raise ValueError("The defensible ensemble requires three fixed Bursa seeds")
    if profile not in {"full", "quick"}:
        raise ValueError("profile must be full or quick")
    runtime_limit = float("inf") if max_runtime_minutes is None else float(max_runtime_minutes)
    if runtime_limit <= 0.0:
        raise ValueError("max_runtime_minutes must be positive")
    if text_mode not in SUPPORTED_TEXT_MODES:
        raise ValueError(f"Unsupported sentiment text mode: {text_mode!r}")
    if sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode!r}")
    if loss_mode not in SUPPORTED_LOSS_MODES:
        raise ValueError(f"Unsupported loss mode: {loss_mode!r}")
    if calibration_mode not in SUPPORTED_CALIBRATION_MODES:
        raise ValueError(f"Unsupported calibration mode: {calibration_mode!r}")
    if storage_mode not in {"all", "sequential"}:
        raise ValueError("storage_mode must be 'all' or 'sequential'")
    if float(min_free_gib) <= 0.0:
        raise ValueError("min_free_gib must be positive")
    recipe_grid = normalize_training_recipe_grid(
        training_recipe_grid,
        text_mode=text_mode,
        sampling_mode=sampling_mode,
        loss_mode=loss_mode,
        calibration_mode=calibration_mode,
    )

    dataset_path = Path(dataset_path)
    policy_path = Path(policy_path)
    phrasebank_model_directory = Path(phrasebank_model_directory)
    policy = load_policy(policy_path)
    evidence = build_evidence_manifest(
        dataset_path, policy_path=policy_path, phrasebank_model_directory=phrasebank_model_directory
    )
    parent_dir = RESULTS_ROOT / str(run_id)
    manifest_path = parent_dir / "evidence-manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        _assert_manifest_matches(prior, evidence)
        evidence = prior
    if dry_run:
        return {
            "status": "dry_run",
            "run_id": str(run_id),
            "evidence_manifest": evidence,
            "phrasebank_replay_shares": list(replay_shares),
            "development_folds": list(DEVELOPMENT_FOLDS),
            "final_fold": FINAL_FOLD.name,
            "training_recipe_grid": list(recipe_grid),
            "storage_mode": storage_mode,
            "min_free_gib": float(min_free_gib),
            "production_changed": False,
        }
    parent_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        _atomic_json(manifest_path, evidence)
    grid_path = parent_dir / "training-recipe-grid.json"
    grid_payload = {"schema_version": "defensible-finbert-training-grid-v1", "recipes": list(recipe_grid)}
    if grid_path.exists():
        if _read_json(grid_path) != grid_payload:
            raise DefensibleEnsembleError(
                "The training recipe grid changed after this run ID was started; use a new run ID."
            )
    else:
        legacy_artifacts = (
            (parent_dir / "development-report.json").exists()
            or (parent_dir / "report.json").exists()
            or any(parent_dir.glob("*-development"))
        )
        if legacy_artifacts:
            raise DefensibleEnsembleError(
                "This run ID contains a pre-grid training result. Use a new run ID so "
                "the recipe search cannot be mixed with older evidence."
            )
        _atomic_json(grid_path, grid_payload)
    # ``None`` is an explicit opt-in for a no-wall-clock-limit run.  The
    # default remains 45 minutes; checkpoints still occur between every child
    # fold/seed and are the resume boundary for the unbounded mode.
    deadline = time.monotonic() + runtime_limit * 60.0

    def remaining_minutes() -> float:
        return max(0.0, deadline - time.monotonic()) / 60.0

    def pause_before(phase: str, **details: Any) -> dict[str, Any]:
        checkpoint = {
            "status": "paused",
            "run_id": str(run_id),
            "phase": phase,
            **details,
            "safe_to_resume": True,
            "production_changed": False,
        }
        _atomic_json(parent_dir / "checkpoint.json", checkpoint)
        return checkpoint

    candidate_summaries: list[dict[str, Any]] = []
    development_reports: dict[str, Any] = {}
    for recipe in recipe_grid:
        for replay_share in replay_shares:
            child_run_id = (
                f"{run_id}-{recipe['id']}-{_replay_key(replay_share)}-development"
            )
            if remaining_minutes() <= 0.0:
                return pause_before(
                    "development",
                    training_recipe=dict(recipe),
                    phrasebank_replay_share=float(replay_share),
                    child_run_id=child_run_id,
                )
            if storage_mode == "sequential" and _free_space_gib() < float(min_free_gib):
                return pause_before(
                    "storage",
                    training_recipe=dict(recipe),
                    phrasebank_replay_share=float(replay_share),
                    child_run_id=child_run_id,
                    free_space_gib=round(_free_space_gib(), 3),
                    minimum_free_gib=float(min_free_gib),
                    storage_mode=storage_mode,
                )
            result = run_chronological_finbert_ensemble(
                dataset_path,
                policy_path=policy_path,
                phrasebank_model_directory=phrasebank_model_directory,
                run_id=child_run_id,
                profile=profile,
                device=device,
                seeds=selected_seeds,
                bursa_replay_share=float(1.0 - replay_share),
                learning_rate=float(learning_rate),
                max_epochs=max_epochs,
                embargo_days=int(embargo_days),
                bootstrap_iterations=int(bootstrap_iterations),
                max_runtime_minutes=remaining_minutes(),
                selected_folds=DEVELOPMENT_FOLDS,
                text_mode=recipe["text_mode"],
                sampling_mode=recipe["sampling_mode"],
                loss_mode=recipe["loss_mode"],
                calibration_mode=recipe["calibration_mode"],
            )
            if result.get("status") != "complete":
                return pause_before(
                    "development",
                    training_recipe=dict(recipe),
                    phrasebank_replay_share=float(replay_share),
                    child_run_id=child_run_id,
                    child_checkpoint=result,
                )
            summary = _development_summary(
                result, phrasebank_replay_share=float(replay_share), child_run_id=child_run_id
            )
            summary.update(dict(recipe))
            if storage_mode == "sequential":
                retention = _remove_development_model_artifacts(child_run_id)
                summary["artifact_retention"] = retention
                result["artifact_retention"] = retention
                _atomic_json(
                    CHRONOLOGICAL_RESULTS_ROOT / str(child_run_id) / "report.json",
                    result,
                )
            candidate_summaries.append(summary)
            development_reports[child_run_id] = result

    selected = select_replay_candidate(candidate_summaries, policy=policy)
    selected_development_report = development_reports[str(selected["child_run_id"])]
    development_payload = {
        "schema_version": "defensible-finbert-ensemble-development-v1",
        "run_id": str(run_id),
        "candidates": candidate_summaries,
        "selected": selected,
        "selected_development_report": selected_development_report,
        "selection_scope": "2015-2024 only; 2025 excluded",
        "training_recipe_grid": list(recipe_grid),
    }
    _atomic_json(parent_dir / "development-report.json", development_payload)

    final_child_run_id = (
        f"{run_id}-{selected['id']}-{_replay_key(float(selected['phrasebank_replay_share']))}-final-2025"
    )
    if remaining_minutes() <= 0.0:
        return pause_before(
            "final_2025", selected=selected, child_run_id=final_child_run_id
        )
    final_result = run_chronological_finbert_ensemble(
        dataset_path,
        policy_path=policy_path,
        phrasebank_model_directory=phrasebank_model_directory,
        run_id=final_child_run_id,
        profile=profile,
        device=device,
        seeds=selected_seeds,
        bursa_replay_share=float(selected["bursa_training_share"]),
        learning_rate=float(learning_rate),
        max_epochs=max_epochs,
        embargo_days=int(embargo_days),
        bootstrap_iterations=int(bootstrap_iterations),
        max_runtime_minutes=remaining_minutes(),
        selected_folds=(FINAL_FOLD.name,),
        text_mode=str(selected["text_mode"]),
        sampling_mode=str(selected["sampling_mode"]),
        loss_mode=str(selected["loss_mode"]),
        calibration_mode=str(selected["calibration_mode"]),
    )
    if final_result.get("status") != "complete":
        return pause_before(
            "final_2025",
            selected=selected,
            child_run_id=final_child_run_id,
            child_checkpoint=final_result,
        )

    report = _final_report(
        parent_run_id=str(run_id),
        final_child_run_id=final_child_run_id,
        selected=selected,
        final_result=final_result,
        evidence_manifest=evidence,
        policy=policy,
        phrasebank_model_directory=phrasebank_model_directory,
        seeds=selected_seeds,
        text_mode=str(selected["text_mode"]),
    )
    report["training_recipe"] = {
        "id": str(selected["id"]),
        "text_mode": str(selected["text_mode"]),
        "sampling_mode": str(selected["sampling_mode"]),
        "loss_mode": str(selected["loss_mode"]),
        "calibration_mode": str(selected["calibration_mode"]),
        "learning_rate": float(learning_rate),
    }
    report["storage"] = {
        "mode": storage_mode,
        "minimum_free_gib": float(min_free_gib),
        "free_space_gib_at_completion": round(_free_space_gib(), 3),
        "development_model_artifacts_retained": storage_mode != "sequential",
        "selected_final_artifacts_retained": True,
    }
    current_evidence = build_evidence_manifest(
        dataset_path, policy_path=policy_path, phrasebank_model_directory=phrasebank_model_directory
    )
    _assert_manifest_matches(evidence, current_evidence)
    if (parent_dir / "checkpoint.json").exists():
        (parent_dir / "checkpoint.json").unlink()
    _atomic_json(parent_dir / "report.json", report)
    _atomic_json(RESULTS_ROOT / "latest.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run/resume the defensible PhraseBank-anchored Bursa FinBERT ensemble experiment."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--phrasebank-model-directory", default=str(DEFAULT_PHRASEBANK_MODEL))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--phrasebank-replay-share",
        dest="phrasebank_replay_shares",
        type=float,
        action="append",
        help="Repeat for each Bursa training mixture; defaults to 0, 0.10, 0.20 and 0.30.",
    )
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--profile", choices=["full", "quick"], default="full")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--embargo-days", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--max-runtime-minutes", type=float, default=45.0)
    parser.add_argument("--text-mode", choices=SUPPORTED_TEXT_MODES, default=TEXT_MODE_DEFAULT)
    parser.add_argument("--sampling-mode", choices=SUPPORTED_SAMPLING_MODES, default=SAMPLING_MODE_SHUFFLE)
    parser.add_argument("--loss-mode", choices=SUPPORTED_LOSS_MODES, default=LOSS_MODE_INVERSE_FREQUENCY_CE)
    parser.add_argument(
        "--calibration-mode", choices=SUPPORTED_CALIBRATION_MODES,
        default=CALIBRATION_MODE_SCALAR_TEMPERATURE,
    )
    parser.add_argument(
        "--training-recipe-grid",
        default=None,
        help="Optional JSON array of development-only text/sampling/loss recipes.",
    )
    parser.add_argument(
        "--storage-mode",
        choices=["all", "sequential"],
        default="all",
        help=(
            "Keep all model artifacts (all) or hash and remove each completed "
            "development child before continuing (sequential)."
        ),
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=DEFAULT_MIN_FREE_GIB,
        help="Pause sequential runs when filesystem free space falls below this GiB.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_defensible_finbert_ensemble(
            Path(args.dataset),
            policy_path=Path(args.policy),
            phrasebank_model_directory=Path(args.phrasebank_model_directory),
            run_id=args.run_id,
            phrasebank_replay_shares=args.phrasebank_replay_shares,
            seeds=tuple(args.seeds) if args.seeds else DEFAULT_SEEDS,
            profile=args.profile,
            device=args.device,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            embargo_days=args.embargo_days,
            bootstrap_iterations=args.bootstrap_iterations,
            max_runtime_minutes=args.max_runtime_minutes,
            dry_run=args.dry_run,
            text_mode=args.text_mode,
            sampling_mode=args.sampling_mode,
            loss_mode=args.loss_mode,
            calibration_mode=args.calibration_mode,
        training_recipe_grid=(
            load_training_recipe_grid(args.training_recipe_grid)
            if args.training_recipe_grid
            else None
        ),
        storage_mode=args.storage_mode,
        min_free_gib=args.min_free_gib,
    )
    except (DefensibleEnsembleError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if result.get("status") in {"complete", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
