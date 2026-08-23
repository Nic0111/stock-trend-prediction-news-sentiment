"""Sealed prospective evaluation for a Bursa/PhraseBank FinBERT challenger.

The historical 2015--2025 experiments are useful for choosing a training
recipe, but they are not an untouched production evaluation.  This module
keeps 2026 evidence separate from the canonical Bursa training corpus and
permits exactly one evaluation of a bound, immutable runtime recipe.

It deliberately does *not* train a model, update the Bursa dataset, change the
runtime configuration, or promote an artifact.  Those actions remain separate
and reviewed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.benchmark_finbert_sentiment import LABEL_TO_ID, metrics_from_probabilities
from pipeline.bursa_news_dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_POLICY_PATH,
    _article_id,
    _content_hash,
    _normalise_url,
    load_policy,
    read_table,
)
from pipeline.chronological_finbert_ensemble import (
    _blend,
    _bootstrap_macro_f1_delta_by_block,
    _company_report,
)
from pipeline.finbert_sentiment_model import (
    AveragedFinBERTEnsemble,
    WeightedFinBERTEnsemble,
    load_finbert_sentiment_model,
)
from pipeline.sentiment_text import TEXT_MODE_DEFAULT, build_sentiment_text
from pipeline.runtime_provenance import sentiment_artifact_hashes_for_directory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "finbert-ensemble-prospective"
DEFAULT_VARIANT = "bursa_phrasebank_soft_vote_v2"
SENTIMENT_LABELS = ("negative", "neutral", "positive")
LLM_ASSISTED_LABEL_STATUSES = {
    "llm_assisted_label",
}


class ProspectiveHoldoutError(RuntimeError):
    """Raised when the sealed prospective protocol cannot be preserved."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProspectiveHoldoutError(f"Unable to read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProspectiveHoldoutError(f"Expected a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    rendered = str(value).strip()
    return "" if rendered.casefold() in {"nan", "none", "null"} else rendered


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value))
    return _clean(value).casefold() in {"1", "true", "yes", "y", "included", "relevant"}


def _candidate_hash(article_id: str) -> int:
    return int(hashlib.sha256(article_id.encode("utf-8")).hexdigest()[:16], 16)


def _policy_settings(policy: Mapping[str, Any]) -> dict[str, Any]:
    configured = dict(
        policy.get("prospective_holdout")
        or (policy.get("sentiment_ensemble") or {}).get("prospective_holdout")
        or {}
    )
    required_tickers = configured.get("required_tickers") or policy.get("required_tickers") or []
    settings = {
        "year": int(configured.get("year", 2026)),
        "required_tickers": tuple(sorted({_clean(item).upper() for item in required_tickers if _clean(item)})),
        "target_candidates_per_ticker": int(configured.get("target_candidates_per_ticker", 25)),
        "shadow_fraction": float(configured.get("shadow_fraction", 0.20)),
        "minimum_final_rows": int(configured.get("minimum_final_rows", 200)),
        "minimum_final_rows_per_ticker": int(configured.get("minimum_final_rows_per_ticker", 15)),
        "minimum_final_class_rows": int(configured.get("minimum_final_class_rows", 30)),
    }
    if not settings["required_tickers"]:
        raise ProspectiveHoldoutError("Prospective holdout policy has no required tickers")
    if settings["target_candidates_per_ticker"] < 1:
        raise ProspectiveHoldoutError("target_candidates_per_ticker must be positive")
    if not 0.0 < settings["shadow_fraction"] < 0.5:
        raise ProspectiveHoldoutError("shadow_fraction must be between zero and 0.5")
    return settings


def _normalise_candidates(frame: pd.DataFrame, settings: Mapping[str, Any]) -> pd.DataFrame:
    data = frame.copy()
    if "published_at" not in data.columns:
        raise ProspectiveHoldoutError("Prospective candidates must contain published_at")
    for column in ("ticker", "company_name", "title", "description", "content_snippet", "url", "language"):
        if column not in data.columns:
            data[column] = ""
    data["ticker"] = data["ticker"].map(_clean).str.upper()
    data["company_name"] = data["company_name"].map(_clean)
    data["title"] = data["title"].map(_clean)
    data["description"] = data["description"].map(_clean)
    data["content_snippet"] = data["content_snippet"].map(_clean)
    data["url"] = data["url"].map(_clean)
    data["language"] = data["language"].map(_clean).str.casefold()
    data["published_at"] = pd.to_datetime(data.get("published_at"), utc=True, errors="coerce")
    data = data.loc[
        data["ticker"].isin(settings["required_tickers"])
        & data["published_at"].notna()
        & (data["published_at"].dt.year == int(settings["year"]))
        & (data["language"].eq("") | data["language"].str.startswith("en"))
        & ((data["title"] != "") | (data["description"] != "") | (data["content_snippet"] != ""))
    ].copy()
    if "automatic_entity_relevant" in data.columns:
        # Candidate discovery may leave this field blank.  Explicit automatic
        # rejections are excluded, while labels still require LLM-assisted review.
        data = data.loc[
            data["automatic_entity_relevant"].map(lambda value: _as_bool(value, default=True))
        ].copy()
    data["article_id"] = data.apply(_article_id, axis=1)
    data["content_sha256"] = data.apply(_content_hash, axis=1)
    data["normalised_url"] = data["url"].map(_normalise_url)
    data["published_at"] = data["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return data


def _without_master_duplicates(candidates: pd.DataFrame, master_dataset_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not master_dataset_path.exists():
        raise ProspectiveHoldoutError(f"Master Bursa dataset is missing: {master_dataset_path}")
    master = read_table(master_dataset_path)
    master["article_id"] = master.apply(_article_id, axis=1)
    master["content_sha256"] = master.apply(_content_hash, axis=1)
    master_urls = {_normalise_url(value) for value in master.get("url", pd.Series(dtype=str))}
    master_ids = set(master["article_id"].astype(str))
    master_hashes = set(master["content_sha256"].astype(str))
    duplicate_mask = (
        candidates["article_id"].isin(master_ids)
        | candidates["content_sha256"].isin(master_hashes)
        | candidates["normalised_url"].isin(master_urls - {""})
    )
    filtered = candidates.loc[~duplicate_mask].copy()
    # A content hash is ticker-specific by design, so retaining URL as a second
    # internal key prevents mirrored links from entering more than once.
    before_internal = len(filtered)
    filtered = filtered.sort_values(["ticker", "published_at", "article_id"], kind="stable")
    filtered = filtered.drop_duplicates(subset=["article_id"], keep="first")
    filtered = filtered.drop_duplicates(subset=["content_sha256"], keep="first")
    filtered = filtered.loc[
        ~filtered["normalised_url"].duplicated(keep="first") | filtered["normalised_url"].eq("")
    ].copy()
    return filtered, {
        "master_duplicates_removed": int(duplicate_mask.sum()),
        "internal_duplicates_removed": int(before_internal - len(filtered)),
    }


def _balanced_candidate_slice(frame: pd.DataFrame, *, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Select a bounded, deterministic domain/month-diverse cohort per ticker."""

    selected: list[pd.DataFrame] = []
    limit = int(settings["target_candidates_per_ticker"])
    for ticker in settings["required_tickers"]:
        group = frame.loc[frame["ticker"] == ticker].copy()
        if group.empty:
            continue
        group["month"] = group["published_at"].str[:7]
        group["domain"] = group.get("source_domain", pd.Series("", index=group.index)).map(_clean).str.casefold()
        group["stable_rank"] = group["article_id"].map(_candidate_hash)
        # First take one candidate from each month/domain cell.  Then fill the
        # remainder deterministically from the unused rows.
        first = (
            group.sort_values(["month", "domain", "stable_rank"], kind="stable")
            .groupby(["month", "domain"], dropna=False, sort=True)
            .head(1)
        )
        first = first.sort_values("stable_rank", kind="stable").head(limit)
        remaining = group.loc[~group.index.isin(first.index)].sort_values("stable_rank", kind="stable")
        picked = pd.concat([first, remaining.head(max(0, limit - len(first)))], ignore_index=False)
        selected.append(picked)
    if not selected:
        return frame.iloc[0:0].copy()
    result = pd.concat(selected, ignore_index=True).drop(columns=["month", "domain", "stable_rank"], errors="ignore")
    return result.sort_values(["ticker", "published_at", "article_id"], kind="stable").reset_index(drop=True)


def _assign_final_and_shadow(frame: pd.DataFrame, *, shadow_fraction: float) -> pd.DataFrame:
    assigned: list[pd.DataFrame] = []
    for _, group in frame.groupby("ticker", sort=True):
        ordered = group.assign(_rank=group["article_id"].map(_candidate_hash)).sort_values("_rank", kind="stable")
        shadow_rows = max(1, int(round(len(ordered) * float(shadow_fraction)))) if len(ordered) >= 5 else 0
        ordered["holdout_partition"] = "final"
        if shadow_rows:
            ordered.iloc[:shadow_rows, ordered.columns.get_loc("holdout_partition")] = "shadow"
        assigned.append(ordered.drop(columns=["_rank"]))
    return pd.concat(assigned, ignore_index=True).sort_values(["ticker", "published_at", "article_id"], kind="stable")


def prepare_prospective_holdout(
    candidates_path: Path | str,
    *,
    output_dir: Path | str,
    policy_path: Path | str = DEFAULT_POLICY_PATH,
    master_dataset_path: Path | str = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """Prepare a 2026 LLM-assisted labelling pack without touching training data."""

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise ProspectiveHoldoutError(
            f"Prospective holdout already exists at {output}; use its status/import commands rather than replacing it."
        )
    policy_file = Path(policy_path)
    candidate_file = Path(candidates_path)
    master_file = Path(master_dataset_path)
    if not candidate_file.exists():
        raise ProspectiveHoldoutError(f"Candidate file is missing: {candidate_file}")
    settings = _policy_settings(load_policy(policy_file))
    raw = read_table(candidate_file)
    normalised = _normalise_candidates(raw, settings)
    deduplicated, duplicate_report = _without_master_duplicates(normalised, master_file)
    selected = _balanced_candidate_slice(deduplicated, settings=settings)
    selected = _assign_final_and_shadow(selected, shadow_fraction=float(settings["shadow_fraction"]))
    if selected.empty:
        raise ProspectiveHoldoutError("No eligible prospective 2026 candidates remain after filtering and deduplication")
    output.mkdir(parents=True, exist_ok=True)
    candidate_output = output / "candidates.csv"
    selected.to_csv(candidate_output, index=False)
    label_columns = [
        "article_id", "ticker", "company_name", "published_at", "holdout_partition", "source_name", "source_domain",
        "title", "description", "content_snippet", "url", "text_source", "recovery_status", "label_evidence_scope",
        "is_pseudo_document", "automatic_entity_status", "automatic_entity_score", "automatic_entity_reasons",
        "llm_assisted_entity_relevant", "llm_assisted_sentiment_label", "label_status", "label_notes",
    ]
    for column in label_columns:
        if column not in selected.columns:
            selected[column] = ""
    template = selected[label_columns].copy()
    template[["llm_assisted_entity_relevant", "llm_assisted_sentiment_label", "label_status", "label_notes"]] = ""
    label_template = output / "llm-assisted-label-template.csv"
    template.to_csv(label_template, index=False)
    counts = {
        "selected_total": int(len(selected)),
        "by_ticker": {ticker: int((selected["ticker"] == ticker).sum()) for ticker in settings["required_tickers"]},
        "by_partition": {name: int(count) for name, count in selected["holdout_partition"].value_counts().items()},
    }
    manifest = {
        "schema_version": "prospective-finbert-holdout-v1",
        "status": "awaiting_llm_assisted_labels",
        "created_at": _now(),
        "policy_path": str(policy_file),
        "policy_sha256": _sha256_file(policy_file),
        "master_dataset_path": str(master_file),
        "master_dataset_sha256": _sha256_file(master_file),
        "source_candidates_path": str(candidate_file),
        "source_candidates_sha256": _sha256_file(candidate_file),
        "settings": settings,
        "candidate_file": str(candidate_output),
        "candidate_sha256": _sha256_file(candidate_output),
        "llm_assisted_label_template": str(label_template),
        "selection": {**counts, **duplicate_report, "normalised_eligible_rows": int(len(normalised))},
        "rules": {
            "training_data_mutation": "prohibited",
            "final_holdout_retraining": "prohibited",
            "evaluation_limit": 1,
            "final_partition_is_sealed_before_labelling": True,
        },
    }
    _atomic_json(manifest_path, manifest)
    return {"prepared": True, "output_dir": str(output), "manifest": manifest}


def _coverage(frame: pd.DataFrame, settings: Mapping[str, Any]) -> dict[str, Any]:
    final = frame.loc[frame["holdout_partition"] == "final"].copy()
    counts = {ticker: int((final["ticker"] == ticker).sum()) for ticker in settings["required_tickers"]}
    class_counts = {label: int((final["sentiment_label"] == label).sum()) for label in SENTIMENT_LABELS}
    ticker_ok = all(count >= int(settings["minimum_final_rows_per_ticker"]) for count in counts.values())
    class_ok = all(count >= int(settings["minimum_final_class_rows"]) for count in class_counts.values())
    return {
        "final_rows": int(len(final)),
        "final_rows_by_ticker": counts,
        "final_rows_by_class": class_counts,
        "minimum_final_rows": int(settings["minimum_final_rows"]),
        "minimum_final_rows_per_ticker": int(settings["minimum_final_rows_per_ticker"]),
        "minimum_final_class_rows": int(settings["minimum_final_class_rows"]),
        "ready": bool(len(final) >= int(settings["minimum_final_rows"]) and ticker_ok and class_ok),
    }


def import_prospective_labels(
    labels_path: Path | str,
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Seal LLM-assisted labels for the pre-assigned final/shadow partitions."""

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    manifest = _read_json(manifest_path)
    if str(manifest.get("status")) not in {"awaiting_llm_assisted_labels", "labels_import_failed"}:
        raise ProspectiveHoldoutError("Prospective labels are already sealed; create a new holdout for a new label set")
    candidate_path = Path(str(manifest.get("candidate_file", "")))
    if not candidate_path.exists() or _sha256_file(candidate_path) != str(manifest.get("candidate_sha256")):
        raise ProspectiveHoldoutError("Candidate cohort changed after preparation; it cannot be used as a sealed holdout")
    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise ProspectiveHoldoutError(f"LLM-assisted label file is missing: {labels_file}")
    candidates = pd.read_csv(candidate_path)
    labels = read_table(labels_file)
    if "article_id" not in labels.columns:
        raise ProspectiveHoldoutError("LLM-assisted labels must contain article_id")
    if labels["article_id"].duplicated().any():
        raise ProspectiveHoldoutError("LLM-assisted labels contain duplicated article_id values")
    label_column = "llm_assisted_sentiment_label" if "llm_assisted_sentiment_label" in labels.columns else "sentiment_label"
    entity_column = "llm_assisted_entity_relevant" if "llm_assisted_entity_relevant" in labels.columns else "entity_relevant"
    status_column = "label_status" if "label_status" in labels.columns else "validation_status"
    required = [label_column, entity_column, status_column]
    missing = [name for name in required if name not in labels.columns]
    if missing:
        raise ProspectiveHoldoutError(
            "LLM-assisted labels must provide entity relevance, sentiment, and label status; missing " + ", ".join(missing)
        )
    selected = labels[["article_id", label_column, entity_column, status_column]].copy()
    selected.columns = ["article_id", "_label", "_entity", "_status"]
    merged = candidates.merge(selected, on="article_id", how="left", validate="one_to_one")
    merged["_label"] = merged["_label"].map(_clean).str.casefold()
    merged["_status"] = merged["_status"].map(_clean).str.casefold().str.replace("-", "_", regex=False)
    merged["_entity"] = merged["_entity"].map(lambda value: _as_bool(value, default=False))
    relevant = merged.loc[merged["_entity"]].copy()
    invalid_labels = relevant.loc[~relevant["_label"].isin(SENTIMENT_LABELS)]
    invalid_status = relevant.loc[~relevant["_status"].isin(LLM_ASSISTED_LABEL_STATUSES)]
    if not invalid_labels.empty or not invalid_status.empty:
        errors: list[str] = []
        if not invalid_labels.empty:
            errors.append(f"{len(invalid_labels)} relevant rows lack a valid sentiment label")
        if not invalid_status.empty:
            errors.append(f"{len(invalid_status)} relevant rows lack an LLM-assisted label status")
        manifest["status"] = "labels_import_failed"
        manifest["last_import_error"] = "; ".join(errors)
        _atomic_json(manifest_path, manifest)
        raise ProspectiveHoldoutError(manifest["last_import_error"])
    relevant["sentiment_label"] = relevant["_label"]
    relevant["label_source"] = "llm_assisted_label"
    relevant["llm_assisted_label"] = True
    relevant["labelled_at"] = _now()
    relevant["llm_assisted_label_status"] = relevant["_status"]
    labelled_path = output / "llm-assisted-labelled.csv"
    relevant.drop(columns=["_label", "_entity", "_status"], errors="ignore").to_csv(labelled_path, index=False)
    settings = dict(manifest.get("settings") or {})
    coverage = _coverage(relevant, settings)
    manifest.update(
        {
            "status": "labels_sealed",
            "labels_imported_at": _now(),
            "llm_assisted_labels_path": str(labels_file),
            "llm_assisted_labels_sha256": _sha256_file(labels_file),
            "labelled_file": str(labelled_path),
            "labelled_sha256": _sha256_file(labelled_path),
            "label_counts": {
                "submitted_rows": int(len(labels)),
                "relevant_rows": int(len(relevant)),
                "excluded_by_llm_assisted_entity_review": int(len(merged) - len(relevant)),
                "coverage": coverage,
            },
        }
    )
    _atomic_json(manifest_path, manifest)
    return {"imported": True, "output_dir": str(output), "coverage": coverage, "ready_for_evaluation": coverage["ready"]}


def bind_prospective_candidate(
    benchmark_report_path: Path | str,
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Bind one immutable runtime recipe before the one-time final evaluation."""

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    manifest = _read_json(manifest_path)
    if str(manifest.get("status")) != "labels_sealed":
        raise ProspectiveHoldoutError("Seal LLM-assisted labels before binding a challenger")
    binding_path = output / "candidate-binding.json"
    if binding_path.exists():
        raise ProspectiveHoldoutError("A challenger is already bound to this holdout")
    labelled_path = Path(str(manifest.get("labelled_file", "")))
    if not labelled_path.exists() or _sha256_file(labelled_path) != str(manifest.get("labelled_sha256")):
        raise ProspectiveHoldoutError("Sealed LLM-assisted labels have changed")
    report_path = Path(benchmark_report_path)
    report = _read_json(report_path)
    development_checks = (report.get("gates") or {}).get("checks") or {}
    required_development_checks = (
        "validation_macro_f1_uplift",
        "validation_balanced_accuracy_uplift",
        "phrasebank_generalisation_noninferiority",
        "bursa_seed_stability",
        "component_non_degeneracy",
        "weight_stable_across_development_folds",
    )
    failed_development_checks = [
        name for name in required_development_checks if not bool(development_checks.get(name))
    ]
    if failed_development_checks:
        raise ProspectiveHoldoutError(
            "Benchmark candidate did not pass its locked development-stage checks: "
            + ", ".join(failed_development_checks)
        )
    candidate = report.get("runtime_candidate")
    if not isinstance(candidate, Mapping):
        raise ProspectiveHoldoutError("Benchmark report does not contain an immutable runtime candidate")
    if str(candidate.get("backend", "")).casefold() != "weighted_finbert_ensemble":
        raise ProspectiveHoldoutError("Only the exact weighted Bursa/PhraseBank ensemble may be prospectively evaluated")
    components = candidate.get("components")
    bursa = components.get("bursa") if isinstance(components, Mapping) else None
    if not isinstance(bursa, Mapping) or len(bursa.get("seed_models") or []) != 3:
        raise ProspectiveHoldoutError("Candidate must use exactly three Bursa seed models before the PhraseBank blend")
    binding = {
        "schema_version": "prospective-finbert-candidate-binding-v1",
        "bound_at": _now(),
        "benchmark_report_path": str(report_path),
        "benchmark_report_sha256": _sha256_file(report_path),
        "variant_id": str(report.get("variant_id", "")),
        "runtime_candidate": dict(candidate),
        "artifact_hashes": report.get("artifact_hashes") or {},
        "training_recipe": report.get("training_recipe") or {},
        "exact_runtime_recipe": "three_bursa_seed_average_then_phrasebank_probability_blend",
        "labels_sha256": _sha256_file(labelled_path),
    }
    _atomic_json(binding_path, binding)
    manifest["status"] = "candidate_bound"
    manifest["candidate_binding"] = str(binding_path)
    _atomic_json(manifest_path, manifest)
    return {"bound": True, "binding": binding, "output_dir": str(output)}


def _load_bound_ensemble(candidate: Mapping[str, Any], *, variant_id: str) -> WeightedFinBERTEnsemble:
    components_config = candidate.get("components")
    weights = candidate.get("weights")
    if not isinstance(components_config, Mapping) or not isinstance(weights, Mapping):
        raise ProspectiveHoldoutError("Bound runtime candidate is incomplete")
    components: dict[str, Any] = {}

    def assert_hashes(model_path: Path, expected: Any, *, label: str) -> None:
        if not isinstance(expected, Mapping) or not expected:
            raise ProspectiveHoldoutError(f"Bound {label} model has no artifact integrity hashes")
        actual = sentiment_artifact_hashes_for_directory(model_path)
        mismatches = [
            str(key) for key, value in expected.items() if actual.get(str(key)) != str(value)
        ]
        if mismatches:
            raise ProspectiveHoldoutError(
                f"Bound {label} model artifacts changed after evaluation binding: " + ", ".join(mismatches)
            )

    for component_name, component in components_config.items():
        if not isinstance(component, Mapping):
            raise ProspectiveHoldoutError(f"Invalid component: {component_name}")
        seed_models = component.get("seed_models")
        if seed_models is not None:
            if not isinstance(seed_models, list) or len(seed_models) != 3:
                raise ProspectiveHoldoutError("Bursa component must contain exactly three fixed seeds")
            seeds: dict[str, Any] = {}
            for seed in seed_models:
                if not isinstance(seed, Mapping) or not seed.get("model_directory"):
                    raise ProspectiveHoldoutError("Bound Bursa seed is missing its model directory")
                name = str(seed.get("seed"))
                if not name or name in seeds:
                    raise ProspectiveHoldoutError("Bound Bursa seed identifiers are invalid")
                model_path = Path(str(seed["model_directory"]))
                if not model_path.is_absolute():
                    model_path = PROJECT_ROOT / model_path
                assert_hashes(model_path, seed.get("artifact_hashes"), label=f"Bursa seed {name}")
                seeds[name] = load_finbert_sentiment_model(
                    str(model_path), temperature=float(seed.get("temperature", 1.0))
                )
            components[str(component_name)] = AveragedFinBERTEnsemble(
                seeds, variant_id=f"{variant_id}:{component_name}:seed-average"
            )
            continue
        model_directory = component.get("model_directory")
        if not model_directory:
            raise ProspectiveHoldoutError(f"Component {component_name} has no model directory")
        model_path = Path(str(model_directory))
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        assert_hashes(model_path, component.get("artifact_hashes"), label=str(component_name))
        components[str(component_name)] = load_finbert_sentiment_model(
            str(model_path), temperature=float(component.get("temperature", 1.0))
        )
    model = WeightedFinBERTEnsemble(
        components,
        {str(name): float(value) for name, value in weights.items()},
        variant_id=variant_id,
        ensemble_temperature=float(candidate.get("ensemble_temperature", 1.0)),
    )
    setattr(model, "sentiment_text_mode", str(candidate.get("text_mode") or TEXT_MODE_DEFAULT))
    return model


def _prospective_gates(
    *,
    policy: Mapping[str, Any],
    coverage: Mapping[str, Any],
    ensemble_metrics: Mapping[str, Any],
    phrasebank_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    per_company: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    gates = dict(((policy.get("sentiment_ensemble") or {}).get("promotion_gates") or {}))
    minimum_class_f1 = float(gates.get("minimum_class_f1", 0.60))
    maximum_company_loss = float(gates.get("maximum_company_macro_f1_loss", 0.05))
    minimum_per_ticker = int(gates.get("minimum_holdout_rows_per_ticker", 15))
    company_rows_ok = all(int(item.get("rows", 0)) >= minimum_per_ticker for item in per_company.values())
    company_noninferior = all(
        float(item.get("macro_f1_delta", -1.0)) >= -maximum_company_loss
        for item in per_company.values()
        if int(item.get("rows", 0)) >= minimum_per_ticker
    )
    origin_report = _read_json(Path(str(binding["benchmark_report_path"])))
    origin_gates = origin_report.get("gates") or {}
    seed_std = float(((origin_report.get("final_2025") or {}).get("seed_macro_f1_std", float("inf"))))
    weight_stability = origin_report.get("weight_stability") or {}
    bursa_weight = float(((binding.get("runtime_candidate") or {}).get("weights") or {}).get("bursa", -1.0))
    checks = {
        "independent_2026_holdout": bool(coverage.get("ready")),
        "final_macro_f1": float(ensemble_metrics["macro_f1"]) >= float(gates.get("minimum_macro_f1", 0.70)),
        "final_balanced_accuracy": float(ensemble_metrics["balanced_accuracy"]) >= float(gates.get("minimum_balanced_accuracy", 0.70)),
        "final_ece": float(ensemble_metrics["ece"]) <= float(gates.get("maximum_ece", 0.10)),
        "minimum_class_f1": min(float(ensemble_metrics[f"{label}_f1"]) for label in SENTIMENT_LABELS) >= minimum_class_f1,
        "bootstrap_macro_f1_noninferiority": float(bootstrap["ci95_lower"]) >= float(gates.get("minimum_bootstrap_ci_lower", 0.0)),
        "per_company_coverage": company_rows_ok,
        "per_company_noninferiority": company_noninferior,
        "component_non_degeneracy": float(gates.get("minimum_component_weight", 0.20)) <= bursa_weight <= float(gates.get("maximum_component_weight", 0.80)),
        "bursa_seed_stability": seed_std <= float(gates.get("maximum_seed_macro_f1_std", 0.02)),
        "weight_stable_across_development_folds": int(weight_stability.get("matching_fold_count", 0)) >= int(gates.get("minimum_matching_fold_count", 3)),
        # The frozen PhraseBank check was carried out during the model's source
        # benchmark.  It must remain passing for a prospective promotion.
        "phrasebank_generalisation_noninferiority": bool((origin_gates.get("checks") or {}).get("phrasebank_generalisation_noninferiority", False)),
    }
    return {
        "checks": checks,
        "thresholds": gates,
        "eligible_for_promotion": bool(all(checks.values())),
        "final_macro_f1_uplift": float(ensemble_metrics["macro_f1"] - phrasebank_metrics["macro_f1"]),
        "final_balanced_accuracy_uplift": float(ensemble_metrics["balanced_accuracy"] - phrasebank_metrics["balanced_accuracy"]),
        "exact_recipe": str(binding.get("exact_runtime_recipe", "")),
        "origin_run_id": origin_report.get("run_id"),
    }


def evaluate_prospective_holdout(*, output_dir: Path | str) -> dict[str, Any]:
    """Evaluate the sealed final partition once, without retraining or promotion."""

    output = Path(output_dir)
    manifest = _read_json(output / "manifest.json")
    if str(manifest.get("status")) != "candidate_bound":
        raise ProspectiveHoldoutError("Seal labels and bind a candidate before final evaluation")
    report_path = output / "prospective-evaluation-report.json"
    if report_path.exists():
        raise ProspectiveHoldoutError("This sealed final holdout has already been evaluated once")
    labelled_path = Path(str(manifest.get("labelled_file", "")))
    binding_path = Path(str(manifest.get("candidate_binding", "")))
    if not labelled_path.exists() or _sha256_file(labelled_path) != str(manifest.get("labelled_sha256")):
        raise ProspectiveHoldoutError("Sealed LLM-assisted label data changed")
    binding = _read_json(binding_path)
    benchmark_path = Path(str(binding.get("benchmark_report_path", "")))
    if not benchmark_path.exists() or _sha256_file(benchmark_path) != str(binding.get("benchmark_report_sha256")):
        raise ProspectiveHoldoutError("Bound benchmark report changed after candidate binding")
    if _sha256_file(labelled_path) != str(binding.get("labels_sha256")):
        raise ProspectiveHoldoutError("Bound labels changed after candidate binding")
    policy_path = Path(str(manifest.get("policy_path", "")))
    if not policy_path.exists() or _sha256_file(policy_path) != str(manifest.get("policy_sha256")):
        raise ProspectiveHoldoutError("Policy changed after holdout preparation")
    policy = load_policy(policy_path)
    settings = dict(manifest.get("settings") or {})
    labelled = pd.read_csv(labelled_path)
    final = labelled.loc[labelled["holdout_partition"] == "final"].copy()
    coverage = _coverage(labelled, settings)
    if not coverage["ready"]:
        raise ProspectiveHoldoutError("Prospective final holdout does not meet its predeclared coverage requirements")
    model = _load_bound_ensemble(binding["runtime_candidate"], variant_id=str(binding.get("variant_id") or DEFAULT_VARIANT))
    text_mode = str(getattr(model, "sentiment_text_mode", TEXT_MODE_DEFAULT))
    texts = [build_sentiment_text(record, mode=text_mode) for record in final.to_dict(orient="records")]
    component_probabilities = model.predict_component_proba(texts)
    ensemble_probabilities = model.blend_component_proba(component_probabilities)
    classes = [str(value).casefold() for value in model.classes_]
    if classes != list(SENTIMENT_LABELS):
        raise ProspectiveHoldoutError("Bound ensemble has an incompatible sentiment class order")
    labels = final["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    phrasebank_probabilities = np.asarray(component_probabilities["phrasebank"], dtype=float)
    metrics = metrics_from_probabilities("prospective_bursa_phrasebank_ensemble", ensemble_probabilities, labels)
    phrasebank_metrics = metrics_from_probabilities("prospective_phrasebank", phrasebank_probabilities, labels)
    bootstrap = _bootstrap_macro_f1_delta_by_block(
        ensemble_probabilities.argmax(axis=1),
        phrasebank_probabilities.argmax(axis=1),
        labels,
        # Articles from one issuer/month are correlated evidence. Bootstrap by
        # issuer-month rather than by row so the non-inferiority interval does
        # not overstate the precision of a burst of related coverage.
        (
            final["ticker"].astype(str).str.upper()
            + "|"
            + pd.to_datetime(final["published_at"], utc=True).dt.strftime("%Y-%m")
        ).to_numpy(),
        iterations=2000,
    )
    per_company = _company_report(final.reset_index(drop=True), ensemble_probabilities, phrasebank_probabilities)
    gates = _prospective_gates(
        policy=policy,
        coverage=coverage,
        ensemble_metrics=metrics,
        phrasebank_metrics=phrasebank_metrics,
        bootstrap=bootstrap,
        per_company=per_company,
        binding=binding,
    )
    probability_frame = final[["article_id", "ticker", "published_at", "sentiment_label"]].copy()
    for index, label in enumerate(SENTIMENT_LABELS):
        probability_frame[f"ensemble_probability_{label}"] = ensemble_probabilities[:, index]
        probability_frame[f"phrasebank_probability_{label}"] = phrasebank_probabilities[:, index]
    probability_path = output / "prospective-final-probabilities.csv"
    probability_frame.to_csv(probability_path, index=False)
    report = {
        "schema_version": "prospective-finbert-evaluation-v1",
        "status": "complete",
        "variant_id": str(binding.get("variant_id") or DEFAULT_VARIANT),
        "evaluated_at": _now(),
        "promotion_protocol": {
            "kind": "prospective_2026_sealed_holdout",
            "independent_holdout": True,
            "labels_sealed": True,
            "evaluation_only_once": True,
            "final_holdout_year": int(settings["year"]),
            "exact_runtime_recipe": "three_bursa_seed_average_then_phrasebank_probability_blend",
            "origin_benchmark_sha256": str(binding["benchmark_report_sha256"]),
        },
        "source_benchmark": {
            "path": str(benchmark_path),
            "sha256": str(binding["benchmark_report_sha256"]),
            "training_recipe": binding.get("training_recipe") or {},
        },
        "runtime_candidate": binding["runtime_candidate"],
        "artifact_hashes": binding.get("artifact_hashes") or {},
        "holdout": coverage,
        "metrics": {"ensemble": metrics, "phrasebank": phrasebank_metrics},
        "bootstrap_macro_f1_delta_vs_phrasebank": bootstrap,
        "per_company": per_company,
        "gates": gates,
        "text_mode": text_mode,
        "probability_file": str(probability_path),
        "production_changed": False,
        "manual_promotion_required": True,
    }
    _atomic_json(report_path, report)
    manifest["status"] = "evaluated"
    manifest["prospective_report"] = str(report_path)
    manifest["prospective_report_sha256"] = _sha256_file(report_path)
    _atomic_json(output / "manifest.json", manifest)
    return report


def prospective_holdout_status(*, output_dir: Path | str) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = _read_json(output / "manifest.json")
    result = {
        "output_dir": str(output),
        "status": manifest.get("status"),
        "selection": manifest.get("selection") or {},
        "coverage": ((manifest.get("label_counts") or {}).get("coverage") or {}),
        "candidate_binding": manifest.get("candidate_binding"),
        "prospective_report": manifest.get("prospective_report"),
    }
    return result
