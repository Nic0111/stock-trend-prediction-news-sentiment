"""Fail-closed validation shared by inference, readiness and system audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from pipeline.model_artifacts import CalibratedStockModel
from pipeline.runtime_provenance import (
    FINBERT_MODEL_DIR,
    PROJECT_ROOT,
    SENTIMENT_RUNTIME_CONFIG_FILE,
    sentiment_artifact_hashes,
    sentiment_artifact_hashes_for_directory,
    sha256_file,
)
from pipeline.stock_features import FEATURE_SCHEMA_VERSION


def validate_stock_artifact(
    model: Any,
    entry: Mapping[str, Any],
    *,
    horizon_days: int,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(model, CalibratedStockModel):
        return ["artifact is not a CalibratedStockModel"]
    expected = {
        "horizon_days": int(horizon_days),
        "refinement_round": int(entry.get("refinement_round", -1)),
        "model_name": str(entry.get("model_name", "")),
    }
    for field, value in expected.items():
        if getattr(model, field, None) != value:
            errors.append(f"{field} does not match the registry")
    if list(model.input_columns) != list(entry.get("input_columns") or []):
        errors.append("input_columns do not match the registry")
    if list(model.numeric_feature_columns) != list(entry.get("numeric_feature_columns") or []):
        errors.append("numeric_feature_columns do not match the registry")
    threshold = float(getattr(model, "threshold", -1.0))
    if not 0.0 < threshold < 1.0:
        errors.append("decision threshold is outside (0, 1)")
    expected_threshold = entry.get("decision_threshold", entry.get("threshold"))
    if expected_threshold is not None and abs(threshold - float(expected_threshold)) > 1e-12:
        errors.append("decision threshold does not match the registry")
    # ``threshold_constraints_passed`` in the registry is the nested OOF
    # constraint result.  The persisted model flag is the separate
    # deployment-threshold result produced by ``select_directional_threshold``.
    # Comparing those two fields makes every otherwise valid promoted artifact
    # fail the runtime integrity check (the two checks intentionally answer
    # different questions).  Compare the deployment result only when newer
    # registries explicitly persist it; older registries remain valid under
    # their existing provenance contract.
    expected_constraints = entry.get("deployment_threshold_constraints_passed")
    if expected_constraints is not None and bool(model.threshold_constraints_passed) != bool(
        expected_constraints
    ):
        errors.append("threshold constraint status does not match the registry")
    return errors


def stock_registry_integrity(
    registry: Mapping[str, Any],
    *,
    load_models: bool = False,
) -> dict[str, Any]:
    import joblib

    missing_provenance: list[str] = []
    mismatches: list[str] = []
    if registry.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        mismatches.append("stock-feature-schema")
    availability: dict[str, bool] = {}
    for horizon in (1, 5, 10):
        name = f"stock-{horizon}d"
        entry = (registry.get("horizons") or {}).get(str(horizon), {})
        reference = str(entry.get("artifact_path", "")).strip()
        path = PROJECT_ROOT / reference if reference else PROJECT_ROOT / "__missing__"
        expected_hash = str(entry.get("artifact_sha256") or "").strip()
        valid = bool(entry and reference and path.is_file() and expected_hash)
        if not expected_hash:
            missing_provenance.append(name)
        elif not path.is_file() or sha256_file(path) != expected_hash:
            mismatches.append(name)
            valid = False
        if valid and load_models:
            try:
                errors = validate_stock_artifact(
                    joblib.load(path), entry, horizon_days=horizon
                )
            except Exception:
                errors = ["artifact could not be loaded"]
            if errors:
                mismatches.append(name + "-contract")
                valid = False
        availability[str(horizon)] = valid
    return {
        "ok": not missing_provenance and not mismatches,
        "availability": availability,
        "missing_provenance": sorted(set(missing_provenance)),
        "mismatches": sorted(set(mismatches)),
    }


def sentiment_integrity() -> dict[str, Any]:
    errors: list[str] = []
    try:
        runtime_config = json.loads(SENTIMENT_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"ok": False, "errors": ["finbert-runtime-config"]}
    selected_variant = os.getenv("STOCKSENSE_SENTIMENT_VARIANT", "").strip()
    if selected_variant and selected_variant != "finbert_phrasebank_v1":
        challenger = (runtime_config.get("challengers") or {}).get(selected_variant)
        if not isinstance(challenger, Mapping):
            return {
                "ok": False,
                "errors": ["unknown-sentiment-challenger"],
                "variant": selected_variant,
            }
        challenger_errors: list[str] = []
        for component_name, component in (challenger.get("components") or {}).items():
            if not isinstance(component, Mapping) or not component.get("model_directory"):
                challenger_errors.append(f"{component_name}-configuration")
                continue
            directory = Path(str(component["model_directory"]))
            if not directory.is_absolute():
                directory = PROJECT_ROOT / directory
            expected_component = {
                str(Path(path)): str(digest)
                for path, digest in dict(component.get("artifact_hashes") or {}).items()
            }
            actual_component = {
                str(Path(path)): digest
                for path, digest in sentiment_artifact_hashes_for_directory(directory).items()
            }
            if not expected_component or set(expected_component) != set(actual_component):
                challenger_errors.append(f"{component_name}-file-set")
            elif any(actual_component[path] != digest for path, digest in expected_component.items()):
                challenger_errors.append(f"{component_name}-provenance")
        return {
            "ok": not challenger_errors,
            "errors": sorted(set(challenger_errors)),
            "variant": selected_variant,
            "components": list((challenger.get("components") or {}).keys()),
        }
    expected = {
        str(Path(path)): str(digest)
        for path, digest in dict(runtime_config.get("artifact_hashes") or {}).items()
    }
    actual = {str(Path(path)): digest for path, digest in sentiment_artifact_hashes().items()}
    if not expected or set(expected) != set(actual):
        errors.append("finbert-file-set")
    elif any(actual[path] != digest for path, digest in expected.items()):
        errors.append("finbert-provenance")
    required = {
        str(Path("models/experimental/finbert-sentiment") / name)
        for name in (
            "config.json",
            "metadata.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "model.safetensors",
        )
    }
    normalized_actual = {path.replace("\\", "/") for path in actual}
    if {path.replace("\\", "/") for path in required} - normalized_actual:
        errors.append("finbert-required-files")
    try:
        config = json.loads((FINBERT_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        metadata = json.loads((FINBERT_MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        labels = [
            str(config.get("id2label", {}).get(str(index), "")).casefold()
            for index in range(3)
        ]
        class_order = [str(value).casefold() for value in metadata.get("class_order", [])]
        if labels != class_order or set(labels) != {"negative", "neutral", "positive"}:
            errors.append("finbert-label-contract")
        if int(metadata.get("max_length", 0)) < 1:
            errors.append("finbert-max-length")
    except (OSError, ValueError, TypeError):
        errors.append("finbert-metadata-contract")
    return {"ok": not errors, "errors": sorted(set(errors)), "signature": actual}
