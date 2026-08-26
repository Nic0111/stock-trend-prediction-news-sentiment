"""Generate row-level diagnostics for the active Fusion v4 evaluation.

The public Fusion report intentionally stores aggregate benchmark metrics and
does not carry every benchmark row.  The Model Performance page also exposes
three explanatory views that need the row-level predictions and evidence
groups.  This module reconstructs those views from the *same* frozen dataset,
selected window, policy and promoted model artifacts, then writes a small
sidecar artifact.  It never retrains a model, changes the registry, or changes
runtime forecast behaviour.

The sidecar is bound to the report and dataset hashes so a stale chart cannot
silently be paired with a newer evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from pipeline.fusion_evaluation_v3 import (
    _capacity_matched_frames,
    _confidence_accuracy_curve,
    _number,
    _subgroup_metrics,
)
from pipeline.fusion_evaluation_v4 import (
    DEFAULT_POLICY,
    load_policy,
    prepare_v4_features,
)
from pipeline.fusion_model import dataset_manifest_hash, load_fusion_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "fusion"
    / "fusion-window-candidates-v4-expanded-v14-auto-corrected.parquet"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "fusion"
    / "fusion-evaluation-v4-expanded-v14-auto-corrected.json"
)
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "fusion" / "active-v4-override-v14"
DEFAULT_OUTPUT = DEFAULT_REPORT.with_name(
    f"{DEFAULT_REPORT.stem}-diagnostics.json"
)


class FusionDiagnosticsError(ValueError):
    """Raised when row-level diagnostics cannot be proven provenance-safe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_reference(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("/", "\\")
    except ValueError:
        return str(path)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _selected_frame(
    frame: pd.DataFrame,
    horizon: int,
    window_days: float,
    half_life_days: float,
) -> pd.DataFrame:
    horizon_values = pd.to_numeric(frame["horizon_days"], errors="coerce")
    window_values = pd.to_numeric(frame["news_window_days"], errors="coerce")
    half_life_values = pd.to_numeric(frame["news_half_life_days"], errors="coerce")
    selected = frame.loc[
        horizon_values.eq(int(horizon))
        & np.isclose(window_values.to_numpy(dtype=float), float(window_days), atol=1e-9)
        & np.isclose(half_life_values.to_numpy(dtype=float), float(half_life_days), atol=1e-9)
    ].copy()
    if selected.empty:
        raise FusionDiagnosticsError(
            f"No rows found for horizon={horizon}, window={window_days}, "
            f"half_life={half_life_days}"
        )
    return selected.reset_index(drop=True)


def _period_frames(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = pd.to_datetime(selected["observation_date"], errors="coerce")
    targets = pd.to_datetime(selected["target_date"], errors="coerce")
    development = selected.loc[
        observations.dt.year.between(2019, 2023)
        & targets.lt(pd.Timestamp("2024-01-01"))
    ].reset_index(drop=True)
    calibration = selected.loc[
        observations.dt.year.eq(2024)
        & targets.lt(pd.Timestamp("2025-01-01"))
    ].reset_index(drop=True)
    benchmark = selected.loc[observations.dt.year.eq(2025)].reset_index(drop=True)
    if min(len(development), len(calibration), len(benchmark)) == 0:
        raise FusionDiagnosticsError(
            "The selected variant does not contain complete 2019-2025 periods"
        )
    return development, calibration, benchmark


def _relationship_state(benchmark: pd.DataFrame) -> pd.Series:
    state = pd.Series("available_other", index=benchmark.index, dtype=object)
    available = _number(benchmark, "news_available").gt(0)
    state.loc[~available] = "unavailable"
    state.loc[
        available & _number(benchmark, "stock_news_agreement", 0.5).ge(0.75)
    ] = "stock_news_aligned"
    state.loc[
        available & _number(benchmark, "stock_news_agreement", 0.5).le(0.25)
    ] = "stock_news_opposed"
    state.loc[_number(benchmark, "news_conflicting_signal").gt(0)] = (
        "conflicting_news"
    )
    return state


def _reliability_state(benchmark: pd.DataFrame) -> pd.Series:
    reliability = _number(benchmark, "news_event_reliability_mean").clip(0.0, 1.0)
    available = _number(benchmark, "news_available").gt(0)
    state = pd.Series("unavailable", index=benchmark.index, dtype=object)
    state.loc[available & reliability.lt(0.40)] = "low"
    state.loc[available & reliability.ge(0.40) & reliability.lt(0.70)] = "medium"
    state.loc[available & reliability.ge(0.70)] = "high"
    return state


def _artifact_path(artifact_dir: Path, horizon: int) -> Path:
    suffix = {1: "1d", 5: "5d", 10: "10d"}.get(int(horizon), f"{int(horizon)}d")
    candidates = sorted(artifact_dir.glob(f"*{suffix}.joblib"))
    if not candidates:
        raise FusionDiagnosticsError(
            f"No active Fusion artifact found for horizon {horizon} in {artifact_dir}"
        )
    # The active override directory is expected to contain one artifact per
    # horizon.  Sorting keeps this deterministic if an operator leaves a
    # historical copy beside the active file.
    return candidates[-1]


def _alias_arm_metrics(groups: dict[str, Any]) -> dict[str, Any]:
    """Add the names consumed by both old and current frontend contracts."""
    for value in groups.values():
        if not isinstance(value, dict):
            continue
        if "full_fusion" in value and "compact_full" not in value:
            value["compact_full"] = value["full_fusion"]
        if "stock_only" in value and "stock_only_meta" not in value:
            value["stock_only_meta"] = value["stock_only"]
    return groups


def generate_fusion_v4_diagnostics(
    dataset_path: Path = DEFAULT_DATASET,
    report_path: Path = DEFAULT_REPORT,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    policy_path: Path = DEFAULT_POLICY,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reconstruct and persist the row-level data needed by Fusion charts."""

    dataset_path = Path(dataset_path)
    report_path = Path(report_path)
    artifact_dir = Path(artifact_dir)
    output_path = Path(output_path)
    policy_path = Path(policy_path)
    if not dataset_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Fusion diagnostics require the active dataset and report")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not str(report.get("schema_version", "")).startswith("stocksense-fusion-evaluation-v4"):
        raise FusionDiagnosticsError("Only a Fusion v4 report can receive this artifact")
    policy, policy_hash = load_policy(policy_path)
    dataset_hash = dataset_manifest_hash(dataset_path)
    report_hash = _sha256_file(report_path)
    if report.get("dataset_sha256") != dataset_hash:
        raise FusionDiagnosticsError(
            "Dataset hash does not match the active Fusion report; refusing stale diagnostics"
        )
    if report.get("policy_sha256") != policy_hash:
        raise FusionDiagnosticsError(
            "Policy hash does not match the active Fusion report; refusing stale diagnostics"
        )

    requested = tuple(int(value) for value in (horizons or policy["supported_horizons"]))
    source = load_fusion_dataset(dataset_path)
    prepared, impact_provenance = prepare_v4_features(
        source, horizons=requested, development_end=2023
    )
    output_horizons: dict[str, Any] = {}

    for horizon in requested:
        key = str(horizon)
        report_horizon = report.get("horizons", {}).get(key)
        if not isinstance(report_horizon, dict):
            raise FusionDiagnosticsError(f"Fusion report has no horizon {horizon}")
        window_days = float(report_horizon["selected_window_days"])
        half_life_days = float(report_horizon["selected_half_life_days"])
        selected = _selected_frame(prepared, horizon, window_days, half_life_days)
        development_raw, calibration_raw, benchmark_raw = _period_frames(selected)
        development, calibration, benchmark, stock_config, _ = _capacity_matched_frames(
            development_raw, calibration_raw, benchmark_raw, policy
        )
        artifact_path = _artifact_path(artifact_dir, horizon)
        payload = joblib.load(artifact_path)
        if not isinstance(payload, dict):
            raise FusionDiagnosticsError(f"Invalid Fusion artifact payload: {artifact_path}")
        if payload.get("dataset_sha256") not in {None, dataset_hash}:
            raise FusionDiagnosticsError(f"Artifact dataset identity mismatch: {artifact_path}")
        if payload.get("policy_sha256") not in {None, policy_hash}:
            raise FusionDiagnosticsError(f"Artifact policy identity mismatch: {artifact_path}")
        if int(payload.get("horizon_days", horizon)) != int(horizon):
            raise FusionDiagnosticsError(f"Artifact horizon mismatch: {artifact_path}")
        model = payload.get("model")
        if model is None or not hasattr(model, "predict"):
            raise FusionDiagnosticsError(f"Artifact has no prediction model: {artifact_path}")

        target = pd.to_numeric(benchmark["target_up"], errors="coerce").astype(int).to_numpy()
        stock_probability = pd.to_numeric(
            benchmark["stock_probability_up"], errors="coerce"
        ).to_numpy(dtype=float)
        fusion_prediction = model.predict(benchmark)
        fusion_probability = np.asarray(fusion_prediction.probability_up, dtype=float)
        threshold = float(report_horizon["shared_decision_threshold"])
        coverage = benchmark.get(
            "news_coverage", pd.Series("Unavailable", index=benchmark.index)
        ).fillna("Unavailable").astype(str)
        relationship = _relationship_state(benchmark)
        reliability = _reliability_state(benchmark)
        coverage_groups = _alias_arm_metrics(
            _subgroup_metrics(
                benchmark, target, stock_probability, fusion_probability, threshold, coverage
            )
        )
        relationship_groups = _alias_arm_metrics(
            _subgroup_metrics(
                benchmark, target, stock_probability, fusion_probability, threshold, relationship
            )
        )
        reliability_groups = _alias_arm_metrics(
            _subgroup_metrics(
                benchmark, target, stock_probability, fusion_probability, threshold, reliability
            )
        )
        stock_curve = _confidence_accuracy_curve(target, stock_probability, threshold)
        fusion_curve = _confidence_accuracy_curve(target, fusion_probability, threshold)
        output_horizons[key] = {
            "benchmark_rows": int(len(benchmark)),
            "news_present_rows": int(_number(benchmark, "news_available").gt(0).sum()),
            "selected_window_days": window_days,
            "selected_half_life_days": half_life_days,
            "stock_meta_configuration": stock_config,
            "subgroups": {
                "news_coverage": coverage_groups,
                "agreement_conflict": relationship_groups,
                "news_reliability": reliability_groups,
            },
            "confidence_accuracy": {
                "stock_only_meta": stock_curve,
                "stock_only": stock_curve,
                "compact_full": fusion_curve,
                "full_fusion": fusion_curve,
            },
            "prediction_provenance": {
                "artifact_path": _project_reference(artifact_path),
                "artifact_sha256": _sha256_file(artifact_path),
                "dataset_sha256": dataset_hash,
                "policy_sha256": policy_hash,
                "threshold": threshold,
                "rows_reconstructed": int(len(benchmark)),
                "observation_year": 2025,
                "target_year_cutoff": 2025,
            },
        }

    output: dict[str, Any] = {
        "schema_version": "fusion-v4-row-diagnostics-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_report_path": _project_reference(report_path),
        "source_report_sha256": report_hash,
        "dataset_path": _project_reference(dataset_path),
        "dataset_sha256": dataset_hash,
        "policy_path": _project_reference(policy_path),
        "policy_sha256": policy_hash,
        "claim_scope": report.get("claim_scope", "Retrospective evidence, 2019-2025"),
        "horizons": output_horizons,
        "provenance": {
            "active_report_schema": report.get("schema_version"),
            "impact_provenance": impact_provenance,
            "reconstruction": "exact selected v4 benchmark rows; no retraining",
        },
    }
    _atomic_json_write(output_path, output)
    return {
        "completed": True,
        "output_path": _project_reference(output_path),
        "schema_version": output["schema_version"],
        "dataset_sha256": dataset_hash,
        "source_report_sha256": report_hash,
        "horizons": {
            key: {
                "benchmark_rows": value["benchmark_rows"],
                "news_present_rows": value["news_present_rows"],
                "coverage_groups": len(value["subgroups"]["news_coverage"]),
                "reliability_groups": len(value["subgroups"]["news_reliability"]),
                "confidence_buckets": len(value["confidence_accuracy"]["stock_only"]),
            }
            for key, value in output_horizons.items()
        },
    }


__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_DATASET",
    "DEFAULT_OUTPUT",
    "DEFAULT_POLICY",
    "DEFAULT_REPORT",
    "FusionDiagnosticsError",
    "generate_fusion_v4_diagnostics",
]
