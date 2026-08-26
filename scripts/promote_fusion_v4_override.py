"""Materialise and promote the user-authorised Fusion v4/v14 override.

This command does not change the v14 scientific report or its failed gates.
It creates the missing runtime event-impact models and trailing baseline
state, validates the candidate registry through the normal runtime loader,
backs up the current production registry, and only then activates v14.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.fusion_evaluation_v4 import (
    DEFAULT_POLICY,
    IMPACT_INPUTS,
    V4_FEATURES,
    TargetSpecificEventImpact,
    _read_json,
    load_policy,
    prepare_v4_features,
)
from pipeline.fusion_model import load_fusion_dataset
from pipeline.fusion_runtime import load_accepted_fusion_artifact
DEFAULT_REPORT = ROOT / "results/fusion/fusion-evaluation-v4-expanded-v14-auto-corrected.json"
DEFAULT_SOURCE_REGISTRY = ROOT / "models/fusion/staged-fusion-v4-expanded-v14-auto-corrected-registry.json"
DEFAULT_DATASET = ROOT / "data/fusion/fusion-window-candidates-v4-expanded-v14-auto-corrected.parquet"
DEFAULT_ACTIVE_REGISTRY = ROOT / "models/fusion/fusion-registry.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_reference(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("/", "\\")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _selected_rows(prepared: pd.DataFrame, horizon: int, value: dict[str, Any]) -> pd.DataFrame:
    mask = pd.to_numeric(prepared["horizon_days"], errors="coerce").eq(int(horizon))
    window = float(value["selected_window_days"])
    half_life = float(value["selected_half_life_days"])
    mask &= np.isclose(pd.to_numeric(prepared["news_window_days"], errors="coerce"), window)
    mask &= np.isclose(pd.to_numeric(prepared["news_half_life_days"], errors="coerce"), half_life)
    result = prepared.loc[mask].copy().reset_index(drop=True)
    if result.empty:
        raise RuntimeError(f"No prepared rows match the v14 selected variant for {horizon}D")
    return result


def _development(rows: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(rows["observation_date"], errors="coerce")
    targets = pd.to_datetime(rows["target_date"], errors="coerce")
    return rows.loc[
        dates.dt.year.between(2019, 2023)
        & targets.lt(pd.Timestamp("2024-01-01"))
    ].reset_index(drop=True)


def _runtime_state(rows: pd.DataFrame, horizon: int, window: float, half_life: float) -> dict[str, Any]:
    """Persist the latest trailing 252-session baselines per company."""

    state: dict[str, Any] = {
        "schema_version": "fusion-runtime-state-v4",
        "horizon_days": int(horizon),
        "selected_window_days": float(window),
        "selected_half_life_days": float(half_life),
        "baseline_lookback_sessions": 252,
        "tickers": {},
    }
    for ticker, group in rows.groupby(rows["ticker"].astype(str), sort=True):
        ordered = group.sort_values("observation_date", kind="stable").copy()
        event = pd.to_numeric(ordered["v4_log_event_count"], errors="coerce").fillna(0.0)
        tone = pd.to_numeric(ordered["news_weighted_sentiment"], errors="coerce").fillna(0.0)
        volume_baseline = event.shift(1).rolling(252, min_periods=20).mean()
        tone_baseline = tone.shift(1).rolling(252, min_periods=20).mean()
        volume_value = float(volume_baseline.iloc[-1]) if pd.notna(volume_baseline.iloc[-1]) else 0.0
        tone_value = float(tone_baseline.iloc[-1]) if pd.notna(tone_baseline.iloc[-1]) else 0.0
        state["tickers"][str(ticker)] = {
            "volume_baseline_mean": volume_value,
            "tone_baseline_mean": tone_value,
            "surprise_baseline_ready": True,
            "baseline_rows": int(len(ordered)),
            "state_as_of": str(pd.to_datetime(ordered["observation_date"]).max().date()),
        }
    return state


def _augment_artifact(
    source_path: Path,
    output_path: Path,
    *,
    event_impact_model: TargetSpecificEventImpact,
    horizon: int,
    policy_hash: str,
    dataset_hash: str,
    selected_window: float,
    selected_half_life: float,
) -> str:
    payload = joblib.load(source_path)
    if not isinstance(payload, dict) or payload.get("schema_version") != "fusion-model-artifact-v4":
        raise RuntimeError(f"Unexpected v14 artifact schema: {source_path}")
    if list(payload.get("feature_names") or []) != list(V4_FEATURES):
        raise RuntimeError(f"Unexpected v14 feature contract: {source_path}")
    payload["event_impact_model"] = event_impact_model
    payload["runtime_contract"] = {
        "schema_version": "fusion-runtime-contract-v4",
        "feature_names": list(V4_FEATURES),
        "event_impact_inputs": list(IMPACT_INPUTS),
        "baseline_lookback_sessions": 252,
        "selected_window_days": float(selected_window),
        "selected_half_life_days": float(selected_half_life),
        "no_news_exact_stock_fallback": True,
        "point_in_time": True,
        "scientific_gates_passed": False,
    }
    payload["runtime_promotion"] = {
        "source_artifact": project_reference(source_path),
        "source_artifact_sha256": sha256(source_path),
        "override": True,
        "reason": "User-authorised v14 production override",
        "horizon_days": int(horizon),
        "policy_sha256": policy_hash,
        "dataset_sha256": dataset_hash,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    return sha256(output_path)


def promote(
    report_path: Path = DEFAULT_REPORT,
    source_registry_path: Path = DEFAULT_SOURCE_REGISTRY,
    dataset_path: Path = DEFAULT_DATASET,
    policy_path: Path = DEFAULT_POLICY,
    active_registry_path: Path = DEFAULT_ACTIVE_REGISTRY,
) -> dict[str, Any]:
    report = _read_json(report_path)
    source_registry = _read_json(source_registry_path)
    if report.get("schema_version") != "stocksense-fusion-evaluation-v4":
        raise RuntimeError("The supplied report is not a Fusion v4 report")
    if source_registry.get("schema_version") != "fusion-registry-v6":
        raise RuntimeError("The supplied registry is not a Fusion v6 registry")
    policy, policy_hash = load_policy(policy_path)
    if policy_hash != str(report.get("policy_sha256") or ""):
        raise RuntimeError("v14 report and policy hash do not match")
    if sha256(dataset_path) != str(report.get("dataset_sha256") or ""):
        raise RuntimeError("v14 report and dataset hash do not match")

    print("Preparing v4 point-in-time features for runtime materialisation...", flush=True)
    source = load_fusion_dataset(dataset_path)
    prepared, _ = prepare_v4_features(source, horizons=policy["supported_horizons"])

    output_dir = ROOT / "models/fusion/active-v4-override-v14"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_payload: dict[str, Any] = {
        "schema_version": "fusion-runtime-state-v4",
        "dataset_sha256": report["dataset_sha256"],
        "policy_sha256": policy_hash,
        "source_report": project_reference(report_path),
        "state_generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "baseline_lookback_sessions": 252,
        "horizons": {},
    }
    registry = json.loads(json.dumps(source_registry))
    registry["status"] = "override_accepted"
    registry["overall_status"] = "acceptance_overridden"
    registry["overall_all_horizon_accepted"] = True
    registry["runtime_action"] = "enable_fusion_override"
    registry["runtime_feature_contract_ready"] = True
    registry["runtime_note"] = (
        "Fusion v4 is enabled by explicit operator override. Scientific acceptance "
        "gates remain recorded as failed in the retrospective report."
    )
    registry["accepted_horizons"] = [1, 5, 10]
    registry["promotion_note"] = (
        "User-authorised v14 runtime override; report remains retrospective and "
        "scientific gates are not passed."
    )

    for horizon_text, report_horizon in (report.get("horizons") or {}).items():
        horizon = int(horizon_text)
        selected_rows = _selected_rows(prepared, horizon, report_horizon)
        development = _development(selected_rows)
        event_model = TargetSpecificEventImpact().fit(development)
        if not event_model.ready:
            raise RuntimeError(f"v4 event-impact model is not ready for {horizon}D")
        window = float(report_horizon["selected_window_days"])
        half_life = float(report_horizon["selected_half_life_days"])
        state_payload["horizons"][str(horizon)] = {
            "window_days": window,
            "half_life_days": half_life,
            **_runtime_state(selected_rows, horizon, window, half_life),
        }
        source_artifact = ROOT / str(report_horizon["artifact_path"])
        target_artifact = output_dir / source_artifact.name
        artifact_hash = _augment_artifact(
            source_artifact,
            target_artifact,
            event_impact_model=event_model,
            horizon=horizon,
            policy_hash=policy_hash,
            dataset_hash=report["dataset_sha256"],
            selected_window=window,
            selected_half_life=half_life,
        )
        entry = dict((registry.get("horizons") or {}).get(str(horizon)) or {})
        entry.update(
            {
                "accepted": True,
                "scientific_accepted": False,
                "acceptance_overridden": True,
                "artifact_path": project_reference(target_artifact),
                "artifact_sha256": artifact_hash,
                "runtime_label": "Fusion-supported",
                "validation_status": "accepted",
                "runtime_enabled": True,
                "runtime_contract": "fusion-runtime-contract-v4",
                "event_impact_training_rows": int(event_model.training_rows),
            }
        )
        registry.setdefault("horizons", {})[str(horizon)] = entry

    state_path = ROOT / "models/fusion/fusion-v4-runtime-state-v14-override.json"
    atomic_json(state_path, state_payload)
    registry["runtime_state_path"] = project_reference(state_path)
    registry["runtime_state_sha256"] = sha256(state_path)
    registry["acceptance_override"] = {
        "active": True,
        "operator_authorized": True,
        "reason": "User-authorised production override of Final v14 evaluation",
        "overridden_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_registry": project_reference(source_registry_path),
        "source_registry_sha256": sha256(source_registry_path),
        "evaluation_report": project_reference(report_path),
        "evaluation_report_sha256": sha256(report_path),
        "scientific_gates_passed": False,
        "runtime_contract": "fusion-runtime-contract-v4",
    }

    # Validate the candidate through the exact loader before changing active state.
    candidate_registry = ROOT / "models/fusion/fusion-registry-v6-v14-override-candidate.json"
    atomic_json(candidate_registry, registry)
    for horizon in (1, 5, 10):
        artifact, reason = load_accepted_fusion_artifact(horizon, candidate_registry)
        if artifact is None:
            raise RuntimeError(f"Candidate v4 artifact failed runtime validation for {horizon}D: {reason}")
        if getattr(artifact, "horizon_days", None) != horizon:
            raise RuntimeError(f"Candidate v4 artifact horizon mismatch for {horizon}D")

        stock = SimpleNamespace(probability_up=0.55, decision_threshold=0.5)
        no_news = pd.DataFrame(
            [{"stock_probability_up": 0.55, "news_available": 0.0, **{feature: 0.0 for feature in V4_FEATURES}}]
        )
        full = artifact.predict_proba(no_news)[0, 1]
        if not np.isclose(full, 0.55, atol=1e-12):
            raise RuntimeError(f"No-news exact stock fallback failed for {horizon}D")

    if active_registry_path.is_file():
        backup = active_registry_path.with_name(
            f"fusion-registry-v5-before-v14-override-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )
        shutil.copy2(active_registry_path, backup)
        registry["previous_active_registry_backup"] = project_reference(backup)
    atomic_json(candidate_registry, registry)
    shutil.copy2(candidate_registry, active_registry_path)
    return {
        "active_registry": project_reference(active_registry_path),
        "candidate_registry": project_reference(candidate_registry),
        "runtime_state": project_reference(state_path),
        "artifact_directory": project_reference(output_dir),
        "active_registry_sha256": sha256(active_registry_path),
        "horizons": {
            key: {
                "artifact": value.get("artifact_path"),
                "artifact_sha256": value.get("artifact_sha256"),
                "window_days": value.get("selected_window_days"),
                "half_life_days": value.get("selected_half_life_days"),
            }
            for key, value in (registry.get("horizons") or {}).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--active-registry", type=Path, default=DEFAULT_ACTIVE_REGISTRY)
    args = parser.parse_args()
    result = promote(
        args.report,
        args.source_registry,
        args.dataset,
        args.policy,
        args.active_registry,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
