"""Fail-closed staging helpers for Fusion v3 runtime promotion.

This module intentionally does not pretend that an offline model is ready for
online inference.  It materialises the trailing state needed by surprise
features and promotes only when an independently generated runtime-parity
report proves that the online NewsEvidence adapter reproduces the frozen
offline feature contract.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipeline.fusion_evaluation_v2 import _read_json
from pipeline.fusion_evaluation_v3 import FULL_FEATURES, _enhance_features
from pipeline.fusion_model import load_fusion_dataset
from pipeline.download_financial_news_data import NewsContextResult
from pipeline.fusion_runtime import build_runtime_fusion_frame
from pipeline.predict_stock_trend import PredictionResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVE_REGISTRY = PROJECT_ROOT / "models" / "fusion" / "fusion-registry.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def prepare_fusion_v3_acceptance_override(
    staged_registry_path: Path,
    evaluation_report_path: Path,
    *,
    output_path: Path,
    reason: str,
) -> dict[str, Any]:
    """Create an explicit, auditable runtime-enabled override registry.

    The normal v3 registry is deliberately fail-closed: failed acceptance
    checks leave every horizon stock-led.  This helper does not rewrite the
    scientific report or claim that any gate passed.  It creates a separate
    registry whose enabled horizons are clearly marked as an operator-
    authorised exception, while retaining the original gate outcomes and
    report hash for later review.
    """

    registry = _read_json(staged_registry_path)
    report = _read_json(evaluation_report_path)
    if registry.get("schema_version") != "fusion-registry-v5":
        raise ValueError("Only Fusion v3 registries can receive an acceptance override")
    if report.get("schema_version") != "stocksense-fusion-evaluation-v3":
        raise ValueError("The override must reference a Fusion v3 evaluation report")
    reason_text = str(reason or "").strip()
    if not reason_text:
        raise ValueError("An explicit reason is required for an acceptance override")

    original_horizons = [int(value) for value in registry.get("accepted_horizons") or []]
    horizon_entries = dict(registry.get("horizons") or {})
    if not horizon_entries:
        raise ValueError("The staged registry has no horizon artifacts to enable")

    enabled_horizons: list[int] = []
    failed_checks: dict[str, Any] = {}
    for key, raw_entry in sorted(horizon_entries.items(), key=lambda item: int(item[0])):
        horizon = int(key)
        entry = dict(raw_entry or {})
        if not str(entry.get("artifact_path") or "").strip():
            raise ValueError(f"Fusion v3 horizon {horizon} has no artifact path")
        artifact_path = Path(str(entry["artifact_path"]))
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        if not artifact_path.is_file():
            raise ValueError(f"Fusion v3 horizon {horizon} artifact is missing: {artifact_path}")
        entry["accepted"] = True
        entry["validation_status"] = "accepted"
        entry["acceptance_gate_overridden"] = True
        entry["runtime_label"] = "Fusion-supported (acceptance override)"
        horizon_entries[str(horizon)] = entry
        enabled_horizons.append(horizon)
        report_entry = dict((report.get("horizons") or {}).get(str(horizon)) or {})
        checks = dict(report_entry.get("acceptance_checks") or {})
        failed_checks[str(horizon)] = {
            "acceptance_checks": checks,
            "failed_checks": sorted(name for name, passed in checks.items() if not bool(passed)),
            "runtime_label_before_override": report_entry.get("runtime_label"),
        }

    source_registry_sha256 = _sha256(staged_registry_path)
    report_sha256 = _sha256(evaluation_report_path)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    output = dict(registry)
    output["horizons"] = horizon_entries
    output["accepted_horizons"] = enabled_horizons
    output["gate_accepted_horizons"] = original_horizons
    output["status"] = "override_staged"
    output["overall_status"] = "acceptance_overridden"
    output["overall_all_horizon_accepted"] = False
    output["runtime_action"] = "enable_fusion_override"
    output["runtime_enabled_horizons"] = enabled_horizons
    output["acceptance_override"] = {
        "active": True,
        "operator_authorized": True,
        "reason": reason_text,
        "overridden_at": now,
        "source_registry": _reference(staged_registry_path),
        "source_registry_sha256": source_registry_sha256,
        "evaluation_report": _reference(evaluation_report_path),
        "evaluation_report_sha256": report_sha256,
        "original_status": registry.get("status"),
        "original_overall_status": registry.get("overall_status"),
        "original_accepted_horizons": original_horizons,
        "failed_horizon_checks": failed_checks,
        "warning": (
            "Runtime fusion is enabled by explicit operator override. The frozen "
            "v3 acceptance gates did not pass; this is retrospective research output."
        ),
    }
    _atomic_json(output_path, output)
    return output


def materialize_runtime_surprise_state(
    dataset_path: Path,
    staged_registry_path: Path,
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Persist the 252-session histories required for online surprise features."""

    registry = _read_json(staged_registry_path)
    if registry.get("schema_version") != "fusion-registry-v5":
        raise ValueError("Runtime surprise state requires a fusion-registry-v5 input")
    accepted = [int(value) for value in registry.get("accepted_horizons") or []]
    if not accepted:
        raise ValueError("No Fusion v3 horizon is accepted; runtime state is unnecessary")
    dataset_hash = _sha256(dataset_path)
    if dataset_hash != str(registry.get("dataset_sha256") or ""):
        raise ValueError("The staged registry does not match the supplied Fusion v3 dataset")
    frame = _enhance_features(load_fusion_dataset(dataset_path))
    horizons: dict[str, Any] = {}
    for horizon in accepted:
        entry = dict((registry.get("horizons") or {}).get(str(horizon)) or {})
        window = float(entry.get("selected_window_days", 0.0))
        half_life = float(entry.get("selected_half_life_days", 0.0))
        rows = frame.loc[
            pd.to_numeric(frame["horizon_days"], errors="coerce").eq(horizon)
            & pd.to_numeric(frame["news_window_days"], errors="coerce").eq(window)
            & pd.to_numeric(frame["news_half_life_days"], errors="coerce").eq(half_life)
        ].copy()
        if rows.empty:
            raise ValueError(f"No rows match the accepted {horizon}-day window")
        ticker_state: dict[str, Any] = {}
        for ticker, ticker_rows in rows.groupby("ticker", sort=True):
            ordered = ticker_rows.sort_values("observation_date").tail(252)
            volume = np.log1p(
                pd.to_numeric(
                    ordered["news_independent_event_count"], errors="coerce"
                ).fillna(0.0)
            )
            tone = pd.to_numeric(
                ordered["news_weighted_sentiment"], errors="coerce"
            ).fillna(0.0)
            ticker_state[str(ticker)] = {
                "observation_dates": pd.to_datetime(
                    ordered["observation_date"]
                ).dt.strftime("%Y-%m-%d").tolist(),
                "news_log_event_count": volume.astype(float).tolist(),
                "news_weighted_sentiment": tone.astype(float).tolist(),
                "rows": len(ordered),
                "surprise_baseline_ready": len(ordered) >= 20,
                "volume_baseline_mean": float(volume.mean()) if len(volume) else 0.0,
                "tone_baseline_mean": float(tone.mean()) if len(tone) else 0.0,
            }
        horizons[str(horizon)] = {
            "selected_window_days": window,
            "selected_half_life_days": half_life,
            "tickers": ticker_state,
        }
    payload = {
        "schema_version": "fusion-v3-runtime-surprise-state-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_sha256": dataset_hash,
        "policy_sha256": registry.get("policy_sha256"),
        "sentiment_config_sha256": (
            registry.get("provenance") or {}
        ).get("sentiment_config_sha256"),
        "rolling_sessions": 252,
        "accepted_horizons": accepted,
        "horizons": horizons,
        # Offline state alone is insufficient. The online adapter must still
        # pass a row-level parity replay before promotion is permitted.
        "runtime_feature_contract_ready": False,
        "required_next_check": "online_news_evidence_feature_parity",
    }
    _atomic_json(output_path, payload)
    return payload


def run_runtime_feature_parity(
    dataset_path: Path,
    staged_registry_path: Path,
    runtime_state_path: Path,
    *,
    output_path: Path,
    sample_per_ticker: int = 10,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Replay frozen rows through the live feature adapter before promotion.

    This is a deterministic contract check, not a new performance estimate.
    It verifies that every v3 feature is present, that available-news values
    match the frozen feature table within the declared tolerance, and that an
    unavailable-news row remains an exact stock-only input.
    """

    registry = _read_json(staged_registry_path)
    if registry.get("schema_version") != "fusion-registry-v5":
        raise ValueError("Runtime parity requires a fusion-registry-v5 input")
    accepted = [int(value) for value in registry.get("accepted_horizons") or []]
    if not accepted:
        raise ValueError("No accepted horizon is available for runtime parity")
    dataset_hash = _sha256(dataset_path)
    if dataset_hash != str(registry.get("dataset_sha256") or ""):
        raise ValueError("The staged registry does not match the supplied dataset")
    state_hash = _sha256(runtime_state_path)
    state = _read_json(runtime_state_path)
    if str(state.get("dataset_sha256") or "") != dataset_hash:
        raise ValueError("Runtime surprise state does not match the supplied dataset")

    frame = _enhance_features(load_fusion_dataset(dataset_path))
    feature_max_error: dict[str, float] = {name: 0.0 for name in FULL_FEATURES}
    sampled_rows = 0
    schema_passed = True
    numeric_passed = True
    no_news_passed = True
    runtime_state_structure_passed = True
    horizon_reports: dict[str, Any] = {}
    for horizon in accepted:
        entry = dict((registry.get("horizons") or {}).get(str(horizon)) or {})
        window = float(entry.get("selected_window_days", 0.0))
        half_life = float(entry.get("selected_half_life_days", 0.0))
        rows = frame.loc[
            pd.to_numeric(frame["horizon_days"], errors="coerce").eq(horizon)
            & pd.to_numeric(frame["news_window_days"], errors="coerce").eq(window)
            & pd.to_numeric(frame["news_half_life_days"], errors="coerce").eq(half_life)
        ].sort_values(["ticker", "observation_date"])
        rows = rows.groupby("ticker", sort=True, group_keys=False).head(int(sample_per_ticker))
        horizon_reports[str(horizon)] = {"rows": int(len(rows)), "max_feature_error": 0.0}
        state_horizon = (state.get("horizons") or {}).get(str(horizon), {})
        state_tickers = state_horizon.get("tickers") or {}
        if (
            float(state_horizon.get("selected_window_days", window)) != window
            or float(state_horizon.get("selected_half_life_days", half_life)) != half_life
        ):
            runtime_state_structure_passed = False
        for _, row in rows.iterrows():
            ticker = str(row["ticker"])
            # Surprise features are point-in-time quantities.  Reconstruct the
            # trailing baseline strictly before this observation rather than
            # using the terminal runtime state for every historical row.
            ticker_history = frame.loc[
                (frame["ticker"].astype(str) == ticker)
                & pd.to_datetime(frame["observation_date"], errors="coerce").lt(
                    pd.Timestamp(row["observation_date"])
                )
                & pd.to_numeric(frame["horizon_days"], errors="coerce").eq(horizon)
                & pd.to_numeric(frame["news_window_days"], errors="coerce").eq(window)
                & pd.to_numeric(frame["news_half_life_days"], errors="coerce").eq(half_life)
            ].sort_values("observation_date").tail(252)
            point_state = {
                # Match the offline shift(1).rolling(252, min_periods=20)
                # contract.  The first 19 historical rows intentionally have
                # a zero baseline rather than an unstable short mean.
                "volume_baseline_mean": float(
                    pd.to_numeric(ticker_history.get("news_log_event_count", 0.0), errors="coerce")
                    .fillna(0.0)
                    .mean()
                )
                if len(ticker_history) >= 20
                else 0.0,
                "tone_baseline_mean": float(
                    pd.to_numeric(ticker_history.get("news_weighted_sentiment", 0.0), errors="coerce")
                    .fillna(0.0)
                    .mean()
                )
                if len(ticker_history) >= 20
                else 0.0,
                "surprise_baseline_ready": len(ticker_history) >= 20,
            }
            stock_probability = float(row["stock_probability_up"])
            threshold = float(row.get("stock_threshold", 0.5) or 0.5)
            evidence = "accepted" if float(row.get("stock_evidence_strength", 0.0) or 0.0) >= 0.75 else "limited"
            stock = PredictionResult(
                ticker=ticker,
                company_name=str(row.get("company_name", ticker)),
                horizon_days=horizon,
                as_of_date=pd.Timestamp(row["observation_date"]).strftime("%Y-%m-%d"),
                prediction_label="Up" if stock_probability >= threshold else "Down",
                probability_up=stock_probability,
                decision_threshold=threshold,
                model_name="frozen-stock-oof",
                model_version=str(row.get("stock_registry_run_id", "frozen")),
                refinement_round=0,
                benchmark_passed=True,
                status="accepted",
                stale_data_warning=False,
                latest_data_age_days=0,
                threshold_constraints_passed=True,
                evidence_strength=evidence,
            )
            context: dict[str, Any] = {
                "weighted_sentiment_score": row.get("news_weighted_sentiment"),
                "coverage_confidence": row.get("coverage_confidence", "Unavailable"),
                "usable_article_count": int(row.get("news_usable_article_count", 0) or 0),
                "candidate_article_count": int(row.get("news_usable_article_count", 0) or 0),
                "unique_source_count": int(row.get("news_unique_source_count", 0) or 0),
                "independent_event_count": int(row.get("news_independent_event_count", 0) or 0),
                "positive_count": int(round(float(row.get("news_positive_share", 0.0) or 0.0) * int(row.get("news_usable_article_count", 0) or 0))),
                "neutral_count": int(round(float(row.get("news_neutral_share", 0.0) or 0.0) * int(row.get("news_usable_article_count", 0) or 0))),
                "negative_count": int(round(float(row.get("news_negative_share", 0.0) or 0.0) * int(row.get("news_usable_article_count", 0) or 0))),
                "news_positive_share": row.get("news_positive_share", 0.0),
                "news_neutral_share": row.get("news_neutral_share", 0.0),
                "news_negative_share": row.get("news_negative_share", 0.0),
                "mean_sentiment_confidence": row.get("news_mean_confidence", 0.0),
                "max_sentiment_confidence": row.get("news_max_confidence", 0.0),
                "newest_article_age_days": row.get("news_newest_age_days", 0.0),
                "mean_article_age_days": row.get("news_mean_age_days", 0.0),
                "half_life_days": half_life,
                "news_event_reliability_mean": row.get("news_event_reliability_mean", 0.0),
                "news_positive_event_impact": row.get("news_positive_event_impact", 0.0),
                "news_negative_event_impact": row.get("news_negative_event_impact", 0.0),
                "news_official_disclosure_share": row.get("news_official_disclosure_share", 0.0),
            }
            for name in FULL_FEATURES:
                if name.startswith("news_event_category_"):
                    context[name] = row.get(name, 0.0)
            news = NewsContextResult(
                ticker=ticker,
                company_name=str(row.get("company_name", ticker)),
                query_text=ticker,
                as_of_date=pd.Timestamp(row["observation_date"]).strftime("%Y-%m-%d"),
                retrieved_at=pd.Timestamp(row["observation_date"]).isoformat(),
                raw_article_count=int(context["candidate_article_count"]),
                relevant_article_count=int(context["usable_article_count"]),
                results_truncated=False,
                contexts={str(horizon): context},
                articles=[],
                output_paths={},
            )
            if ticker not in state_tickers:
                runtime_state_structure_passed = False
            runtime = build_runtime_fusion_frame(
                stock, news, horizon, runtime_state=point_state
            ).iloc[0]
            missing = [name for name in FULL_FEATURES if name not in runtime.index]
            if missing:
                schema_passed = False
                continue
            row_error = 0.0
            for name in FULL_FEATURES:
                expected = float(row.get(name, 0.0) or 0.0)
                observed = float(runtime.get(name, 0.0) or 0.0)
                error = abs(expected - observed)
                feature_max_error[name] = max(feature_max_error[name], error)
                row_error = max(row_error, error)
            horizon_reports[str(horizon)]["max_feature_error"] = max(
                horizon_reports[str(horizon)]["max_feature_error"], row_error
            )
            sampled_rows += 1
            numeric_passed = numeric_passed and row_error <= float(tolerance)
            no_news_context = dict(context)
            no_news_context.update(
                {
                    "weighted_sentiment_score": None,
                    "coverage_confidence": "Unavailable",
                    "usable_article_count": 0,
                    "candidate_article_count": 0,
                    "unique_source_count": 0,
                    "independent_event_count": 0,
                    "positive_count": 0,
                    "neutral_count": 0,
                    "negative_count": 0,
                }
            )
            no_news = NewsContextResult(
                ticker=ticker,
                company_name=str(row.get("company_name", ticker)),
                query_text=ticker,
                as_of_date=pd.Timestamp(row["observation_date"]).strftime("%Y-%m-%d"),
                retrieved_at=pd.Timestamp(row["observation_date"]).isoformat(),
                raw_article_count=0,
                relevant_article_count=0,
                results_truncated=False,
                contexts={str(horizon): no_news_context},
                articles=[],
                output_paths={},
            )
            no_news_frame = build_runtime_fusion_frame(stock, no_news, horizon)
            no_news_passed = no_news_passed and float(no_news_frame.iloc[0]["stock_probability_up"]) == stock_probability and float(no_news_frame.iloc[0]["news_available"]) == 0.0
    payload = {
        "schema_version": "fusion-v3-runtime-parity-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_sha256": dataset_hash,
        "runtime_state_sha256": state_hash,
        "accepted_horizons": accepted,
        "sample_per_ticker": int(sample_per_ticker),
        "sampled_rows": int(sampled_rows),
        "tolerance": float(tolerance),
        "feature_max_error": feature_max_error,
        "horizons": horizon_reports,
        "checks": {
            "feature_schema": schema_passed,
            "numeric_feature_parity": numeric_passed,
            "no_news_exact_stock_input": no_news_passed,
            "runtime_state_structure": runtime_state_structure_passed,
        },
        "passed": bool(
            sampled_rows > 0
            and schema_passed
            and numeric_passed
            and no_news_passed
            and runtime_state_structure_passed
        ),
    }
    _atomic_json(output_path, payload)
    return payload


def promote_fusion_v3_registry(
    staged_registry_path: Path,
    reproducibility_report_path: Path,
    runtime_state_path: Path,
    runtime_parity_report_path: Path,
    *,
    active_registry_path: Path = DEFAULT_ACTIVE_REGISTRY,
    allow_acceptance_override: bool = False,
) -> dict[str, Any]:
    """Promote a staged registry only after every runtime guard is satisfied."""

    registry = _read_json(staged_registry_path)
    reproducibility = _read_json(reproducibility_report_path)
    state = _read_json(runtime_state_path)
    parity = _read_json(runtime_parity_report_path)
    if registry.get("schema_version") != "fusion-registry-v5":
        raise ValueError("Only Fusion v3 registries can be promoted here")
    override = dict(registry.get("acceptance_override") or {})
    override_active = bool(override.get("active"))
    if override_active and not allow_acceptance_override:
        raise ValueError(
            "This registry contains an acceptance override; pass the explicit "
            "allow_acceptance_override flag to promote it"
        )
    if override_active and not bool(override.get("operator_authorized")):
        raise ValueError("The acceptance override is not marked operator-authorized")
    if not registry.get("accepted_horizons"):
        raise ValueError("No horizon passed the frozen Fusion v3 gates")
    if not bool(reproducibility.get("passed")):
        raise ValueError("The two complete Fusion v3 runs are not reproducible")
    if str(state.get("dataset_sha256") or "") != str(
        registry.get("dataset_sha256") or ""
    ):
        raise ValueError("Runtime surprise state does not match the staged registry")
    if not bool(parity.get("passed")):
        raise ValueError("Online/offline Fusion v3 feature parity has not passed")
    if str(parity.get("runtime_state_sha256") or "") != _sha256(runtime_state_path):
        raise ValueError("Runtime parity report refers to a different surprise state")
    promoted = dict(registry)
    promoted["status"] = "override_accepted" if override_active else "partially_accepted"
    promoted["overall_status"] = (
        "acceptance_overridden" if override_active else promoted.get("overall_status", "horizon_specific_acceptance")
    )
    promoted["overall_all_horizon_accepted"] = False if override_active else bool(
        promoted.get("overall_all_horizon_accepted", False)
    )
    promoted["runtime_action"] = (
        "enable_fusion_override" if override_active else promoted.get("runtime_action", "enable_fusion")
    )
    promoted["runtime_feature_contract_ready"] = True
    promoted["runtime_state_path"] = _reference(runtime_state_path)
    promoted["runtime_state_sha256"] = _sha256(runtime_state_path)
    promoted["runtime_parity_report_path"] = _reference(runtime_parity_report_path)
    promoted["reproducibility_report_path"] = _reference(
        reproducibility_report_path
    )
    promoted["promoted_at"] = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    if active_registry_path.exists():
        backup = active_registry_path.with_suffix(
            active_registry_path.suffix + ".pre-v3.bak"
        )
        shutil.copy2(active_registry_path, backup)
    _atomic_json(active_registry_path, promoted)
    return promoted
