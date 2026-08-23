from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.download_raw_stock_data import STOCKS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOCK_REGISTRY = PROJECT_ROOT / "models" / "model-registry.json"
SENTIMENT_RUNTIME_CONFIG = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
DEFAULT_SENTIMENT_REPORT = PROJECT_ROOT / "results" / "finbert-ensemble-defensible" / "latest.json"
MODEL_PERFORMANCE_ARTIFACT = PROJECT_ROOT / "results" / "model-performance" / "model-performance.json"
MODEL_PERFORMANCE_RUNS_DIR = PROJECT_ROOT / "results" / "model-performance" / "runs"
FUSION_ABLATION_ARTIFACT = PROJECT_ROOT / "results" / "fusion" / "fusion-ablation.json"
FUSION_EVALUATION_V2_ARTIFACT = (
    PROJECT_ROOT / "results" / "fusion" / "fusion-evaluation-v2.json"
)
FUSION_EVALUATION_V3_ARTIFACT = (
    PROJECT_ROOT / "results" / "fusion" / "fusion-evaluation-v3.json"
)


def _fusion_ablation_path(run_id: str | None) -> Path | None:
    """Return only a provenance-matched, threshold-correct v3 report.

    V2 is intentionally not a display fallback: its no-news decision-threshold
    defect can create a misleading directional gain even when probabilities
    are copied from the stock model.
    """

    for evaluation_path in (FUSION_EVALUATION_V3_ARTIFACT,):
        if not evaluation_path.exists():
            continue
        payload = _read_json(evaluation_path)
        provenance = dict(payload.get("provenance") or {})
        active_registry = _read_json(STOCK_REGISTRY)
        current_sentiment_hash = (
            hashlib.sha256(SENTIMENT_RUNTIME_CONFIG.read_bytes()).hexdigest()
            if SENTIMENT_RUNTIME_CONFIG.is_file()
            else None
        )
        identities_match = (
            bool(provenance.get("strict_identity_passed"))
            and run_id == provenance.get("active_stock_run_id")
            and run_id == active_registry.get("run_id")
            and provenance.get("stock_dataset_sha256")
            == active_registry.get("dataset_sha256")
            and provenance.get("sentiment_config_sha256")
            == current_sentiment_hash
        )
        if run_id and identities_match:
            return evaluation_path
    # Do not expose a report whose stock/sentiment identities do not match the
    # selected run.  Returning None keeps stale diagnostics out of the API even
    # when a newer canonical report exists for the active run.
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_project_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = (PROJECT_ROOT / value).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return candidate


def _stock_diagnostics(registry_path: Path = STOCK_REGISTRY) -> dict[str, Any]:
    registry = _read_json(registry_path)
    horizons: list[dict[str, Any]] = []
    for horizon_text, record in sorted(
        (registry.get("horizons") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        if not isinstance(record, dict):
            continue
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        baseline = record.get("baseline_metrics") if isinstance(record.get("baseline_metrics"), dict) else {}
        observed = record.get("observed_acceptance") if isinstance(record.get("observed_acceptance"), dict) else {}
        shadow = record.get("shadow_acceptance") if isinstance(record.get("shadow_acceptance"), dict) else {}
        calibration = observed.get("calibration") if isinstance(observed.get("calibration"), dict) else {}
        bootstrap = observed.get("bootstrap") if isinstance(observed.get("bootstrap"), dict) else None
        observed_failures = [str(value) for value in observed.get("failed_checks", [])]
        shadow_failures = [str(value) for value in shadow.get("failed_checks", [])]
        horizons.append(
            {
                "horizon_days": int(horizon_text),
                "model_name": str(record.get("model_name") or "Unavailable"),
                "status": str(record.get("effective_status") or record.get("status") or "unavailable"),
                "observed_status": str(record.get("status") or "unavailable"),
                "shadow_status": str(shadow.get("status") or "unavailable"),
                "decision_threshold": _number(record.get("decision_threshold")),
                "strongest_baseline": str(record.get("strongest_baseline") or "Naive baseline"),
                "metrics": {
                    "accuracy": _number(metrics.get("accuracy")),
                    "balanced_accuracy": _number(metrics.get("balanced_accuracy")),
                    "macro_f1": _number(metrics.get("macro_f1")),
                    "mcc": _number(metrics.get("mcc")),
                    "roc_auc": _number(metrics.get("roc_auc")),
                    "ece": _number(calibration.get("ece_10")),
                    "brier_score": _number(calibration.get("brier_score")),
                },
                "baseline_metrics": {
                    "accuracy": _number(baseline.get("accuracy")),
                    "balanced_accuracy": _number(baseline.get("balanced_accuracy")),
                    "macro_f1": _number(baseline.get("macro_f1")),
                    "mcc": _number(baseline.get("mcc")),
                },
                "balanced_accuracy_gain": _number(observed.get("balanced_accuracy_gain")),
                "observed_bootstrap_gain_ci": (
                    {
                        "lower": _number(bootstrap.get("lower")),
                        "median": _number(bootstrap.get("median")),
                        "upper": _number(bootstrap.get("upper")),
                        "repetitions": bootstrap.get("repetitions"),
                        "block_size": bootstrap.get("block_size"),
                    }
                    if bootstrap
                    else None
                ),
                "cv_fold_balanced_accuracy": [
                    float(value)
                    for value in record.get("cv_fold_balanced_accuracy", [])
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ],
                "observed_failed_checks": observed_failures,
                "shadow_failed_checks": shadow_failures,
                "observed_unique_dates": observed.get("unique_dates"),
                "shadow_rows": shadow.get("rows"),
            }
        )
    benchmark_context = registry.get("benchmark_context") if isinstance(registry.get("benchmark_context"), dict) else {}
    return {
        "available": bool(horizons),
        "run_id": registry.get("run_id"),
        "search_profile": registry.get("search_profile"),
        "feature_schema_version": registry.get("feature_schema_version"),
        "development_period": {
            "start": registry.get("development_start_date"),
            "end": registry.get("development_end_date"),
        },
        "retrospective_test_period": {
            "start": registry.get("test_start_date"),
            "end": registry.get("test_end_date"),
            "disclosure": benchmark_context.get("test_warning"),
        },
        "shadow_period": (registry.get("shadow_evaluation") or {}),
        "acceptance_policy_version": benchmark_context.get("acceptance_policy_version"),
        "horizons": horizons,
    }


def _active_sentiment_report(runtime: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    active_variant = runtime.get("active_variant")
    challengers = runtime.get("challengers") if isinstance(runtime.get("challengers"), dict) else {}
    active = challengers.get(active_variant) if isinstance(challengers.get(active_variant), dict) else {}
    manual_override = active.get("manual_override") if isinstance(active.get("manual_override"), dict) else {}
    report_path = _safe_project_path(manual_override.get("benchmark_report"))
    if report_path and report_path.exists():
        return report_path, _read_json(report_path)
    return DEFAULT_SENTIMENT_REPORT, _read_json(DEFAULT_SENTIMENT_REPORT)


def _metric_set(key: str, label: str, metrics: Any) -> dict[str, Any]:
    value = metrics if isinstance(metrics, dict) else {}
    return {
        "key": key,
        "label": label,
        "accuracy": _number(value.get("accuracy")),
        "balanced_accuracy": _number(value.get("balanced_accuracy")),
        "macro_f1": _number(value.get("macro_f1")),
        "mcc": _number(value.get("mcc")),
        "log_loss": _number(value.get("log_loss")),
        "brier_score": _number(value.get("brier_score")),
        "ece": _number(value.get("ece")),
        "class_f1": {
            "negative": _number(value.get("negative_f1")),
            "neutral": _number(value.get("neutral_f1")),
            "positive": _number(value.get("positive_f1")),
        },
    }


def _sentiment_diagnostics(
    runtime_path: Path = SENTIMENT_RUNTIME_CONFIG,
) -> dict[str, Any]:
    runtime = _read_json(runtime_path)
    report_path, report = _active_sentiment_report(runtime)
    active_variant = str(runtime.get("active_variant") or "unavailable")
    challengers = runtime.get("challengers") if isinstance(runtime.get("challengers"), dict) else {}
    active = challengers.get(active_variant) if isinstance(challengers.get(active_variant), dict) else {}
    final = report.get("final_2025") if isinstance(report.get("final_2025"), dict) else {}
    gates = report.get("gates") if isinstance(report.get("gates"), dict) else {}
    checks = gates.get("checks") if isinstance(gates.get("checks"), dict) else {}
    dataset = report.get("evidence_manifest") if isinstance(report.get("evidence_manifest"), dict) else {}
    diagnostics = dataset.get("dataset_diagnostics") if isinstance(dataset.get("dataset_diagnostics"), dict) else {}
    weights = active.get("weights") if isinstance(active.get("weights"), dict) else {}
    models = [
        _metric_set("phrasebank", "PhraseBank only", final.get("phrasebank")),
        _metric_set("bursa", "Bursa-adapted", final.get("bursa_seed_average")),
        _metric_set("ensemble", "Active ensemble", final.get("ensemble")),
    ]
    return {
        "available": bool(report and final),
        "active_variant": active_variant,
        "backend": active.get("backend") or runtime.get("backend"),
        "promotion_status": active.get("promotion_status") or "unavailable",
        "manual_override_active": bool((active.get("manual_override") or {}).get("active")),
        "weights": {
            "bursa": _number(weights.get("bursa")),
            "phrasebank": _number(weights.get("phrasebank")),
        },
        "ensemble_temperature": _number(active.get("ensemble_temperature")),
        "run_id": report.get("run_id"),
        "report_path": str(report_path.relative_to(PROJECT_ROOT)) if report_path.exists() else None,
        "selection_scope": report.get("selection_scope") or active.get("selection_scope"),
        "evaluation_scope": report.get("final_evaluation_scope"),
        "evaluation_year": 2025,
        "evaluation_rows": final.get("rows"),
        "models": models,
        "dataset": {
            "eligible_rows": dataset.get("eligible_rows"),
            "year_counts": diagnostics.get("year_counts") or {},
            "class_counts": diagnostics.get("class_counts") or {},
        },
        "acceptance": {
            "eligible_for_promotion": bool(gates.get("eligible_for_promotion", False)),
            "passed_checks": [str(key) for key, passed in checks.items() if passed is True],
            "failed_checks": [str(key) for key, passed in checks.items() if passed is False],
            "thresholds": gates.get("thresholds") if isinstance(gates.get("thresholds"), dict) else {},
        },
    }


def load_model_diagnostics() -> dict[str, Any]:
    """Build a compact, read-only view of the evaluation evidence used by the UI."""

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stock": _stock_diagnostics(),
        "sentiment": _sentiment_diagnostics(),
        "disclosure": (
            "Metrics describe recorded evaluation datasets, not the correctness of today's individual prediction. "
            "Retrospective and shadow results are kept separate from live inference."
        ),
    }


def _public_stock_performance(
    *, registry_path: Path = STOCK_REGISTRY, artifact_path: Path = MODEL_PERFORMANCE_ARTIFACT
) -> dict[str, Any]:
    compact = _stock_diagnostics(registry_path)
    shadow = compact.get("shadow_period") if isinstance(compact.get("shadow_period"), dict) else {}
    compact["shadow_period"] = {
        "start_date": shadow.get("start_date"),
        "end_date": shadow.get("end_date"),
        "policy_version": shadow.get("policy_version"),
        "run_id": shadow.get("run_id"),
    }
    registry = _read_json(registry_path)
    artifact = _read_json(artifact_path)
    expected_run_id = str(registry.get("run_id") or "").strip()
    artifact_run_id = str(
        artifact.get("stock_run_id") or artifact.get("run_id") or ""
    ).strip()
    artifact_matches_registry = bool(
        expected_run_id and artifact_run_id and artifact_run_id == expected_run_id
    )
    by_horizon = {
        int(row.get("horizon_days")): row
        for row in artifact.get("horizons", [])
        if isinstance(row, dict) and isinstance(row.get("horizon_days"), int)
    } if artifact_matches_registry else {}
    for row in compact.get("horizons", []):
        detail = by_horizon.get(int(row["horizon_days"]), {})
        row.update(
            {
                "learning_curve": detail.get("learning_curve") or [],
                "confusion_matrix": detail.get("confusion_matrix"),
                "calibration_curve": detail.get("calibration_curve") or [],
                "roc_curve": detail.get("roc_curve") or [],
                "precision_recall_curve": detail.get("precision_recall_curve") or [],
                "threshold_curve": detail.get("threshold_curve") or [],
                "probability_distribution": detail.get("probability_distribution") or [],
                "permutation_importance": detail.get("permutation_importance") or [],
                "candidate_families": detail.get("candidate_families") or [],
                "hyperparameter_trials": detail.get("hyperparameter_trials") or [],
                "seed_stability": detail.get("seed_stability") or [],
                "per_company": detail.get("per_company") or [],
            }
        )
    compact["artifact_run_id"] = artifact_run_id or None
    compact["artifact_status"] = (
        "current"
        if artifact_matches_registry and by_horizon
        else "stale"
        if artifact_run_id
        else "unavailable"
    )
    compact["artifact_generated_at"] = (
        artifact.get("generated_at") if artifact_matches_registry else None
    )
    compact["methodology"] = artifact.get("methodology") if artifact_matches_registry else {}
    compact["detailed_artifact_available"] = bool(artifact_matches_registry and by_horizon)
    compact["artifact_warning"] = (
        None
        if artifact_matches_registry and by_horizon
        else "Detailed charts are hidden because the recorded diagnostic artifact does not match the active model run."
        if artifact_run_id
        else "Detailed chart artifact has not been generated for the active model run."
    )
    return compact


def _public_sentiment_performance() -> dict[str, Any]:
    diagnostics = _sentiment_diagnostics()
    diagnostics.pop("report_path", None)
    runtime = _read_json(SENTIMENT_RUNTIME_CONFIG)
    _, report = _active_sentiment_report(runtime)
    final = report.get("final_2025") if isinstance(report.get("final_2025"), dict) else {}
    per_company_source = final.get("per_company") if isinstance(final.get("per_company"), dict) else {}
    per_company: list[dict[str, Any]] = []
    for ticker, value in sorted(per_company_source.items()):
        if not isinstance(value, dict):
            continue
        ensemble = value.get("ensemble") if isinstance(value.get("ensemble"), dict) else {}
        per_company.append(
            {
                "ticker": str(ticker),
                "company_name": STOCKS.get(str(ticker), str(ticker)),
                "rows": value.get("rows"),
                "balanced_accuracy": _number(ensemble.get("balanced_accuracy")),
                "macro_f1": _number(ensemble.get("macro_f1")),
                "mcc": _number(ensemble.get("mcc")),
                "ece": _number(ensemble.get("ece")),
            }
        )
    diagnostics["per_company"] = per_company
    diagnostics["confusion_matrix"] = None
    diagnostics["calibration_curve"] = []
    diagnostics["artifact_notes"] = {
        "confusion_matrix": "The active report retains class metrics but not row-level predictions, so a confusion matrix is not reconstructed from incomplete evidence.",
        "calibration_curve": "ECE is available; reliability-bin points were not retained by the production benchmark.",
    }
    return diagnostics


def list_model_performance_runs() -> list[dict[str, Any]]:
    active = _read_json(STOCK_REGISTRY)
    active_run_id = str(active.get("run_id") or "").strip()
    active_artifact = _read_json(MODEL_PERFORMANCE_ARTIFACT)
    active_artifact_run_id = str(
        active_artifact.get("stock_run_id") or active_artifact.get("run_id") or ""
    ).strip()
    runs = [
        {
            "run_id": active_run_id,
            "label": "Active production",
            "search_profile": active.get("search_profile"),
            "active": True,
            "artifact_available": bool(
                active_run_id
                and active_artifact_run_id == active_run_id
                and active_artifact.get("horizons")
            ),
        }
    ]
    staged_root = PROJECT_ROOT / "models" / "staged-stock-runs"
    if staged_root.exists():
        for registry in sorted(staged_root.glob("*/model-registry.json"), reverse=True):
            payload = _read_json(registry)
            run_id = str(payload.get("run_id") or registry.parent.name)
            runs.append(
                {
                    "run_id": run_id,
                    "label": f"Staged {payload.get('search_profile') or 'run'}",
                    "search_profile": payload.get("search_profile"),
                    "active": False,
                    "artifact_available": (MODEL_PERFORMANCE_RUNS_DIR / run_id / "model-performance.json").exists(),
                }
            )
    return [row for row in runs if row.get("run_id")]


def load_model_performance(run_id: str | None = None) -> dict[str, Any]:
    """Return presentation diagnostics without exposing artifact paths or model internals."""

    active = _read_json(STOCK_REGISTRY)
    registry_path = STOCK_REGISTRY
    artifact_path = MODEL_PERFORMANCE_ARTIFACT
    if run_id and run_id != active.get("run_id"):
        candidate_registry = PROJECT_ROOT / "models" / "staged-stock-runs" / run_id / "model-registry.json"
        if candidate_registry.exists():
            registry_path = candidate_registry
            artifact_path = MODEL_PERFORMANCE_RUNS_DIR / run_id / "model-performance.json"
    selected_run_id = str(_read_json(registry_path).get("run_id") or "").strip() or None
    fusion_path = _fusion_ablation_path(selected_run_id)
    fusion_source = _read_json(fusion_path) if fusion_path else {}
    fusion_horizons = fusion_source.get("horizons") or {}
    fusion_accepted_horizons = fusion_source.get("accepted_horizons") or []
    all_horizon_accepted = fusion_source.get("overall_all_horizon_accepted")
    if all_horizon_accepted is None:
        all_horizon_accepted = bool(fusion_horizons) and (
            len(fusion_accepted_horizons) == len(fusion_horizons)
        )
    runtime_action = fusion_source.get("runtime_action")
    if not runtime_action:
        runtime_action = (
            "enable_fusion" if all_horizon_accepted else "stock_led_fallback"
        )
    fusion_ablation = {
        "schema_version": fusion_source.get("schema_version"),
        "created_at": fusion_source.get("created_at"),
        "claim_scope": fusion_source.get("claim_scope"),
        "dataset_sha256": fusion_source.get("dataset_sha256"),
        "policy_sha256": fusion_source.get("policy_sha256"),
        "provenance": fusion_source.get("provenance") or {},
        "periods": fusion_source.get("periods") or {},
        "arms": fusion_source.get("arms") or [],
        "horizons": fusion_horizons,
        "overall_all_horizon_accepted": all_horizon_accepted,
        "accepted_horizons": fusion_accepted_horizons,
        "overall_status": fusion_source.get("overall_status"),
        "runtime_action": runtime_action,
    } if fusion_source else {}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stock": _public_stock_performance(registry_path=registry_path, artifact_path=artifact_path),
        "sentiment": _public_sentiment_performance(),
        "available_runs": list_model_performance_runs(),
        "fusion_ablation": fusion_ablation,
        "disclosure": (
            "Historical evaluation describes recorded datasets and cannot guarantee an individual forecast. "
            "Development, retrospective and stock-shadow evidence remain separate."
        ),
    }
    return payload
