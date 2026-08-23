"""Benchmark the opt-in Bursa/PhraseBank probability ensemble.

The benchmark selects a blend weight on the chronological validation split and
evaluates that one selected weight once on the frozen test split.  It never
changes the active runtime artifact; enabling the challenger remains an
explicit ``STOCKSENSE_SENTIMENT_VARIANT`` opt-in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.benchmark_bursa_finbert import _load_bursa_splits
from pipeline.benchmark_finbert_sentiment import (
    LABEL_TO_ID,
    _bootstrap_macro_f1_delta,
    expected_calibration_error,
    metrics_from_probabilities,
)
from pipeline.bursa_news_dataset import DEFAULT_DATASET_PATH, DEFAULT_POLICY_PATH, load_policy
from pipeline.finbert_sentiment_model import WeightedFinBERTEnsemble, load_finbert_sentiment_model
from pipeline.runtime_provenance import sentiment_artifact_hashes_for_directory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "finbert-ensemble-benchmark"
DEFAULT_BURSA_MODEL = PROJECT_ROOT / "models" / "experimental" / "bursa-finbert-staging-20260801T040002" / "seed-42"
DEFAULT_PHRASEBANK_MODEL = PROJECT_ROOT / "models" / "experimental" / "finbert-sentiment"
DEFAULT_VARIANT_ID = "bursa_phrasebank_soft_vote_v2"
DEFAULT_REFERENCE_BURSA_WEIGHT = 0.60
DEFAULT_BURSA_BENCHMARK_REPORT = (
    PROJECT_ROOT / "results" / "bursa-finbert-benchmark" / "latest.json"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _runtime_path(path: str | Path) -> str:
    """Store project-relative paths where possible so runtime config is portable."""

    resolved = _resolve(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _probabilities(model: object, frame: Any) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(frame["model_text"].astype(str).tolist()), dtype=float
    )
    return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def _blend_probabilities(
    bursa_probabilities: np.ndarray,
    phrasebank_probabilities: np.ndarray,
    bursa_weight: float,
) -> np.ndarray:
    weight = float(bursa_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("bursa_weight must be between 0 and 1")
    blended = weight * bursa_probabilities + (1.0 - weight) * phrasebank_probabilities
    return blended / np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)


def _ensemble_gates(policy: Mapping[str, Any]) -> dict[str, Any]:
    configured = (policy.get("sentiment_ensemble") or {}).get("promotion_gates")
    if isinstance(configured, dict):
        return dict(configured)
    legacy = policy.get("sentiment_promotion_gates") or {}
    return {
        "minimum_validation_macro_f1_uplift": 0.02,
        "minimum_validation_balanced_accuracy_uplift": 0.02,
        "minimum_macro_f1": legacy.get("minimum_macro_f1", 0.70),
        "minimum_balanced_accuracy": legacy.get("minimum_balanced_accuracy", 0.70),
        "maximum_ece": legacy.get("maximum_ece", 0.10),
        "minimum_class_f1": 0.60,
        "minimum_bootstrap_ci_lower": legacy.get("minimum_bootstrap_ci_lower", 0.0),
        "maximum_phrasebank_macro_f1_degradation": legacy.get(
            "maximum_phrasebank_macro_f1_degradation", 0.02
        ),
        "maximum_seed_macro_f1_std": legacy.get("maximum_seed_macro_f1_std", 0.02),
        "maximum_company_macro_f1_loss": 0.05,
        "minimum_holdout_rows": 200,
        "minimum_holdout_rows_per_ticker": 15,
        "require_independent_holdout": True,
    }


def _load_optional_holdout(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    source = _resolve(path)
    if not source.exists():
        raise FileNotFoundError(f"Independent holdout dataset not found: {source}")
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".json", ".jsonl"}:
        frame = pd.read_json(source, lines=suffix == ".jsonl")
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        raise ValueError("Independent holdout must be CSV, JSONL, JSON, or Parquet")
    required = {"model_text", "sentiment_label", "ticker"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Independent holdout is missing columns: {sorted(missing)}")
    frame["sentiment_label"] = frame["sentiment_label"].astype(str).str.strip().str.casefold()
    frame = frame.loc[frame["sentiment_label"].isin(LABEL_TO_ID)].copy()
    frame = frame.loc[frame["model_text"].astype(str).str.strip().ne("")].copy()
    if frame.empty:
        raise ValueError("Independent holdout contains no labelled text rows")
    return frame.reset_index(drop=True)


def _load_component_benchmark(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = _resolve(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _company_loss_check(
    frame: pd.DataFrame,
    ensemble_probabilities: np.ndarray,
    phrasebank_probabilities: np.ndarray,
    *,
    minimum_rows: int,
    maximum_loss: float,
) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    passed = True
    for ticker, indices in frame.groupby(frame["ticker"].astype(str).str.upper()).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        if len(positions) < int(minimum_rows):
            details[str(ticker)] = {"rows": int(len(positions)), "evaluated": False}
            continue
        labels = frame.iloc[positions]["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
        ensemble_metrics = metrics_from_probabilities("ensemble", ensemble_probabilities[positions], labels)
        phrasebank_metrics = metrics_from_probabilities("phrasebank", phrasebank_probabilities[positions], labels)
        delta = float(ensemble_metrics["macro_f1"] - phrasebank_metrics["macro_f1"])
        eligible = delta >= -float(maximum_loss)
        details[str(ticker)] = {
            "rows": int(len(positions)),
            "evaluated": True,
            "macro_f1_delta": delta,
            "passed": eligible,
        }
        passed = passed and eligible
    return passed, details


def _gate_report(
    selected_validation: dict[str, Any],
    production_validation: dict[str, Any],
    selected_final: dict[str, Any],
    production_final: dict[str, Any],
    bootstrap: dict[str, float],
    *,
    policy: Mapping[str, Any],
    independent_holdout: bool,
    component_benchmark: Mapping[str, Any],
    company_check: bool,
) -> dict[str, Any]:
    """Centralized, conservative gates; no gate causes automatic deployment."""
    gates = _ensemble_gates(policy)
    validation_uplift = float(
        selected_validation["macro_f1"] - production_validation["macro_f1"]
    )
    final_uplift = float(selected_final["macro_f1"] - production_final["macro_f1"])
    seed_std = component_benchmark.get("seed_macro_f1_std")
    phrasebank_degradation = component_benchmark.get("phrasebank_macro_f1_degradation")
    checks = {
        "validation_macro_f1_uplift": validation_uplift >= float(gates["minimum_validation_macro_f1_uplift"]),
        "validation_balanced_accuracy_uplift": selected_validation["balanced_accuracy"]
        >= production_validation["balanced_accuracy"] + float(gates["minimum_validation_balanced_accuracy_uplift"]),
        "final_macro_f1": selected_final["macro_f1"] >= float(gates["minimum_macro_f1"]),
        "final_balanced_accuracy": selected_final["balanced_accuracy"] >= float(gates["minimum_balanced_accuracy"]),
        "final_ece": selected_final["ece"] <= float(gates["maximum_ece"]),
        "bootstrap_ci_lower": bootstrap["ci95_lower"] >= float(gates["minimum_bootstrap_ci_lower"]),
        "minimum_class_f1": min(
            selected_final["negative_f1"],
            selected_final["neutral_f1"],
            selected_final["positive_f1"],
        ) >= float(gates["minimum_class_f1"]),
        "component_seed_stability": seed_std is not None
        and float(seed_std) <= float(gates["maximum_seed_macro_f1_std"]),
        "phrasebank_non_inferiority": phrasebank_degradation is not None
        and float(phrasebank_degradation) <= float(gates["maximum_phrasebank_macro_f1_degradation"]),
        "per_company_non_inferiority": bool(company_check),
        "independent_holdout": (not bool(gates.get("require_independent_holdout", True)))
        or bool(independent_holdout),
    }
    return {
        "checks": checks,
        "eligible_for_promotion": bool(all(checks.values())),
        "validation_macro_f1_uplift": validation_uplift,
        "final_macro_f1_uplift": final_uplift,
        "thresholds": gates,
    }


def run_finbert_ensemble_benchmark(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
    bursa_model_directory: Path = DEFAULT_BURSA_MODEL,
    phrasebank_model_directory: Path = DEFAULT_PHRASEBANK_MODEL,
    bursa_temperature: float = 1.27,
    phrasebank_temperature: float = 1.52,
    weight_step: float = 0.05,
    variant_id: str = DEFAULT_VARIANT_ID,
    independent_holdout_path: Path | str | None = None,
    bursa_benchmark_report: Path | str | None = DEFAULT_BURSA_BENCHMARK_REPORT,
) -> dict[str, Any]:
    if not 0.0 < float(weight_step) <= 1.0:
        raise ValueError("weight_step must be in (0, 1]")
    policy = load_policy(Path(policy_path))
    splits = _load_bursa_splits(Path(dataset_path), policy_path=Path(policy_path))
    bursa_model = load_finbert_sentiment_model(
        str(_resolve(bursa_model_directory)), temperature=float(bursa_temperature)
    )
    phrasebank_model = load_finbert_sentiment_model(
        str(_resolve(phrasebank_model_directory)), temperature=float(phrasebank_temperature)
    )
    validation = splits["validation"].reset_index(drop=True)
    test = splits["test"].reset_index(drop=True)
    bursa_validation = _probabilities(bursa_model, validation)
    phrasebank_validation = _probabilities(phrasebank_model, validation)
    bursa_test = _probabilities(bursa_model, test)
    phrasebank_test = _probabilities(phrasebank_model, test)
    validation_labels = validation["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    test_labels = test["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)

    production_validation = metrics_from_probabilities(
        "phrasebank_production_validation", phrasebank_validation, validation_labels
    )
    production_test = metrics_from_probabilities(
        "phrasebank_production_test", phrasebank_test, test_labels
    )
    weights = np.arange(0.0, 1.0 + float(weight_step) / 2.0, float(weight_step))
    grid: list[dict[str, Any]] = []
    for weight in weights:
        candidate = _blend_probabilities(bursa_validation, phrasebank_validation, float(weight))
        metrics = metrics_from_probabilities(
            f"bursa_{float(weight):.2f}_phrasebank_{1.0 - float(weight):.2f}_validation",
            candidate,
            validation_labels,
        )
        grid.append({"bursa_weight": float(weight), "phrasebank_weight": float(1.0 - weight), **metrics})
    selected = sorted(
        grid,
        key=lambda item: (
            -float(item["macro_f1"]),
            -float(item["balanced_accuracy"]),
            float(item["ece"]),
            float(item["log_loss"]),
            # 60/40 is the pre-registered starting hypothesis for this
            # project.  It is only a deterministic tie-breaker; validation
            # quality still decides the actual selected blend.
            abs(float(item["bursa_weight"]) - DEFAULT_REFERENCE_BURSA_WEIGHT),
        ),
    )[0]
    selected_weight = float(selected["bursa_weight"])
    selected_validation = dict(selected)
    selected_validation["model"] = f"{variant_id}_validation"
    selected_test_probabilities = _blend_probabilities(
        bursa_test, phrasebank_test, selected_weight
    )
    selected_test = metrics_from_probabilities(
        f"{variant_id}_test", selected_test_probabilities, test_labels
    )
    holdout = _load_optional_holdout(
        Path(independent_holdout_path) if independent_holdout_path else None
    )
    if holdout is not None:
        bursa_holdout = _probabilities(bursa_model, holdout)
        phrasebank_holdout = _probabilities(phrasebank_model, holdout)
        holdout_labels = holdout["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
        selected_final_probabilities = _blend_probabilities(
            bursa_holdout, phrasebank_holdout, selected_weight
        )
        selected_final = metrics_from_probabilities(
            f"{variant_id}_independent_holdout", selected_final_probabilities, holdout_labels
        )
        production_final = metrics_from_probabilities(
            "phrasebank_independent_holdout", phrasebank_holdout, holdout_labels
        )
        final_frame = holdout
        final_phrasebank = phrasebank_holdout
        final_scope = "independent labelled holdout"
        independent_holdout = True
    else:
        selected_final_probabilities = selected_test_probabilities
        selected_final = selected_test
        production_final = production_test
        final_frame = test
        final_phrasebank = phrasebank_test
        final_scope = "2025 retrospective chronological test split"
        independent_holdout = False
    bootstrap = _bootstrap_macro_f1_delta(
        selected_final_probabilities.argmax(axis=1),
        final_phrasebank.argmax(axis=1),
        final_frame["sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int),
        iterations=5000,
    )
    gates_policy = _ensemble_gates(policy)
    company_check, company_metrics = _company_loss_check(
        final_frame,
        selected_final_probabilities,
        final_phrasebank,
        minimum_rows=int(gates_policy["minimum_holdout_rows_per_ticker"]),
        maximum_loss=float(gates_policy["maximum_company_macro_f1_loss"]),
    )
    if len(final_frame) < int(gates_policy["minimum_holdout_rows"]):
        company_check = False
    component_report = _load_component_benchmark(
        Path(bursa_benchmark_report) if bursa_benchmark_report else None
    )
    gates = _gate_report(
        selected_validation,
        production_validation,
        selected_final,
        production_final,
        bootstrap,
        policy=policy,
        independent_holdout=independent_holdout,
        component_benchmark=component_report,
        company_check=company_check,
    )
    run_id = time.strftime("%Y%m%dT%H%M%S")
    report = {
        "schema_version": "finbert-ensemble-benchmark-v2",
        "run_id": run_id,
        "variant_id": variant_id,
        "dataset_path": str(Path(dataset_path)),
        "policy_path": str(Path(policy_path)),
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "selection": {
            "selected_bursa_weight": selected_weight,
            "selected_phrasebank_weight": float(1.0 - selected_weight),
            "validation_scope": "2024 chronological validation split",
            "retrospective_test_scope": "2025 chronological test split",
            "weight_step": float(weight_step),
            "grid": grid,
        },
        "production_validation": production_validation,
        "selected_validation": selected_validation,
        "production_test": production_test,
        "selected_test": selected_test,
        "final_evaluation": {
            "scope": final_scope,
            "independent_holdout": independent_holdout,
            "rows": len(final_frame),
            "production": production_final,
            "selected": selected_final,
            "per_company": company_metrics,
        },
        "bootstrap_macro_f1_delta": bootstrap,
        "component_benchmark": {
            "path": str(bursa_benchmark_report) if bursa_benchmark_report else None,
            "seed_macro_f1_std": component_report.get("seed_macro_f1_std"),
            "phrasebank_macro_f1_degradation": component_report.get("phrasebank_macro_f1_degradation"),
        },
        "gates": gates,
        "artifact_hashes": {
            "bursa": sentiment_artifact_hashes_for_directory(_resolve(bursa_model_directory)),
            "phrasebank": sentiment_artifact_hashes_for_directory(_resolve(phrasebank_model_directory)),
        },
        "runtime_candidate": {
            "backend": "weighted_finbert_ensemble",
            "weights": {
                "bursa": selected_weight,
                "phrasebank": float(1.0 - selected_weight),
            },
            "selection_scope": "2024 chronological validation; 2025 is a retrospective check only",
            "promotion_status": "benchmark_complete_not_active",
            "benchmark_run_id": run_id,
            "validation_macro_f1": float(selected_validation["macro_f1"]),
            "retrospective_test_macro_f1": float(selected_test["macro_f1"]),
            "components": {
                "bursa": {
                    "model_directory": _runtime_path(bursa_model_directory),
                    "temperature": float(bursa_temperature),
                    "artifact_hashes": sentiment_artifact_hashes_for_directory(
                        _resolve(bursa_model_directory)
                    ),
                },
                "phrasebank": {
                    "model_directory": _runtime_path(phrasebank_model_directory),
                    "temperature": float(phrasebank_temperature),
                    "artifact_hashes": sentiment_artifact_hashes_for_directory(
                        _resolve(phrasebank_model_directory)
                    ),
                },
            },
        },
        "production_changed": False,
        "opt_in_environment_variable": "STOCKSENSE_SENTIMENT_VARIANT",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / f"{run_id}.json", report)
    _write_json(RESULTS_DIR / "latest.json", report)
    print(json.dumps(report, indent=2, default=_json_default))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the opt-in FinBERT probability ensemble")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--bursa-model-directory", default=str(DEFAULT_BURSA_MODEL))
    parser.add_argument("--bursa-run-id", help="Resolve seed-42 from a staged Bursa run id")
    parser.add_argument("--phrasebank-model-directory", default=str(DEFAULT_PHRASEBANK_MODEL))
    parser.add_argument("--bursa-temperature", type=float, default=1.27)
    parser.add_argument("--phrasebank-temperature", type=float, default=1.52)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    parser.add_argument(
        "--independent-holdout",
        default=None,
        help=(
            "A separately frozen labelled holdout dataset. Without it the 2025 "
            "result is reported retrospectively and cannot promote the challenger."
        ),
    )
    parser.add_argument(
        "--bursa-benchmark-report", default=str(DEFAULT_BURSA_BENCHMARK_REPORT)
    )
    args = parser.parse_args(argv)
    bursa_directory = Path(args.bursa_model_directory)
    if args.bursa_run_id:
        bursa_directory = PROJECT_ROOT / "models" / "experimental" / f"bursa-finbert-staging-{args.bursa_run_id}" / "seed-42"
    run_finbert_ensemble_benchmark(
        Path(args.dataset),
        policy_path=Path(args.policy),
        bursa_model_directory=bursa_directory,
        phrasebank_model_directory=Path(args.phrasebank_model_directory),
        bursa_temperature=args.bursa_temperature,
        phrasebank_temperature=args.phrasebank_temperature,
        weight_step=args.weight_step,
        variant_id=args.variant_id,
        independent_holdout_path=args.independent_holdout,
        bursa_benchmark_report=args.bursa_benchmark_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
