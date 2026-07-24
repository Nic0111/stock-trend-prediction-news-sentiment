"""Recompute Horizon Acceptance V2 for existing frozen stock artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from pipeline.horizon_acceptance import POLICY, evaluate_acceptance, weaker_status
from pipeline.train_stock_models import (
    BASELINE_FILE,
    PER_TICKER_FILE,
    SUMMARY_FILE,
    build_baseline_predictions,
    calculate_metrics,
    load_and_combine_model_input_data,
    prepare_horizon_frame,
    split_development_and_test,
)
from pipeline.runtime_provenance import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = PROJECT_ROOT / "models" / "model-registry.json"
RESULTS_DIR = PROJECT_ROOT / "results"


def recompute_acceptance() -> dict[str, Any]:
    registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    combined = load_and_combine_model_input_data()
    summary_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    for horizon in (1, 5, 10):
        entry = registry["horizons"][str(horizon)]
        feature_columns = list(entry["numeric_feature_columns"])
        input_columns = list(entry["input_columns"])
        frame = prepare_horizon_frame(combined, horizon, feature_columns)
        development, test = split_development_and_test(frame, horizon)
        artifact = joblib.load(PROJECT_ROOT / str(entry["artifact_path"]))
        target = f"Target_{horizon}D"
        probability = artifact.predict_proba(test[input_columns])[:, 1]
        predictions = (probability >= float(artifact.threshold)).astype(int)
        y_test = test[target].astype(int).to_numpy()
        baselines = build_baseline_predictions(development, test, horizon)
        development_baselines = build_baseline_predictions(development, development, horizon)
        baseline_name = str(entry.get("strongest_baseline", ""))
        if baseline_name not in baselines:
            baseline_name = max(
                development_baselines,
                key=lambda name: (
                    calculate_metrics(development[target].astype(int), development_baselines[name])["balanced_accuracy"],
                    calculate_metrics(development[target].astype(int), development_baselines[name])["mcc"],
                ),
            )
        baseline = baselines[baseline_name]
        metrics = calculate_metrics(y_test, predictions, probability)
        baseline_metrics = calculate_metrics(y_test, baseline)
        acceptance = evaluate_acceptance(
            horizon=horizon,
            y_true=y_test,
            predictions=predictions,
            probability_up=probability,
            baseline_predictions=baseline,
            dates=test["Date"].to_numpy(),
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            oof_metrics=entry.get("oof_metrics", {}),
            cv_positive_mcc_folds=int(entry.get("cv_positive_mcc_folds", 0)),
            phase="observed",
        )
        ticker_acceptance: dict[str, dict[str, Any]] = {}
        evaluated = test.assign(_prediction=predictions, _probability=probability, _baseline=baseline)
        for ticker, ticker_frame in evaluated.groupby("Ticker"):
            ticker_metrics = calculate_metrics(
                ticker_frame[target].astype(int),
                ticker_frame["_prediction"].to_numpy(),
                ticker_frame["_probability"].to_numpy(),
            )
            ticker_result = evaluate_acceptance(
                horizon=horizon,
                y_true=ticker_frame[target].astype(int).to_numpy(),
                predictions=ticker_frame["_prediction"].to_numpy(),
                probability_up=ticker_frame["_probability"].to_numpy(),
                baseline_predictions=ticker_frame["_baseline"].to_numpy(),
                dates=ticker_frame["Date"].to_numpy(),
                metrics=ticker_metrics,
                baseline_metrics=calculate_metrics(
                    ticker_frame[target].astype(int).to_numpy(), ticker_frame["_baseline"].to_numpy()
                ),
                oof_metrics=entry.get("oof_metrics", {}),
                cv_positive_mcc_folds=int(entry.get("cv_positive_mcc_folds", 0)),
                phase="observed",
                ticker=str(ticker),
            )
            ticker_result["status"] = weaker_status(str(acceptance["status"]), str(ticker_result["status"]))
            ticker_acceptance[str(ticker)] = ticker_result
            ticker_rows.append(
                {
                    "horizon_days": horizon,
                    "ticker": ticker,
                    "rows": len(ticker_frame),
                    "status": ticker_result["status"],
                    **ticker_metrics,
                    "bootstrap_gain_lower": ticker_result["bootstrap"]["lower"],
                    "bootstrap_gain_upper": ticker_result["bootstrap"]["upper"],
                    "failed_checks": ";".join(ticker_result["failed_checks"]),
                }
            )
        entry["global_status"] = acceptance["status"]
        entry["status"] = acceptance["status"]
        entry["threshold_passed"] = acceptance["status"] == "accepted"
        entry["observed_acceptance"] = acceptance
        entry["shadow_acceptance"] = entry.get("shadow_acceptance")
        entry["ticker_acceptance"] = ticker_acceptance
        entry["strongest_baseline"] = baseline_name
        artifact_path = PROJECT_ROOT / str(entry.get("artifact_path", ""))
        if artifact_path.exists():
            entry["artifact_sha256"] = sha256_file(artifact_path)
        entry["baseline_majority_class"] = int(development[target].astype(int).mean() >= 0.5)
        summary_rows.append(
            {
                "horizon_days": horizon,
                "model_name": entry.get("model_name", "unknown"),
                "status": acceptance["status"],
                "policy_version": POLICY["policy_version"],
                "strongest_baseline": baseline_name,
                "balanced_accuracy_gain": acceptance["balanced_accuracy_gain"],
                "bootstrap_gain_lower": acceptance["bootstrap"]["lower"],
                "bootstrap_gain_upper": acceptance["bootstrap"]["upper"],
                "brier_skill": acceptance["calibration"]["brier_skill"],
                "failed_checks": ";".join(acceptance["failed_checks"]),
            }
        )
    registry["schema_version"] = 3
    registry.setdefault("benchmark_context", {})["acceptance_policy_version"] = POLICY["policy_version"]
    registry["benchmark_context"]["acceptance_policy_path"] = "config/horizon_acceptance_policy.json"
    registry["benchmark_context"]["status_tiers"] = ["accepted", "limited", "experimental"]
    temporary = REGISTRY_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(REGISTRY_FILE)
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "horizon-acceptance-summary.csv", index=False)
    pd.DataFrame(ticker_rows).to_csv(RESULTS_DIR / "stock-model-per-ticker-acceptance.csv", index=False)
    report = {
        "run_id": registry.get("run_id"),
        "policy_version": POLICY["policy_version"],
        "status_by_horizon": {str(row["horizon_days"]): row["status"] for row in summary_rows},
        "summary_path": "results/horizon-acceptance-summary.csv",
        "ticker_summary_path": "results/stock-model-per-ticker-acceptance.csv",
    }
    report["registry_signature"] = sha256_file(REGISTRY_FILE)
    (RESULTS_DIR / "horizon-acceptance-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(recompute_acceptance(), indent=2))
