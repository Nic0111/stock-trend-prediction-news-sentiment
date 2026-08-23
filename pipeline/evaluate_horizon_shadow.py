"""Evaluate frozen stock champions on a forward 2026 shadow period.

This command never retrains, selects a model or changes a probability
threshold. It only evaluates the current registry artifacts and records the
shadow status alongside the observed 2025 benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yfinance as yf

from pipeline.horizon_acceptance import POLICY, evaluate_acceptance, weaker_status
from pipeline.stock_features import add_stock_targets, build_stock_features
from pipeline.train_stock_models import (
    BASELINE_FILE,
    build_baseline_predictions,
    calculate_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = PROJECT_ROOT / "models" / "model-registry.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "horizon-shadow"
METADATA_FILE = PROJECT_ROOT / "data" / "stock-metadata.csv"
MARKET_TICKER = "^KLSE"


def _download(ticker: str, start: date, end_exclusive: date) -> pd.DataFrame:
    frame = yf.download(
        ticker,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise ValueError(f"No shadow data returned for {ticker}")
    frame = frame.reset_index()
    if "Date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "Date"})
    frame["Ticker"] = ticker
    return frame


def build_shadow_frame(start: date, end: date) -> pd.DataFrame:
    """Download enough buffer to construct complete shadow features/targets."""
    buffer_start = start - timedelta(days=420)
    download_end = min(date.today(), end + timedelta(days=45))
    metadata = pd.read_csv(METADATA_FILE)
    tickers = [str(value).strip().upper() for value in metadata["Ticker"].tolist()]
    stocks = [_download(ticker, buffer_start, download_end) for ticker in tickers]
    market = _download(MARKET_TICKER, buffer_start, download_end)
    features = add_stock_targets(build_stock_features(pd.concat(stocks, ignore_index=True), market))
    return features[
        (features["Date"] >= pd.Timestamp(start))
        & (features["Date"] <= pd.Timestamp(end))
    ].copy()


def _write_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run_shadow_evaluation(
    start: date | str = date(2026, 1, 1),
    end: date | str = date(2026, 6, 30),
    *,
    registry_path: Path = REGISTRY_FILE,
) -> dict[str, Any]:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if end_date < start_date:
        raise ValueError("Shadow end date must not precede its start date")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    frame = build_shadow_frame(start_date, end_date)
    frame_hash = hashlib.sha256(
        frame.sort_values(["Date", "Ticker"]).to_csv(index=False).encode("utf-8")
    ).hexdigest()
    output: dict[str, Any] = {
        "policy_version": POLICY["policy_version"],
        "phase": "prospective_shadow" if start_date >= date(2026, 7, 23) else "consumed_shadow",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "dataset_sha256": frame_hash,
        "horizons": {},
    }
    for horizon in (1, 5, 10):
        horizon_registry = registry.get("horizons", {}).get(str(horizon), {})
        feature_columns = list(horizon_registry.get("numeric_feature_columns", []))
        input_columns = list(horizon_registry.get("input_columns", []))
        target_column = f"Target_{horizon}D"
        valid = frame.dropna(subset=[target_column, *input_columns]).copy()
        if valid.empty:
            raise ValueError(f"No complete shadow rows are available for {horizon}D")
        artifact_path = PROJECT_ROOT / str(horizon_registry["artifact_path"])
        model = joblib.load(artifact_path)
        probability = model.predict_proba(valid[input_columns])[:, 1]
        predictions = (probability >= float(model.threshold)).astype(int)
        y_true = valid[target_column].astype(int).to_numpy()
        baseline_name = str(horizon_registry.get("strongest_baseline", "majority-class"))
        baselines = build_baseline_predictions(
            valid,
            valid,
            horizon,
            majority_class=int(horizon_registry.get("baseline_majority_class", 0)),
        )
        if baseline_name not in baselines:
            raise ValueError(f"Locked baseline {baseline_name} is unavailable for {horizon}D shadow data")
        baseline = baselines[baseline_name]
        metrics = calculate_metrics(y_true, predictions, probability)
        baseline_metrics = calculate_metrics(y_true, baseline)
        acceptance = evaluate_acceptance(
            horizon=horizon,
            y_true=y_true,
            predictions=predictions,
            probability_up=probability,
            baseline_predictions=baseline,
            dates=valid["Date"].to_numpy(),
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            oof_metrics=horizon_registry.get("oof_metrics", {}),
            cv_positive_mcc_folds=int(horizon_registry.get("cv_positive_mcc_folds", 0)),
            baseline_prevalence=float(
                horizon_registry.get(
                    "baseline_prevalence",
                    horizon_registry.get("baseline_majority_class", 0),
                )
            ),
            phase="shadow",
        )
        unique_dates = int(pd.to_datetime(valid["Date"], errors="coerce").nunique())
        output["horizons"][str(horizon)] = {
            **acceptance,
            "rows": int(len(valid)),
            "unique_dates": unique_dates,
            "matured_shadow_sessions": unique_dates,
            "target_dates": {
                "minimum": str(valid[f"Target_Date_{horizon}D"].min().date()),
                "maximum": str(valid[f"Target_Date_{horizon}D"].max().date()),
            },
        }
        horizon_registry["shadow_acceptance"] = output["horizons"][str(horizon)]
        horizon_registry["effective_status"] = weaker_status(
            str(horizon_registry.get("global_status", horizon_registry.get("status", "experimental"))),
            str(acceptance["status"]),
        )
    run_id = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    output["run_id"] = run_id
    report_path = RESULTS_DIR / f"shadow-{run_id}.json"
    _write_atomic(output, report_path)
    registry["shadow_evaluation"] = {
        "run_id": run_id,
        "phase": output["phase"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "dataset_sha256": frame_hash,
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
        "policy_version": POLICY["policy_version"],
        "horizons": output["horizons"],
    }
    registry["benchmark_context"]["shadow_status"] = "evaluated"
    _write_atomic(registry, registry_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen stock models on 2026 shadow data")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--run-id", help="Evaluate a staged official run instead of production")
    args = parser.parse_args()
    registry_path = REGISTRY_FILE
    if args.run_id:
        registry_path = PROJECT_ROOT / "models" / "staged-stock-runs" / args.run_id / "model-registry.json"
    print(json.dumps(run_shadow_evaluation(args.start, args.end, registry_path=registry_path), indent=2))


if __name__ == "__main__":
    main()
