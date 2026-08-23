"""Generate leakage-safe stock probabilities for the historical fusion table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from pipeline.model_artifacts import CalibratedStockModel, probability_to_logit
from pipeline.predict_stock_trend import REGISTRY_FILE, SUPPORTED_HORIZONS
from pipeline.train_stock_models import select_directional_threshold
from pipeline.train_stock_models import (
    _official_candidate_sort_key,
    available_model_names,
    build_purged_date_splits,
    feature_columns_for_round,
    run_candidate_search,
    search_plan_for_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUSION_OOF_SCHEMA_VERSION = "fusion-stock-oof-v1"
SCIENTIFIC_FUSION_OOF_SCHEMA_VERSION = "fusion-stock-oof-v2"


def read_stock_feature_input(path: Path | str) -> pd.DataFrame:
    """Read one stock-feature CSV or all model-input CSVs in a directory."""

    source = Path(path)
    files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    if not files or any(not file.exists() for file in files):
        raise FileNotFoundError(source)
    frame = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    if frame.empty:
        raise ValueError(f"Stock feature input is empty: {source}")
    return frame


def _load_templates(registry_path: Path) -> tuple[dict[str, Any], dict[int, CalibratedStockModel]]:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    templates: dict[int, CalibratedStockModel] = {}
    for horizon in SUPPORTED_HORIZONS:
        entry = registry.get("horizons", {}).get(str(horizon), {})
        reference = str(entry.get("artifact_path", "")).strip()
        if not reference:
            raise ValueError(f"Stock registry has no artifact for horizon {horizon}")
        path = Path(reference)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        artifact = joblib.load(path)
        if not isinstance(artifact, CalibratedStockModel):
            raise TypeError(f"Stock artifact for horizon {horizon} has an unexpected type")
        templates[horizon] = artifact
    return registry, templates


def _inner_purged_splits(
    frame: pd.DataFrame,
    target_date_column: str,
    *,
    fold_count: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    targets = pd.to_datetime(frame[target_date_column], errors="raise").dt.normalize()
    unique_dates = np.asarray(sorted(dates.unique()))
    blocks = [block for block in np.array_split(unique_dates, fold_count + 1) if len(block)]
    if len(blocks) < 3:
        raise ValueError("Not enough dates to calibrate an outer stock fold")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks[1:]:
        start = pd.Timestamp(block[0])
        train = np.flatnonzero(((dates < start) & (targets < start)).to_numpy())
        validation = np.flatnonzero(dates.isin(block).to_numpy())
        if len(train) and len(validation):
            splits.append((train, validation))
    if len(splits) < 2:
        raise ValueError("Not enough non-empty purged inner folds for stock calibration")
    return splits


def _fit_outer_stock_model(
    template: CalibratedStockModel,
    training: pd.DataFrame,
    *,
    horizon_days: int,
) -> tuple[Any, LogisticRegression, float]:
    target_column = f"Target_{horizon_days}D"
    target_date_column = f"Target_Date_{horizon_days}D"
    X = training[list(template.input_columns)]
    y = training[target_column].astype(int)
    raw_probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for train_indices, validation_indices in _inner_purged_splits(
        training, target_date_column
    ):
        estimator = clone(template.estimator)
        estimator.fit(X.iloc[train_indices], y.iloc[train_indices])
        raw_probabilities.append(
            estimator.predict_proba(X.iloc[validation_indices])[:, 1]
        )
        labels.append(y.iloc[validation_indices].to_numpy())
    raw = np.concatenate(raw_probabilities)
    target = np.concatenate(labels)
    if np.unique(target).size < 2:
        raise ValueError("Nested stock calibration requires both target classes")
    calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=42)
    calibrator.fit(probability_to_logit(raw), target)
    calibrated = calibrator.predict_proba(probability_to_logit(raw))[:, 1]
    threshold, _, _ = select_directional_threshold(target, calibrated)
    estimator = clone(template.estimator)
    estimator.fit(X, y)
    return estimator, calibrator, float(threshold)


def build_stock_oof_probabilities(
    stock_frame: pd.DataFrame,
    *,
    registry_path: Path = REGISTRY_FILE,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
    output_start_year: int = 2019,
    output_end_year: int = 2025,
) -> pd.DataFrame:
    """Cross-fit frozen stock architectures before fusion-model training.

    Each output block is predicted by an estimator trained only on earlier
    target-complete rows.  Calibration and threshold selection are nested
    inside that earlier training period, so the output row's label is never
    used to produce its stock probability.
    """

    registry, templates = _load_templates(Path(registry_path))
    frame = stock_frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["Ticker"] = frame["Ticker"].astype(str).str.upper().str.strip()
    frame = frame.dropna(subset=["Date", "Ticker"]).sort_values(["Date", "Ticker"])
    rows: list[dict[str, Any]] = []
    for horizon_days in horizons:
        if horizon_days not in templates:
            raise ValueError(f"Unsupported OOF horizon: {horizon_days}")
        template = templates[horizon_days]
        target_column = f"Target_{horizon_days}D"
        target_date_column = f"Target_Date_{horizon_days}D"
        required = {
            "Date",
            "Ticker",
            target_column,
            target_date_column,
            *template.input_columns,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Stock OOF frame is missing columns for {horizon_days}D: {missing}")
        horizon_frame = frame.dropna(
            subset=[target_column, target_date_column, *template.input_columns]
        ).copy()
        target_dates = pd.to_datetime(
            horizon_frame[target_date_column], errors="coerce"
        ).dt.normalize()
        horizon_frame[target_date_column] = target_dates
        for year in range(output_start_year, output_end_year + 1):
            validation = horizon_frame.loc[horizon_frame["Date"].dt.year.eq(year)].copy()
            if validation.empty:
                continue
            validation_start = validation["Date"].min()
            training = horizon_frame.loc[
                (horizon_frame["Date"] < validation_start)
                & (horizon_frame[target_date_column] < validation_start)
            ].copy()
            if training.empty:
                raise ValueError(
                    f"No earlier target-complete stock rows for {horizon_days}D year {year}"
                )
            estimator, calibrator, threshold = _fit_outer_stock_model(
                template, training, horizon_days=horizon_days
            )
            raw = estimator.predict_proba(
                validation[list(template.input_columns)]
            )[:, 1]
            probability = calibrator.predict_proba(probability_to_logit(raw))[:, 1]
            for position, (_, source) in enumerate(validation.iterrows()):
                margin = float(probability[position]) - threshold
                strength = (
                    "Strong"
                    if abs(margin) >= 0.05
                    else "Moderate"
                    if abs(margin) >= 0.02
                    else "Limited"
                )
                rows.append(
                    {
                        "Date": pd.Timestamp(source["Date"]).date().isoformat(),
                        "Ticker": str(source["Ticker"]),
                        "horizon": int(horizon_days),
                        "probability_up": float(probability[position]),
                        "decision_threshold": threshold,
                        "threshold_margin": margin,
                        "evidence_strength": strength,
                        "prediction_role": "nested_out_of_fold",
                        "compatibility_role": "active_production_architecture",
                        "training_end": training["Date"].max().date().isoformat(),
                        "validation_start": validation_start.date().isoformat(),
                        "stock_model_name": template.model_name,
                        "stock_model_hyperparameters": json.dumps(
                            template.estimator.get_params(deep=True),
                            sort_keys=True,
                            default=str,
                        ),
                        "stock_registry_run_id": registry.get("run_id"),
                        "stock_registry_dataset_sha256": registry.get("dataset_sha256"),
                        "stock_feature_schema_sha256": hashlib.sha256(
                            json.dumps(list(template.input_columns)).encode("utf-8")
                        ).hexdigest(),
                        "fold_id": f"runtime-compatibility-{horizon_days}d-{year}",
                        "schema_version": FUSION_OOF_SCHEMA_VERSION,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["Date", "Ticker", "horizon"]
    ).reset_index(drop=True)


def build_scientific_stock_oof_probabilities(
    stock_frame: pd.DataFrame,
    *,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
    output_start_year: int = 2019,
    output_end_year: int = 2025,
    search_profile: str = "focused",
    stock_dataset_sha256: str | None = None,
) -> pd.DataFrame:
    """Select the stock architecture independently inside every outer year.

    This is the headline fusion-study input.  Candidate selection, calibration
    and threshold selection see only target-complete rows before the output
    year.  It is intentionally slower than the runtime-compatibility builder.
    """

    frame = stock_frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame["Ticker"] = frame["Ticker"].astype(str).str.upper().str.strip()
    available = available_model_names()
    search_plan = search_plan_for_profile(search_profile, available)
    plan_hash = hashlib.sha256(
        json.dumps(search_plan, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()
    scientific_run_id = (
        f"scientific-oof-{plan_hash[:12]}-{(stock_dataset_sha256 or 'unknown')[:12]}"
    )
    rows: list[dict[str, Any]] = []
    for horizon_days in horizons:
        target_column = f"Target_{horizon_days}D"
        target_date_column = f"Target_Date_{horizon_days}D"
        for year in range(output_start_year, output_end_year + 1):
            validation = frame.loc[frame["Date"].dt.year.eq(year)].copy()
            if validation.empty:
                continue
            validation_start = validation["Date"].min()
            candidate_results: list[tuple[Any, dict[str, Any], pd.DataFrame]] = []
            for refinement_round, model_names in search_plan[int(horizon_days)].items():
                feature_columns = feature_columns_for_round(int(refinement_round))
                required = [target_column, target_date_column, *feature_columns]
                candidate_frame = frame.dropna(subset=required).copy()
                candidate_frame[target_date_column] = pd.to_datetime(
                    candidate_frame[target_date_column], errors="coerce"
                ).dt.normalize()
                training = candidate_frame.loc[
                    (candidate_frame["Date"] < validation_start)
                    & (candidate_frame[target_date_column] < validation_start)
                ].sort_values(["Date", "Ticker"]).reset_index(drop=True)
                if training.empty:
                    continue
                splits = build_purged_date_splits(
                    training, target_date_column, n_splits=5
                )
                for model_name in model_names:
                    estimator, summary, _ = run_candidate_search(
                        training,
                        int(horizon_days),
                        int(refinement_round),
                        str(model_name),
                        feature_columns,
                        splits,
                        search_profile,
                    )
                    candidate_results.append((estimator, summary, training))
            if not candidate_results:
                raise ValueError(
                    f"No nested stock candidate could be fitted for {horizon_days}D/{year}"
                )
            estimator, summary, training = min(
                candidate_results, key=lambda item: _official_candidate_sort_key(item[1])
            )
            input_columns = list(summary["input_columns"])
            # Recreate leakage-safe inner calibration using the selected outer
            # architecture; no output-year label enters this operation.
            raw_probabilities: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            for train_index, calibration_index in _inner_purged_splits(
                training, target_date_column
            ):
                inner = clone(estimator)
                inner.fit(
                    training.iloc[train_index][input_columns],
                    training.iloc[train_index][target_column].astype(int),
                )
                raw_probabilities.append(
                    inner.predict_proba(training.iloc[calibration_index][input_columns])[:, 1]
                )
                labels.append(
                    training.iloc[calibration_index][target_column].astype(int).to_numpy()
                )
            raw = np.concatenate(raw_probabilities)
            target = np.concatenate(labels)
            calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", random_state=42)
            calibrator.fit(probability_to_logit(raw), target)
            calibrated = calibrator.predict_proba(probability_to_logit(raw))[:, 1]
            threshold, _, _ = select_directional_threshold(target, calibrated)
            estimator.fit(training[input_columns], training[target_column].astype(int))
            usable_validation = validation.dropna(subset=input_columns).copy()
            predicted_raw = estimator.predict_proba(usable_validation[input_columns])[:, 1]
            probability = calibrator.predict_proba(probability_to_logit(predicted_raw))[:, 1]
            parameters = json.loads(summary["best_parameters"])
            feature_hash = hashlib.sha256(
                json.dumps(input_columns, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for position, (_, source) in enumerate(usable_validation.iterrows()):
                margin = float(probability[position]) - float(threshold)
                rows.append(
                    {
                        "Date": pd.Timestamp(source["Date"]).date().isoformat(),
                        "Ticker": str(source["Ticker"]),
                        "horizon": int(horizon_days),
                        "probability_up": float(probability[position]),
                        "decision_threshold": float(threshold),
                        "threshold_margin": margin,
                        "evidence_strength": "Strong" if abs(margin) >= 0.05 else "Moderate" if abs(margin) >= 0.02 else "Limited",
                        "prediction_role": "scientific_nested_model_selection_oof",
                        "fold_id": f"scientific-{horizon_days}d-{year}",
                        "training_end": training["Date"].max().date().isoformat(),
                        "calibration_end": training["Date"].max().date().isoformat(),
                        "validation_start": validation_start.date().isoformat(),
                        "stock_model_name": summary["model_name"],
                        "stock_model_hyperparameters": json.dumps(parameters, sort_keys=True),
                        "stock_registry_run_id": scientific_run_id,
                        "stock_registry_dataset_sha256": stock_dataset_sha256,
                        "stock_feature_schema_sha256": feature_hash,
                        "selection_policy_sha256": plan_hash,
                        "schema_version": SCIENTIFIC_FUSION_OOF_SCHEMA_VERSION,
                    }
                )
    return pd.DataFrame(rows).sort_values(["Date", "Ticker", "horizon"]).reset_index(drop=True)


def run_build_stock_oof_probabilities(
    stock_path: str,
    *,
    registry_path: str | None = None,
    output_path: str,
    mode: str = "runtime-compatibility",
    search_profile: str = "focused",
) -> dict[str, Any]:
    stock = read_stock_feature_input(stock_path)
    source = Path(stock_path)
    source_files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    digest = hashlib.sha256()
    for file_path in source_files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(file_path.read_bytes())
    stock_dataset_sha256 = digest.hexdigest()
    if mode == "scientific":
        result = build_scientific_stock_oof_probabilities(
            stock,
            search_profile=search_profile,
            stock_dataset_sha256=stock_dataset_sha256,
        )
        schema_version = SCIENTIFIC_FUSION_OOF_SCHEMA_VERSION
    elif mode == "runtime-compatibility":
        result = build_stock_oof_probabilities(
            stock,
            registry_path=Path(registry_path) if registry_path else REGISTRY_FILE,
        )
        schema_version = FUSION_OOF_SCHEMA_VERSION
    else:
        raise ValueError("mode must be 'scientific' or 'runtime-compatibility'")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.casefold() in {".parquet", ".pq"}:
        result.to_parquet(target, index=False)
    else:
        result.to_csv(target, index=False)
    payload = {
        "schema_version": schema_version,
        "mode": mode,
        "search_profile": search_profile if mode == "scientific" else None,
        "stock_dataset_sha256": stock_dataset_sha256,
        "output_path": str(target),
        "rows": int(len(result)),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    print(json.dumps(payload, indent=2))
    return payload
