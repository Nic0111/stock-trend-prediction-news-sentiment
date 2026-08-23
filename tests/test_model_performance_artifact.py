from __future__ import annotations

import csv

import numpy as np

from pipeline.generate_model_performance_artifact import (
    _calibration_points,
    _candidate_family_summary,
    _curve_points,
)


def test_curve_points_downsamples_without_losing_endpoints() -> None:
    x = np.linspace(0.0, 1.0, 1000)
    y = x**2

    points = _curve_points(x, y, x_key="x", y_key="y")

    assert len(points) <= 101
    assert points[0] == {"x": 0.0, "y": 0.0}
    assert points[-1] == {"x": 1.0, "y": 1.0}


def test_calibration_points_are_bounded_and_presentation_ready() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    probabilities = np.array([0.05, 0.15, 0.30, 0.65, 0.75, 0.90, 0.80, 0.40])

    points = _calibration_points(labels, probabilities)

    assert points
    assert all(0.0 <= point["predicted"] <= 1.0 for point in points)
    assert all(0.0 <= point["observed"] <= 1.0 for point in points)


def test_candidate_family_summary_keeps_only_each_family_best_row(tmp_path) -> None:
    path = tmp_path / "stock-model-candidates.csv"
    fieldnames = ["horizon_days", "model_name", "cv_balanced_accuracy", "cv_macro_f1", "cv_mcc", "refinement_round"]
    rows = [
        {"horizon_days": "1", "model_name": "logistic", "cv_balanced_accuracy": "0.51", "cv_macro_f1": "0.50", "cv_mcc": "0.02", "refinement_round": "0"},
        {"horizon_days": "1", "model_name": "logistic", "cv_balanced_accuracy": "0.54", "cv_macro_f1": "0.53", "cv_mcc": "0.08", "refinement_round": "1"},
        {"horizon_days": "1", "model_name": "xgboost", "cv_balanced_accuracy": "0.52", "cv_macro_f1": "0.51", "cv_mcc": "0.04", "refinement_round": "0"},
        {"horizon_days": "5", "model_name": "logistic", "cv_balanced_accuracy": "0.99", "cv_macro_f1": "0.99", "cv_mcc": "0.98", "refinement_round": "0"},
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _candidate_family_summary(tmp_path, 1)

    assert [row["model_name"] for row in summary] == ["logistic", "xgboost"]
    assert summary[0]["balanced_accuracy"] == 0.54
    assert summary[0]["refinement_round"] == 1
