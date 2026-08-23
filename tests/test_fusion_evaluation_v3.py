from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fusion_evaluation_v3 import (
    FULL_FEATURES,
    _CapacityMatchedStockModel,
    _capacity_matched_frames,
    _metrics,
    _placebo_frame,
    _readiness,
    compare_fusion_v3_reports,
    load_policy,
    scientific_result_sha256,
)


def _placebo_input() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    )
    frame = pd.DataFrame(
        {
            "ticker": ["1155.KL"] * 4,
            "observation_date": dates,
            "news_available": [0.0, 1.0, 1.0, 0.0],
            "news_positive_impact": [0.0, 0.8, 0.2, 0.0],
            "news_negative_impact": [0.0, 0.1, 0.6, 0.0],
        }
    )
    for feature in FULL_FEATURES:
        if feature not in frame:
            frame[feature] = np.arange(len(frame), dtype=float)
    return frame


def test_v3_policy_is_frozen_and_hash_is_repeatable() -> None:
    policy, first_hash = load_policy(Path("config/fusion_evaluation_v3.json"))
    _, second_hash = load_policy(Path("config/fusion_evaluation_v3.json"))
    assert policy["schema_version"] == "fusion-evaluation-v3"
    assert policy["periods"]["development"] == [2019, 2023]
    assert policy["periods"]["calibration"] == [2024, 2024]
    assert policy["periods"]["benchmark"] == [2025, 2025]
    assert first_hash == second_hash


def test_v3_metrics_confusion_matrix_matches_recall_and_balanced_accuracy() -> None:
    target = np.asarray([0, 0, 0, 1, 1, 1])
    probability = np.asarray([0.1, 0.4, 0.8, 0.2, 0.6, 0.9])
    result = _metrics(target, probability, 0.5)
    matrix = result["confusion_matrix"]
    assert matrix == [[2, 1], [1, 2]]
    assert result["down_recall"] == 2 / 3
    assert result["up_recall"] == 2 / 3
    assert result["balanced_accuracy"] == (
        result["down_recall"] + result["up_recall"]
    ) / 2


def test_placebos_preserve_ticker_month_feature_distributions() -> None:
    source = _placebo_input()
    group = [source["ticker"], source["observation_date"].dt.year, source["observation_date"].dt.month]
    for mode in ("content_only", "complete_event"):
        shuffled = _placebo_frame(source, mode=mode, seed=41)
        for feature in FULL_FEATURES:
            expected = source.groupby(group, sort=False)[feature].apply(
                lambda values: sorted(values.tolist())
            )
            actual = shuffled.groupby(group, sort=False)[feature].apply(
                lambda values: sorted(values.tolist())
            )
            assert actual.equals(expected)
        assert sorted(shuffled["news_available"].tolist()) == sorted(
            source["news_available"].tolist()
        )


def test_content_placebo_preserves_news_availability_row_by_row() -> None:
    source = _placebo_input()
    shuffled = _placebo_frame(source, mode="content_only", seed=7)
    assert shuffled["news_available"].tolist() == source["news_available"].tolist()


def test_scientific_hash_ignores_paths_and_timestamps() -> None:
    first = {
        "created_at": "2026-08-21T00:00:00Z",
        "dataset_sha256": "dataset",
        "artifact_path": "models/first.joblib",
        "horizons": {"1": {"balanced_accuracy_gain": 0.01}},
    }
    second = {
        "created_at": "2026-08-22T00:00:00Z",
        "dataset_sha256": "dataset",
        "artifact_path": "models/second.joblib",
        "horizons": {"1": {"balanced_accuracy_gain": 0.01}},
    }
    assert scientific_result_sha256(first) == scientific_result_sha256(second)


def test_compare_v3_reports_uses_canonical_scientific_hash(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "comparison.json"
    first.write_text(
        json.dumps(
            {"created_at": "one", "dataset_sha256": "same", "accepted_horizons": []}
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {"created_at": "two", "dataset_sha256": "same", "accepted_horizons": []}
        ),
        encoding="utf-8",
    )
    result = compare_fusion_v3_reports(first, second, output_path=output)
    assert result["passed"] is True
    assert output.is_file()


def test_readiness_reports_actionable_company_evidence_gaps() -> None:
    policy, _ = load_policy(Path("config/fusion_evaluation_v3.json"))
    development = pd.DataFrame(
        {
            "ticker": ["1155.KL", "1295.KL"],
            "news_material_event_ids": ['["event-a"]', "[]"],
        }
    )
    calibration = development.iloc[0:0].copy()
    benchmark = pd.DataFrame(
        {
            "ticker": ["1155.KL", "1295.KL"],
            "news_available": [1.0, 0.0],
        }
    )
    result = _readiness(
        development,
        calibration,
        benchmark,
        policy,
        np.asarray([0.01, -0.01]),
    )
    assert result["evidence_gaps"]["news_present_benchmark_rows"] == 299
    assert result["evidence_gaps"]["news_present_by_company"]["1155.KL"] == 19
    assert result["evidence_gaps"]["news_present_by_company"]["1295.KL"] == 20
    assert (
        result["evidence_gaps"]["unique_material_events_by_company"]["1155.KL"]
        == 99
    )


def test_capacity_matched_stock_model_uses_no_news_features() -> None:
    frame = pd.DataFrame(
        {
            "stock_probability_up": [0.2, 0.35, 0.65, 0.8, 0.55, 0.45],
            "stock_threshold_margin": [-0.3, -0.15, 0.15, 0.3, 0.05, -0.05],
            "stock_evidence_strength": ["accepted", "limited", "accepted", "accepted", "experimental", "limited"],
            "news_available": [0, 1, 1, 0, 1, 0],
        }
    )
    model = _CapacityMatchedStockModel(1.0).fit(frame, [0, 0, 1, 1, 1, 0])
    probabilities = model.predict(frame)
    assert len(probabilities) == len(frame)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert model.provenance()["news_features_used"] is False


def test_capacity_matched_frames_preserve_raw_stock_probability() -> None:
    policy, _ = load_policy(Path("config/fusion_evaluation_v3.json"))
    dates = pd.date_range("2019-01-01", periods=80, freq="B")
    frame = pd.DataFrame(
        {
            "observation_date": dates,
            "target_date": dates + pd.Timedelta(days=1),
            "ticker": ["1155.KL"] * len(dates),
            "target_up": [0, 1] * 40,
            "stock_probability_up": np.linspace(0.2, 0.8, len(dates)),
            "stock_threshold_margin": np.linspace(-0.3, 0.3, len(dates)),
            "stock_evidence_strength": ["accepted"] * len(dates),
            "news_available": [1] * len(dates),
        }
    )
    development, calibration, benchmark, selected, trials = _capacity_matched_frames(
        frame.iloc[:60].reset_index(drop=True),
        frame.iloc[60:70].reset_index(drop=True),
        frame.iloc[70:].reset_index(drop=True),
        policy,
    )
    assert "stock_probability_raw" in development
    assert np.allclose(
        development["stock_probability_raw"], frame.iloc[:60]["stock_probability_up"]
    )
    assert len(trials) > 0
    assert selected["family"] == "ridge"
    assert len(calibration) == 10 and len(benchmark) == 10
