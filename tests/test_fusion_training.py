from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fusion_model import FUSION_FEATURE_NAMES, FusionRegistry
from pipeline.fusion_training import (
    build_fusion_coverage_report,
    classification_metrics,
    expanding_purged_folds,
    promote_fusion_registry,
    write_fusion_signal_diagnostics,
)


def _development_frame() -> pd.DataFrame:
    dates = pd.date_range("2019-01-02", "2023-12-29", freq="W-FRI")
    rows = len(dates)
    frame = pd.DataFrame(
        {
            "ticker": ["1155.KL"] * rows,
            "observation_date": dates,
            "target_date": dates + pd.Timedelta(days=2),
            "horizon_days": [1] * rows,
            "target_up": np.arange(rows) % 2,
        }
    )
    for feature in FUSION_FEATURE_NAMES:
        frame[feature] = 0.5
    frame["stock_probability_up"] = np.where(frame["target_up"].eq(1), 0.6, 0.4)
    frame["news_coverage_score"] = 0.5
    frame["news_available"] = 1
    return frame


class FusionTrainingTests(unittest.TestCase):
    def test_expanding_folds_are_chronological_and_purged(self) -> None:
        frame = _development_frame()
        splits = expanding_purged_folds(frame, 1)
        self.assertEqual(len(splits), 5)
        for training, validation in splits:
            validation_start = frame.iloc[validation]["observation_date"].min()
            self.assertTrue(
                (frame.iloc[training]["target_date"] < validation_start).all()
            )

    def test_metrics_include_calibration_and_class_balance_fields(self) -> None:
        metrics = classification_metrics(
            [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.5
        )
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)
        self.assertEqual(metrics["up_recall"], 1.0)
        self.assertEqual(metrics["down_recall"], 1.0)

    def test_promotion_copies_only_accepted_staged_horizons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged_path = root / "staged.json"
            active_path = root / "active.json"
            FusionRegistry(
                "2026-07-24T00:00:00+00:00",
                "dataset",
                {
                    "1": {"validation_status": "accepted", "artifact_path": "one"},
                    "5": {
                        "validation_status": "experimental",
                        "artifact_path": "five",
                    },
                },
            ).write(staged_path)
            with self.assertRaises(ValueError):
                promote_fusion_registry(
                    staged_path, active_registry_path=active_path
                )
            active = promote_fusion_registry(
                staged_path, active_registry_path=active_path, approve=True
            )
        self.assertIn("1", active.horizons)
        self.assertNotIn("5", active.horizons)

    def test_coverage_report_exposes_period_specific_gaps(self) -> None:
        rows: list[pd.DataFrame] = []
        for horizon in (1, 5, 10):
            for year in (2019, 2024, 2025):
                frame = _development_frame().iloc[:2].copy()
                frame["ticker"] = "1155.KL"
                frame["horizon_days"] = horizon
                frame["observation_date"] = pd.date_range(
                    f"{year}-01-02", periods=2, freq="D"
                )
                frame["target_date"] = frame["observation_date"] + pd.Timedelta(
                    days=horizon + 1
                )
                rows.append(frame)
        complete = pd.concat(rows, ignore_index=True)
        # A second ticker has data but no eligible 2024 calibration evidence.
        incomplete = complete.copy()
        incomplete["ticker"] = "1295.KL"
        incomplete.loc[
            incomplete["observation_date"].dt.year.eq(2024),
            "news_coverage_score",
        ] = 0.0
        report = build_fusion_coverage_report(
            pd.concat([complete, incomplete], ignore_index=True)
        )
        self.assertFalse(report["ready_for_requested_universe"])
        self.assertEqual(report["complete_all_horizons"], ["1155.KL"])
        calibration = report["horizons"]["1"]["tickers"]["1295.KL"][
            "periods"
        ]["calibration"]
        self.assertEqual(calibration["any_news_rows"], 2)
        self.assertEqual(calibration["eligible_medium_high_rows"], 0)

    def test_signal_diagnostics_is_explicitly_non_promotional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows: list[pd.DataFrame] = []
            for horizon in (1, 5, 10):
                frame = _development_frame().iloc[:8].copy()
                frame["horizon_days"] = horizon
                frame["news_weighted_sentiment"] = np.linspace(-1.0, 1.0, 8)
                rows.append(frame)
            dataset = root / "fusion.parquet"
            output = root / "signal.json"
            pd.concat(rows, ignore_index=True).to_parquet(dataset, index=False)
            report = write_fusion_signal_diagnostics(
                dataset, output_path=output
            )
            self.assertTrue(output.exists())
        self.assertFalse(report["production_promotion_allowed"])
        self.assertIn(
            "sentiment_roc_auc",
            report["horizons"]["1"]["coverage_tiers"][
                "all_news_available"
            ],
        )


if __name__ == "__main__":
    unittest.main()
