from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline.fusion_stock_oof import (
    build_stock_oof_probabilities,
    read_stock_feature_input,
)
from pipeline.model_artifacts import CalibratedStockModel


class FusionStockOOFTests(unittest.TestCase):
    def test_directory_input_combines_model_feature_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"Date": ["2025-01-01"], "Ticker": ["1155.KL"]}).to_csv(
                root / "a.csv", index=False
            )
            pd.DataFrame({"Date": ["2025-01-01"], "Ticker": ["1295.KL"]}).to_csv(
                root / "b.csv", index=False
            )
            result = read_stock_feature_input(root)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["Ticker"]), {"1155.KL", "1295.KL"})

    def _registry(self, root: Path) -> Path:
        X = pd.DataFrame({"x": [-1.0, 1.0, -0.5, 0.5]})
        y = np.asarray([0, 1, 0, 1])
        estimator = Pipeline([("classifier", LogisticRegression(random_state=42))])
        estimator.fit(X, y)
        calibrator = LogisticRegression(random_state=42).fit(
            np.asarray([[-1.0], [1.0], [-0.5], [0.5]]), y
        )
        entries = {}
        for horizon in (1, 5, 10):
            artifact = CalibratedStockModel(
                estimator=estimator,
                calibrator=calibrator,
                threshold=0.5,
                input_columns=["x"],
                numeric_feature_columns=["x"],
                refinement_round=1,
                model_name="logistic-test",
                horizon_days=horizon,
            )
            path = root / f"stock-{horizon}.joblib"
            joblib.dump(artifact, path)
            entries[str(horizon)] = {"artifact_path": str(path)}
        registry = root / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "run_id": "test",
                    "dataset_sha256": "test-data",
                    "horizons": entries,
                }
            ),
            encoding="utf-8",
        )
        return registry

    def test_nested_oof_rows_use_only_earlier_target_complete_data(self) -> None:
        dates = pd.date_range("2015-01-02", "2025-12-26", freq="W-FRI")
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Ticker": ["1155.KL"] * len(dates),
                "x": np.sin(np.arange(len(dates))),
                "Target_1D": np.arange(len(dates)) % 2,
                "Target_Date_1D": dates + pd.Timedelta(days=3),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            result = build_stock_oof_probabilities(
                frame,
                registry_path=self._registry(Path(directory)),
                horizons=[1],
            )
        self.assertFalse(result.empty)
        self.assertTrue(result["prediction_role"].eq("nested_out_of_fold").all())
        self.assertTrue(
            (
                pd.to_datetime(result["training_end"])
                < pd.to_datetime(result["validation_start"])
            ).all()
        )
        self.assertTrue(result["probability_up"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
