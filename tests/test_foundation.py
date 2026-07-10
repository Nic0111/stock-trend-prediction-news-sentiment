from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.download_financial_news_data import (
    build_company_query_variants,
    build_news_contexts,
    filter_articles,
    is_relevant_article,
)
from pipeline.model_artifacts import CalibratedStockModel
from pipeline.predict_stock_trend import PROJECT_ROOT, predict_stock_trend
from pipeline.stock_features import (
    BASE_FEATURE_COLUMNS,
    STATIONARY_FEATURE_COLUMNS,
    add_stock_targets,
    build_stock_features,
)
from pipeline.train_stock_models import (
    build_purged_date_splits,
    prepare_horizon_frame,
    split_development_and_test,
)


def make_stock_data(
    tickers: tuple[str, ...] = ("1155.KL",),
    start: str = "2021-01-04",
    periods: int = 1300,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    frames = []
    for ticker_index, ticker in enumerate(tickers):
        sequence = np.arange(periods, dtype=float)
        close = 5.0 + ticker_index + 0.002 * sequence + 0.2 * np.sin(sequence / 15)
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "Open": close * 0.998,
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Adj Close": close,
                    "Volume": 1_000_000 + ((sequence * 10_003) % 500_000),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_market_data(start: str = "2021-01-04", periods: int = 1300) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    sequence = np.arange(periods, dtype=float)
    close = 1500 + 0.1 * sequence + 10 * np.sin(sequence / 25)
    return pd.DataFrame({"Date": dates, "Adj Close": close})


class ConstantEstimator:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.4, 0.6], (len(X), 1))


class ConstantCalibrator:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile([0.3, 0.7], (len(X), 1))


class StockFeatureTests(unittest.TestCase):
    def test_targets_keep_horizon_specific_missing_tails_and_dates(self) -> None:
        raw = make_stock_data(periods=90)
        features = build_stock_features(raw)
        targeted = add_stock_targets(features)

        for horizon in (1, 5, 10):
            target = targeted[f"Target_{horizon}D"]
            target_date = targeted[f"Target_Date_{horizon}D"]
            self.assertEqual(int(target.isna().sum()), horizon)
            self.assertEqual(int(target_date.isna().sum()), horizon)

        latest = targeted.sort_values("Date").iloc[-1]
        self.assertTrue(all(pd.notna(latest[column]) for column in STATIONARY_FEATURE_COLUMNS))
        self.assertTrue(pd.isna(latest["Target_1D"]))

    def test_market_features_are_finite_after_warmup(self) -> None:
        features = build_stock_features(make_stock_data(periods=100), make_market_data(periods=100))
        latest = features.sort_values("Date").iloc[-1]
        self.assertTrue(np.isfinite(float(latest["Market_Return_1D"])))
        self.assertTrue(np.isfinite(float(latest["Excess_Return_5D"])))


class TemporalSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw = make_stock_data(("1155.KL", "1295.KL", "1023.KL"))
        cls.combined = add_stock_targets(build_stock_features(raw, make_market_data()))

    def test_purged_folds_have_no_target_or_same_date_leakage(self) -> None:
        frame = prepare_horizon_frame(self.combined, 10, STATIONARY_FEATURE_COLUMNS)
        development, test = split_development_and_test(frame, 10)
        splits = build_purged_date_splits(development, "Target_Date_10D", n_splits=5)
        self.assertGreater(len(test), 0)
        for train_indexes, validation_indexes in splits:
            train = development.iloc[train_indexes]
            validation = development.iloc[validation_indexes]
            self.assertLess(train["Target_Date_10D"].max(), validation["Date"].min())
            self.assertFalse(set(train["Date"]).intersection(set(validation["Date"])))
            counts = validation.groupby("Date")["Ticker"].nunique()
            self.assertTrue((counts == 3).all())

    def test_final_boundary_purges_labels_entering_2025(self) -> None:
        frame = prepare_horizon_frame(self.combined, 10, STATIONARY_FEATURE_COLUMNS)
        development, test = split_development_and_test(frame, 10)
        self.assertLess(development["Target_Date_10D"].max(), pd.Timestamp("2025-01-01"))
        self.assertGreaterEqual(test["Date"].min(), pd.Timestamp("2025-01-01"))


class NewsContextTests(unittest.TestCase):
    def test_short_mb_alias_is_not_generated_or_matched(self) -> None:
        variants = build_company_query_variants("1155.KL", "Malayan Banking Berhad")
        self.assertNotIn("MB", variants)
        self.assertFalse(is_relevant_article("mb-stash 0.1.0", "A Python package", variants))
        self.assertTrue(
            is_relevant_article("Maybank reports profit growth", "Malaysian banking news", variants)
        )

    def test_filtering_records_alias_and_deduplicates(self) -> None:
        variants = build_company_query_variants("1023.KL", "CIMB Group Holdings Berhad")
        article = {
            "source": {"id": "example", "name": "Example"},
            "author": "Reporter",
            "title": "CIMB expands its regional business",
            "description": "CIMB Group announced an expansion.",
            "url": "https://example.com/cimb?utm_source=test",
            "urlToImage": None,
            "publishedAt": "2026-04-17T08:00:00Z",
            "content": "Details",
        }
        rows = filter_articles({"articles": [article, dict(article)]}, variants)
        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0]["matched_alias"], variants)

    def test_context_windows_weight_and_grade_coverage(self) -> None:
        rows = []
        for index in range(6):
            rows.append(
                {
                    "published_at": f"2026-04-{17-index:02d}T08:00:00Z",
                    "source_name": f"Source {index % 3}",
                    "sentiment_label": "positive" if index < 4 else "neutral",
                    "sentiment_score": 1 if index < 4 else 0,
                    "sentiment_probability": 0.8,
                    "sentiment_uncertain": False,
                }
            )
        contexts = build_news_contexts(rows, date(2026, 4, 18))
        self.assertEqual(contexts["1"]["coverage_confidence"], "High")
        self.assertGreater(contexts["1"]["weighted_sentiment_score"], 0)
        self.assertEqual(contexts["10"]["window_days"], 21)


class SentimentTrainingTests(unittest.TestCase):
    def test_quick_training_writes_evaluation_and_metadata(self) -> None:
        from pipeline import train_financial_news_sentiment_model as sentiment_training

        rows = []
        for label, word in [("positive", "gain"), ("neutral", "steady"), ("negative", "loss")]:
            for index in range(30):
                rows.append({"Sentiment": label, "Sentence": f"{word} company outlook {index}"})

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "sentiment.csv"
            pd.DataFrame(rows).to_csv(dataset, index=False)
            models = root / "models"
            results = root / "results"
            patches = {
                "DATASET_FILE": dataset,
                "MODELS_DIR": models,
                "RESULTS_DIR": results,
                "MODEL_FILE": models / "model.joblib",
                "METADATA_FILE": models / "metadata.json",
                "EVALUATION_FILE": results / "evaluation.csv",
                "REPORT_FILE": results / "classification-reports" / "report.txt",
                "CONFUSION_FILE": results / "confusion-matrices" / "matrix.csv",
            }
            patchers = [patch.object(sentiment_training, name, value) for name, value in patches.items()]
            for patcher in patchers:
                patcher.start()
            try:
                metadata = sentiment_training.run_sentiment_training_pipeline("quick")
            finally:
                for patcher in reversed(patchers):
                    patcher.stop()
            self.assertEqual(metadata["dataset_rows"], 90)
            self.assertTrue((models / "model.joblib").exists())
            self.assertTrue((results / "evaluation.csv").exists())


class InferenceTests(unittest.TestCase):
    def test_inference_uses_registry_schema_and_retains_latest_row(self) -> None:
        stock = make_stock_data(periods=100)
        input_columns = list(BASE_FEATURE_COLUMNS)
        artifact = CalibratedStockModel(
            estimator=ConstantEstimator(),
            calibrator=ConstantCalibrator(),
            threshold=0.55,
            input_columns=input_columns,
            numeric_feature_columns=input_columns,
            refinement_round=1,
            model_name="constant",
            horizon_days=5,
        )
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            temporary = Path(temporary_directory)
            artifact_path = temporary / "artifact.joblib"
            artifact_path.touch()
            registry_path = temporary / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "dataset_sha256": "abc1234567890",
                        "horizons": {
                            "5": {
                                "artifact_path": str(artifact_path.relative_to(PROJECT_ROOT)),
                                "model_name": "constant",
                                "refinement_round": 1,
                                "input_columns": input_columns,
                                "numeric_feature_columns": input_columns,
                                "threshold_passed": False,
                                "status": "experimental",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = predict_stock_trend(
                "1155.KL",
                5,
                as_of_date=stock["Date"].max().date(),
                stock_df=stock,
                registry_path=registry_path,
                model_loader=lambda _: artifact,
            )
        self.assertEqual(result.prediction_label, "Up")
        self.assertAlmostEqual(result.probability_up, 0.7)
        self.assertFalse(result.benchmark_passed)


if __name__ == "__main__":
    unittest.main()
