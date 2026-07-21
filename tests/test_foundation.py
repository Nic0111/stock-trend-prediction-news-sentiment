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
    CONTEXT_HALF_LIVES,
    CONTEXT_WINDOWS,
    build_company_query_variants,
    build_news_contexts,
    build_request_url,
    calculate_recency_weight,
    filter_articles,
    is_relevant_article,
)
from pipeline.model_artifacts import CalibratedStockModel
from pipeline.predict_stock_trend import PROJECT_ROOT, predict_stock_trend
from pipeline.stock_features import (
    BASE_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    REGIME_FEATURE_COLUMNS,
    STATIONARY_FEATURE_COLUMNS,
    add_stock_targets,
    build_stock_features,
)
from pipeline.train_stock_models import (
    available_model_names,
    build_purged_date_splits,
    prepare_horizon_frame,
    search_plan_for_profile,
    select_directional_threshold,
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


class FeatureEstimator:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probability = np.clip(0.5 + pd.to_numeric(X["Return_1D"]).to_numpy() * 2, 0.01, 0.99)
        return np.column_stack([1 - probability, probability])


class IdentityLogitCalibrator:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probability = 1 / (1 + np.exp(-np.asarray(X, dtype=float).reshape(-1)))
        return np.column_stack([1 - probability, probability])


class StockFeatureTests(unittest.TestCase):
    def test_market_alignment_removes_provider_only_sessions(self) -> None:
        stock = make_stock_data(periods=180)
        market = make_market_data(periods=180)
        missing_dates = market.loc[[70, 125], "Date"].tolist()
        market = market.drop(index=[70, 125]).reset_index(drop=True)

        features = build_stock_features(stock, market)
        feature_dates = set(pd.to_datetime(features["Date"]))
        self.assertTrue(set(missing_dates).isdisjoint(feature_dates))
        latest = features.sort_values("Date").iloc[-1]
        self.assertTrue(pd.notna(latest["Rolling_Beta_60D"]))
        self.assertTrue(pd.notna(latest["Rolling_Correlation_60D"]))

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
        self.assertTrue(all(np.isfinite(float(latest[column])) for column in REGIME_FEATURE_COLUMNS))


class TemporalSplitTests(unittest.TestCase):
    def test_expanded_model_benchmark_has_diverse_tabular_candidates(self) -> None:
        candidates = set(available_model_names())
        self.assertTrue({"logistic-regression", "elastic-net-logistic", "random-forest", "extra-trees", "hist-gradient-boosting", "xgboost", "soft-voting-ensemble"}.issubset(candidates))
        focused = search_plan_for_profile("focused", tuple(candidates))
        self.assertEqual(set(focused), {1, 5, 10})
        self.assertEqual(set(focused[1]), {3, 4})
        self.assertNotIn("random-forest", focused[10][2])
        self.assertIn("soft-voting-ensemble", focused[5][4])

    def test_threshold_selection_rejects_degenerate_direction_coverage(self) -> None:
        labels = np.array([0, 1] * 50)
        useful_probabilities = np.linspace(0.1, 0.9, 100)
        _, metrics, passed = select_directional_threshold(labels, useful_probabilities)
        self.assertTrue(passed)
        self.assertGreaterEqual(metrics["recall"], 0.35)
        self.assertGreaterEqual(metrics["specificity"], 0.35)
        self.assertGreaterEqual(metrics["predicted_up_rate"], 0.20)
        self.assertLessEqual(metrics["predicted_up_rate"], 0.80)

        _, degenerate_metrics, degenerate_passed = select_directional_threshold(
            np.zeros(100, dtype=int), np.full(100, 0.5)
        )
        self.assertFalse(degenerate_passed)
        self.assertIn(degenerate_metrics["predicted_up_rate"], {0.0, 1.0})

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
    def test_newsapi_request_uses_30_day_retrieval_boundary(self) -> None:
        url = build_request_url('"Maybank"', as_of_date=date(2026, 7, 10))
        self.assertIn("from=2026-06-10", url)
        self.assertIn("to=2026-07-10", url)

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
        self.assertEqual(contexts["10"]["window_days"], 30)
        self.assertEqual(contexts["10"]["half_life_days"], 15.0)

    def test_context_contract_and_boundary_decay(self) -> None:
        self.assertEqual(CONTEXT_WINDOWS, {1: 7, 5: 14, 10: 30})
        self.assertEqual(CONTEXT_HALF_LIVES, {1: 3.5, 5: 7.0, 10: 15.0})
        self.assertAlmostEqual(calculate_recency_weight(30.0, 15.0), 0.25)
        self.assertAlmostEqual(calculate_recency_weight(0.0, 15.0), 1.0)

    def test_context_boundary_includes_30_days_but_excludes_older_articles(self) -> None:
        rows = [
            {
                "published_at": "2026-06-11T00:00:00Z",
                "source_name": "Boundary Source",
                "sentiment_label": "positive",
                "sentiment_score": 1,
                "sentiment_probability": 0.9,
                "sentiment_uncertain": False,
            },
            {
                "published_at": "2026-06-10T23:59:59Z",
                "source_name": "Older Source",
                "sentiment_label": "negative",
                "sentiment_score": -1,
                "sentiment_probability": 0.9,
                "sentiment_uncertain": False,
            },
        ]
        contexts = build_news_contexts(rows, date(2026, 7, 10))
        self.assertEqual(contexts["10"]["candidate_article_count"], 1)


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
    def test_local_feature_contributions_explain_model_sensitivity(self) -> None:
        artifact = CalibratedStockModel(
            estimator=FeatureEstimator(),
            calibrator=IdentityLogitCalibrator(),
            threshold=0.5,
            input_columns=["Return_1D"],
            numeric_feature_columns=["Return_1D"],
            refinement_round=1,
            model_name="feature-test",
            horizon_days=1,
            feature_reference_values={"Return_1D": 0.0},
        )
        factors = artifact.local_feature_contributions(pd.DataFrame({"Return_1D": [0.1]}))
        self.assertEqual(factors[0]["feature"], "Return_1D")
        self.assertGreater(float(factors[0]["contribution"]), 0)

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
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
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
