from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.download_financial_news_data import NewsContextResult
from pipeline.fusion_model import (
    FUSION_FEATURE_NAMES,
    FusionModelArtifact,
    FusionRegistry,
    TransparentLogOddsEstimator,
)
from pipeline.fusion_runtime import (
    _v5_registry_guard,
    build_aggregate_outlook,
    build_runtime_fusion_frame,
    load_accepted_fusion_artifact,
)
from pipeline.fusion_evaluation_v3 import FULL_FEATURES
from pipeline.predict_stock_trend import PredictionResult


def _stock(status: str = "accepted") -> PredictionResult:
    return PredictionResult(
        ticker="1155.KL",
        company_name="Malayan Banking Berhad",
        horizon_days=1,
        as_of_date="2026-07-21",
        prediction_label="Up",
        probability_up=0.55,
        decision_threshold=0.5,
        model_name="test",
        model_version="test",
        refinement_round=1,
        benchmark_passed=status == "accepted",
        status=status,
        stale_data_warning=False,
        latest_data_age_days=0,
        threshold_constraints_passed=True,
        evidence_strength="Moderate",
    )


def _news(coverage: str = "Medium", score: float = 0.8) -> NewsContextResult:
    count = 3 if coverage == "Medium" else 1
    return NewsContextResult(
        ticker="1155.KL",
        company_name="Malayan Banking Berhad",
        query_text="Maybank",
        as_of_date="2026-07-21",
        retrieved_at="2026-07-21T10:00:00Z",
        raw_article_count=count,
        relevant_article_count=count,
        results_truncated=False,
        contexts={
            "1": {
                "weighted_sentiment_score": score,
                "coverage_confidence": coverage,
                "usable_article_count": count,
                "unique_source_count": 2 if count > 1 else 1,
                "positive_count": count if score >= 0 else 0,
                "neutral_count": 0,
                "negative_count": count if score < 0 else 0,
                "mean_sentiment_confidence": 0.9,
                "max_sentiment_confidence": 0.95,
                "newest_article_age_days": 0.5,
                "mean_article_age_days": 1.0,
            }
        },
        articles=[],
        output_paths={},
        provider_metadata={"prediction_cutoff_at": "2026-07-21T17:00:00+08:00"},
    )


class FusionRuntimeTests(unittest.TestCase):
    def _registry(self, root: Path) -> Path:
        artifact = FusionModelArtifact(
            horizon_days=1,
            model_version="accepted-transparent",
            feature_names=tuple(FUSION_FEATURE_NAMES),
            estimator=TransparentLogOddsEstimator(1.0),
            validation_status="accepted",
            fusion_method="transparent_log_odds",
            transparent_alpha=1.0,
        )
        path = root / "fusion.joblib"
        artifact.save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        registry_path = root / "fusion-registry.json"
        FusionRegistry(
            "2026-07-24T00:00:00+00:00",
            "dataset",
            {
                "1": {
                    "artifact_path": str(path),
                    "artifact_sha256": digest,
                    "validation_status": "accepted",
                    "model_version": artifact.model_version,
                }
            },
        ).write(registry_path)
        return registry_path

    def test_accepted_medium_coverage_uses_combined_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = build_aggregate_outlook(
                _stock(),
                _news(),
                ticker="1155.KL",
                company_name="Malayan Banking Berhad",
                horizon_days=1,
                registry_path=self._registry(Path(directory)),
            )
        self.assertEqual(result.mode, "validated_fusion")
        self.assertEqual(result.probability_kind, "combined")
        self.assertGreater(result.probability_up, result.stock_probability_up)
        self.assertIn("news_adjustment_probability_points", result.component_contributions)

    def test_low_coverage_preserves_stock_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = build_aggregate_outlook(
                _stock(),
                _news("Low"),
                ticker="1155.KL",
                company_name="Malayan Banking Berhad",
                horizon_days=1,
                registry_path=self._registry(Path(directory)),
            )
        self.assertEqual(result.mode, "stock_led_synthesis")
        self.assertEqual(result.probability_up, 0.55)
        self.assertIn("requires Medium or High", result.fallback_reason or "")

    def test_experimental_stock_marks_fusion_research_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = build_aggregate_outlook(
                _stock("experimental"),
                _news(),
                ticker="1155.KL",
                company_name="Malayan Banking Berhad",
                horizon_days=1,
                registry_path=self._registry(Path(directory)),
            )
        self.assertEqual(result.mode, "stock_led_synthesis")
        self.assertEqual(result.validation_status, "experimental")
        self.assertIn("numerical fusion is disabled", result.fallback_reason or "")

    def test_accepted_fusion_may_reverse_stock_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = build_aggregate_outlook(
                _stock(),
                _news(score=-1.0),
                ticker="1155.KL",
                company_name="Malayan Banking Berhad",
                horizon_days=1,
                registry_path=self._registry(Path(directory)),
            )
        self.assertEqual(result.stock_direction, "Up")
        self.assertEqual(result.direction, "Down")
        self.assertIn("changed the stock-only direction", result.explanation)

    def test_v2_registry_fails_closed_when_all_horizon_gate_did_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "fusion-v2.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fusion-registry-v4",
                        "status": "experimental",
                        "overall_all_horizon_accepted": False,
                        "horizons": {"1": {"accepted": False}},
                    }
                ),
                encoding="utf-8",
            )
            artifact, reason = load_accepted_fusion_artifact(1, registry_path)
        self.assertIsNone(artifact)
        self.assertIn("strict all-horizon acceptance", reason or "")

    def test_v3_registry_fails_closed_until_runtime_feature_state_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_path = root / "stock-registry.json"
            sentiment_path = root / "sentiment-runtime.json"
            stock_path.write_text(
                json.dumps({"run_id": "stock-v3", "dataset_sha256": "dataset-v3"}),
                encoding="utf-8",
            )
            sentiment_path.write_text('{"variant":"frozen"}', encoding="utf-8")
            payload = {
                "schema_version": "fusion-registry-v5",
                "runtime_feature_contract_ready": False,
                "horizons": {"1": {"accepted": True}},
                "provenance": {
                    "strict_identity_passed": True,
                    "active_stock_run_id": "stock-v3",
                    "stock_dataset_sha256": "dataset-v3",
                    "sentiment_config_sha256": hashlib.sha256(
                        sentiment_path.read_bytes()
                    ).hexdigest(),
                },
            }
            with patch("pipeline.fusion_runtime.ACTIVE_STOCK_REGISTRY", stock_path), patch(
                "pipeline.fusion_runtime.SENTIMENT_RUNTIME_CONFIG", sentiment_path
            ):
                reason = _v5_registry_guard(payload, 1)
        self.assertIn("baseline state has not been materialised", reason or "")

    def test_v3_runtime_frame_has_full_contract_and_exact_no_news_semantics(self) -> None:
        news = _news("High", score=0.4)
        context = news.contexts["1"]
        context.update(
            {
                "independent_event_count": 2,
                "candidate_article_count": 4,
                "news_event_reliability_mean": 0.8,
                "news_positive_event_impact": 0.24,
                "news_negative_event_impact": 0.04,
                "news_official_disclosure_share": 0.5,
            }
        )
        for name in FULL_FEATURES:
            if name.startswith("news_event_category_"):
                context[name] = 0.125
        frame = build_runtime_fusion_frame(
            _stock(),
            news,
            1,
            runtime_state={"volume_baseline_mean": 0.5, "tone_baseline_mean": 0.1},
        )
        self.assertTrue(set(FULL_FEATURES).issubset(frame.columns))
        self.assertEqual(float(frame.iloc[0]["news_available"]), 1.0)

        no_news = _news("Unavailable", score=0.0)
        no_news.contexts["1"].update(
            {
                "weighted_sentiment_score": None,
                "usable_article_count": 0,
                "candidate_article_count": 0,
                "unique_source_count": 0,
                "independent_event_count": 0,
                "positive_count": 0,
                "neutral_count": 0,
                "negative_count": 0,
            }
        )
        no_news_frame = build_runtime_fusion_frame(_stock(), no_news, 1)
        self.assertEqual(float(no_news_frame.iloc[0]["stock_probability_up"]), 0.55)
        self.assertEqual(float(no_news_frame.iloc[0]["news_available"]), 0.0)
        for name in FULL_FEATURES:
            self.assertEqual(float(no_news_frame.iloc[0][name]), 0.0, name)


if __name__ == "__main__":
    unittest.main()
