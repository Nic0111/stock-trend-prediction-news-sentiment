from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.sentiment_ensemble_promotion import (
    SentimentEnsemblePromotionError,
    promote_sentiment_ensemble,
    rollback_sentiment_ensemble,
)


def _candidate() -> dict[str, object]:
    return {
        "backend": "weighted_finbert_ensemble",
        "weights": {"bursa": 0.60, "phrasebank": 0.40},
        "components": {
            "bursa": {
                "model_directory": "models/experimental/bursa",
                "temperature": 1.1,
                "artifact_hashes": {"model.safetensors": "bursa-hash"},
            },
            "phrasebank": {
                "model_directory": "models/experimental/phrasebank",
                "temperature": 1.2,
                "artifact_hashes": {"model.safetensors": "phrasebank-hash"},
            },
        },
    }


class SentimentEnsemblePromotionTests(unittest.TestCase):
    def test_passed_report_creates_and_promotes_its_immutable_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "runtime.json"
            report_path = root / "benchmark.json"
            policy_path = root / "legacy-policy.json"
            policy_path.write_text(
                json.dumps({"sentiment_ensemble": {"promotion_gates": {"require_prospective_holdout": False}}}),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps({"active_variant": "finbert_phrasebank_v1", "challengers": {}}),
                encoding="utf-8",
            )
            candidate = _candidate()
            report_path.write_text(
                json.dumps(
                    {
                        "variant_id": "bursa_phrasebank_soft_vote_v2",
                        "gates": {"eligible_for_promotion": True},
                        "artifact_hashes": {
                            "bursa": candidate["components"]["bursa"]["artifact_hashes"],
                            "phrasebank": candidate["components"]["phrasebank"]["artifact_hashes"],
                        },
                        "runtime_candidate": candidate,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SentimentEnsemblePromotionError):
                promote_sentiment_ensemble(
                    variant="bursa_phrasebank_soft_vote_v2",
                    benchmark_report=report_path,
                    runtime_config=config_path,
                    policy_path=policy_path,
                    approve=False,
                )
            promoted = promote_sentiment_ensemble(
                variant="bursa_phrasebank_soft_vote_v2",
                benchmark_report=report_path,
                runtime_config=config_path,
                policy_path=policy_path,
                approve=True,
            )
            self.assertEqual(promoted["active_variant"], "bursa_phrasebank_soft_vote_v2")
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["challengers"]["bursa_phrasebank_soft_vote_v2"]["weights"]["bursa"], 0.60)
            rolled_back = rollback_sentiment_ensemble(
                runtime_config=config_path, approve=True
            )
            self.assertEqual(rolled_back["active_variant"], "finbert_phrasebank_v1")

    def test_failed_gate_cannot_change_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "runtime.json"
            report_path = root / "benchmark.json"
            original = {"active_variant": "finbert_phrasebank_v1", "challengers": {}}
            config_path.write_text(json.dumps(original), encoding="utf-8")
            report_path.write_text(
                json.dumps(
                    {
                        "variant_id": "bursa_phrasebank_soft_vote_v2",
                        "gates": {"eligible_for_promotion": False},
                        "runtime_candidate": _candidate(),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SentimentEnsemblePromotionError):
                promote_sentiment_ensemble(
                    variant="bursa_phrasebank_soft_vote_v2",
                    benchmark_report=report_path,
                    runtime_config=config_path,
                    approve=True,
                )
            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), original)

    def test_retrospective_report_requires_explicit_policy_permission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "runtime.json"
            report_path = root / "benchmark.json"
            policy_path = root / "policy.json"
            config_path.write_text(
                json.dumps({"active_variant": "finbert_phrasebank_v1", "challengers": {}}),
                encoding="utf-8",
            )
            candidate = _candidate()
            report_path.write_text(
                json.dumps(
                    {
                        "variant_id": "bursa_phrasebank_soft_vote_v2",
                        "gates": {"eligible_for_promotion": True},
                        "artifact_hashes": {
                            "bursa": candidate["components"]["bursa"]["artifact_hashes"],
                            "phrasebank": candidate["components"]["phrasebank"]["artifact_hashes"],
                        },
                        "runtime_candidate": candidate,
                        "promotion_protocol": {
                            "kind": "retrospective_locked_2025",
                            "manual_promotion_only": True,
                            "retrospective_test_disclosure": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            policy_path.write_text(
                json.dumps(
                    {"sentiment_ensemble": {"promotion_gates": {"allow_retrospective_manual_promotion": False}}}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SentimentEnsemblePromotionError):
                promote_sentiment_ensemble(
                    variant="bursa_phrasebank_soft_vote_v2",
                    benchmark_report=report_path,
                    runtime_config=config_path,
                    policy_path=policy_path,
                    approve=True,
                )
            policy_path.write_text(
                json.dumps(
                    {"sentiment_ensemble": {"promotion_gates": {"allow_retrospective_manual_promotion": True}}}
                ),
                encoding="utf-8",
            )
            promoted = promote_sentiment_ensemble(
                variant="bursa_phrasebank_soft_vote_v2",
                benchmark_report=report_path,
                runtime_config=config_path,
                policy_path=policy_path,
                approve=True,
            )
            self.assertTrue(promoted["promoted"])

    def test_current_policy_rejects_a_report_without_a_prospective_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "runtime.json"
            report_path = root / "benchmark.json"
            policy_path = root / "policy.json"
            config_path.write_text(
                json.dumps({"active_variant": "finbert_phrasebank_v1", "challengers": {}}),
                encoding="utf-8",
            )
            candidate = _candidate()
            report_path.write_text(
                json.dumps(
                    {
                        "variant_id": "bursa_phrasebank_soft_vote_v2",
                        "gates": {"eligible_for_promotion": True},
                        "artifact_hashes": {
                            "bursa": candidate["components"]["bursa"]["artifact_hashes"],
                            "phrasebank": candidate["components"]["phrasebank"]["artifact_hashes"],
                        },
                        "runtime_candidate": candidate,
                    }
                ),
                encoding="utf-8",
            )
            policy_path.write_text(
                json.dumps({"sentiment_ensemble": {"promotion_gates": {"require_prospective_holdout": True}}}),
                encoding="utf-8",
            )
            with self.assertRaises(SentimentEnsemblePromotionError):
                promote_sentiment_ensemble(
                    variant="bursa_phrasebank_soft_vote_v2",
                    benchmark_report=report_path,
                    runtime_config=config_path,
                    policy_path=policy_path,
                    approve=True,
                )


if __name__ == "__main__":
    unittest.main()
