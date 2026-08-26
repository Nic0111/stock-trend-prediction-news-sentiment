from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fusion_evaluation_v4 import (
    EVENT_IMPACT_FEATURE,
    V4_FEATURES,
    TargetSpecificEventImpact,
    _compact_base,
    _count_blind_audit_rows,
    _blind_audit_sample,
    _v4_placebo_frame,
    load_policy,
    rebind_v4_sentiment_config,
)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=12, freq="B")
    return pd.DataFrame(
        {
            "ticker": ["1155.KL"] * len(dates),
            "observation_date": dates,
            "target_date": dates + pd.Timedelta(days=1),
            "target_up": [0, 1] * 6,
            "stock_probability_up": np.linspace(0.35, 0.65, len(dates)),
            "news_available": [0, 1] * 6,
            "news_independent_event_count": [0, 1] * 6,
            "news_usable_article_count": [0, 2] * 6,
            "news_positive_event_impact": [0, 0.7] * 6,
            "news_negative_event_impact": [0, 0.1] * 6,
            "news_weighted_sentiment": [0, 0.5] * 6,
            "news_mean_confidence": [0.0, 0.8] * 6,
            "news_event_reliability_mean": [0.0, 0.8] * 6,
            "news_unique_source_count": [0, 2] * 6,
        }
    )


def test_v4_policy_is_versioned_and_all_horizons_are_locked() -> None:
    policy, first = load_policy()
    _, second = load_policy()
    assert policy["schema_version"] == "fusion-evaluation-v4"
    assert policy["periods"] == {"development": [2019, 2023], "calibration": [2024, 2024], "benchmark": [2025, 2025]}
    assert policy["supported_horizons"] == [1, 5, 10]
    assert first == second


def test_compact_contract_separates_event_inputs_and_has_no_news_zeroes() -> None:
    result = _compact_base(_frame())
    assert set(V4_FEATURES).issubset(result.columns)
    assert result.loc[0, "v4_log_event_count"] == 0
    assert result.loc[1, "v4_positive_impact"] > 0
    assert result.loc[1, "v4_negative_impact"] > 0
    assert result.loc[0, EVENT_IMPACT_FEATURE] == 0


def test_blind_audit_is_not_inferred_when_provenance_column_is_missing() -> None:
    frame = _compact_base(_frame())
    assert _count_blind_audit_rows(frame) == (0, None)
    frame["news_blind_audit_passed"] = [True] * len(frame)
    assert _count_blind_audit_rows(frame) == (len(frame), "news_blind_audit_passed")


def test_blind_audit_sample_is_deterministic_and_company_balanced() -> None:
    rows = []
    for ticker in ("1023.KL", "1155.KL", "1295.KL"):
        for year in (2020, 2021, 2022, 2023, 2024, 2025):
            for index in range(4):
                rows.append(
                    {
                        "article_id": f"{ticker}-{year}-{index}",
                        "ticker": ticker,
                        "published_at": f"{year}-01-{index + 1:02d}",
                        "event_category": "contracts_order_wins" if index % 2 else "financial_results_guidance",
                        "event_id": f"event-{ticker}-{year}-{index}",
                    }
                )
    frame = pd.DataFrame(rows)
    first = _blind_audit_sample(frame, sample_size=12, seed=17)
    second = _blind_audit_sample(frame, sample_size=12, seed=17)
    assert first["article_id"].tolist() == second["article_id"].tolist()
    assert len(first) == 12
    assert first.groupby("ticker").size().to_dict() == {
        "1023.KL": 4,
        "1155.KL": 4,
        "1295.KL": 4,
    }


def test_event_impact_model_is_explicitly_unready_without_both_classes() -> None:
    frame = _compact_base(_frame())
    frame["target_up"] = 1
    model = TargetSpecificEventImpact().fit(frame)
    assert model.ready is False
    assert np.allclose(model.predict_proba_up(frame), 0.5)


def test_event_impact_model_predicts_only_for_available_news() -> None:
    frame = _compact_base(_frame())
    model = TargetSpecificEventImpact().fit(frame)
    probabilities = model.predict_proba_up(frame)
    assert np.allclose(probabilities[frame["news_available"].eq(0)], 0.5)
    assert model.provenance()["fit_scope"] == "earlier target-complete development rows only"


def test_v4_placebo_preserves_availability_and_ticker_month() -> None:
    source = _compact_base(_frame())
    shuffled = _v4_placebo_frame(source, mode="content_only", seed=7)
    assert shuffled["news_available"].tolist() == source["news_available"].tolist()
    assert shuffled["ticker"].tolist() == source["ticker"].tolist()


def test_rebind_changes_only_the_stale_config_hash(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "rebound.parquet"
    manifest = tmp_path / "rebound.manifest.json"
    frame = pd.DataFrame(
        {
            "sentiment_runtime_config_sha256": ["old-config", "old-config"],
            "sentiment_model_variant": ["active", "active"],
            "sentiment_model_variant_sha256": ["variant-hash", "variant-hash"],
            "sentiment_component_artifacts_sha256": ["component-hash", "component-hash"],
            "article_sentiment_identity_verified": [True, True],
            "stock_probability_up": [0.4, 0.6],
        }
    )
    frame.to_parquet(source, index=False)
    monkeypatch.setattr(
        "pipeline.fusion_evaluation_v4._sentiment_runtime_identity",
        lambda _path: {
            "config_sha256": "new-config",
            "variant": "active",
            "variant_sha256": "variant-hash",
            "component_artifacts_sha256": "component-hash",
        },
    )
    result = rebind_v4_sentiment_config(
        source,
        output,
        policy_path=Path("config/fusion_evaluation_v4.json"),
        manifest_path=manifest,
    )
    rebound = pd.read_parquet(output)
    assert rebound["sentiment_runtime_config_sha256"].tolist() == ["new-config"] * 2
    assert rebound["stock_probability_up"].tolist() == frame["stock_probability_up"].tolist()
    assert result["feature_value_parity_excluding_config_hash"] is True
    assert result["production_unchanged"] is True
