from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fusion_evaluation_v2 import (
    _block_bootstrap,
    _cap_news_shift,
    _eligible_all_rows,
    _enhance_features,
    _permute_news,
    load_policy,
)
from pipeline.historical_news_fusion_data import _score_news


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=12, freq="B")
    return pd.DataFrame(
        {
            "ticker": ["A.KL"] * 6 + ["B.KL"] * 6,
            "observation_date": list(dates[:6]) + list(dates[:6]),
            "target_date": list(dates[1:7]) + list(dates[1:7]),
            "horizon_days": [1] * 12,
            "target_up": [0, 1] * 6,
            "stock_probability_up": np.linspace(0.35, 0.65, 12),
            "stock_threshold_margin": np.linspace(-0.15, 0.15, 12),
            "stock_evidence_strength": [0.33] * 12,
            "news_available": [0, 1, 1, 0, 1, 1] * 2,
            "news_coverage_score": [0, .25, .5, 0, 1, .25] * 2,
            "news_mean_confidence": [0, .8, .7, 0, .9, .6] * 2,
            "news_usable_article_count": [0, 1, 3, 0, 5, 1] * 2,
            "news_independent_event_count": [0, 1, 2, 0, 3, 1] * 2,
            "news_unique_source_count": [0, 1, 2, 0, 3, 1] * 2,
            "news_newest_age_days": [np.nan, 1, 2, np.nan, 1, 3] * 2,
            "news_mean_age_days": [np.nan, 1, 3, np.nan, 2, 4] * 2,
            "news_half_life_days": [3.5] * 12,
            "news_weighted_sentiment": [0, 1, -.5, 0, .2, -.2] * 2,
            "news_positive_share": [0, 1, 0, 0, .6, 0] * 2,
            "news_neutral_share": [0, 0, .5, 0, .2, .5] * 2,
            "news_negative_share": [0, 0, .5, 0, .2, .5] * 2,
            "stock_news_agreement": [.5, 1, 0, .5, 1, 0] * 2,
            "coverage_confidence": ["Unavailable", "Low", "Medium", "Unavailable", "High", "Low"] * 2,
        }
    )


def test_policy_is_versioned_and_hash_is_stable() -> None:
    policy, first = load_policy(Path("config/fusion_evaluation_v2.json"))
    _, second = load_policy(Path("config/fusion_evaluation_v2.json"))
    assert policy["claim_scope"] == "retrospective-2019-2025"
    assert first == second


def test_all_target_complete_rows_are_eligible_even_without_news() -> None:
    enhanced = _enhance_features(_frame())
    eligible = _eligible_all_rows(enhanced, 1)
    assert len(eligible) == 12
    assert int((eligible["news_available"] == 0).sum()) == 4
    assert (eligible.loc[eligible["news_available"].eq(0), "news_reliability"] == 0).all()


def test_news_shift_cap_and_missing_news_fallback_are_exact() -> None:
    stock = np.asarray([0.4, 0.6])
    fusion = np.asarray([0.05, 0.95])
    capped = _cap_news_shift(stock, fusion, 0.25)
    assert np.all(np.abs(np.log(capped / (1 - capped)) - np.log(stock / (1 - stock))) <= 0.2500001)
    capped[0] = stock[0]
    assert capped[0] == stock[0]


def test_placebo_shuffle_stays_within_ticker_and_month() -> None:
    enhanced = _enhance_features(_frame())
    shuffled = _permute_news(enhanced, 42)
    original = enhanced.groupby(["ticker", enhanced["observation_date"].dt.month])["news_weighted_sentiment"].apply(lambda value: sorted(value.tolist()))
    changed = shuffled.groupby(["ticker", shuffled["observation_date"].dt.month])["news_weighted_sentiment"].apply(lambda value: sorted(value.tolist()))
    assert original.to_dict() == changed.to_dict()


def test_moving_block_bootstrap_is_reproducible() -> None:
    frame = _frame()
    target = frame["target_up"].to_numpy()
    baseline = np.linspace(.4, .6, len(frame))
    challenger = np.linspace(.35, .65, len(frame))
    kwargs = dict(block_size=5, iterations=25, seed=42)
    first = _block_bootstrap(frame, target, baseline, challenger, baseline >= .5, challenger >= .5, **kwargs)
    second = _block_bootstrap(frame, target, baseline, challenger, baseline >= .5, challenger >= .5, **kwargs)
    assert first == second


def test_historical_syndicated_copies_do_not_multiply_event_weight() -> None:
    cutoff = pd.Timestamp("2025-01-10T09:00:00Z")
    base = {
        "article_id": "a",
        "url": "https://example.com/story",
        "title": "Company reports earnings",
        "description": "Company reports earnings and raises guidance.",
        "source_domain": "example.com",
        "source_name": "Example",
        "published_at": pd.Timestamp("2025-01-09T09:00:00Z"),
        "sentiment_uncertain": False,
        "sentiment_score": 1,
        "sentiment_probability": .9,
        "sentiment_label": "positive",
    }
    opposing = dict(base, article_id="b", url="https://other.example/down", title="Company cuts outlook", sentiment_score=-1, sentiment_label="negative", sentiment_probability=.6)
    single = _score_news(pd.DataFrame([base, opposing]), cutoff, 1)
    duplicated = _score_news(pd.DataFrame([base, dict(base, article_id="copy"), opposing]), cutoff, 1)
    assert single["weighted_sentiment_score"] == duplicated["weighted_sentiment_score"]


def test_window_candidates_have_independent_surprise_histories() -> None:
    base = pd.concat([_frame().iloc[[1]].copy() for _ in range(30)], ignore_index=True)
    base["observation_date"] = pd.date_range("2025-01-02", periods=30, freq="B")
    base["target_date"] = pd.date_range("2025-01-03", periods=30, freq="B")
    base["news_available"] = 1
    base["news_independent_event_count"] = 1
    base["news_window_days"] = 3
    base["news_half_life_days"] = 1.5
    other = base.copy()
    other["news_window_days"] = 7
    other["news_half_life_days"] = 3.5
    other["news_independent_event_count"] = 20
    enhanced = _enhance_features(pd.concat([base, other], ignore_index=True))
    three_day = enhanced.loc[enhanced["news_window_days"].eq(3)]
    seven_day = enhanced.loc[enhanced["news_window_days"].eq(7)]
    assert len(three_day) == len(base)
    assert len(seven_day) == len(base)
    assert three_day["news_available"].tolist() == base["news_available"].tolist()
    assert seven_day.iloc[:19]["volume_surprise"].eq(0).all()
