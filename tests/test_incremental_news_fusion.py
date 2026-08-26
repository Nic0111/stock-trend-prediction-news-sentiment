from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from pipeline.fusion_evaluation_v3 import _metrics
from pipeline.incremental_news_fusion import IncrementalNewsFusion


def _training_frame(seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    generator = np.random.default_rng(seed)
    rows = 180
    stock = generator.uniform(0.25, 0.75, rows)
    available = np.zeros(rows)
    available[:120] = 1.0
    tone = generator.normal(size=rows) * available
    true_probability = np.clip(stock + 0.10 * tone, 0.02, 0.98)
    target = (generator.random(rows) < true_probability).astype(int)
    return (
        pd.DataFrame(
            {
                "stock_probability_up": stock,
                "news_available": available,
                "news_positive_impact": np.maximum(tone, 0),
                "news_negative_impact": np.maximum(-tone, 0),
            }
        ),
        target,
    )


def test_no_news_is_exact_stock_probability_and_direction() -> None:
    frame, target = _training_frame()
    model = IncrementalNewsFusion(
        ["news_positive_impact", "news_negative_impact"], adjustment_cap=0.25
    ).fit(frame, target)
    result = model.predict(frame)
    unavailable = frame["news_available"].eq(0).to_numpy()
    stock = frame["stock_probability_up"].to_numpy()
    assert np.array_equal(result.probability_up[unavailable], stock[unavailable])
    assert np.array_equal(
        result.probability_up[unavailable] >= 0.53,
        stock[unavailable] >= 0.53,
    )
    assert np.all(result.news_adjustment_log_odds[unavailable] == 0.0)


def test_adjustment_is_capped() -> None:
    frame, target = _training_frame()
    model = IncrementalNewsFusion(
        ["news_positive_impact", "news_negative_impact"], adjustment_cap=0.10
    ).fit(frame, target)
    result = model.predict(frame)
    assert np.abs(result.news_adjustment_log_odds).max() <= 0.10 + 1e-12


def test_news_present_intercept_is_recorded_without_changing_no_news_rows() -> None:
    frame, target = _training_frame()
    model = IncrementalNewsFusion(
        ["news_positive_impact", "news_negative_impact"], adjustment_cap=0.25
    ).fit(frame, target)
    provenance = model.provenance()
    assert isinstance(provenance["news_present_intercept"], float)
    prediction = model.predict(frame)
    unavailable = frame["news_available"].eq(0).to_numpy()
    assert np.array_equal(
        prediction.probability_up[unavailable],
        frame.loc[unavailable, "stock_probability_up"].to_numpy(),
    )


def test_fitted_state_survives_joblib_round_trip(tmp_path) -> None:
    frame, target = _training_frame()
    model = IncrementalNewsFusion(
        ["news_positive_impact", "news_negative_impact"],
        adjustment_cap=0.25,
        fit_intercept=False,
    ).fit(frame, target)
    expected = model.predict(frame)
    path = tmp_path / "fusion.joblib"
    joblib.dump(model, path)

    loaded = joblib.load(path)
    actual = loaded.predict(frame)
    np.testing.assert_allclose(actual.probability_up, expected.probability_up)
    np.testing.assert_allclose(
        actual.news_adjustment_log_odds, expected.news_adjustment_log_odds
    )
    assert loaded.provenance() == model.provenance()


def test_metrics_share_one_prediction_and_confusion_matrix() -> None:
    target = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.20, 0.60, 0.40, 0.80])
    metrics = _metrics(target, probability, 0.50)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["down_recall"] == 0.5
    assert metrics["up_recall"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
