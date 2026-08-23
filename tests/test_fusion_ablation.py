from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fusion_ablation import _paired_diagnostics, _shuffled_news
from pipeline.fusion_model import NEWS_FUSION_FEATURES


def test_shuffled_news_preserves_stock_and_targets() -> None:
    frame = pd.DataFrame(
        {
            "stock_probability_up": [0.2, 0.4, 0.6, 0.8],
            "target_up": [0, 0, 1, 1],
            **{name: np.arange(4, dtype=float) for name in NEWS_FUSION_FEATURES},
        }
    )
    shuffled = _shuffled_news(frame, 42)
    assert shuffled["stock_probability_up"].tolist() == frame["stock_probability_up"].tolist()
    assert shuffled["target_up"].tolist() == frame["target_up"].tolist()
    assert any(not shuffled[name].equals(frame[name]) for name in NEWS_FUSION_FEATURES)


def test_paired_diagnostics_counts_error_transitions() -> None:
    frame = pd.DataFrame(
        {"ticker": ["A", "A", "A", "A"], "observation_date": pd.date_range("2025-01-01", periods=4)}
    )
    target = np.array([0, 0, 1, 1])
    stock = np.array([0, 1, 0, 1])
    fusion = np.array([0, 0, 1, 0])
    result = _paired_diagnostics(frame, target, stock, fusion)
    assert result["error_transitions"]["stock_wrong_fusion_right"] == 2
    assert result["error_transitions"]["stock_right_fusion_wrong"] == 1
