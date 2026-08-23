from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.chronological_finbert_ensemble import (
    ChronologicalFold,
    build_chronological_fold,
    select_stable_weight,
)


def _frame() -> pd.DataFrame:
    rows = []
    index = 0
    for year in range(2015, 2026):
        for label in ("negative", "neutral", "positive"):
            rows.append(
                {
                    "article_id": f"{year}-{label}",
                    "ticker": "1155.KL",
                    "published_at": pd.Timestamp(f"{year}-06-15", tz="UTC"),
                    "event_id": f"{year}-{label}",
                    "_event_group": f"1155.KL|event:{year}-{label}",
                    "content_sha256": f"content-{index}",
                    "sentiment_label": label,
                    "model_text": f"{label} article for {year}",
                }
            )
            index += 1
    # This duplicated event straddles a calibration/evaluation boundary.  The
    # latest copy must be retained while the calibration copy is purged.
    rows.extend(
        [
            {
                "article_id": "duplicate-calibration",
                "ticker": "1155.KL",
                "published_at": pd.Timestamp("2020-06-20", tz="UTC"),
                "event_id": "same-event",
                "_event_group": "1155.KL|event:same-event",
                "content_sha256": "duplicate-a",
                "sentiment_label": "positive",
                "model_text": "same event earlier copy",
            },
            {
                "article_id": "duplicate-evaluation",
                "ticker": "1155.KL",
                "published_at": pd.Timestamp("2021-06-20", tz="UTC"),
                "event_id": "same-event",
                "_event_group": "1155.KL|event:same-event",
                "content_sha256": "duplicate-b",
                "sentiment_label": "positive",
                "model_text": "same event latest copy",
            },
        ]
    )
    return pd.DataFrame(rows)


class ChronologicalFoldTests(unittest.TestCase):
    def test_fold_uses_forward_dates_and_purges_cross_partition_event(self) -> None:
        partitions, diagnostics = build_chronological_fold(
            _frame(), ChronologicalFold("fold", 2019, 2020, 2021), embargo_days=7
        )
        self.assertLess(partitions["training"]["published_at"].max(), pd.Timestamp("2019-12-25", tz="UTC"))
        self.assertTrue((partitions["calibration"]["published_at"].dt.year == 2020).all())
        self.assertTrue((partitions["evaluation"]["published_at"].dt.year == 2021).all())
        self.assertNotIn("same-event", set(partitions["calibration"]["event_id"]))
        self.assertIn("same-event", set(partitions["evaluation"]["event_id"]))
        self.assertGreater(diagnostics["event_rows_purged"], 0)


class StableWeightTests(unittest.TestCase):
    def test_probability_quality_selects_bursa_when_it_is_clearly_better(self) -> None:
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=int)
        bursa = np.asarray(
            [
                [0.98, 0.01, 0.01],
                [0.01, 0.98, 0.01],
                [0.01, 0.01, 0.98],
                [0.95, 0.03, 0.02],
                [0.02, 0.95, 0.03],
                [0.03, 0.02, 0.95],
            ]
        )
        phrasebank = np.asarray(
            [
                [0.01, 0.01, 0.98],
                [0.98, 0.01, 0.01],
                [0.01, 0.98, 0.01],
                [0.02, 0.03, 0.95],
                [0.94, 0.03, 0.03],
                [0.02, 0.94, 0.04],
            ]
        )
        result = select_stable_weight(
            bursa,
            phrasebank,
            labels,
            ["event-a", "event-b", "event-c", "event-d", "event-e", "event-f"],
            bootstrap_iterations=20,
        )
        self.assertEqual(result["selected_bursa_weight"], 1.0)
        self.assertEqual(result["selected_phrasebank_weight"], 0.0)
        self.assertEqual(result["selection_objective"], "multiclass_log_loss")


if __name__ == "__main__":
    unittest.main()
