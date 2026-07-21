from __future__ import annotations

import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

from pipeline import market_snapshot


class MarketSnapshotTests(unittest.TestCase):
    def test_daily_snapshot_calculates_change_and_history(self) -> None:
        dates = pd.to_datetime([date.today() - timedelta(days=2), date.today() - timedelta(days=1)])
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Close": [10.0, 10.5],
                "High": [10.2, 10.7],
                "Low": [9.8, 10.1],
                "Volume": [1000, 1500],
            }
        ).set_index("Date")
        with patch.object(market_snapshot.yf, "download", return_value=frame):
            result = market_snapshot.fetch_market_snapshot("1155.KL")

        self.assertEqual(result.ticker, "1155.KL")
        self.assertEqual(result.price, 10.5)
        self.assertAlmostEqual(result.change_pct or 0.0, 0.05)
        self.assertEqual(result.day_high, 10.7)
        self.assertEqual(result.volume, 1500)
        self.assertEqual(len(result.history_prices), 2)
        self.assertFalse(result.stale_data_warning)

    def test_unsupported_ticker_is_rejected_before_download(self) -> None:
        with patch.object(market_snapshot.yf, "download") as download:
            with self.assertRaises(ValueError):
                market_snapshot.fetch_market_snapshot("UNKNOWN.KL")
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
