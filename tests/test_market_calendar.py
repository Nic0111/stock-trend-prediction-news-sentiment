from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from pipeline import market_calendar


KUALA_LUMPUR = ZoneInfo("Asia/Kuala_Lumpur")


class MarketCalendarTests(unittest.TestCase):
    def test_before_market_close_uses_the_previous_completed_session(self) -> None:
        with patch.object(market_calendar, "_exchange_calendar", return_value=None):
            result = market_calendar.latest_completed_session(
                now=datetime(2026, 7, 30, 10, 0, tzinfo=KUALA_LUMPUR)
            )
        self.assertEqual(result, date(2026, 7, 29))

    def test_after_market_close_can_use_today(self) -> None:
        with patch.object(market_calendar, "_exchange_calendar", return_value=None):
            result = market_calendar.latest_completed_session(
                now=datetime(2026, 7, 30, 18, 0, tzinfo=KUALA_LUMPUR)
            )
        self.assertEqual(result, date(2026, 7, 30))

    def test_weekend_falls_back_to_friday(self) -> None:
        with patch.object(market_calendar, "_exchange_calendar", return_value=None):
            result = market_calendar.latest_completed_session(
                now=datetime(2026, 8, 2, 18, 0, tzinfo=KUALA_LUMPUR)
            )
        self.assertEqual(result, date(2026, 7, 31))


if __name__ == "__main__":
    unittest.main()
