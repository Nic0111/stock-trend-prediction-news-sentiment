"""Bursa Malaysia session helpers shared by snapshot and prediction runtime."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd


KUALA_LUMPUR = ZoneInfo("Asia/Kuala_Lumpur")
MARKET_CLOSE = time(17, 0)


@lru_cache(maxsize=1)
def _exchange_calendar():
    try:
        import exchange_calendars as xcals

        return xcals.get_calendar("XKLS")
    except (ImportError, KeyError):
        return None


def _weekday_on_or_before(value: date) -> date:
    current = value
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def latest_completed_session(
    requested: date | None = None,
    *,
    now: datetime | None = None,
) -> date:
    """Return the latest completed Bursa session at or before ``requested``.

    ``exchange_calendars`` supplies Bursa holidays when installed. A weekday
    fallback keeps the local application usable, but callers should disclose
    degraded calendar precision through readiness.
    """

    local_now = (now or datetime.now(KUALA_LUMPUR)).astimezone(KUALA_LUMPUR)
    today = local_now.date()
    requested_date = min(requested or today, today)
    if requested_date == today and local_now.time() < MARKET_CLOSE:
        requested_date -= timedelta(days=1)

    calendar = _exchange_calendar()
    if calendar is None:
        return _weekday_on_or_before(requested_date)
    session = calendar.date_to_session(pd.Timestamp(requested_date), direction="previous")
    return pd.Timestamp(session).date()


def completed_sessions_between(start_exclusive: date, end_inclusive: date) -> int:
    if end_inclusive <= start_exclusive:
        return 0
    calendar = _exchange_calendar()
    if calendar is None:
        dates = pd.bdate_range(start_exclusive + timedelta(days=1), end_inclusive)
        return int(len(dates))
    sessions = calendar.sessions_in_range(
        pd.Timestamp(start_exclusive + timedelta(days=1)),
        pd.Timestamp(end_inclusive),
    )
    return int(len(sessions))


def previous_market_session(session_date: date) -> date:
    """Return the Bursa session immediately preceding ``session_date``.

    ``session_date`` is normally an observed market session.  The defensive
    normalization also gives callers a stable result when they pass a weekend
    or holiday.  Unlike :func:`latest_completed_session`, this helper is purely
    historical and therefore does not cap its input at today's date.
    """

    calendar = _exchange_calendar()
    if calendar is None:
        candidate = session_date - timedelta(days=1)
        return _weekday_on_or_before(candidate)
    current = calendar.date_to_session(pd.Timestamp(session_date), direction="previous")
    previous = calendar.previous_session(current)
    return pd.Timestamp(previous).date()


def calendar_precision() -> str:
    return "bursa_exchange_calendar" if _exchange_calendar() is not None else "weekday_fallback"
