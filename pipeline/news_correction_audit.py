"""Append-only audit trail for user changes to news evidence.

The analysis cache is intentionally short lived, but a correction changes the
sentiment context shown to the user.  This ledger preserves who/what/when and
the before/after evidence without storing article text or API credentials.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_DB = PROJECT_ROOT / "data" / "audit" / "news-corrections.sqlite3"


def _context_snapshot(analysis: Any) -> dict[str, Any]:
    news = getattr(analysis, "news", None)
    horizon = int(getattr(analysis, "horizon_days", 0) or 0)
    context = (getattr(news, "contexts", {}) or {}).get(str(horizon), {}) if news else {}
    outlook = getattr(analysis, "combined_outlook", None)
    return {
        "ticker": str(getattr(analysis, "ticker", "")),
        "horizon_days": horizon,
        "weighted_sentiment_score": context.get("weighted_sentiment_score"),
        "coverage_confidence": context.get("coverage_confidence"),
        "usable_article_count": int(context.get("usable_article_count", 0) or 0),
        "unique_source_count": int(context.get("unique_source_count", 0) or 0),
        "independent_event_count": int(context.get("independent_event_count", 0) or 0),
        "aggregate_mode": getattr(outlook, "mode", None),
        "aggregate_direction": getattr(outlook, "direction", None),
        "aggregate_probability_up": getattr(outlook, "probability_up", None),
    }


def record_news_correction(
    *,
    analysis_id: str,
    article_id: str,
    action: str,
    before: Any,
    after: Any,
    registry_signature: str,
    actor: str = "local-user",
    db_path: Path = DEFAULT_AUDIT_DB,
) -> None:
    """Persist one immutable correction event.

    The caller should only publish the corrected analysis after this function
    succeeds, preventing an unlogged evidence change.
    """

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    before_snapshot = _context_snapshot(before)
    after_snapshot = _context_snapshot(after)
    with closing(sqlite3.connect(path, timeout=5.0)) as connection:
        connection.execute("PRAGMA busy_timeout = 5000")
        with connection:
            connection.execute(
            """
            CREATE TABLE IF NOT EXISTS news_correction_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                analysis_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                ticker TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                article_id TEXT NOT NULL,
                action TEXT NOT NULL,
                registry_signature TEXT NOT NULL,
                before_json TEXT NOT NULL,
                after_json TEXT NOT NULL
            )
            """
            )
            connection.execute(
            """
            INSERT INTO news_correction_events (
                occurred_at, analysis_id, actor, ticker, horizon_days,
                article_id, action, registry_signature, before_json, after_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                str(analysis_id),
                str(actor),
                str(before_snapshot["ticker"]),
                int(before_snapshot["horizon_days"]),
                str(article_id),
                str(action),
                str(registry_signature),
                json.dumps(before_snapshot, sort_keys=True, separators=(",", ":")),
                json.dumps(after_snapshot, sort_keys=True, separators=(",", ":")),
            ),
            )


def correction_audit_summary(
    db_path: Path = DEFAULT_AUDIT_DB,
) -> Mapping[str, Any]:
    """Return non-sensitive ledger counts for system status and audits."""

    path = Path(db_path)
    if not path.exists():
        return {"available": True, "event_count": 0, "last_event_at": None}
    with closing(sqlite3.connect(path, timeout=5.0)) as connection:
        connection.execute("PRAGMA busy_timeout = 5000")
        row = connection.execute(
            "SELECT COUNT(*), MAX(occurred_at) FROM news_correction_events"
        ).fetchone()
    return {
        "available": True,
        "event_count": int(row[0] or 0),
        "last_event_at": row[1],
    }
