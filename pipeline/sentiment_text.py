"""Stable text representations shared by Bursa FinBERT training and runtime.

The Bursa challenger must see the same representation during offline training
and live article scoring.  Keeping this small transformation in one module
avoids a common source of silent train/serve skew.
"""

from __future__ import annotations

import re
import math
from typing import Any, Mapping


TEXT_MODE_DEFAULT = "headline_lead"
TEXT_MODE_TARGET_AWARE = "target_aware"
SUPPORTED_TEXT_MODES = (TEXT_MODE_DEFAULT, TEXT_MODE_TARGET_AWARE)


def _clean(value: Any) -> str:
    """Return a stable text value without leaking pandas' ``nan`` token.

    The historical corpus is assembled from CSV/JSON exports, where an absent
    description is often represented as ``float('nan')``.  Treating that as
    literal language would create an avoidable train/serve mismatch.
    """

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    rendered = str(value).strip()
    if rendered.casefold() in {"nan", "none", "null", "<na>"}:
        return ""
    return re.sub(r"\s+", " ", rendered).strip()


def _sentences(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]


def _target_tokens(company_name: str, ticker: str) -> set[str]:
    values = re.findall(r"[A-Za-z0-9]{3,}", f"{company_name} {ticker}".casefold())
    return {value for value in values if value not in {"berhad", "holdings", "group", "limited"}}


def build_sentiment_text(
    row: Mapping[str, Any],
    *,
    mode: str = TEXT_MODE_DEFAULT,
) -> str:
    """Build a bounded, deterministic FinBERT input from one article.

    ``target_aware`` preserves the headline and favours lead sentences that
    explicitly anchor the selected company.  It does not invent text or change
    sentence wording; it simply chooses the most useful existing context.
    """

    selected_mode = str(mode or TEXT_MODE_DEFAULT).strip().casefold()
    if selected_mode not in SUPPORTED_TEXT_MODES:
        raise ValueError(f"Unsupported sentiment text mode: {mode!r}")

    title = _clean(row.get("title"))
    description = _clean(row.get("description"))
    if not description:
        description = _clean(row.get("content_snippet"))
    if not description:
        description = _clean(row.get("article_text"))
    fallback = _clean(row.get("model_text"))
    body = description or fallback
    if selected_mode == TEXT_MODE_DEFAULT:
        return _clean(". ".join(part for part in (title, body) if part))

    company_name = _clean(row.get("company_name"))
    ticker = _clean(row.get("ticker")).upper()
    tokens = _target_tokens(company_name, ticker)
    candidates = _sentences(body)
    anchored = [
        sentence
        for sentence in candidates
        if any(token in sentence.casefold() for token in tokens)
    ]
    # Keep an entity-anchored lead when possible, then enough early context to
    # make the sentiment coherent.  Deduplication preserves source order.
    selected: list[str] = []
    for sentence in [*anchored[:3], *candidates[:3]]:
        if sentence not in selected:
            selected.append(sentence)
    evidence = " ".join(selected) or body
    prefix = ". ".join(
        part
        for part in (
            f"Target company: {company_name} ({ticker})" if company_name or ticker else "",
            title,
        )
        if part
    )
    return _clean(". ".join(part for part in (prefix, evidence) if part))


def apply_sentiment_text_mode(frame: Any, *, mode: str = TEXT_MODE_DEFAULT) -> Any:
    """Return a copy of a tabular frame with one consistent ``model_text`` field.

    Training, validation, test and runtime inputs have slightly different
    source schemas. This adapter keeps their FinBERT representation identical,
    preventing a challenger from training on target-aware text but being
    evaluated on an older headline/lead representation.
    """

    selected_mode = str(mode or TEXT_MODE_DEFAULT).strip().casefold()
    if selected_mode not in SUPPORTED_TEXT_MODES:
        raise ValueError(f"Unsupported sentiment text mode: {mode!r}")

    data = frame.copy()
    for column in (
        "title",
        "description",
        "content_snippet",
        "article_text",
        "model_text",
        "company_name",
        "ticker",
    ):
        if column not in data.columns:
            data[column] = ""
    data["model_text"] = [
        build_sentiment_text(row, mode=selected_mode)
        for row in data.to_dict(orient="records")
    ]
    return data
