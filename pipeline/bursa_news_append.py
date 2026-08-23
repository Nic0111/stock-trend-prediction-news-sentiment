"""Append-only Bursa news expansion helpers.

The historical Bursa corpus is intentionally treated as an immutable baseline.
This module selects only unseen candidates from the already-downloaded GDELT
exports, drives the existing resumable recovery pipeline, packages recovered
text for Codex review, and atomically appends validated records afterwards.

It never trains or promotes a sentiment model.  The output is deliberately
small and versioned so that a failed recovery or label-import run cannot alter
the active historical dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from pipeline.bursa_news_dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_POLICY_PATH,
    EVENT_SIMILARITY_THRESHOLD,
    EVENT_WINDOW_DAYS,
    SENTIMENT_LABELS,
    _derive_event_ids,
    _normalise_title,
    _normalise_url,
    load_policy,
    split_for_timestamp,
    validate_bursa_news_dataset,
)
from pipeline.external_news_label_export import (
    _eligible,
    _input_record,
    export_external_label_package,
)
from pipeline.external_news_label_import import import_external_news_labels
from pipeline.gdelt_ngram_pilot import run_gdelt_ngram_pilot


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECOVERY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "external-corpus-2015-2025"
)
DEFAULT_WORK_DIR = (
    PROJECT_ROOT / "data" / "news-evidence-benchmark" / "bursa-news-append"
)
APPEND_SCHEMA_VERSION = "bursa-news-append-v1"
DEFAULT_TARGET_RELEVANT_PER_TICKER = 100
DEFAULT_RECOVERY_MULTIPLIER = 5
DEFAULT_BATCH_SIZE = 25

# The balance-aware append cycle is deliberately separate from the historical
# total-per-ticker workflow above.  It fills sparse ticker/split/label cells
# without changing the canonical corpus until ``commit`` is explicitly run.
DEFAULT_BALANCE_WORK_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "bursa-news-balance-v2"
)
DEFAULT_TARGET_PER_CLASS_PER_TICKER = 40
DEFAULT_TRAINING_CLASS_TARGET = 28
DEFAULT_VALIDATION_CLASS_TARGET = 6
DEFAULT_TEST_CLASS_TARGET = 6
DEFAULT_BALANCE_RECOVERY_MULTIPLIER = 8
DEFAULT_MAX_CANDIDATES_PER_CYCLE = 1000
BALANCE_SPLITS = ("training", "validation", "test")
BALANCE_LABELS = tuple(str(value).casefold() for value in SENTIMENT_LABELS)


class BursaNewsAppendError(RuntimeError):
    """Raised when an append workflow would change an unexpected baseline."""


@dataclass(frozen=True)
class ExistingIndexes:
    article_ids: frozenset[str]
    ticker_urls: frozenset[tuple[str, str]]
    ticker_hashes: frozenset[tuple[str, str]]
    titles_by_ticker_day: Mapping[str, Mapping[str, tuple[str, ...]]]


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BursaNewsAppendError(f"Unable to read append manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise BursaNewsAppendError(f"Append manifest is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BursaNewsAppendError(f"Invalid JSONL in {path} line {line_number}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(row), ensure_ascii=False, default=str) for row in rows]
    _atomic_write_text(path, ("\n".join(lines) + "\n") if lines else "")


def _read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise BursaNewsAppendError(f"Bursa dataset does not exist: {path}")
    return pd.read_json(path, lines=True)


def _normalise_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalise_hash(value: Any) -> str:
    return str(value or "").strip().casefold()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def _normalise_timestamp(value: Any) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp)


def _candidate_article_id(row: Mapping[str, Any], url: str) -> str:
    existing = str(row.get("article_id") or "").strip()
    if existing:
        return existing
    basis = "|".join(
        (
            _normalise_ticker(row.get("ticker")),
            str(row.get("published_at") or "").strip(),
            url,
        )
    )
    return _sha256_bytes(basis.encode("utf-8"))


def _source_domain(row: Mapping[str, Any], url: str) -> str:
    explicit = str(row.get("source_domain") or "").strip().casefold()
    if explicit:
        return explicit.removeprefix("www.")
    return (urlparse(url).hostname or "").casefold().removeprefix("www.")


def _canonical_candidate(row: Mapping[str, Any]) -> dict[str, Any] | None:
    ticker = _normalise_ticker(row.get("ticker"))
    url = _normalise_url(row.get("url") or row.get("source_url"))
    published = _normalise_timestamp(row.get("published_at"))
    if not ticker or not url or published is None:
        return None
    split = split_for_timestamp(published)
    if not split:
        return None
    title = str(row.get("title") or "").strip()
    canonical = dict(row)
    canonical.update(
        {
            "ticker": ticker,
            "company_name": str(row.get("company_name") or "").strip(),
            "url": url,
            "published_at": published.isoformat(),
            "title": title,
            "source_domain": _source_domain(row, url),
            "split": split,
            "article_id": _candidate_article_id(row, url),
        }
    )
    canonical["candidate_key"] = _sha256_bytes(
        f"{ticker}|{canonical['published_at']}|{url}".encode("utf-8")
    )
    return canonical


def _existing_indexes(frame: pd.DataFrame) -> ExistingIndexes:
    titles: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    article_ids: set[str] = set()
    ticker_urls: set[tuple[str, str]] = set()
    ticker_hashes: set[tuple[str, str]] = set()
    for row in frame.to_dict(orient="records"):
        ticker = _normalise_ticker(row.get("ticker"))
        article_id = str(row.get("article_id") or "").strip()
        if article_id:
            article_ids.add(article_id)
        url = _normalise_url(row.get("url") or row.get("source_url"))
        if ticker and url:
            ticker_urls.add((ticker, url))
        content_hash = _normalise_hash(row.get("content_sha256"))
        if ticker and content_hash:
            ticker_hashes.add((ticker, content_hash))
        published = _normalise_timestamp(row.get("published_at"))
        title = _normalise_title(row.get("title") or row.get("model_text"))
        if ticker and published is not None and title:
            titles[ticker][published.date().isoformat()].append(title)
    return ExistingIndexes(
        article_ids=frozenset(article_ids),
        ticker_urls=frozenset(ticker_urls),
        ticker_hashes=frozenset(ticker_hashes),
        titles_by_ticker_day={
            ticker: {day: tuple(values) for day, values in days.items()}
            for ticker, days in titles.items()
        },
    )


def _matches_title_event(
    ticker: str,
    published_at: pd.Timestamp,
    title: str,
    indexes: ExistingIndexes,
    *,
    fuzzy: bool = True,
) -> bool:
    normalised = _normalise_title(title)
    if not normalised:
        return False
    for offset in range(-EVENT_WINDOW_DAYS, EVENT_WINDOW_DAYS + 1):
        day = (published_at + pd.Timedelta(days=offset)).date().isoformat()
        for prior_title in indexes.titles_by_ticker_day.get(ticker, {}).get(day, ()):
            if normalised == prior_title:
                return True
            if not fuzzy:
                continue
            # Avoid paying the SequenceMatcher cost for obviously different
            # headlines.  It is used only for plausible same-event variants.
            if len(set(normalised.split()) & set(prior_title.split())) < 3:
                continue
            if SequenceMatcher(None, normalised, prior_title).ratio() >= EVENT_SIMILARITY_THRESHOLD:
                return True
    return False


def _relevant_counts(frame: pd.DataFrame) -> dict[str, int]:
    entity = frame.get("entity_relevant", pd.Series(False, index=frame.index))
    entity = entity.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().casefold() in {"1", "true", "yes", "y"}
    )
    labels = (
        frame.get("sentiment_label", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    selected = frame.loc[entity & labels.isin(SENTIMENT_LABELS)]
    return {
        str(ticker): int(count)
        for ticker, count in selected["ticker"].astype(str).str.upper().value_counts().items()
    }


def _split_counts(frame: pd.DataFrame, ticker: str) -> dict[str, int]:
    entity = frame.get("entity_relevant", pd.Series(False, index=frame.index))
    entity = entity.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().casefold() in {"1", "true", "yes", "y"}
    )
    labels = (
        frame.get("sentiment_label", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    selected = frame.loc[
        entity
        & labels.isin(SENTIMENT_LABELS)
        & frame["ticker"].astype(str).str.upper().eq(ticker)
    ]
    return {str(key): int(value) for key, value in selected["split"].value_counts().items()}


def _round_robin_month_domain(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Return a deterministic, month/domain-diverse bounded sample."""

    if count <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    working = frame.copy()
    working["_month"] = pd.to_datetime(working["published_at"], utc=True).dt.strftime("%Y-%m")
    working["_key"] = working.apply(
        lambda row: _sha256_bytes(
            f"{row['ticker']}|{row['_month']}|{row['source_domain']}|{row['url']}".encode("utf-8")
        ),
        axis=1,
    )
    groups = [
        group.sort_values(["_key", "url"], kind="stable")
        for _, group in working.sort_values(["_month", "source_domain", "_key"], kind="stable").groupby(
            ["_month", "source_domain"], sort=True
        )
    ]
    positions = [0] * len(groups)
    # Preserve the source index type.  Candidate frames normally use an
    # integer RangeIndex, but tests and future collectors may use a string or
    # UUID index.  Coercing it to int would silently make a valid candidate
    # unselectable.
    selected: list[Any] = []
    while len(selected) < count:
        progressed = False
        for position, group in enumerate(groups):
            cursor = positions[position]
            if cursor >= len(group):
                continue
            selected.append(group.index[cursor])
            positions[position] += 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return working.loc[selected].drop(columns=["_month", "_key"], errors="ignore")


def _select_ticker_candidates(
    frame: pd.DataFrame,
    *,
    ticker: str,
    current_total: int,
    current_by_split: Mapping[str, int],
    target_relevant_per_ticker: int,
    recovery_multiplier: int,
) -> pd.DataFrame:
    deficit = max(0, int(target_relevant_per_ticker) - int(current_total))
    if deficit <= 0:
        return frame.iloc[0:0].copy()
    desired = {"training": 70, "validation": 15, "test": 15}
    selections: list[pd.DataFrame] = []
    selected_indices: set[Any] = set()
    for split in ("training", "validation", "test"):
        shortage = max(0, desired[split] - int(current_by_split.get(split, 0)))
        budget = shortage * max(1, int(recovery_multiplier))
        available = frame.loc[frame["split"].eq(split)]
        sample = _round_robin_month_domain(available, budget)
        selections.append(sample)
        selected_indices.update(sample.index.tolist())
    total_budget = deficit * max(1, int(recovery_multiplier))
    selected = pd.concat(selections, ignore_index=False) if selections else frame.iloc[0:0].copy()
    if len(selected) < total_budget:
        remaining = frame.loc[~frame.index.isin(selected_indices)]
        fill = _round_robin_month_domain(remaining, total_budget - len(selected))
        selected = pd.concat([selected, fill], ignore_index=False)
    return selected.head(total_budget).copy()


def _candidate_files(root: Path, years: Sequence[int]) -> list[Path]:
    files: list[Path] = []
    for year in dict.fromkeys(int(value) for value in years):
        direct = root / str(year) / "exports"
        if direct.exists():
            files.extend(sorted(direct.glob("*.jsonl")))
        if year == 2023 and not direct.exists():
            for name in ("pilot-2023-high-recall-v3b", "pilot-2023-high-recall-v3"):
                fallback = root.parent / name / "exports"
                if fallback.exists():
                    files.extend(sorted(fallback.glob("*.jsonl")))
                    break
    return [path for path in files if path.name != "query-state.jsonl"]


def _load_candidates(root: Path, years: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in _candidate_files(root, years):
        rows.extend(_read_jsonl(path))
    if not rows:
        return pd.DataFrame(columns=["ticker", "company_name", "url", "published_at", "title", "source_domain", "split", "article_id", "candidate_key"])
    raw = pd.DataFrame(rows)
    if "url" not in raw and "source_url" in raw:
        raw["url"] = raw["source_url"]
    required = {"ticker", "url", "published_at"}
    if not required.issubset(raw.columns):
        return pd.DataFrame(columns=["ticker", "company_name", "url", "published_at", "title", "source_domain", "split", "article_id", "candidate_key"])
    raw["published_at"] = pd.to_datetime(raw["published_at"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["published_at", "ticker", "url"])
    canonical_rows: list[dict[str, Any]] = []
    for row in raw.to_dict(orient="records"):
        canonical = _canonical_candidate(row)
        if canonical is not None:
            canonical_rows.append(canonical)
    return pd.DataFrame(canonical_rows)


def _candidate_is_duplicate(
    row: Mapping[str, Any],
    indexes: ExistingIndexes,
    *,
    seen_ids: set[str],
    seen_urls: set[tuple[str, str]],
    seen_titles: dict[str, dict[str, set[str]]],
    fuzzy_title: bool = True,
) -> str | None:
    ticker = _normalise_ticker(row.get("ticker"))
    article_id = str(row.get("article_id") or "").strip()
    url = _normalise_url(row.get("url") or row.get("source_url"))
    published = _normalise_timestamp(row.get("published_at"))
    title = _normalise_title(row.get("title"))
    if article_id and (article_id in indexes.article_ids or article_id in seen_ids):
        return "article_id"
    if ticker and url and ((ticker, url) in indexes.ticker_urls or (ticker, url) in seen_urls):
        return "url"
    if published is not None and _matches_title_event(
        ticker, published, title, indexes, fuzzy=fuzzy_title
    ):
        return "title_event"
    if published is not None and title:
        for offset in range(-EVENT_WINDOW_DAYS, EVENT_WINDOW_DAYS + 1):
            day = (published + pd.Timedelta(days=offset)).date().isoformat()
            prior_titles = seen_titles.get(ticker, {}).get(day, set())
            if title in prior_titles:
                return "candidate_title_event"
            if not fuzzy_title:
                continue
            for prior_title in prior_titles:
                if len(set(title.split()) & set(prior_title.split())) < 3:
                    continue
                if SequenceMatcher(None, title, prior_title).ratio() >= EVENT_SIMILARITY_THRESHOLD:
                    return "candidate_title_event"
    return None


def _mark_seen_candidate(
    row: Mapping[str, Any],
    *,
    seen_ids: set[str],
    seen_urls: set[tuple[str, str]],
    seen_titles: dict[str, dict[str, set[str]]],
) -> None:
    ticker = _normalise_ticker(row.get("ticker"))
    article_id = str(row.get("article_id") or "").strip()
    url = _normalise_url(row.get("url") or row.get("source_url"))
    published = _normalise_timestamp(row.get("published_at"))
    title = _normalise_title(row.get("title"))
    if article_id:
        seen_ids.add(article_id)
    if ticker and url:
        seen_urls.add((ticker, url))
    if ticker and published is not None and title:
        seen_titles[ticker][published.date().isoformat()].add(title)


def _base_manifest(dataset_path: Path, dataset: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "dataset_rows": int(len(dataset)),
        "created_at": _utc_now(),
    }


def _assert_baseline(manifest: Mapping[str, Any], dataset_path: Path) -> None:
    if not dataset_path.exists():
        raise BursaNewsAppendError(f"Canonical dataset disappeared: {dataset_path}")
    expected = str(manifest.get("base_dataset", {}).get("dataset_sha256") or "")
    actual = _sha256_file(dataset_path)
    if not expected or expected != actual:
        raise BursaNewsAppendError(
            "The canonical Bursa dataset changed after this append run was prepared. "
            "Start a new append run rather than merging into a different baseline."
        )


def _balance_targets(
    *,
    target_per_class_per_ticker: int,
    training_target: int,
    validation_target: int,
    test_target: int,
) -> dict[str, int]:
    """Validate and return the per-class chronological target matrix."""

    values = {
        "training": int(training_target),
        "validation": int(validation_target),
        "test": int(test_target),
    }
    if int(target_per_class_per_ticker) <= 0:
        raise ValueError("target_per_class_per_ticker must be positive")
    if any(value < 0 for value in values.values()):
        raise ValueError("per-split class targets cannot be negative")
    if sum(values.values()) != int(target_per_class_per_ticker):
        raise ValueError(
            "training, validation and test class targets must sum to "
            "target_per_class_per_ticker"
        )
    return values


def _empty_balance_matrix(
    tickers: Sequence[str],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        str(ticker).upper(): {
            split: {label: 0 for label in BALANCE_LABELS}
            for split in BALANCE_SPLITS
        }
        for ticker in tickers
    }


def _label_split_counts(
    frame: pd.DataFrame,
    tickers: Sequence[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Count only relevant, labelled rows in the requested cells."""

    counts = _empty_balance_matrix(tickers)
    if frame.empty:
        return counts
    entity = frame.get("entity_relevant", pd.Series(False, index=frame.index)).map(
        _coerce_bool
    )
    labels = (
        frame.get("sentiment_label", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    for row in frame.loc[entity & labels.isin(BALANCE_LABELS)].to_dict(orient="records"):
        ticker = _normalise_ticker(row.get("ticker"))
        split = str(row.get("split") or "").strip().casefold()
        label = str(row.get("sentiment_label") or "").strip().casefold()
        if ticker in counts and split in BALANCE_SPLITS and label in BALANCE_LABELS:
            counts[ticker][split][label] += 1
    return counts


def _balance_deficits(
    counts: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    split_targets: Mapping[str, int],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        ticker: {
            split: {
                label: max(
                    0,
                    int(split_targets[split])
                    - int(counts.get(ticker, {}).get(split, {}).get(label, 0)),
                )
                for label in BALANCE_LABELS
            }
            for split in BALANCE_SPLITS
        }
        for ticker in counts
    }


def _balance_candidate_sample(
    frame: pd.DataFrame,
    *,
    deficits: Mapping[str, Mapping[str, int]],
    recovery_multiplier: int,
) -> pd.DataFrame:
    """Select a diverse raw pool for a ticker/split.

    Sentiment is intentionally unknown at candidate time.  The budget is the
    sum of class deficits multiplied by the recovery factor; the subsequent
    local-model triage only prioritises which recovered rows are sent for
    Codex review.  This avoids leaking model predictions into labels.
    """

    if frame.empty:
        return frame.iloc[0:0].copy()
    deficit_total = sum(max(0, int(value)) for value in deficits.values())
    budget = deficit_total * max(1, int(recovery_multiplier))
    if budget <= 0:
        return frame.iloc[0:0].copy()
    return _round_robin_month_domain(frame, budget).head(budget).copy()


def _write_selected_candidate_files(
    output: Path,
    selected: pd.DataFrame,
    *,
    candidate_base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    candidate_dir = candidate_base_dir or (output / "candidates")
    candidate_files: dict[str, dict[str, Any]] = {}
    if selected.empty:
        return candidate_files
    selected = selected.copy()
    selected["published_at"] = pd.to_datetime(
        selected["published_at"], utc=True, errors="coerce"
    )
    for year, group in selected.dropna(subset=["published_at"]).groupby(
        selected.dropna(subset=["published_at"])["published_at"].dt.year
    ):
        destination = candidate_dir / str(int(year)) / "candidates.jsonl"
        records = group.drop(columns=["candidate_key"], errors="ignore").to_dict(
            orient="records"
        )
        _write_jsonl(destination, records)
        candidate_files[str(int(year))] = {
            "path": str(destination),
            "rows": len(records),
            "sha256": _sha256_file(destination),
        }
    return candidate_files


def _load_prior_candidate_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load prior-cycle candidate identities for explicit cycle continuation."""

    rows: list[dict[str, Any]] = []
    for value in (manifest.get("candidate_files") or {}).values():
        if isinstance(value, Mapping) and value.get("path"):
            rows.extend(_read_jsonl(Path(str(value["path"]))))
    return rows


def _cycle_name(manifest: Mapping[str, Any]) -> str | None:
    if str(manifest.get("selection_mode", "")).casefold() != "class_split_balanced":
        return None
    return f"cycle-{int(manifest.get('cycle_index', 1) or 1):03d}"


def _cycle_path(output: Path, manifest: Mapping[str, Any], name: str) -> Path:
    cycle = _cycle_name(manifest)
    return output / name / cycle if cycle else output / name


def _prepare_balanced_append_plan(
    *,
    dataset_file: Path,
    root: Path,
    output: Path,
    normalized_years: Sequence[int],
    required_tickers: Sequence[str],
    dataset: pd.DataFrame,
    unique: pd.DataFrame,
    rejections: Mapping[str, int],
    target_per_class_per_ticker: int,
    split_targets: Mapping[str, int],
    recovery_multiplier: int,
    max_candidates_per_cycle: int,
    cycle_index: int = 1,
) -> dict[str, Any]:
    counts = _label_split_counts(dataset, required_tickers)
    deficits = _balance_deficits(counts, split_targets=split_targets)
    selections: list[pd.DataFrame] = []
    for ticker in required_tickers:
        ticker_frame = unique.loc[unique["ticker"].eq(ticker)] if not unique.empty else unique
        for split in BALANCE_SPLITS:
            available = ticker_frame.loc[ticker_frame["split"].eq(split)]
            sample = _balance_candidate_sample(
                available,
                deficits=deficits[ticker][split],
                recovery_multiplier=recovery_multiplier,
            )
            if not sample.empty:
                selections.append(sample)

    # Keep every ticker/split represented where possible, then cap the whole
    # cycle deterministically.  This makes reruns idempotent and protects the
    # free recovery budget from a single abundant ticker.
    selected_frames = [frame.reset_index(drop=True) for frame in selections if not frame.empty]
    selected_rows: list[dict[str, Any]] = []
    position = 0
    while len(selected_rows) < max(0, int(max_candidates_per_cycle)):
        progressed = False
        for frame in selected_frames:
            if position < len(frame):
                selected_rows.append(frame.iloc[position].to_dict())
                progressed = True
                if len(selected_rows) >= int(max_candidates_per_cycle):
                    break
        if not progressed:
            break
        position += 1
    selected = pd.DataFrame(selected_rows)
    output.mkdir(parents=True, exist_ok=True)
    candidate_files = _write_selected_candidate_files(
        output,
        selected,
        candidate_base_dir=output / "candidates" / f"cycle-{int(cycle_index):03d}",
    )
    targets: dict[str, Any] = {}
    for ticker in required_tickers:
        ticker_pool = unique.loc[unique["ticker"].eq(ticker)] if not unique.empty else unique
        selected_count = int(selected["ticker"].eq(ticker).sum()) if not selected.empty else 0
        targets[ticker] = {
            "target_per_class_per_ticker": int(target_per_class_per_ticker),
            "split_targets_per_class": dict(split_targets),
            "current_by_split_label": counts[ticker],
            "deficit_by_split_label": deficits[ticker],
            "candidate_pool_after_deduplication": int(len(ticker_pool)),
            "selected_for_recovery": selected_count,
        }
    manifest = {
        "schema_version": APPEND_SCHEMA_VERSION,
        "status": "prepared",
        "selection_mode": "class_split_balanced",
        "cycle_index": int(cycle_index),
        "recovery_root_dir": str(output / "recovery-root" / f"cycle-{int(cycle_index):03d}"),
        "base_dataset": _base_manifest(dataset_file, dataset),
        "candidate_root": str(root),
        "years": list(normalized_years),
        "target_per_class_per_ticker": int(target_per_class_per_ticker),
        "split_targets_per_class": dict(split_targets),
        "recovery_multiplier": int(recovery_multiplier),
        "max_candidates_per_cycle": int(max_candidates_per_cycle),
        "candidate_rows_loaded": int(len(unique) + sum(rejections.values())),
        "candidate_rows_after_deduplication": int(len(unique)),
        "candidate_rejections": dict(sorted(rejections.items())),
        "targets": targets,
        "selected_rows": int(len(selected)),
        "candidate_files": candidate_files,
        "created_at": _utc_now(),
        "production_changes": False,
    }
    manifest_path = output / "append-manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def prepare_bursa_news_append(
    *,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    candidate_root: Path | str = DEFAULT_RECOVERY_ROOT,
    work_dir: Path | str = DEFAULT_WORK_DIR,
    years: Sequence[int] = tuple(range(2015, 2026)),
    target_relevant_per_ticker: int = DEFAULT_TARGET_RELEVANT_PER_TICKER,
    recovery_multiplier: int = DEFAULT_RECOVERY_MULTIPLIER,
    target_per_class_per_ticker: int | None = None,
    training_target_per_class: int = DEFAULT_TRAINING_CLASS_TARGET,
    validation_target_per_class: int = DEFAULT_VALIDATION_CLASS_TARGET,
    test_target_per_class: int = DEFAULT_TEST_CLASS_TARGET,
    max_candidates_per_cycle: int = DEFAULT_MAX_CANDIDATES_PER_CYCLE,
    new_cycle: bool = False,
    tickers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a deduplicated, balanced candidate plan without network calls.

    The legacy mode targets a total relevant count per ticker.  Passing
    ``target_per_class_per_ticker`` activates the class/split matrix planner
    used by the balance-v2 append cycle.  Both modes share the same immutable
    baseline and exact candidate de-duplication rules.
    """

    if target_relevant_per_ticker <= 0:
        raise ValueError("target_relevant_per_ticker must be positive")
    if recovery_multiplier <= 0:
        raise ValueError("recovery_multiplier must be positive")
    if target_per_class_per_ticker is not None:
        split_targets = _balance_targets(
            target_per_class_per_ticker=int(target_per_class_per_ticker),
            training_target=int(training_target_per_class),
            validation_target=int(validation_target_per_class),
            test_target=int(test_target_per_class),
        )
        if int(max_candidates_per_cycle) <= 0:
            raise ValueError("max_candidates_per_cycle must be positive")
    normalized_years = tuple(dict.fromkeys(int(value) for value in years))
    if any(year < 2015 or year > 2025 for year in normalized_years):
        raise ValueError("append collection supports only 2015-2025")
    dataset_file = Path(dataset_path).resolve()
    root = Path(candidate_root).resolve()
    output = Path(work_dir).resolve()
    manifest_path = output / "append-manifest.json"
    prior_manifest: dict[str, Any] | None = None
    cycle_index = 1
    prior_candidate_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        # A prepared workspace is a checkpoint.  Recreating it in place could
        # orphan completed recovery requests and violate the no-repeat promise.
        # Start a fresh work directory if the sampling settings need to change.
        if str(existing.get("status", "")).casefold() != "committed" and not new_cycle:
            _assert_baseline(existing, dataset_file)
            return {
                **existing,
                "manifest_path": str(manifest_path),
                "resumed_existing_workspace": True,
            }
        if new_cycle:
            if str(existing.get("status", "")).casefold() == "committed":
                raise BursaNewsAppendError(
                    "The append workspace is already committed; start a new work directory "
                    "for another balance cycle."
                )
            _assert_baseline(existing, dataset_file)
            prior_manifest = existing
            prior_candidate_rows = _load_prior_candidate_rows(existing)
            cycle_index = int(existing.get("cycle_index", 1) or 1) + 1
    dataset = _read_dataset(dataset_file)
    policy = load_policy(DEFAULT_POLICY_PATH)
    policy_tickers = tuple(str(value).upper() for value in policy["required_tickers"])
    requested_tickers = tuple(
        dict.fromkeys(_normalise_ticker(value) for value in (tickers or policy_tickers))
    )
    unsupported = sorted(set(requested_tickers) - set(policy_tickers))
    if unsupported:
        raise ValueError(f"Unsupported remediation tickers: {unsupported}")
    if not requested_tickers:
        raise ValueError("At least one remediation ticker is required")
    required_tickers = requested_tickers
    current_counts = _relevant_counts(dataset)
    indexes = _existing_indexes(dataset)
    candidates = _load_candidates(root, normalized_years)
    rejections: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()
    seen_urls: set[tuple[str, str]] = set()
    seen_titles: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    # An explicit cycle continuation must not select a URL/article that was
    # already sent through a previous recovery pass in this workspace.
    for prior in prior_candidate_rows:
        _mark_seen_candidate(
            prior,
            seen_ids=seen_ids,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
        )
    unique_rows: list[dict[str, Any]] = []
    for row in candidates.sort_values(["ticker", "published_at", "url"], kind="stable").to_dict(orient="records"):
        ticker = _normalise_ticker(row.get("ticker"))
        if ticker not in required_tickers:
            rejections["unsupported_ticker"] += 1
            continue
        reason = _candidate_is_duplicate(
            row,
            indexes,
            seen_ids=seen_ids,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
            fuzzy_title=False,
        )
        if reason:
            rejections[reason] += 1
            continue
        _mark_seen_candidate(
            row,
            seen_ids=seen_ids,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
        )
        unique_rows.append(row)
    unique = pd.DataFrame(unique_rows)
    if target_per_class_per_ticker is not None:
        return _prepare_balanced_append_plan(
            dataset_file=dataset_file,
            root=root,
            output=output,
            normalized_years=normalized_years,
            required_tickers=required_tickers,
            dataset=dataset,
            unique=unique,
            rejections=rejections,
            target_per_class_per_ticker=int(target_per_class_per_ticker),
            split_targets=split_targets,
            recovery_multiplier=int(recovery_multiplier),
            max_candidates_per_cycle=int(max_candidates_per_cycle),
            cycle_index=cycle_index,
        )
    selections: list[pd.DataFrame] = []
    targets: dict[str, Any] = {}
    for ticker in required_tickers:
        current = int(current_counts.get(ticker, 0))
        by_split = _split_counts(dataset, ticker)
        target_addition = max(0, int(target_relevant_per_ticker) - current)
        ticker_frame = unique.loc[unique["ticker"].eq(ticker)] if not unique.empty else unique
        sample = _select_ticker_candidates(
            ticker_frame,
            ticker=ticker,
            current_total=current,
            current_by_split=by_split,
            target_relevant_per_ticker=target_relevant_per_ticker,
            recovery_multiplier=recovery_multiplier,
        )
        selections.append(sample)
        targets[ticker] = {
            "current_relevant": current,
            "target_relevant": int(target_relevant_per_ticker),
            "target_addition": target_addition,
            "current_by_split": by_split,
            "candidate_pool_after_deduplication": int(len(ticker_frame)),
            "selected_for_recovery": int(len(sample)),
        }
    selected_by_ticker = [frame.reset_index(drop=True) for frame in selections if not frame.empty]
    selected_rows: list[dict[str, Any]] = []
    maximum = max((len(frame) for frame in selected_by_ticker), default=0)
    for position in range(maximum):
        for frame in selected_by_ticker:
            if position < len(frame):
                selected_rows.append(frame.iloc[position].to_dict())
    selected = pd.DataFrame(selected_rows)
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = output / "candidates"
    candidate_files: dict[str, dict[str, Any]] = {}
    if not selected.empty:
        for year, group in selected.groupby(pd.to_datetime(selected["published_at"], utc=True).dt.year):
            destination = candidate_dir / str(int(year)) / "candidates.jsonl"
            records = group.drop(columns=["candidate_key"], errors="ignore").to_dict(orient="records")
            _write_jsonl(destination, records)
            candidate_files[str(int(year))] = {
                "path": str(destination),
                "rows": len(records),
                "sha256": _sha256_file(destination),
            }
    manifest = {
        "schema_version": APPEND_SCHEMA_VERSION,
        "status": "prepared",
        "base_dataset": _base_manifest(dataset_file, dataset),
        "candidate_root": str(root),
        "years": list(normalized_years),
        "target_relevant_per_ticker": int(target_relevant_per_ticker),
        "recovery_multiplier": int(recovery_multiplier),
        "candidate_rows_loaded": int(len(candidates)),
        "candidate_rows_after_deduplication": int(len(unique)),
        "candidate_rejections": dict(sorted(rejections.items())),
        "targets": targets,
        "selected_rows": int(len(selected)),
        "candidate_files": candidate_files,
        "created_at": _utc_now(),
        "production_changes": False,
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def recover_bursa_news_append(
    *,
    work_dir: Path | str = DEFAULT_WORK_DIR,
    max_runtime_minutes: float = 45.0,
    max_download_gib: float = 5.0,
    request_interval_seconds: float = 1.0,
    skip_archives: bool = False,
    recovery_runner: Callable[..., dict[str, Any]] = run_gdelt_ngram_pilot,
) -> dict[str, Any]:
    """Run one bounded, resumable recovery pass for each prepared year."""

    output = Path(work_dir).resolve()
    manifest_path = output / "append-manifest.json"
    manifest = _read_json(manifest_path)
    dataset_path = Path(manifest["base_dataset"]["dataset_path"])
    _assert_baseline(manifest, dataset_path)
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + max(0.0, float(max_runtime_minutes)) * 60.0
    paused_at_safety_limit = False
    recovery_root = Path(
        manifest.get("recovery_root_dir")
        or (output / "recovery-root")
    )
    for year_text, record in sorted((manifest.get("candidate_files") or {}).items()):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            paused_at_safety_limit = True
            break
        year = int(year_text)
        candidate_path = Path(record["path"])
        rows = _read_jsonl(candidate_path)
        if not rows:
            continue
        ticker_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            ticker_counts[_normalise_ticker(row.get("ticker"))] += 1
        per_ticker = max(ticker_counts.values(), default=1)
        result = recovery_runner(
            year=year,
            candidate_dir=candidate_path.parent,
            sample_per_ticker=per_ticker,
            paired_per_ticker=0,
            output_dir=recovery_root / str(year) / "recovery",
            # The caller's limit is a command-wide safety budget, not a fresh
            # 45-minute allowance for every year.  The recovery runner has its
            # own checkpoint and will resume the unfinished year next time.
            max_runtime_minutes=max(0.05, remaining_seconds / 60.0),
            max_download_gib=float(max_download_gib),
            request_interval_seconds=float(request_interval_seconds),
            run_labels=False,
            skip_archives=bool(skip_archives),
        )
        summaries.append({"year": year, **result})
    recovery_report = {
        "schema_version": APPEND_SCHEMA_VERSION,
        "status": "recovery_attempted",
        "created_at": _utc_now(),
        "summaries": summaries,
        "resumable": True,
        "paused_at_safety_limit": paused_at_safety_limit,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "production_changes": False,
    }
    _write_json(output / "recovery-report.json", recovery_report)
    return recovery_report


def _filter_recovered_rows(
    recovered_rows: Iterable[Mapping[str, Any]],
    baseline: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    indexes = _existing_indexes(baseline)
    accepted: list[dict[str, Any]] = []
    rejected: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()
    seen_urls: set[tuple[str, str]] = set()
    seen_titles: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    seen_hashes: set[tuple[str, str]] = set()
    ordered = sorted(
        (dict(row) for row in recovered_rows),
        key=lambda row: (str(row.get("published_at", "")), str(row.get("ticker", "")), str(row.get("url", ""))),
    )
    for raw in ordered:
        eligible, reason = _eligible(raw)
        if not eligible:
            rejected[reason] += 1
            continue
        record = _input_record(raw)
        duplicate = _candidate_is_duplicate(
            record,
            indexes,
            seen_ids=seen_ids,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
        )
        if duplicate:
            rejected[f"deduplicated_{duplicate}"] += 1
            continue
        ticker_hash = (_normalise_ticker(record.get("ticker")), _normalise_hash(record.get("content_sha256")))
        if ticker_hash in indexes.ticker_hashes or ticker_hash in seen_hashes:
            rejected["deduplicated_content_hash"] += 1
            continue
        _mark_seen_candidate(
            record,
            seen_ids=seen_ids,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
        )
        seen_hashes.add(ticker_hash)
        accepted.append(raw)
    return accepted, dict(sorted(rejected.items()))


def triage_bursa_news_append(
    *,
    work_dir: Path | str = DEFAULT_BALANCE_WORK_DIR,
    random_control_share: float = 0.10,
) -> dict[str, Any]:
    """Prioritise recovered rows for review using a frozen local model.

    Triage is a sampling hint only.  It never writes a sentiment label and the
    hint/probability fields are deliberately removed by ``_input_record``
    before a Codex package is produced.  If the local PhraseBank artifact is
    unavailable, all rows remain reviewable in deterministic source order.
    """

    if not 0.0 <= float(random_control_share) <= 1.0:
        raise ValueError("random_control_share must be between 0 and 1")
    output = Path(work_dir).resolve()
    manifest = _read_json(output / "append-manifest.json")
    dataset_path = Path(manifest["base_dataset"]["dataset_path"])
    _assert_baseline(manifest, dataset_path)
    baseline = _read_dataset(dataset_path)
    recovery_root = Path(manifest.get("recovery_root_dir") or (output / "recovery-root"))
    raw_rows: list[dict[str, Any]] = []
    for year in manifest.get("years", []):
        source = recovery_root / str(int(year)) / "recovery" / "raw" / "hybrid-recovered.jsonl"
        raw_rows.extend(_read_jsonl(source))
    accepted, rejected = _filter_recovered_rows(raw_rows, baseline)
    if not accepted:
        raise BursaNewsAppendError(
            "No eligible recovered rows are available for triage. Resume recovery first."
        )

    policy = load_policy(DEFAULT_POLICY_PATH)
    required_tickers = tuple(str(value).upper() for value in policy["required_tickers"])
    counts = _label_split_counts(baseline, required_tickers)
    split_targets = manifest.get("split_targets_per_class") or {
        "training": DEFAULT_TRAINING_CLASS_TARGET,
        "validation": DEFAULT_VALIDATION_CLASS_TARGET,
        "test": DEFAULT_TEST_CLASS_TARGET,
    }
    deficits = _balance_deficits(counts, split_targets=split_targets)

    probabilities: dict[str, list[float]] = {}
    model_available = False
    model_error = ""
    try:
        # This is the frozen PhraseBank artifact used only to order a review
        # queue; it is not the active runtime ensemble and is never promoted.
        from pipeline.finbert_sentiment_model import load_finbert_sentiment_model

        model = load_finbert_sentiment_model()
        texts = [
            ". ".join(
                value
                for value in (
                    str(row.get("title") or "").strip(),
                    str(
                        row.get("article_text")
                        or row.get("content_snippet")
                        or row.get("description")
                        or ""
                    ).strip(),
                )
                if value
            )
            for row in accepted
        ]
        predicted = model.predict_proba(texts)
        classes = [str(value).casefold() for value in model.classes_]
        probabilities = {
            str(row.get("article_id") or ""): [float(value) for value in vector]
            for row, vector in zip(accepted, predicted)
        }
        model_available = True
    except Exception as exc:  # triage must not block a free recovery cycle
        model_error = str(exc)

    queue: list[dict[str, Any]] = []
    lane_counts: dict[str, int] = defaultdict(int)
    for row in accepted:
        record = dict(row)
        article_id = str(record.get("article_id") or "")
        ticker = _normalise_ticker(record.get("ticker"))
        split = str(record.get("split") or split_for_timestamp(_normalise_timestamp(record.get("published_at"))) or "").casefold()
        vector = probabilities.get(article_id)
        if vector and model_available:
            class_probs = {classes[index]: float(value) for index, value in enumerate(vector)}
            ordered = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
            hint, top_probability = ordered[0]
            margin = top_probability - ordered[1][1] if len(ordered) > 1 else top_probability
        else:
            class_probs = {}
            hint, top_probability, margin = "unknown", 0.0, 0.0
        deficits_for_cell = deficits.get(ticker, {}).get(split, {})
        deficit_label = max(
            BALANCE_LABELS,
            key=lambda label: int(deficits_for_cell.get(label, 0)),
            default="unknown",
        )
        if sum(int(value) for value in deficits_for_cell.values()) <= 0:
            deficit_label = "reserve"
        is_control = (
            int(_sha256_bytes(article_id.encode("utf-8"))[:8], 16) / 0xFFFFFFFF
            < float(random_control_share)
        )
        if is_control:
            lane = "random_control"
            priority = 0
        elif hint == "unknown":
            lane = "unscored"
            priority = 1
        elif top_probability < 0.55 or margin < 0.15:
            lane = "uncertain"
            priority = 2
        elif hint in deficits_for_cell and int(deficits_for_cell.get(hint, 0)) > 0:
            lane = f"likely_{hint}"
            priority = 3
        else:
            lane = f"deficit_{deficit_label}"
            priority = 4
        record.update(
            {
                "triage_lane": lane,
                "triage_priority": priority,
                "triage_model_hint": hint,
                "triage_model_top_probability": round(float(top_probability), 6),
                "triage_model_margin": round(float(margin), 6),
                "triage_target_deficit_label": deficit_label,
                "triage_random_control": bool(is_control),
                "triage_model_predictions_are_not_labels": True,
            }
        )
        lane_counts[lane] += 1
        queue.append(record)
    queue.sort(
        key=lambda row: (
            int(row.get("triage_priority", 9)),
            _normalise_ticker(row.get("ticker")),
            str(row.get("published_at", "")),
            str(row.get("article_id", "")),
        )
    )
    triage_dir = _cycle_path(output, manifest, "triage")
    queue_path = triage_dir / "label-queue.jsonl"
    _write_jsonl(queue_path, queue)
    report = {
        "schema_version": APPEND_SCHEMA_VERSION,
        "status": "triage_ready",
        "created_at": _utc_now(),
        "input_recovered_rows": len(raw_rows),
        "eligible_rows": len(accepted),
        "rejected_rows": rejected,
        "queue_rows": len(queue),
        "lane_counts": dict(sorted(lane_counts.items())),
        "model_available": model_available,
        "model": "frozen_phrasebank_finbert_sampling_hint",
        "model_error": model_error,
        "random_control_share": float(random_control_share),
        "target_deficits_at_triage": deficits,
        "queue_path": str(queue_path),
        "codex_prompt_excludes_triage_predictions": True,
        "production_changes": False,
    }
    _write_json(triage_dir / "triage-report.json", report)
    return report


def export_bursa_news_append_labels(
    *,
    work_dir: Path | str = DEFAULT_WORK_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Build a fresh, baseline-deduplicated Codex label package from recovery."""

    output = Path(work_dir).resolve()
    manifest = _read_json(output / "append-manifest.json")
    dataset_path = Path(manifest["base_dataset"]["dataset_path"])
    _assert_baseline(manifest, dataset_path)
    baseline = _read_dataset(dataset_path)
    recovery_root = Path(manifest.get("recovery_root_dir") or (output / "recovery-root"))
    filtered_root = _cycle_path(output, manifest, "filtered-recovery")
    queue_path = _cycle_path(output, manifest, "triage") / "label-queue.jsonl"
    raw_rows: list[dict[str, Any]] = (
        _read_jsonl(queue_path) if queue_path.exists() else []
    )
    if not raw_rows:
        for year in manifest.get("years", []):
            source = recovery_root / str(int(year)) / "recovery" / "raw" / "hybrid-recovered.jsonl"
            raw_rows.extend(_read_jsonl(source))
    accepted, rejected = _filter_recovered_rows(raw_rows, baseline)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        published = _normalise_timestamp(row.get("published_at"))
        if published is not None:
            grouped[int(published.year)].append(row)
    for year, rows in grouped.items():
        _write_jsonl(
            filtered_root / str(year) / "recovery" / "raw" / "hybrid-recovered.jsonl",
            rows,
        )
    if not grouped:
        raise BursaNewsAppendError(
            "No high-quality, non-duplicate recovered articles are ready for "
            "labelling. Complete or resume recovery before exporting labels."
        )
    package_dir = _cycle_path(output, manifest, "codex-label-package")
    package = export_external_label_package(
        filtered_root,
        output_dir=package_dir,
        years=tuple(sorted(grouped)),
        extra_files=(),
        batch_size=int(batch_size),
    )
    report = {
        "schema_version": APPEND_SCHEMA_VERSION,
        "status": "label_package_ready",
        "created_at": _utc_now(),
        "raw_recovered_rows": len(raw_rows),
        "deduplicated_recovered_rows": len(accepted),
        "rejected_rows": rejected,
        "label_package": package,
        "production_changes": False,
    }
    manifest["label_package_dir"] = str(package_dir)
    _write_json(output / "append-manifest.json", manifest)
    report_path = (
        output / "cycles" / str(_cycle_name(manifest)) / "label-package-report.json"
        if _cycle_name(manifest)
        else output / "label-package-report.json"
    )
    _write_json(report_path, report)
    return report


def import_bursa_news_append_labels(
    returned_labels: Path | str,
    *,
    work_dir: Path | str = DEFAULT_WORK_DIR,
) -> dict[str, Any]:
    """Validate Codex labels into an append staging file, never the canonical dataset."""

    output = Path(work_dir).resolve()
    manifest = _read_json(output / "append-manifest.json")
    dataset_path = Path(manifest["base_dataset"]["dataset_path"])
    _assert_baseline(manifest, dataset_path)
    package = Path(manifest.get("label_package_dir") or _cycle_path(output, manifest, "codex-label-package"))
    staging_root = _cycle_path(output, manifest, "staging")
    staging = staging_root / "codex-reviewed.jsonl"
    queue = staging_root / "codex-review-queue.jsonl"
    report_path = staging_root / "codex-import-report.json"
    report = import_external_news_labels(
        package / "external-label-input.jsonl",
        Path(returned_labels),
        output_dataset=staging,
        review_queue=queue,
        report_path=report_path,
        allow_backfill=True,
    )
    rows = _read_jsonl(staging)
    low_confidence_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            confidence = float(row.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        status = str(row.get("external_label_status") or "labelled").casefold()
        if confidence < 0.70 or status == "needs_review":
            unresolved = dict(row)
            unresolved["append_review_reason"] = (
                "confidence_below_0.70" if confidence < 0.70 else "external_label_needs_review"
            )
            low_confidence_rows.append(unresolved)
            continue
        accepted_rows.append(row)
    _write_jsonl(staging, accepted_rows)
    unresolved_path = staging_root / "unresolved-confidence.jsonl"
    _write_jsonl(unresolved_path, low_confidence_rows)
    for row in accepted_rows:
        # The label was produced through an LLM-assisted review workflow. It
        # is useful weak supervision, but it is not independent human ground
        # truth; keep that provenance explicit in every active field.
        row["label_source"] = "llm_assisted_label"
        row["label_source_original"] = "codex_single_review"
        row["label_provenance"] = "llm_assisted_label"
        row["provenance"] = "llm_assisted_label"
        row["external_labelled_by"] = "codex"
        row["consensus_reason"] = (
            "LLM-assisted single review accepted as weak supervision; "
            "not independent human ground truth"
        )
        row["validation_status"] = "llm_single_pass"
        row["annotation_agreement"] = "llm_single_review"
        row["llm_assisted_label"] = True
        row["llm_assisted_label_at"] = row.get("labelled_at", "")
    _write_jsonl(staging, accepted_rows)
    append_report = {
        "schema_version": APPEND_SCHEMA_VERSION,
        "status": "labels_imported_to_staging",
        "created_at": _utc_now(),
        "import": report,
        "staging_path": str(staging),
        "review_queue": str(queue),
        "unresolved_confidence_path": str(unresolved_path),
        "unresolved_confidence_rows": len(low_confidence_rows),
        "minimum_confidence_for_canonical_append": 0.70,
        "production_changes": False,
    }
    _write_json(staging_root / "append-import-report.json", append_report)
    return append_report


def _filter_append_rows(
    rows: Iterable[Mapping[str, Any]], baseline: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    indexes = _existing_indexes(baseline)
    accepted: list[dict[str, Any]] = []
    rejected: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()
    seen_urls: set[tuple[str, str]] = set()
    seen_titles: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    seen_hashes: set[tuple[str, str]] = set()
    for raw in sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row.get("published_at", "")), str(row.get("ticker", "")), str(row.get("article_id", ""))),
    ):
        record = dict(raw)
        record["url"] = _normalise_url(record.get("url") or record.get("source_url"))
        record["ticker"] = _normalise_ticker(record.get("ticker"))
        record["published_at"] = str(record.get("published_at") or "")
        published = _normalise_timestamp(record["published_at"])
        if not record["ticker"] or not record["url"] or published is None:
            rejected["invalid_identity"] += 1
            continue
        expected_split = split_for_timestamp(published)
        if not expected_split or str(record.get("split") or "") != expected_split:
            rejected["split_mismatch"] += 1
            continue
        relevant = _coerce_bool(record.get("entity_relevant"))
        record["entity_relevant"] = relevant
        label = str(record.get("sentiment_label") or "").strip().casefold()
        if relevant and label not in SENTIMENT_LABELS:
            rejected["invalid_sentiment"] += 1
            continue
        duplicate = _candidate_is_duplicate(
            record,
            indexes,
            seen_ids=seen_ids,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
        )
        if duplicate:
            rejected[f"deduplicated_{duplicate}"] += 1
            continue
        ticker_hash = (record["ticker"], _normalise_hash(record.get("content_sha256")))
        if not ticker_hash[1] or ticker_hash in indexes.ticker_hashes or ticker_hash in seen_hashes:
            rejected["deduplicated_content_hash"] += 1
            continue
        _mark_seen_candidate(
            record,
            seen_ids=seen_ids,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
        )
        seen_hashes.add(ticker_hash)
        accepted.append(record)
    if accepted:
        append_frame = pd.DataFrame(accepted)
        derived = _derive_event_ids(append_frame)
        for position, event_id in enumerate(derived.tolist()):
            if not str(accepted[position].get("event_id") or "").strip():
                accepted[position]["event_id"] = event_id
    return accepted, dict(sorted(rejected.items()))


def _route_balanced_rows(
    rows: Sequence[Mapping[str, Any]],
    baseline: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    """Keep only rows that fill a missing class/split cell.

    Valid but surplus relevant rows and entity-negative examples are retained
    in immutable reserve files for a later cycle; they are not silently mixed
    into the balanced training corpus.
    """

    if str(manifest.get("selection_mode", "")).casefold() != "class_split_balanced":
        return [dict(row) for row in rows], {}, {}
    policy = load_policy(DEFAULT_POLICY_PATH)
    tickers = tuple(str(value).upper() for value in policy["required_tickers"])
    split_targets = manifest.get("split_targets_per_class") or {
        "training": DEFAULT_TRAINING_CLASS_TARGET,
        "validation": DEFAULT_VALIDATION_CLASS_TARGET,
        "test": DEFAULT_TEST_CLASS_TARGET,
    }
    counts = _label_split_counts(baseline, tickers)
    accepted: list[dict[str, Any]] = []
    reserved: dict[str, list[dict[str, Any]]] = {
        "labelled_excess": [],
        "entity_negative": [],
        # The current FinBERT corpus is English-only.  Preserve a valid
        # non-English company article for audit/future multilingual work,
        # rather than allowing it to reach dataset validation and abort an
        # otherwise valid append cycle.
        "non_english_relevant": [],
    }
    routed_counts: dict[str, int] = defaultdict(int)
    for raw in rows:
        row = dict(raw)
        ticker = _normalise_ticker(row.get("ticker"))
        split = str(row.get("split") or "").strip().casefold()
        relevant = _coerce_bool(row.get("entity_relevant"))
        label = str(row.get("sentiment_label") or "").strip().casefold()
        if not relevant:
            reserved["entity_negative"].append(row)
            routed_counts["entity_negative"] += 1
            continue
        language = str(row.get("language") or "").strip().casefold()
        if language != "en":
            reserved["non_english_relevant"].append(row)
            routed_counts["non_english_relevant"] += 1
            continue
        if (
            ticker in counts
            and split in BALANCE_SPLITS
            and label in BALANCE_LABELS
            and counts[ticker][split][label] < int(split_targets[split])
        ):
            accepted.append(row)
            counts[ticker][split][label] += 1
            routed_counts["canonical"] += 1
        else:
            reserved["labelled_excess"].append(row)
            routed_counts["labelled_excess"] += 1
    reserve_paths: dict[str, str] = {}
    reserve_root = _cycle_path(output, manifest, "reserves")
    for name, reserve_rows in reserved.items():
        path = reserve_root / f"{name}.jsonl"
        _write_jsonl(path, reserve_rows)
        reserve_paths[name] = str(path)
    return accepted, dict(sorted(routed_counts.items())), reserve_paths


def commit_bursa_news_append(
    *,
    work_dir: Path | str = DEFAULT_WORK_DIR,
    dataset_path: Path | str | None = None,
) -> dict[str, Any]:
    """Atomically append validated staging records after final deduplication."""

    output = Path(work_dir).resolve()
    manifest = _read_json(output / "append-manifest.json")
    canonical = Path(dataset_path or manifest["base_dataset"]["dataset_path"]).resolve()
    _assert_baseline(manifest, canonical)
    baseline = _read_dataset(canonical)
    staging = _cycle_path(output, manifest, "staging") / "codex-reviewed.jsonl"
    if not staging.exists():
        raise BursaNewsAppendError("No imported Codex labels found; run import-labels first")
    accepted, rejected = _filter_append_rows(_read_jsonl(staging), baseline)
    accepted, routed, reserve_paths = _route_balanced_rows(
        accepted,
        baseline,
        manifest=manifest,
        output=output,
    )
    if routed:
        rejected = {**rejected, **{f"routed_{key}": value for key, value in routed.items() if key != "canonical"}}
    if not accepted:
        raise BursaNewsAppendError(
            "No validated, non-duplicate rows are available to append; surplus and "
            "entity-negative rows were written to the balance reserves."
        )
    combined = pd.concat([baseline, pd.DataFrame(accepted)], ignore_index=True, sort=False)
    readiness = validate_bursa_news_dataset(
        combined,
        policy=load_policy(DEFAULT_POLICY_PATH),
        dataset_path=str(canonical),
    )
    if not readiness.ready:
        raise BursaNewsAppendError(
            "Append validation failed: " + "; ".join(readiness.errors)
        )
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    backup = output / "backups" / f"{canonical.stem}-{timestamp}{canonical.suffix}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(canonical, backup)
    existing_text = canonical.read_text(encoding="utf-8")
    appended_text = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in accepted)
    if existing_text and not existing_text.endswith("\n"):
        existing_text += "\n"
    temporary = canonical.with_suffix(canonical.suffix + ".append-tmp")
    temporary.write_text(existing_text + appended_text + "\n", encoding="utf-8")
    os.replace(temporary, canonical)
    report = {
        "schema_version": APPEND_SCHEMA_VERSION,
        "status": "committed",
        "created_at": _utc_now(),
        "dataset_path": str(canonical),
        "backup_path": str(backup),
        "rows_before": int(len(baseline)),
        "rows_appended": int(len(accepted)),
        "rows_after": int(len(combined)),
        "final_deduplication_rejections": rejected,
        "balance_routing": routed,
        "reserve_paths": reserve_paths,
        "relevant_counts_after": _relevant_counts(combined),
        "readiness": readiness.to_dict(),
        "canonical_dataset_changed": True,
        "runtime_model_changed": False,
    }
    _write_json(output / "append-commit-report.json", report)
    return report


def append_status(*, work_dir: Path | str = DEFAULT_WORK_DIR) -> dict[str, Any]:
    """Summarise the current resumable append workspace without mutating it."""

    output = Path(work_dir).resolve()
    manifest_path = output / "append-manifest.json"
    if not manifest_path.exists():
        return {"prepared": False, "work_dir": str(output)}
    manifest = _read_json(manifest_path)
    package_root = Path(manifest.get("label_package_dir") or _cycle_path(output, manifest, "codex-label-package"))
    staging_root = _cycle_path(output, manifest, "staging")
    return {
        "prepared": True,
        "work_dir": str(output),
        "manifest": manifest,
        "recovery_report_exists": (output / "recovery-report.json").exists(),
        "triage_report_exists": (_cycle_path(output, manifest, "triage") / "triage-report.json").exists(),
        "triage_queue_exists": (_cycle_path(output, manifest, "triage") / "label-queue.jsonl").exists(),
        "label_package_exists": (package_root / "external-label-input.jsonl").exists(),
        "staging_exists": (staging_root / "codex-reviewed.jsonl").exists(),
        "commit_report_exists": (output / "append-commit-report.json").exists(),
    }
