"""Free-only hybrid recovery and Web NGram usability pilot.

This module is deliberately offline-only.  It never changes the runtime news
providers, deployed FinBERT artifact, or fusion registry.  It recovers a
deterministic historical cohort through publisher pages, GDELT Web NGrams
(2020 onward), Common Crawl and Wayback.  It can measure whether reconstructed
pseudo-documents are usable as weak sentiment-labelling evidence, or simply
recover a resumable corpus for external labelling.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

try:  # Optional dependency used only by this offline acquisition workflow.
    import httpx
except ImportError:  # pragma: no cover - minimal production environments
    httpx = None  # type: ignore[assignment]

try:  # Preferred article-body extractor; the stdlib parser remains a fallback.
    import trafilatura
except ImportError:  # pragma: no cover - tested through the fallback extractor
    trafilatura = None  # type: ignore[assignment]

from pipeline.download_financial_news_data import load_sentiment_model
from pipeline.gdelt_historical_news import (
    CommonCrawlClient,
    CommonCrawlRateLimitError,
    extract_html_text,
    load_gdelt_export,
    select_pilot_candidates,
)
from pipeline.llm_bursa_news_labeler import run_llm_consensus_labelling
from pipeline.news_relevance import COMPANY_ENTITY_PROFILES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATE_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "pilot-2023-high-recall-v3b"
    / "exports"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "pilot-2023-ngram"
)
# GDELT's official Web NGrams documentation publishes this download endpoint
# over HTTP.  The host currently presents a TLS certificate that does not
# match data.gdeltproject.org, so using HTTPS would either fail safely or
# require disabling certificate verification.  Keep verification intact and
# use the documented transport instead.
WEBNGRAMS_BASE_URL = "http://data.gdeltproject.org/gdeltv3/webngrams"
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_REPLAY_URL = "https://web.archive.org/web/{timestamp}id_/{url}"
USER_AGENT = "StockSense academic research/1.0"
TOKEN_PATTERN = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", flags=re.UNICODE)
SENTENCE_END_PATTERN = re.compile(r"[.!?](?:[\"'”’)\]]+)?(?:\s|$)")
ENGLISH_COMMON_WORDS = frozenset(
    {
        "the",
        "and",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "as",
        "is",
        "was",
        "that",
        "by",
        "from",
        "at",
        "its",
        "has",
        "will",
    }
)
PROVENANCE_FIELDS = (
    "text_source",
    "is_pseudo_document",
    "recovery_status",
    "recovery_failure_reason",
    "recovered_at",
    "content_sha256",
    "ngram_file_timestamps",
    "ngram_record_count",
    "deciles_present",
    "unique_token_count",
    "overlap_attachment_rate",
    "reconstruction_quality_score",
    "reconstruction_quality_tier",
    "publisher_full_text_available",
)


class DownloadBudgetExceeded(RuntimeError):
    """Raised before an offline run exceeds its declared download ceiling."""


@dataclass(frozen=True)
class ReconstructionResult:
    text: str
    quality_score: float
    quality_tier: str
    deciles_present: tuple[int, ...]
    unique_token_count: int
    overlap_attachment_rate: float
    entity_present: bool
    sentence_boundary_coverage: float
    record_count: int
    file_timestamps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}
        ],
        doseq=True,
    )
    return urlunparse(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path.rstrip("/"),
            "",
            query,
            "",
        )
    )


def _row_key(row: Mapping[str, Any]) -> str:
    basis = (
        f"{str(row.get('ticker', '')).upper().strip()}|"
        f"{_normalise_url(row.get('url', ''))}"
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _candidate_title(row: Mapping[str, Any]) -> str:
    for key in ("title", "page_title", "gdelt_page_title"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _aliases_for_ticker(ticker: str) -> tuple[str, ...]:
    profile = COMPANY_ENTITY_PROFILES.get(str(ticker).upper(), {})
    values = [
        str(value).strip()
        for key in ("distinctive", "ambiguous")
        for value in profile.get(key, ())
        if str(value).strip()
    ]
    return tuple(dict.fromkeys(values))


def _normalise_phrase(value: str) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def _contains_company_alias(text: str, ticker: str) -> bool:
    haystack = _normalise_phrase(text)
    for alias in _aliases_for_ticker(ticker):
        needle = _normalise_phrase(alias)
        if not needle:
            continue
        if re.search(
            r"(?<!\w)" + r"\s+".join(re.escape(part) for part in needle.split()) + r"(?!\w)",
            haystack,
        ):
            return True
    return False


def _probably_english(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", str(text or "").casefold())
    if len(words) < 30:
        return False
    letters = [char for char in text if char.isalpha()]
    ascii_ratio = (
        sum(ord(char) < 128 for char in letters) / len(letters) if letters else 0.0
    )
    common = sum(word in ENGLISH_COMMON_WORDS for word in words)
    return ascii_ratio >= 0.90 and common >= max(4, len(words) // 100)


def _tokenise(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text or ""))


def _untokenise(tokens: Sequence[str]) -> str:
    output = ""
    no_space_before = {".", ",", ";", ":", "!", "?", "%", ")", "]", "}"}
    no_space_after = {"(", "[", "{", "£", "$", "RM"}
    for token in tokens:
        if not output:
            output = token
        elif token in no_space_before or output[-1:] in no_space_after:
            output += token
        else:
            output += " " + token
    return re.sub(r"\s+", " ", output).strip()


def _overlap(left: Sequence[str], right: Sequence[str], minimum: int = 5) -> int:
    maximum = min(len(left), len(right), 20)
    for size in range(maximum, minimum - 1, -1):
        if tuple(token.casefold() for token in left[-size:]) == tuple(
            token.casefold() for token in right[:size]
        ):
            return size
    return 0


def merge_overlapping_windows(
    windows: Sequence[Sequence[str]], *, minimum_overlap: int = 5
) -> tuple[list[list[str]], float]:
    """Merge rolling windows without an all-pairs cubic reconstruction pass."""

    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for window in windows:
        value = [str(token) for token in window if str(token).strip()]
        key = tuple(token.casefold() for token in value)
        if len(value) < minimum_overlap or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    if not unique:
        return [], 0.0
    chains = unique
    original_count = len(chains)
    while len(chains) > 1:
        prefix_index: dict[tuple[int, tuple[str, ...]], list[int]] = {}
        for index, chain in enumerate(chains):
            folded = [token.casefold() for token in chain]
            for size in range(minimum_overlap, min(20, len(chain)) + 1):
                prefix_index.setdefault((size, tuple(folded[:size])), []).append(index)
        merged = False
        for left_index, left in enumerate(chains):
            folded_left = [token.casefold() for token in left]
            for size in range(min(20, len(left)), minimum_overlap - 1, -1):
                candidates = prefix_index.get(
                    (size, tuple(folded_left[-size:])), ()
                )
                right_index = next(
                    (index for index in candidates if index != left_index), None
                )
                if right_index is None:
                    continue
                merged_chain = left + chains[right_index][size:]
                chains = [
                    chain
                    for index, chain in enumerate(chains)
                    if index not in {left_index, right_index}
                ] + [merged_chain]
                merged = True
                break
            if merged:
                break
        if not merged:
            break
    attachment = (
        1.0
        if original_count == 1
        else 1.0 - ((len(chains) - 1) / (original_count - 1))
    )
    return sorted(chains, key=lambda value: (-len(value), _untokenise(value))), float(
        max(0.0, min(1.0, attachment))
    )


def reconstruct_webngrams(
    records: Sequence[Mapping[str, Any]],
    *,
    ticker: str,
    file_timestamps: Sequence[str] = (),
) -> ReconstructionResult:
    grouped: dict[int, list[list[str]]] = {}
    for record in records:
        language = str(record.get("lang", "")).casefold()
        if language not in {"en", "eng", "english"}:
            continue
        try:
            decile = int(record.get("pos"))
        except (TypeError, ValueError):
            continue
        snippet = " ".join(
            str(record.get(key, "") or "").strip() for key in ("pre", "ngram", "post")
        ).strip()
        tokens = _tokenise(snippet)
        if tokens:
            grouped.setdefault(decile, []).append(tokens)

    paragraphs: list[str] = []
    attachment_values: list[float] = []
    for decile in sorted(grouped):
        chains, attachment = merge_overlapping_windows(grouped[decile])
        attachment_values.append(attachment)
        paragraphs.extend(_untokenise(chain) for chain in chains if chain)
    text = "\n\n".join(value for value in paragraphs if value)
    word_tokens = [
        token.casefold()
        for token in _tokenise(text)
        if re.search(r"\w", token, flags=re.UNICODE)
    ]
    unique_tokens = len(set(word_tokens))
    deciles = tuple(sorted(grouped))
    decile_coverage = min(len(deciles) / 10.0, 1.0)
    overlap_attachment = (
        float(np.mean(attachment_values)) if attachment_values else 0.0
    )
    unique_token_coverage = min(unique_tokens / 300.0, 1.0)
    entity_present = _contains_company_alias(text, ticker)
    sentence_boundaries = len(SENTENCE_END_PATTERN.findall(text))
    sentence_boundary_coverage = min(
        sentence_boundaries / max(1, len(deciles)), 1.0
    )
    score = (
        0.30 * decile_coverage
        + 0.25 * overlap_attachment
        + 0.20 * unique_token_coverage
        + 0.15 * float(entity_present)
        + 0.10 * sentence_boundary_coverage
    )
    if score >= 0.75 and entity_present and len(deciles) >= 6 and unique_tokens >= 180:
        tier = "high"
    elif (
        score >= 0.55
        and entity_present
        and len(deciles) >= 4
        and unique_tokens >= 100
    ):
        tier = "medium"
    else:
        tier = "low"
    return ReconstructionResult(
        text=text,
        quality_score=round(float(score), 6),
        quality_tier=tier,
        deciles_present=deciles,
        unique_token_count=unique_tokens,
        overlap_attachment_rate=round(overlap_attachment, 6),
        entity_present=entity_present,
        sentence_boundary_coverage=round(sentence_boundary_coverage, 6),
        record_count=sum(len(values) for values in grouped.values()),
        file_timestamps=tuple(sorted(set(str(value) for value in file_timestamps))),
    )


def _extract_article(html: str) -> tuple[str, str]:
    if trafilatura is not None:
        try:
            body = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                output_format="txt",
            )
            metadata = trafilatura.extract_metadata(html)
            title = str(getattr(metadata, "title", "") or "").strip()
            if body:
                return title, re.sub(r"\s+", " ", body).strip()
        except (AttributeError, TypeError, ValueError):
            pass
    return extract_html_text(html)


def _recovery_payload(
    item: Mapping[str, Any],
    *,
    title: str,
    body: str,
    text_source: str,
    status: str = "recovered",
    is_pseudo: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **dict(item),
        "title": title or _candidate_title(item),
        "description": body[:4000],
        "content_snippet": body[:12000],
        "text_source": text_source,
        "is_pseudo_document": bool(is_pseudo),
        "recovery_status": status,
        "recovery_failure_reason": "",
        "recovered_at": _utc_now(),
        "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "publisher_full_text_available": text_source == "publisher",
    }
    for field in PROVENANCE_FIELDS:
        payload.setdefault(field, "" if field != "is_pseudo_document" else False)
    if extra:
        payload.update(dict(extra))
    return payload


def _validate_recovered_body(
    item: Mapping[str, Any], title: str, body: str
) -> str:
    if len(re.findall(r"\b\w+\b", body)) < 150:
        return "recovered_text_too_short"
    if not _probably_english(body):
        return "recovered_text_not_english"
    if not _contains_company_alias(
        f"{title} {body}", str(item.get("ticker", ""))
    ):
        return "recovered_text_entity_missing"
    return ""


class PublisherClient:
    """Respectful, bounded publisher-page retriever."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        max_response_bytes: int = 5 * 1024 * 1024,
        request_interval_seconds: float = 1.0,
        transport: Any | None = None,
    ) -> None:
        if httpx is None:
            raise RuntimeError("Publisher recovery requires httpx")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        # Use explicit phase limits instead of httpx's scalar timeout.  A
        # publisher robots endpoint that accepts a TCP connection but never
        # produces a response must not hold the resumable recovery worker for
        # the full global safety window.
        request_timeout = httpx.Timeout(
            connect=min(8.0, self.timeout_seconds),
            read=min(12.0, self.timeout_seconds),
            write=min(12.0, self.timeout_seconds),
            pool=min(8.0, self.timeout_seconds),
        )
        self.client = httpx.Client(
            timeout=request_timeout,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": USER_AGENT},
        )
        self.max_response_bytes = int(max_response_bytes)
        self.request_interval_seconds = max(0.0, float(request_interval_seconds))
        self._last_request_by_domain: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        # A failed robots request must not be retried for every candidate on
        # the same origin.  Apart from wasting the bounded recovery budget,
        # repeated robots calls can make a checkpoint appear stuck when a
        # publisher's robots endpoint is slow or unreachable.  We fail closed
        # for that origin (respecting robots uncertainty) and reuse the
        # decision for the remainder of this recovery pass.
        self._robots_blocked: set[str] = set()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "PublisherClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _throttle(self, domain: str) -> None:
        previous = self._last_request_by_domain.get(domain)
        if previous is not None:
            remaining = self.request_interval_seconds - (time.monotonic() - previous)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_by_domain[domain] = time.monotonic()

    def _robots_allowed(self, url: str) -> tuple[bool, str]:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._robots_blocked:
            return False, "robots_request_failed"
        if origin not in self._robots:
            robots_url = f"{origin}/robots.txt"
            try:
                self._throttle(parsed.netloc.casefold())
                response = self.client.get(robots_url)
                if response.status_code == 404:
                    self._robots[origin] = None
                elif response.is_success:
                    parser = RobotFileParser()
                    parser.set_url(robots_url)
                    parser.parse(response.text.splitlines())
                    self._robots[origin] = parser
                else:
                    self._robots_blocked.add(origin)
                    return False, f"robots_http_{response.status_code}"
            except Exception as exc:
                if httpx is not None and isinstance(exc, httpx.HTTPError):
                    self._robots_blocked.add(origin)
                    return False, "robots_request_failed"
                raise
        parser = self._robots.get(origin)
        if parser is None:
            return True, ""
        return parser.can_fetch(USER_AGENT, url), "robots_disallowed"

    def recover(self, item: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
        url = str(item.get("url", "") or "").strip()
        ticker = str(item.get("ticker", "") or "").upper()
        allowed, reason = self._robots_allowed(url)
        if not allowed:
            return None, reason
        domain = (urlparse(url).hostname or "").casefold()
        try:
            self._throttle(domain)
            with self.client.stream("GET", url) as response:
                if not response.is_success:
                    return None, f"publisher_http_{response.status_code}"
                content_type = response.headers.get("content-type", "").casefold()
                if "html" not in content_type:
                    return None, "publisher_not_html"
                declared = int(response.headers.get("content-length", "0") or 0)
                if declared > self.max_response_bytes:
                    return None, "publisher_response_too_large"
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        return None, "publisher_response_too_large"
                    chunks.append(chunk)
                html = b"".join(chunks).decode(
                    response.encoding or "utf-8", errors="replace"
                )
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.HTTPError):
                return None, "publisher_request_failed"
            raise
        title, body = _extract_article(html)
        validation_error = _validate_recovered_body(item, title, body)
        if validation_error:
            return None, f"publisher_{validation_error.removeprefix('recovered_text_')}"
        return (
            _recovery_payload(
                item,
                title=title,
                body=body,
                text_source="publisher",
            ),
            "",
        )


class GDELTWebNGramClient:
    """Bounded direct downloader for GDELT Web NGram minute files."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        max_download_bytes: int,
        timeout_seconds: float = 30.0,
        request_interval_seconds: float = 0.05,
        transport: Any | None = None,
    ) -> None:
        if httpx is None:
            raise RuntimeError("GDELT NGram recovery requires httpx")
        self.client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": USER_AGENT},
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_download_bytes = int(max_download_bytes)
        self.downloaded_bytes = 0
        self.request_interval_seconds = max(0.0, float(request_interval_seconds))
        self._missing: set[str] = set()
        self._records: dict[str, list[dict[str, Any]]] = {}
        # Keep only the tiny set of rows matching a requested article URL.
        # Minute files can contain millions of records; caching the complete
        # decompressed file made the historical recovery process grow to
        # several gigabytes of RAM and get terminated before the checkpoint.
        self._url_matches: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.last_failure_reason = ""

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "GDELTWebNGramClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def _stamp(value: pd.Timestamp) -> str:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").floor("min").strftime("%Y%m%d%H%M00")

    def _cache_path(self, stamp: str) -> Path:
        return self.cache_dir / f"{stamp}.webngrams.json.gz"

    def _download(self, stamp: str) -> Path | None:
        destination = self._cache_path(stamp)
        if destination.exists():
            return destination
        if stamp in self._missing:
            return None
        url = f"{WEBNGRAMS_BASE_URL}/{stamp}.webngrams.json.gz"
        time.sleep(self.request_interval_seconds)
        with self.client.stream("GET", url) as response:
            if response.status_code == 404:
                self._missing.add(stamp)
                return None
            response.raise_for_status()
            declared = int(response.headers.get("content-length", "0") or 0)
            if (
                declared
                and self.downloaded_bytes + declared > self.max_download_bytes
            ):
                raise DownloadBudgetExceeded("GDELT NGram download ceiling reached")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            size = 0
            with temporary.open("wb") as handle:
                chunks = (
                    (response.content,)
                    if response.is_stream_consumed
                    else response.iter_raw()
                )
                for chunk in chunks:
                    size += len(chunk)
                    if self.downloaded_bytes + size > self.max_download_bytes:
                        handle.close()
                        temporary.unlink(missing_ok=True)
                        raise DownloadBudgetExceeded(
                            "GDELT NGram download ceiling reached"
                        )
                    handle.write(chunk)
            self.downloaded_bytes += size
            temporary.replace(destination)
        return destination

    def _read(self, stamp: str) -> list[dict[str, Any]]:
        if stamp in self._records:
            return self._records[stamp]
        path = self._download(stamp)
        if path is None:
            self._records[stamp] = []
            return []
        rows: list[dict[str, Any]] = []
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        rows.append(value)
        except (OSError, EOFError):
            path.unlink(missing_ok=True)
            rows = []
        self._records[stamp] = rows
        return rows

    def _read_url_matches(self, stamp: str, target_url: str) -> list[dict[str, Any]]:
        """Stream a minute file and retain only records for ``target_url``.

        Web NGram minute files are much larger than the article being
        reconstructed.  The recovery path needs exact URL matches only, so
        parsing line-by-line avoids materialising the entire decompressed
        file.  Results are cached per (minute, URL) because paired or nearby
        candidates may share a minute file.
        """

        cache_key = (stamp, target_url)
        if cache_key in self._url_matches:
            return self._url_matches[cache_key]
        path = self._download(stamp)
        if path is None:
            self._url_matches[cache_key] = []
            return []
        rows: list[dict[str, Any]] = []
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(value, dict):
                        continue
                    if _normalise_url(value.get("url")) == target_url:
                        rows.append(value)
        except (OSError, EOFError):
            path.unlink(missing_ok=True)
            rows = []
        self._url_matches[cache_key] = rows
        return rows

    @staticmethod
    def _offsets(limit_minutes: int, *, start: int = 0) -> Iterable[int]:
        if start <= 0:
            yield 0
            start = 1
        for value in range(max(1, start), limit_minutes + 1):
            yield -value
            yield value

    def recover(
        self,
        item: Mapping[str, Any],
        *,
        primary_window_minutes: int = 30,
        extended_window_minutes: int = 120,
    ) -> ReconstructionResult | None:
        self.last_failure_reason = ""
        published = pd.to_datetime(item.get("published_at"), utc=True, errors="coerce")
        if pd.isna(published):
            self.last_failure_reason = "ngram_timestamp_missing"
            return None
        target_url = _normalise_url(item.get("url"))
        matches: list[dict[str, Any]] = []
        matched_stamps: list[str] = []
        searched: set[str] = set()
        offset_groups = (
            self._offsets(primary_window_minutes),
            self._offsets(
                extended_window_minutes, start=primary_window_minutes + 1
            ),
        )
        for offsets in offset_groups:
            for offset in offsets:
                stamp = self._stamp(published + pd.Timedelta(minutes=offset))
                if stamp in searched:
                    continue
                searched.add(stamp)
                exact_url_rows = self._read_url_matches(stamp, target_url)
                selected = [
                    row
                    for row in exact_url_rows
                    if str(row.get("lang", "")).casefold()
                    in {"en", "eng", "english"}
                ]
                if selected:
                    matches.extend(selected)
                    matched_stamps.append(stamp)
                    break
                # A URL is assigned one source language in a minute file. Once
                # the exact article is found in a non-English language, wider
                # timestamp searches cannot make it eligible for this
                # English-only FinBERT pilot and would waste substantial
                # bandwidth.
                if exact_url_rows:
                    self.last_failure_reason = "ngram_non_english"
                    return None
            if matches:
                break
        if not matches:
            self.last_failure_reason = "ngram_url_not_found"
            return None
        return reconstruct_webngrams(
            matches,
            ticker=str(item.get("ticker", "")).upper(),
            file_timestamps=matched_stamps,
        )


class WaybackClient:
    """Small, rate-limited Internet Archive fallback."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        request_interval_seconds: float = 1.0,
        transport: Any | None = None,
    ) -> None:
        if httpx is None:
            raise RuntimeError("Wayback recovery requires httpx")
        self.client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": USER_AGENT},
        )
        self.request_interval_seconds = max(0.0, float(request_interval_seconds))

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "WaybackClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def recover(self, item: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
        published = pd.to_datetime(item.get("published_at"), utc=True, errors="coerce")
        if pd.isna(published):
            return None, "wayback_invalid_timestamp"
        url = str(item.get("url", ""))
        time.sleep(self.request_interval_seconds)
        try:
            response = self.client.get(
                WAYBACK_CDX_URL,
                params={
                    "url": url,
                    "output": "json",
                    "filter": ["statuscode:200", "mimetype:text/html"],
                    "fl": "timestamp,original,digest,statuscode,mimetype",
                    "from": published.strftime("%Y%m%d"),
                    "to": (published + pd.Timedelta(days=365)).strftime("%Y%m%d"),
                    "collapse": "digest",
                },
            )
            if not response.is_success:
                return None, f"wayback_cdx_http_{response.status_code}"
            values = response.json()
        except Exception as exc:
            if httpx is not None and isinstance(
                exc, (httpx.HTTPError, json.JSONDecodeError)
            ):
                return None, "wayback_cdx_failed"
            raise
        if not isinstance(values, list) or len(values) < 2:
            return None, "wayback_no_capture"
        candidates: list[tuple[pd.Timestamp, list[Any]]] = []
        for row in values[1:]:
            if not isinstance(row, list) or len(row) < 2:
                continue
            captured = pd.to_datetime(
                str(row[0]), format="%Y%m%d%H%M%S", utc=True, errors="coerce"
            )
            if pd.isna(captured) or captured < published:
                continue
            candidates.append((captured, row))
        if not candidates:
            return None, "wayback_no_post_publication_capture"
        captured, row = min(candidates, key=lambda value: value[0])
        replay_url = WAYBACK_REPLAY_URL.format(timestamp=row[0], url=row[1])
        time.sleep(self.request_interval_seconds)
        try:
            replay = self.client.get(replay_url)
            if not replay.is_success:
                return None, f"wayback_replay_http_{replay.status_code}"
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.HTTPError):
                return None, "wayback_replay_failed"
            raise
        title, body = _extract_article(replay.text)
        validation_error = _validate_recovered_body(item, title, body)
        if validation_error:
            return None, f"wayback_{validation_error.removeprefix('recovered_text_')}"
        payload = _recovery_payload(
            item, title=title, body=body, text_source="wayback"
        )
        payload["wayback_capture_at"] = captured.isoformat()
        return payload, ""


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("key"):
            values[str(value["key"])] = value
    return values


def _append_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, default=str) + "\n")


TRANSIENT_FAILURE_MARKERS = (
    "robots_request_failed",
    "publisher_request_failed",
    "common_crawl_rate_limited",
    "wayback_cdx_failed",
    "wayback_replay_failed",
    "request_timeout",
    "connection_error",
    "connection_reset",
    "temporarily_unavailable",
    "http_408",
    "http_425",
    "http_429",
    "http_500",
    "http_502",
    "http_503",
    "http_504",
)


def _checkpoint_attempt_count(record: Mapping[str, Any] | None) -> int:
    if not record:
        return 0
    try:
        return max(1, int(record.get("attempt_count", 1)))
    except (TypeError, ValueError):
        return 1


def _checkpoint_is_retryable(
    record: Mapping[str, Any] | None,
    *,
    max_attempts: int,
) -> bool:
    """Return True only for bounded, clearly transient recovery failures.

    Older checkpoints predate ``attempt_count``.  They represent one attempt,
    which lets a resume repair a temporary outage without repeating completed
    or deterministic requests.
    """

    if not record or record.get("status") == "recovered":
        return False
    if _checkpoint_attempt_count(record) >= max(1, int(max_attempts)):
        return False
    reason = str(record.get("reason", "")).casefold()
    if any(marker in reason for marker in TRANSIENT_FAILURE_MARKERS):
        return True
    return bool(
        re.search(
            r"(?:publisher|robots|wayback|ngram|common_crawl)_http_5\d\d",
            reason,
        )
    )


def _checkpoint_state(
    *,
    key: str,
    payload: Mapping[str, Any] | None,
    reason: str,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "status": "recovered" if payload else "unavailable",
        "reason": reason,
        "payload": dict(payload) if payload is not None else None,
        "attempt_count": _checkpoint_attempt_count(previous) + 1,
        "updated_at": _utc_now(),
    }


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    best_effort: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
    ) as handle:
        handle.write(
            json.dumps(dict(value), indent=2, ensure_ascii=False, default=str)
        )
        temporary = Path(handle.name)
    last_error: PermissionError | None = None
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(0.05 * (2**attempt))
    temporary.unlink(missing_ok=True)
    if best_effort:
        return
    assert last_error is not None
    raise last_error


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _recovery_precedence(row: Mapping[str, Any]) -> int:
    """Rank records so a less useful retry cannot replace recovered text.

    A resumed bounded batch may select a smaller cohort than an earlier batch.
    The recovered corpus is cumulative, so an unavailable result from the new
    cohort must never discard an existing publisher or archive recovery for the
    same ticker/URL.  Higher-quality sources also remain preferred when two
    recovery attempts disagree.
    """

    text = str(row.get("content_snippet", "") or "").strip()
    if not text:
        return 0
    source = str(row.get("text_source", "") or "").strip().casefold()
    if source == "publisher":
        return 4
    if source in {"common_crawl", "wayback"}:
        return 3
    if source == "gdelt_webngrams":
        tier = str(row.get("reconstruction_quality_tier", "") or "").casefold()
        return 2 if tier == "high" else 1
    return 1


def _merge_recovered_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    latest_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return a cumulative ticker/URL corpus and count rows retained from disk.

    ``hybrid-recovered.jsonl`` is both a recovery checkpoint output and the
    source for the external labelling package.  A later time-bounded run must
    update only its selected rows, not silently remove other recovered articles
    that were selected by an earlier run.  On a key collision, use the best
    available text representation; equal-quality latest results replace the
    existing row so refreshed provenance remains visible.
    """

    merged: dict[str, dict[str, Any]] = {}
    preserved_from_existing: set[str] = set()

    for row in existing_rows:
        ticker = str(row.get("ticker", "") or "").strip()
        url = str(row.get("url", "") or "").strip()
        if not ticker or not url:
            continue
        key = _row_key(row)
        merged[key] = dict(row)
        preserved_from_existing.add(key)

    for row in latest_rows:
        ticker = str(row.get("ticker", "") or "").strip()
        url = str(row.get("url", "") or "").strip()
        if not ticker or not url:
            continue
        key = _row_key(row)
        existing = merged.get(key)
        if existing is None or _recovery_precedence(row) >= _recovery_precedence(existing):
            merged[key] = dict(row)
            preserved_from_existing.discard(key)

    ordered = sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("ticker", "")).upper(),
            str(row.get("published_at", "")),
            _normalise_url(row.get("url", "")),
        ),
    )
    return ordered, len(preserved_from_existing)


def _checkpoint_recovered_payloads(
    direct_state: Mapping[str, Mapping[str, Any]],
    ngram_state: Mapping[str, Mapping[str, Any]],
    archive_state: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rehydrate all recoverable rows stored in append-only stage checkpoints.

    The stage checkpoint files retain successful payloads across bounded runs,
    including candidates that may not occur in the latest selected cohort.  Use
    them as a second cumulative source when writing ``hybrid-recovered.jsonl``
    so changing a sample cap cannot make previously recovered text disappear.
    """

    rows: list[dict[str, Any]] = []
    for key in sorted(set(direct_state) | set(ngram_state) | set(archive_state)):
        direct_payload = direct_state.get(key, {}).get("payload")
        ngram_payload = ngram_state.get(key, {}).get("payload")
        archive_payload = archive_state.get(key, {}).get("payload")
        if isinstance(direct_payload, dict):
            rows.append(dict(direct_payload))
        elif (
            isinstance(ngram_payload, dict)
            and ngram_payload.get("reconstruction_quality_tier") == "high"
        ):
            rows.append(dict(ngram_payload))
        elif isinstance(archive_payload, dict):
            rows.append(dict(archive_payload))
        elif isinstance(ngram_payload, dict):
            rows.append(dict(ngram_payload))
    return rows


def _read_jsonl_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.DataFrame()


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def refresh_cumulative_recovered_output(
    output_dir: Path | str,
    *,
    latest_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild a cumulative recovery file solely from local checkpoints.

    This performs no network activity and does not touch paired evaluation or
    label files.  It is safe to run after an interrupted batch or after a
    sample-cap change, and protects the downstream labelling package from
    losing earlier successful recoveries.
    """

    raw_dir = Path(output_dir) / "raw"
    recovered_path = raw_dir / "hybrid-recovered.jsonl"
    existing_rows = _read_jsonl_rows(recovered_path)
    direct_state = _load_checkpoint(raw_dir / "publisher-state.jsonl")
    ngram_state = _load_checkpoint(raw_dir / "ngram-state.jsonl")
    archive_state = _load_checkpoint(raw_dir / "archive-state.jsonl")
    checkpoint_rows = _checkpoint_recovered_payloads(
        direct_state,
        ngram_state,
        archive_state,
    )
    cumulative_rows, preserved_existing_rows = _merge_recovered_rows(
        existing_rows,
        [*checkpoint_rows, *latest_rows],
    )
    _write_jsonl(recovered_path, cumulative_rows)
    return {
        "recovered_path": str(recovered_path),
        "existing_recovered_rows": len(existing_rows),
        "checkpoint_recovered_rows": len(checkpoint_rows),
        "latest_recovered_rows": len(latest_rows),
        "cumulative_recovered_rows": len(cumulative_rows),
        "preserved_existing_recovered_rows": preserved_existing_rows,
    }


def _with_ngram_payload(
    item: Mapping[str, Any], reconstruction: ReconstructionResult
) -> dict[str, Any]:
    payload = _recovery_payload(
        item,
        title=_candidate_title(item),
        body=reconstruction.text,
        text_source="gdelt_webngrams",
        status=(
            "recovered"
            if reconstruction.quality_tier == "high"
            else "recovered_not_label_eligible"
        ),
        is_pseudo=True,
        extra={
            "ngram_file_timestamps": list(reconstruction.file_timestamps),
            "ngram_record_count": reconstruction.record_count,
            "deciles_present": list(reconstruction.deciles_present),
            "unique_token_count": reconstruction.unique_token_count,
            "overlap_attachment_rate": reconstruction.overlap_attachment_rate,
            "reconstruction_quality_score": reconstruction.quality_score,
            "reconstruction_quality_tier": reconstruction.quality_tier,
        },
    )
    payload["publisher_full_text_available"] = False
    return payload


def _label_input(
    row: Mapping[str, Any],
    *,
    pair_id: str,
    representation: str,
) -> dict[str, Any]:
    body = str(row.get("content_snippet", "") or row.get("description", "")).strip()
    title = str(row.get("title", "") or _candidate_title(row)).strip()
    model_text = ". ".join(value for value in (title, body) if value)
    article_id = hashlib.sha256(
        f"{pair_id}|{representation}".encode("utf-8")
    ).hexdigest()
    return {
        **dict(row),
        "article_id": article_id,
        "pair_id": pair_id,
        "representation": representation,
        "model_text": model_text,
        "split": "training",
        "automatic_entity_relevant": True,
    }


def _score_finbert_pairs(
    full: pd.DataFrame, pseudo: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if full.empty or pseudo.empty:
        return pd.DataFrame(), {"available": False, "reason": "no paired rows"}
    joined = full[["pair_id", "model_text"]].merge(
        pseudo[["pair_id", "model_text"]],
        on="pair_id",
        suffixes=("_full", "_pseudo"),
    )
    try:
        model = load_sentiment_model()
        full_probability = model.predict_proba(joined["model_text_full"].tolist())
        pseudo_probability = model.predict_proba(
            joined["model_text_pseudo"].tolist()
        )
    except Exception as exc:
        return pd.DataFrame(), {"available": False, "reason": str(exc)}
    classes = np.asarray([str(value).casefold() for value in model.classes_])
    full_index = full_probability.argmax(axis=1)
    pseudo_index = pseudo_probability.argmax(axis=1)
    output = joined[["pair_id"]].copy()
    output["full_label"] = classes[full_index]
    output["pseudo_label"] = classes[pseudo_index]
    output["class_agreement"] = output["full_label"].eq(output["pseudo_label"])
    output["mean_absolute_probability_difference"] = np.abs(
        full_probability - pseudo_probability
    ).mean(axis=1)
    metrics = {
        "available": True,
        "paired_rows": int(len(output)),
        "class_agreement": float(output["class_agreement"].mean()),
        "mean_absolute_probability_difference": float(
            output["mean_absolute_probability_difference"].mean()
        ),
    }
    return output, metrics


def _llm_pair_metrics(
    full_labels: pd.DataFrame, pseudo_labels: pd.DataFrame
) -> dict[str, Any]:
    if full_labels.empty or pseudo_labels.empty:
        return {"available": False, "reason": "paired labels are incomplete"}
    columns = ["pair_id", "consensus_status", "entity_relevant", "sentiment_label"]
    joined = full_labels[columns].merge(
        pseudo_labels[columns],
        on="pair_id",
        suffixes=("_full", "_pseudo"),
    )
    accepted = joined.loc[
        joined["consensus_status_full"].astype(str).eq("accepted")
        & joined["consensus_status_pseudo"].astype(str).eq("accepted")
    ].copy()
    if accepted.empty:
        return {"available": False, "reason": "no accepted paired consensus labels"}
    for column in ("entity_relevant_full", "entity_relevant_pseudo"):
        accepted[column] = accepted[column].map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).casefold() in {"true", "1", "yes"}
        )
    entity_agreement = float(
        accepted["entity_relevant_full"]
        .eq(accepted["entity_relevant_pseudo"])
        .mean()
    )
    relevant = accepted.loc[
        accepted["entity_relevant_full"] & accepted["entity_relevant_pseudo"]
    ].copy()
    if relevant.empty:
        return {
            "available": False,
            "reason": "no paired entity-relevant consensus labels",
            "entity_agreement": entity_agreement,
        }
    expected = relevant["sentiment_label_full"].astype(str)
    predicted = relevant["sentiment_label_pseudo"].astype(str)
    labels = ["negative", "neutral", "positive"]
    report = classification_report(
        expected, predicted, labels=labels, output_dict=True, zero_division=0
    )
    class_f1 = {label: float(report[label]["f1-score"]) for label in labels}
    return {
        "available": True,
        "accepted_pair_rows": int(len(accepted)),
        "relevant_pair_rows": int(len(relevant)),
        "entity_agreement": entity_agreement,
        "sentiment_agreement": float(accuracy_score(expected, predicted)),
        "sentiment_macro_f1": float(
            f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "sentiment_kappa": float(cohen_kappa_score(expected, predicted)),
        "class_f1": class_f1,
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=labels
        ).tolist(),
        "labels": labels,
        "full_text_class_distribution": dict(Counter(expected)),
        "pseudo_document_class_distribution": dict(Counter(predicted)),
    }


def build_ngram_usability_report(
    *,
    year: int,
    selection: pd.DataFrame,
    direct_records: Mapping[str, Mapping[str, Any]],
    ngram_records: Mapping[str, Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    paired_full: pd.DataFrame,
    paired_pseudo: pd.DataFrame,
    full_labels: pd.DataFrame,
    pseudo_labels: pd.DataFrame,
    finbert_metrics: Mapping[str, Any],
    downloaded_bytes: int,
    elapsed_seconds: float,
    completed: bool,
    stop_reason: str | None,
) -> dict[str, Any]:
    attempted_keys = set(ngram_records)
    matched = [
        value
        for value in ngram_records.values()
        if value.get("status") == "recovered"
    ]
    matched_frame = pd.DataFrame(
        [value.get("payload", {}) for value in matched]
    )
    per_ticker: dict[str, Any] = {}
    for ticker in sorted(selection["ticker"].astype(str).unique()):
        keys = {
            _row_key(row)
            for row in selection.loc[selection["ticker"].eq(ticker)].to_dict(
                orient="records"
            )
        }
        attempted = keys & attempted_keys
        ticker_matches = [
            value
            for key, value in ngram_records.items()
            if key in attempted and value.get("status") == "recovered"
        ]
        high = sum(
            str(value.get("payload", {}).get("reconstruction_quality_tier", ""))
            == "high"
            for value in ticker_matches
        )
        paired_high = (
            int(paired_pseudo.loc[paired_pseudo["ticker"].eq(ticker)].shape[0])
            if not paired_pseudo.empty
            else 0
        )
        per_ticker[ticker] = {
            "selected": int(len(keys)),
            "ngram_attempted": int(len(attempted)),
            "ngram_matched": int(len(ticker_matches)),
            "ngram_recovery_rate": (
                len(ticker_matches) / len(attempted) if attempted else 0.0
            ),
            "high_quality": int(high),
            "paired_high_quality": paired_high,
        }
    matched_count = len(matched)
    high_count = (
        int(
            matched_frame.get(
                "reconstruction_quality_tier", pd.Series(dtype=str)
            )
            .astype(str)
            .eq("high")
            .sum()
        )
        if not matched_frame.empty
        else 0
    )
    llm_metrics = _llm_pair_metrics(full_labels, pseudo_labels)
    timestamps = pd.to_datetime(
        [row.get("published_at") for row in final_rows], utc=True, errors="coerce"
    )
    timestamp_completeness = (
        float(pd.Series(timestamps).notna().mean()) if final_rows else 0.0
    )
    gates = {
        "all_tickers_have_five_paired_high_quality": all(
            values["paired_high_quality"] >= 5 for values in per_ticker.values()
        )
        and len(per_ticker) == 10,
        "ngram_recovery_overall": (
            matched_count / len(attempted_keys) >= 0.60 if attempted_keys else False
        ),
        "ngram_recovery_each_ticker": all(
            values["ngram_recovery_rate"] >= 0.40 for values in per_ticker.values()
        )
        and bool(per_ticker),
        "high_quality_share": (
            high_count / matched_count >= 0.60 if matched_count else False
        ),
        "entity_agreement": bool(llm_metrics.get("available"))
        and float(llm_metrics.get("entity_agreement", 0.0)) >= 0.90,
        "sentiment_agreement": bool(llm_metrics.get("available"))
        and float(llm_metrics.get("sentiment_agreement", 0.0)) >= 0.80,
        "sentiment_macro_f1": bool(llm_metrics.get("available"))
        and float(llm_metrics.get("sentiment_macro_f1", 0.0)) >= 0.75,
        "sentiment_kappa": bool(llm_metrics.get("available"))
        and float(llm_metrics.get("sentiment_kappa", -1.0)) >= 0.70,
        "class_f1": bool(llm_metrics.get("available"))
        and all(
            float(value) >= 0.60
            for value in llm_metrics.get("class_f1", {}).values()
        )
        and len(llm_metrics.get("class_f1", {})) == 3,
        "finbert_class_agreement": bool(finbert_metrics.get("available"))
        and float(finbert_metrics.get("class_agreement", 0.0)) >= 0.80,
        "finbert_probability_difference": bool(finbert_metrics.get("available"))
        and float(
            finbert_metrics.get("mean_absolute_probability_difference", math.inf)
        )
        <= 0.12,
        "timestamp_completeness": timestamp_completeness >= 0.95,
    }
    if all(gates.values()):
        decision = "usable_for_weak_sentiment_labels"
    elif (
        gates["entity_agreement"]
        and gates["ngram_recovery_overall"]
        and gates["ngram_recovery_each_ticker"]
    ):
        decision = "entity_filtering_only"
    else:
        decision = "archive_recovery_only"
    method_counts = Counter(
        str(row.get("text_source", "unrecovered") or "unrecovered")
        for row in final_rows
    )
    final_frame = pd.DataFrame(final_rows)
    recovery_by_month: dict[str, Any] = {}
    recovery_by_domain: dict[str, Any] = {}
    recovery_breakdown: list[dict[str, Any]] = []
    if not final_frame.empty:
        final_frame["_published"] = pd.to_datetime(
            final_frame.get("published_at"), utc=True, errors="coerce"
        )
        final_frame["_month"] = final_frame["_published"].dt.strftime("%Y-%m")
        if "source_domain" not in final_frame:
            final_frame["source_domain"] = final_frame.get(
                "url", pd.Series("", index=final_frame.index)
            ).map(lambda value: (urlparse(str(value)).hostname or "").casefold())
        final_frame["_method"] = (
            final_frame.get("text_source", pd.Series("", index=final_frame.index))
            .astype(str)
            .replace("", "unrecovered")
        )
        for month, group in final_frame.groupby("_month", dropna=False):
            recovery_by_month[str(month)] = {
                "rows": int(len(group)),
                "recovered": int(group["_method"].ne("unrecovered").sum()),
                "methods": group["_method"].value_counts().to_dict(),
            }
        for domain, group in final_frame.groupby("source_domain", dropna=False):
            recovery_by_domain[str(domain)] = {
                "rows": int(len(group)),
                "recovered": int(group["_method"].ne("unrecovered").sum()),
                "methods": group["_method"].value_counts().to_dict(),
            }
        recovery_breakdown = [
            {
                "ticker": str(ticker),
                "month": str(month),
                "source_domain": str(domain),
                "text_source": str(method),
                "rows": int(len(group)),
            }
            for (ticker, month, domain, method), group in final_frame.groupby(
                ["ticker", "_month", "source_domain", "_method"], dropna=False
            )
        ]
    reconstruction_examples: dict[str, list[dict[str, Any]]] = {
        "high": [],
        "medium": [],
        "low": [],
        "failed": [],
    }
    if not matched_frame.empty:
        for tier in ("high", "medium", "low"):
            examples = matched_frame.loc[
                matched_frame.get(
                    "reconstruction_quality_tier", pd.Series("", index=matched_frame.index)
                )
                .astype(str)
                .eq(tier)
            ].head(3)
            reconstruction_examples[tier] = [
                {
                    "ticker": str(row.get("ticker", "")),
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                    "quality_score": float(
                        row.get("reconstruction_quality_score", 0.0) or 0.0
                    ),
                    "deciles_present": row.get("deciles_present", []),
                    "unique_token_count": int(
                        row.get("unique_token_count", 0) or 0
                    ),
                }
                for row in examples.to_dict(orient="records")
            ]
    failed_values = [
        value
        for value in ngram_records.values()
        if value.get("status") != "recovered"
    ][:5]
    reconstruction_examples["failed"] = [
        {
            "reason": str(value.get("reason", "")),
            "key": str(value.get("key", "")),
        }
        for value in failed_values
    ]
    direct_recovered = sum(
        value.get("status") == "recovered" for value in direct_records.values()
    )
    return {
        "schema_version": "gdelt-ngram-usability-pilot-v1",
        "created_at": _utc_now(),
        "pilot_year": int(year),
        "completed": bool(completed),
        "stop_reason": stop_reason,
        "selected_rows": int(len(selection)),
        "selected_tickers": sorted(selection["ticker"].astype(str).unique()),
        "recovery_method_counts": dict(method_counts),
        "direct_publisher_attempted": int(len(direct_records)),
        "direct_publisher_recovered": int(direct_recovered),
        "direct_publisher_recovery_rate": (
            direct_recovered / len(direct_records) if direct_records else 0.0
        ),
        "recovery_by_month": recovery_by_month,
        "recovery_by_domain": recovery_by_domain,
        "recovery_breakdown": recovery_breakdown,
        "ngram_attempted": int(len(attempted_keys)),
        "ngram_matched": int(matched_count),
        "ngram_recovery_rate": (
            matched_count / len(attempted_keys) if attempted_keys else 0.0
        ),
        "high_quality_reconstructions": int(high_count),
        "high_quality_share": (
            high_count / matched_count if matched_count else 0.0
        ),
        "per_ticker": per_ticker,
        "reconstruction_quality_distribution": (
            matched_frame.get(
                "reconstruction_quality_tier", pd.Series(dtype=str)
            )
            .astype(str)
            .value_counts()
            .to_dict()
            if not matched_frame.empty
            else {}
        ),
        "reconstruction_examples": reconstruction_examples,
        "llm_pair_metrics": llm_metrics,
        "finbert_pair_metrics": dict(finbert_metrics),
        "timestamp_completeness": timestamp_completeness,
        "downloaded_bytes_this_run": int(downloaded_bytes),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "estimated_2020_2025_download_gib": round(
            (downloaded_bytes * 6) / (1024**3), 3
        ),
        "gates": gates,
        "decision": decision,
        "promotion_performed": False,
        "decision_notes": [
            "Only labels accepted under the LLM-assisted-label policy are eligible for the active corpus.",
            "GDELT tone fields are not used.",
            "No FinBERT or fusion registry is modified by this pilot.",
        ],
    }


def run_gdelt_ngram_pilot(
    *,
    year: int = 2023,
    candidate_dir: Path | str = DEFAULT_CANDIDATE_DIR,
    sample_per_ticker: int = 30,
    paired_per_ticker: int = 10,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_runtime_minutes: float = 45.0,
    max_download_gib: float = 5.0,
    max_articles: int | None = None,
    request_interval_seconds: float = 1.0,
    run_labels: bool = True,
    max_label_requests: int | None = None,
    skip_archives: bool = False,
    skip_ngrams: bool = False,
    candidate_only: bool = False,
    max_stage_attempts: int = 3,
    network_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Export candidates or resume free-only recovery for one historical year."""

    year = int(year)
    current_year = datetime.now(timezone.utc).year
    if year < 2015 or year > current_year:
        raise ValueError(
            f"Offline corpus recovery supports 2015-{current_year} only"
        )
    if sample_per_ticker <= 0 or paired_per_ticker < 0:
        raise ValueError(
            "sample-per-ticker must be positive and paired-per-ticker non-negative"
        )
    if paired_per_ticker > sample_per_ticker:
        raise ValueError("paired-per-ticker cannot exceed sample-per-ticker")
    if max_stage_attempts <= 0:
        raise ValueError("max-stage-attempts must be positive")
    root = Path(output_dir)
    raw_dir = root / "raw"
    paired_dir = root / "paired"
    report_path = root / f"pilot-{year}-ngram-usability-report.json"
    heartbeat_path = raw_dir / "hybrid-recovery-heartbeat.json"
    root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paired_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    runtime_seconds = max(0.0, float(max_runtime_minutes)) * 60.0
    deadline = started + runtime_seconds
    # A single global deadline used to let large candidate corpora spend every
    # bounded pass in the publisher stage, indefinitely postponing the two
    # fallback recoveries. Divide the same fixed budget into bounded stage
    # windows so a pass can progress publisher -> Web NGram -> archive for the
    # rows it has already attempted.
    publisher_deadline = started + runtime_seconds * 0.55
    ngram_deadline = started + runtime_seconds * 0.80

    indexed = load_gdelt_export(Path(candidate_dir))
    indexed = indexed.loc[indexed["published_at"].dt.year.eq(int(year))].copy()
    if candidate_only:
        candidates = indexed.sort_values(
            ["ticker", "published_at", "url"], kind="stable"
        ).reset_index(drop=True)
        candidates["title"] = candidates.apply(_candidate_title, axis=1)
        candidate_path = raw_dir / "candidate-corpus.jsonl"
        _write_jsonl(candidate_path, candidates.to_dict(orient="records"))

        monthly = (
            candidates.assign(
                month=candidates["published_at"].dt.strftime("%Y-%m")
            )
            .groupby(["ticker", "month"], dropna=False)
            .size()
            .rename("candidates")
            .reset_index()
        )
        summary = {
            "completed": True,
            "mode": "candidate_only",
            "year": int(year),
            "candidate_count": int(len(candidates)),
            "ticker_count": int(candidates["ticker"].nunique()),
            "candidates_by_ticker": {
                str(key): int(value)
                for key, value in candidates["ticker"].value_counts().sort_index().items()
            },
            "candidates_by_ticker_month": monthly.to_dict(orient="records"),
            "candidate_path": str(candidate_path),
            "source_dir": str(Path(candidate_dir)),
            "acceptance_gates_evaluated": False,
            "recovery_attempted": False,
            "sentiment_labelling_attempted": False,
            "eligible_for_sentiment_training": False,
            "production_changes": False,
            "created_at": _utc_now(),
            "notes": [
                "These rows are discovery candidates, not recovered article text.",
                "Fallback-reliability acceptance gates are intentionally not applied.",
                "Entity validation, text recovery and sentiment labelling remain downstream tasks.",
            ],
        }
        _atomic_json(root / "candidate-only-summary.json", summary)
        return summary

    selection = select_pilot_candidates(indexed, per_ticker=sample_per_ticker)
    if max_articles is not None:
        selection = selection.head(max(0, int(max_articles))).copy()
    selection["title"] = selection.apply(_candidate_title, axis=1)
    selection_path = raw_dir / "selected-candidates.jsonl"
    _write_jsonl(selection_path, selection.to_dict(orient="records"))

    def expired(stage_deadline: float) -> bool:
        return time.monotonic() >= min(deadline, stage_deadline)

    stop_reason: str | None = None
    stage_pause_reasons: list[str] = []

    def record_stage_pause(reason: str) -> None:
        if reason not in stage_pause_reasons:
            stage_pause_reasons.append(reason)
    direct_path = raw_dir / "publisher-state.jsonl"
    direct_state = _load_checkpoint(direct_path)
    with PublisherClient(
        request_interval_seconds=request_interval_seconds,
        timeout_seconds=network_timeout_seconds,
    ) as publisher:
        for item in selection.to_dict(orient="records"):
            key = _row_key(item)
            previous = direct_state.get(key)
            if previous is not None and not _checkpoint_is_retryable(
                previous,
                max_attempts=max_stage_attempts,
            ):
                continue
            if expired(publisher_deadline):
                record_stage_pause(
                    "publisher stage time slice reached; resume continues from its checkpoint"
                )
                break
            try:
                payload, reason = publisher.recover(item)
            except Exception as exc:
                payload, reason = None, f"publisher_error:{exc}"
            state = _checkpoint_state(
                key=key,
                payload=payload,
                reason=reason,
                previous=previous,
            )
            _append_checkpoint(direct_path, state)
            direct_state[key] = state
            _atomic_json(
                heartbeat_path,
                {
                    "status": "running",
                    "stage": "publisher",
                    "completed": len(direct_state),
                    "selected": len(selection),
                    "updated_at": _utc_now(),
                },
                best_effort=True,
            )

    direct_success_rows = [
        value["payload"]
        for value in direct_state.values()
        if value.get("status") == "recovered" and isinstance(value.get("payload"), dict)
    ]
    direct_success = pd.DataFrame(direct_success_rows)
    paired_keys: set[str] = set()
    if not direct_success.empty:
        direct_success["_key"] = direct_success.apply(_row_key, axis=1)
        for _, group in direct_success.sort_values(
            ["ticker", "published_at", "url"]
        ).groupby("ticker", sort=True):
            paired_keys.update(group.head(paired_per_ticker)["_key"].tolist())

    ngram_path = raw_dir / "ngram-state.jsonl"
    ngram_state = _load_checkpoint(ngram_path)
    # GDELT Web NGrams 3.0 begins in 2020. Older rows move directly from
    # publisher recovery to Common Crawl/Wayback rather than issuing known
    # impossible requests.
    ngram_targets = (
        [
            item
            for item in selection.to_dict(orient="records")
            if (
                _row_key(item) in direct_state
                and (
                    direct_state.get(_row_key(item), {}).get("status")
                    != "recovered"
                    or _row_key(item) in paired_keys
                )
            )
        ]
        if year >= 2020 and not skip_ngrams
        else []
    )
    downloaded_bytes = 0
    if not expired(ngram_deadline):
        try:
            with GDELTWebNGramClient(
                raw_dir / "webngrams-cache",
                max_download_bytes=int(max_download_gib * 1024**3),
            ) as ngram_client:
                for item in ngram_targets:
                    key = _row_key(item)
                    previous = ngram_state.get(key)
                    if previous is not None and not _checkpoint_is_retryable(
                        previous,
                        max_attempts=max_stage_attempts,
                    ):
                        continue
                    if expired(ngram_deadline):
                        record_stage_pause(
                            "Web NGram stage time slice reached; resume continues from its checkpoint"
                        )
                        break
                    try:
                        result = ngram_client.recover(item)
                        payload = _with_ngram_payload(item, result) if result else None
                        reason = (
                            ""
                            if result
                            else ngram_client.last_failure_reason
                            or "ngram_url_not_found"
                        )
                    except DownloadBudgetExceeded as exc:
                        record_stage_pause(str(exc))
                        break
                    except Exception as exc:
                        payload, reason = None, f"ngram_error:{exc}"
                    state = _checkpoint_state(
                        key=key,
                        payload=payload,
                        reason=reason,
                        previous=previous,
                    )
                    _append_checkpoint(ngram_path, state)
                    ngram_state[key] = state
                    _atomic_json(
                        heartbeat_path,
                        {
                            "status": "running",
                            "stage": "webngrams",
                            "completed": len(ngram_state),
                            "targets": len(ngram_targets),
                            "downloaded_bytes": ngram_client.downloaded_bytes,
                            "updated_at": _utc_now(),
                        },
                        best_effort=True,
                    )
                downloaded_bytes = ngram_client.downloaded_bytes
        except DownloadBudgetExceeded as exc:
            record_stage_pause(str(exc))
    elif ngram_targets:
        record_stage_pause(
            "Web NGram stage time slice reached before fallback recovery began"
        )

    archive_path = raw_dir / "archive-state.jsonl"
    archive_state = _load_checkpoint(archive_path)
    if not skip_archives and not expired(deadline):
        unresolved = []
        for item in selection.to_dict(orient="records"):
            key = _row_key(item)
            # Do not bypass publisher/NGram for candidates that have not yet
            # reached those earlier stages in a safety-bounded pass.
            if key not in direct_state:
                continue
            if direct_state.get(key, {}).get("status") == "recovered":
                continue
            if year >= 2020 and not skip_ngrams and key not in ngram_state:
                continue
            ngram_payload = ngram_state.get(key, {}).get("payload")
            if (
                isinstance(ngram_payload, dict)
                and ngram_payload.get("reconstruction_quality_tier") == "high"
            ):
                continue
            unresolved.append(item)
        try:
            common_crawl = CommonCrawlClient(
                request_interval_seconds=request_interval_seconds,
                timeout_seconds=network_timeout_seconds,
            )
            collections = common_crawl.collections()
        except Exception:
            common_crawl = None
            collections = []
        try:
            with WaybackClient(
                request_interval_seconds=request_interval_seconds,
                timeout_seconds=network_timeout_seconds,
            ) as wayback:
                for item in unresolved:
                    key = _row_key(item)
                    previous = archive_state.get(key)
                    if previous is not None and not _checkpoint_is_retryable(
                        previous,
                        max_attempts=max_stage_attempts,
                    ):
                        continue
                    if expired(deadline):
                        record_stage_pause(
                            "archive stage time limit reached; resume continues from its checkpoint"
                        )
                        break
                    payload: dict[str, Any] | None = None
                    reasons: list[str] = []
                    if common_crawl is not None:
                        try:
                            recovered = common_crawl.recover(
                                str(item["url"]),
                                pd.Timestamp(item["published_at"]),
                                collections=collections,
                            )
                            if recovered:
                                recovered_title = str(recovered.get("title", ""))
                                recovered_body = str(
                                    recovered.get("content_snippet", "")
                                )
                                validation_error = _validate_recovered_body(
                                    item, recovered_title, recovered_body
                                )
                                if validation_error:
                                    reasons.append(
                                        f"common_crawl_{validation_error.removeprefix('recovered_text_')}"
                                    )
                                else:
                                    payload = _recovery_payload(
                                        item,
                                        title=recovered_title,
                                        body=recovered_body,
                                        text_source="common_crawl",
                                        extra=recovered,
                                    )
                            else:
                                reasons.append("common_crawl_no_capture")
                        except CommonCrawlRateLimitError:
                            reasons.append("common_crawl_rate_limited")
                        except Exception as exc:
                            reasons.append(f"common_crawl_error:{exc}")
                    if payload is None:
                        payload, reason = wayback.recover(item)
                        if reason:
                            reasons.append(reason)
                    state = _checkpoint_state(
                        key=key,
                        payload=payload,
                        reason="; ".join(reasons),
                        previous=previous,
                    )
                    _append_checkpoint(archive_path, state)
                    archive_state[key] = state
        finally:
            if common_crawl is not None:
                common_crawl.close()
    elif not skip_archives:
        record_stage_pause(
            "archive stage time limit reached before fallback recovery began"
        )

    if stage_pause_reasons:
        stop_reason = "; ".join(stage_pause_reasons)

    final_rows: list[dict[str, Any]] = []
    paired_full_rows: list[dict[str, Any]] = []
    paired_pseudo_rows: list[dict[str, Any]] = []
    for item in selection.to_dict(orient="records"):
        key = _row_key(item)
        direct_payload = direct_state.get(key, {}).get("payload")
        ngram_payload = ngram_state.get(key, {}).get("payload")
        archive_payload = archive_state.get(key, {}).get("payload")
        if isinstance(direct_payload, dict):
            final = direct_payload
        elif (
            isinstance(ngram_payload, dict)
            and ngram_payload.get("reconstruction_quality_tier") == "high"
        ):
            final = ngram_payload
        elif isinstance(archive_payload, dict):
            final = archive_payload
        elif isinstance(ngram_payload, dict):
            final = ngram_payload
        else:
            final = {
                **item,
                "text_source": "",
                "is_pseudo_document": False,
                "recovery_status": "unavailable",
                "recovery_failure_reason": "; ".join(
                    str(value.get("reason", ""))
                    for value in (
                        direct_state.get(key, {}),
                        ngram_state.get(key, {}),
                        archive_state.get(key, {}),
                    )
                    if value.get("reason")
                ),
                "recovered_at": _utc_now(),
                "publisher_full_text_available": False,
            }
        for field in PROVENANCE_FIELDS:
            final.setdefault(field, "" if field != "is_pseudo_document" else False)
        final_rows.append(final)
        if (
            key in paired_keys
            and isinstance(direct_payload, dict)
            and isinstance(ngram_payload, dict)
            and ngram_payload.get("reconstruction_quality_tier") == "high"
        ):
            pair_id = hashlib.sha256(
                f"{str(item.get('ticker', '')).upper()}|{_normalise_url(item.get('url'))}".encode(
                    "utf-8"
                )
            ).hexdigest()
            paired_full_rows.append(
                _label_input(direct_payload, pair_id=pair_id, representation="full_text")
            )
            paired_pseudo_rows.append(
                _label_input(
                    ngram_payload,
                    pair_id=pair_id,
                    representation="pseudo_document",
                )
            )

    cumulative_refresh = refresh_cumulative_recovered_output(
        root,
        latest_rows=final_rows,
    )
    recovered_path = Path(cumulative_refresh["recovered_path"])
    full_input_path = paired_dir / "full-text-input.jsonl"
    pseudo_input_path = paired_dir / "pseudo-document-input.jsonl"
    _write_jsonl(full_input_path, paired_full_rows)
    _write_jsonl(pseudo_input_path, paired_pseudo_rows)
    paired_full = pd.DataFrame(paired_full_rows)
    paired_pseudo = pd.DataFrame(paired_pseudo_rows)

    finbert_rows, finbert_metrics = _score_finbert_pairs(
        paired_full, paired_pseudo
    )
    if not finbert_rows.empty:
        _write_jsonl(
            paired_dir / "finbert-pair-comparison.jsonl",
            finbert_rows.to_dict(orient="records"),
        )

    full_labels = pd.DataFrame()
    pseudo_labels = pd.DataFrame()
    recovery_complete = (
        len(direct_state) >= len(selection)
        and len(ngram_state) >= len(ngram_targets)
        and (
            skip_archives
            or len(archive_state)
            >= sum(
                1
                for item in selection.to_dict(orient="records")
                if direct_state.get(_row_key(item), {}).get("status") != "recovered"
                and (
                    not isinstance(
                        ngram_state.get(_row_key(item), {}).get("payload"), dict
                    )
                    or ngram_state.get(_row_key(item), {})
                    .get("payload", {})
                    .get("reconstruction_quality_tier")
                    != "high"
                )
            )
        )
    )
    if (
        run_labels
        and recovery_complete
        and not paired_full.empty
        and not paired_pseudo.empty
    ):
        full_output = paired_dir / "full-text-labels.jsonl"
        pseudo_output = paired_dir / "pseudo-document-labels.jsonl"
        full_report = run_llm_consensus_labelling(
            full_input_path,
            output_path=full_output,
            report_path=paired_dir / "full-text-label-report.json",
            checkpoint_path=paired_dir / "full-text-label-checkpoint.jsonl",
            heartbeat_path=paired_dir / "full-text-label-heartbeat.json",
            approval_queue_path=paired_dir / "full-text-approval-queue.jsonl",
            max_runtime_minutes=max_runtime_minutes,
            max_requests=max_label_requests,
            request_interval_seconds=request_interval_seconds,
        )
        full_labels = _read_jsonl_safe(full_output)
        if full_report.get("stop_reason") is None:
            run_llm_consensus_labelling(
                pseudo_input_path,
                output_path=pseudo_output,
                report_path=paired_dir / "pseudo-document-label-report.json",
                checkpoint_path=paired_dir
                / "pseudo-document-label-checkpoint.jsonl",
                heartbeat_path=paired_dir / "pseudo-document-label-heartbeat.json",
                approval_queue_path=paired_dir
                / "pseudo-document-approval-queue.jsonl",
                max_runtime_minutes=max_runtime_minutes,
                max_requests=max_label_requests,
                request_interval_seconds=request_interval_seconds,
            )
        pseudo_labels = _read_jsonl_safe(pseudo_output)

    completed = recovery_complete and (
        not run_labels
        or (
            not full_labels.empty
            and not pseudo_labels.empty
            and len(full_labels) >= len(paired_full)
            and len(pseudo_labels) >= len(paired_pseudo)
        )
    )
    if run_labels and recovery_complete and not completed and stop_reason is None:
        stop_reason = "paired LLM labelling is incomplete; rerun to resume"
    report = build_ngram_usability_report(
        year=year,
        selection=selection,
        direct_records=direct_state,
        ngram_records=ngram_state,
        final_rows=final_rows,
        paired_full=paired_full,
        paired_pseudo=paired_pseudo,
        full_labels=full_labels,
        pseudo_labels=pseudo_labels,
        finbert_metrics=finbert_metrics,
        downloaded_bytes=downloaded_bytes,
        elapsed_seconds=time.monotonic() - started,
        completed=completed,
        stop_reason=stop_reason,
    )
    report.update(
        {
            "selection_path": str(selection_path),
            "recovered_path": str(recovered_path),
            "current_selection_recovered_rows": len(final_rows),
            **{
                key: value
                for key, value in cumulative_refresh.items()
                if key != "recovered_path"
            },
            "paired_full_input": str(full_input_path),
            "paired_pseudo_input": str(pseudo_input_path),
            "report_path": str(report_path),
        }
    )
    _atomic_json(report_path, report)
    _atomic_json(
        heartbeat_path,
        {
            "status": "completed" if completed else "paused",
            "stop_reason": stop_reason,
            "report_path": str(report_path),
            "updated_at": _utc_now(),
        },
        best_effort=True,
    )
    return report
