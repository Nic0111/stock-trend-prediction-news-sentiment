"""Package recovered historical Bursa news for external sentiment labelling.

This module is offline-only.  It does not call an LLM, retrain FinBERT, update
the fusion registry, or alter the runtime news providers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECOVERY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "external-corpus-2015-2025"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RECOVERY_ROOT / "external-labelling"
# The 2023 pilot was recovered before the year-partitioned external corpus was
# created.  Include those checkpoints automatically for the repository's
# canonical corpus, while keeping temporary/test roots isolated from them.
DEFAULT_PILOT_RECOVERY_FILES = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "pilot-2023-corpus-recovery"
    / "raw"
    / "hybrid-recovered.jsonl",
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "pilot-2023-corpus-recovery"
    / "external-labelling"
    / "recovered-articles.jsonl",
)
TRACKING_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}
ALLOWED_FULL_TEXT_SOURCES = {"publisher", "common_crawl", "wayback"}
ALLOWED_SENTIMENTS = ("negative", "neutral", "positive")
# FinBERT's production adapter truncates inputs at 128 model tokens.  The
# export package intentionally does not depend on a transformers tokenizer;
# this deterministic whitespace approximation keeps the externally supplied
# labelling view bounded while preserving the complete article text below it
# for LLM-assisted labelling and later import.
LABEL_TEXT_MAX_TOKENS = 128
# Chat uploads are deliberately conservative.  ``batch_size`` remains a
# public argument for compatibility, but is clamped to this upper bound.
MAX_BATCH_ROWS = 25
MAX_BATCH_CHARS = 60_000
# Keep a little room for the compact label-facing JSON envelope when one
# pathological token is itself extremely long.  Normal news text is governed
# by the token cap above and never approaches this bound.
LABEL_TEXT_MAX_CHARS = MAX_BATCH_CHARS - 2_048
# Descriptive aliases make the bounds discoverable to downstream packaging
# scripts without requiring callers to know the internal constant spelling.
MAX_LABEL_TOKENS = LABEL_TEXT_MAX_TOKENS
MAX_BATCH_SIZE = MAX_BATCH_ROWS
MAX_BATCH_CONTENT_CHARS = MAX_BATCH_CHARS
LABEL_PROMPT_VERSION = "external-bursa-labelling-v1"
EVENT_CATEGORIES = (
    "earnings",
    "guidance",
    "contract",
    "corporate_action",
    "financing",
    "operations",
    "regulatory_legal",
    "management",
    "product_service",
    "macro_sector_exposure",
    "esg",
    "other",
)


def _normalise_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_KEYS
        and not key.casefold().startswith("utm_")
    ]
    return urlunparse(
        (
            parsed.scheme.casefold() or "https",
            parsed.netloc.casefold(),
            re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/",
            "",
            urlencode(sorted(query)),
            "",
        )
    )


def _clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    # Repair the common UTF-8-as-Windows-1252 corruption only when it is
    # strongly indicated; otherwise preserve the publisher text unchanged.
    if any(marker in text for marker in ("â€™", "â€œ", "â€", "Â")):
        try:
            repaired = text.encode("cp1252").decode("utf-8")
            if repaired.count("�") <= text.count("�"):
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text


def _label_text(title: str, article_text: str, *, max_tokens: int = LABEL_TEXT_MAX_TOKENS) -> str:
    """Build the bounded text view sent to an external label reviewer.

    Runtime FinBERT uses a tokenizer with ``max_length=128``.  Pulling a
    transformers tokenizer into this offline packaging step would make the
    recovery/export path needlessly heavyweight and would make output depend
    on a local model artifact.  A stable whitespace token view gives us the
    same practical bound and, importantly, is reproducible across machines.
    The title is always preferred before the article body because it carries
    the strongest entity/event signal.  ``article_text`` itself is never
    truncated in the exported snapshot.
    """

    limit = max(1, int(max_tokens))
    title_tokens = str(title or "").split()
    body_tokens = str(article_text or "").split()
    tokens = title_tokens[:limit]
    if len(tokens) < limit:
        tokens.extend(body_tokens[: limit - len(tokens)])
    text = " ".join(tokens)
    if len(text) > LABEL_TEXT_MAX_CHARS:
        text = text[:LABEL_TEXT_MAX_CHARS].rsplit(" ", 1)[0].rstrip()
    return text


def _label_payload_chars(row: Mapping[str, Any]) -> int:
    """Return deterministic upload-content size for one row.

    This helper is retained for compatibility with callers that report the
    label-facing payload size.  Chat batches are written through
    ``_batch_upload_record`` below, which excludes the potentially long
    review-only article body so the same ceiling applies to the actual upload.
    """

    payload = {
        "article_id": row.get("article_id", ""),
        "ticker": row.get("ticker", ""),
        "company_name": row.get("company_name", ""),
        "published_at": row.get("published_at", ""),
        "title": row.get("title", ""),
        "content_sha256": row.get("content_sha256", ""),
        "label_text": row.get("label_text", ""),
        "label_text_sha256": row.get("label_text_sha256", ""),
    }
    return len(json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")))


# Batch files are the artefacts uploaded to an external reviewer.  Keep the
# full recovered body in ``external-label-input.jsonl`` and ``review-pack.csv``
# for provenance/manual review, but do not duplicate it in every chat batch.
# This makes the advertised character ceiling apply to the actual upload
# rather than only to an internal estimate.
_BATCH_UPLOAD_FIELDS = (
    "batch_id",
    "article_id",
    "ticker",
    "company_name",
    "published_at",
    "title",
    "source_domain",
    "source_url",
    "text_source",
    "is_pseudo_document",
    "publisher_full_text_available",
    "reconstruction_quality_score",
    "reconstruction_quality_tier",
    "content_sha256",
    "label_text",
    "label_text_sha256",
)


def _batch_upload_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact, label-facing representation written to a batch."""

    # The placeholder keeps the pre-assignment size estimate conservative; the
    # actual batch ID has the same fixed-width format.
    return {
        field: row.get(field, "batch-0000" if field == "batch_id" else "")
        for field in _BATCH_UPLOAD_FIELDS
    }


def _batch_upload_chars(row: Mapping[str, Any]) -> int:
    """Count one serialized compact batch row including its newline."""

    return len(
        json.dumps(
            _batch_upload_record(row),
            ensure_ascii=False,
            default=str,
        )
    ) + 1


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a small package metadata file on Windows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _json_safe(value: Any) -> Any:
    """Convert pandas missing scalars into strict-JSON nulls recursively."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA:
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        missing = pd.isna(value)
        # ``pd.isna`` returns an array for list-like values; those were handled
        # above.  Scalar NumPy booleans are safely converted here.
        if not hasattr(missing, "__len__") and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    return value


def discover_recovery_files(
    recovery_root: Path | str,
    *,
    years: Sequence[int] = tuple(range(2015, 2026)),
    extra_files: Sequence[Path | str] = (),
) -> list[Path]:
    """Find one hybrid recovery checkpoint per requested year."""

    root = Path(recovery_root)
    files: list[Path] = []
    for year in dict.fromkeys(int(value) for value in years):
        candidates = (
            root / str(year) / "recovery" / "raw" / "hybrid-recovered.jsonl",
            root / str(year) / "raw" / "hybrid-recovered.jsonl",
        )
        selected = next((path for path in candidates if path.exists()), None)
        if selected is not None:
            files.append(selected)
    for value in extra_files:
        path = Path(value)
        if path.exists() and path not in files:
            files.append(path)
    return files


def _eligible(row: Mapping[str, Any]) -> tuple[bool, str]:
    source = str(row.get("text_source", "")).strip().casefold()
    body = _clean_text(
        row.get("content_snippet")
        or row.get("article_text")
        or row.get("description")
    )
    if not body:
        return False, "missing_text"
    if source in ALLOWED_FULL_TEXT_SOURCES:
        return True, ""
    if source == "gdelt_webngrams":
        if (
            str(row.get("reconstruction_quality_tier", "")).casefold() == "high"
            and int(row.get("unique_token_count", 0) or 0) >= 180
        ):
            return True, ""
        return False, "ngram_not_high_quality"
    return False, f"unsupported_text_source:{source or 'unavailable'}"


def _article_id(row: Mapping[str, Any], normalised_url: str) -> str:
    existing = str(row.get("article_id", "")).strip()
    if existing:
        return existing
    basis = "|".join(
        (
            str(row.get("ticker", "")).strip().upper(),
            str(row.get("published_at", "")).strip(),
            normalised_url,
        )
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _input_record(row: Mapping[str, Any]) -> dict[str, Any]:
    ticker = str(row.get("ticker", "")).strip().upper()
    url = _normalise_url(row.get("url") or row.get("source_url"))
    body = _clean_text(
        row.get("article_text")
        or row.get("content_snippet")
        or row.get("description")
    )
    title = _clean_text(row.get("title"))
    source = str(row.get("text_source", "")).strip().casefold()
    content_hash = str(row.get("content_sha256", "")).strip()
    if not content_hash:
        content_hash = hashlib.sha256(body.casefold().encode("utf-8")).hexdigest()
    published = pd.to_datetime(row.get("published_at"), utc=True, errors="coerce")
    # Keep ``model_text`` as the historical/full compatibility field, while
    # ``label_text`` mirrors the bounded runtime text view used by FinBERT.
    model_text = ". ".join(value for value in (title, body) if value)
    label_text = _label_text(title, body)
    record = {
        "article_id": _article_id(row, url),
        "ticker": ticker,
        "company_name": _clean_text(row.get("company_name")),
        "published_at": (
            published.isoformat() if not pd.isna(published) else ""
        ),
        "title": title,
        "source_domain": str(
            row.get("source_domain")
            or (urlparse(url).hostname or "").casefold().removeprefix("www.")
        ),
        "source_url": url,
        "text_source": source,
        "is_pseudo_document": bool(row.get("is_pseudo_document", False)),
        "publisher_full_text_available": bool(
            row.get("publisher_full_text_available", source == "publisher")
        ),
        "reconstruction_quality_score": (
            float(row.get("reconstruction_quality_score", 0.0) or 0.0)
            if source == "gdelt_webngrams"
            else None
        ),
        "reconstruction_quality_tier": (
            str(row.get("reconstruction_quality_tier", ""))
            if source == "gdelt_webngrams"
            else None
        ),
        "content_sha256": content_hash,
        "article_text": body,
        "label_text": label_text,
        "label_text_sha256": hashlib.sha256(label_text.encode("utf-8")).hexdigest(),
        "model_text": model_text,
    }
    # Recovery provenance is useful during review (especially for reconstructed
    # Web-NGram pseudo-documents).  Keep fields when present without changing
    # the shape of older/minimal recovery fixtures.
    for field in (
        "provider",
        "source_name",
        "recovery_status",
        "recovery_failure_reason",
        "recovered_at",
        "retrieved_at",
        "ngram_file_timestamps",
        "ngram_record_count",
        "deciles_present",
        "unique_token_count",
        "overlap_attachment_rate",
        "automatic_relevance",
        "relevance_score",
        "relevance_reasons",
    ):
        if field in row:
            record[field] = row.get(field)
    return record


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_safe(dict(row)), ensure_ascii=False, default=str)
                + "\n"
            )


def _label_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "StockSense external Bursa news label",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "article_id",
            "content_sha256",
            "label_text_sha256",
            "entity_relevant",
            "sentiment",
            "event_category",
            "language",
            "confidence",
            "evidence",
            "rationale",
            "label_status",
            "labelled_by",
            "labelled_at",
        ],
        "properties": {
            "article_id": {"type": "string", "minLength": 1},
            "content_sha256": {
                "type": "string",
                "pattern": "^[0-9a-fA-F]{64}$",
            },
            "label_text_sha256": {
                "type": "string",
                "pattern": "^[0-9a-fA-F]{64}$",
            },
            "batch_id": {"type": "string"},
            "batch_sha256": {
                "type": "string",
                "pattern": "^[0-9a-fA-F]{64}$",
            },
            "prompt_version": {"type": "string"},
            "prompt_sha256": {
                "type": "string",
                "pattern": "^[0-9a-fA-F]{64}$",
            },
            "run_id": {"type": "string"},
            "label_run_id": {"type": "string"},
            "output_sha256": {
                "type": "string",
                "pattern": "^[0-9a-fA-F]{64}$",
            },
            "entity_relevant": {"type": "boolean"},
            "sentiment": {
                "type": ["string", "null"],
                "enum": [*ALLOWED_SENTIMENTS, None],
            },
            "event_category": {
                "type": "string",
                "enum": list(EVENT_CATEGORIES),
            },
            "language": {"type": "string", "enum": ["en", "other"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 240,
                },
            },
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": 800,
            },
            "label_status": {
                "type": "string",
                "enum": ["labelled", "needs_review"],
            },
            "labelled_by": {"type": "string", "minLength": 1},
            "labelled_at": {"type": "string", "format": "date-time"},
        },
        "allOf": [
            {
                "if": {
                    "properties": {"entity_relevant": {"const": False}}
                },
                "then": {"properties": {"sentiment": {"type": "null"}}},
            },
            {
                "if": {
                    "properties": {"entity_relevant": {"const": True}}
                },
                "then": {
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": list(ALLOWED_SENTIMENTS),
                        }
                    }
                },
            },
        ],
    }


def _guide(
    row_count: int,
    requested_years: Sequence[int],
    recovered_years: Sequence[int],
) -> str:
    coverage = (
        f"{min(recovered_years)}–{max(recovered_years)}"
        if recovered_years
        else "no recovered years yet"
    )
    return f"""# StockSense historical Bursa news external labelling

This package contains **{row_count:,} recovered documents** from
**{coverage}**. The requested collection range is
**{min(requested_years)}–{max(requested_years)}**. Check
`package-summary.json` before labelling; do not label a partial package if all
requested years are required. It is an offline research dataset and is not
used by production automatically.

## What to upload

Upload one file from `batches/`, its matching `.manifest.json` sidecar, and
`external-label-output.schema.json`. Ask ChatGPT to read every input line and
return exactly one JSON object per input line in the same order. Use
`label_text` as the bounded, runtime-aligned labelling view. `article_text` is
retained in the snapshot for review context and provenance; do not copy it
into the returned labels file.

For manual review, `review-pack.csv` carries the same immutable identity,
`label_text`, and full `article_text` fields with blank annotation columns.

## Labelling rules

1. First decide whether the article is materially about the named company.
   A broker name, incidental alias, acronym collision or passing mention is
   not sufficient.
2. If `entity_relevant=false`, set `sentiment=null`.
3. If relevant, label the likely company-specific financial implication as
   `negative`, `neutral`, or `positive`. Do not label writing tone.
4. Use only the supplied text. Never infer missing facts.
5. Set `needs_review` when the entity, event, language, reconstruction, or
   sentiment is ambiguous.
6. `gdelt_webngrams` rows are reconstructed pseudo-documents. Treat broken
   ordering or missing context conservatively.
7. Copy `article_id`, `content_sha256`, and `label_text_sha256` exactly, plus
   `batch_id`/`batch_sha256` when present. Do not add Markdown or explanatory
   text outside JSONL.

## Recommended batching

Large chat uploads may exceed a model context window. The exporter caps each
batch at **25 rows** and **60,000 label-facing characters** (the exact limits
are recorded in `package-summary.json` and the snapshot manifest). Label one
batch at a time, then concatenate the returned JSONL files without headers.

## After labelling

Return the combined JSONL to Codex. Viability checks will cover schema
validity, ID coverage, duplicates, entity precision, uncertain-label rate,
class balance, temporal leakage, per-company/year coverage and chronological
train/validation/test suitability. After review, accepted labels are written
under the active `llm_assisted_label` provenance policy.
"""


def _chatgpt_prompt() -> str:
    return """# Prompt for external Bursa-news labelling

You are labelling a weak-supervision research dataset for StockSense, a Bursa
Malaysia stock-trend education system.

I will upload one JSONL batch, its matching batch manifest sidecar, and
`external-label-output.schema.json`.
For every input line (use its bounded `label_text`; `article_text`, when
present, is review context only):

1. Decide whether the article is materially about the supplied `company_name`
   and `ticker`. A broker name, incidental mention, generic acronym collision,
   or unrelated use of an alias is not relevant.
2. If irrelevant, output `entity_relevant=false` and `sentiment=null`.
3. If relevant, classify the likely company-specific financial implication as
   `negative`, `neutral`, or `positive`. Classify financial implication, not
   emotional writing style.
4. Use only supplied text. For a reconstructed GDELT pseudo-document, abstain
   with `label_status="needs_review"` whenever missing or disordered context
   could change the conclusion.
5. Give confidence from 0 to 1 and at most two short evidence excerpts.
6. Copy every `article_id`, `content_sha256`, and `label_text_sha256` exactly.
   Copy `batch_id` and `batch_sha256` from the input/sidecar when present.
   Return one output object for every input object in the same order. These
   hashes are required for stale-snapshot and text-view validation.

Return raw JSONL only: no Markdown fence, no headings, no commentary, and do
not repeat article text. Each line must validate against the supplied schema.
Use `labelled_by="chatgpt_external"` and an ISO-8601 UTC `labelled_at`.
"""


def _assign_batches(
    rows: Sequence[Mapping[str, Any]], *, requested_batch_size: int
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, int],
    dict[str, int],
]:
    """Assign stable batch IDs under both row and label-content limits."""

    requested = max(1, int(requested_batch_size))
    effective = min(requested, MAX_BATCH_ROWS)
    grouped: dict[str, list[dict[str, Any]]] = {}
    label_char_counts: dict[str, int] = {}
    label_text_char_counts: dict[str, int] = {}
    current: list[dict[str, Any]] = []
    current_chars = 0
    current_label_text_chars = 0

    def flush() -> None:
        nonlocal current, current_chars, current_label_text_chars
        if not current:
            return
        batch_id = f"batch-{len(grouped) + 1:04d}"
        for item in current:
            item["batch_id"] = batch_id
        grouped[batch_id] = current
        label_char_counts[batch_id] = current_chars
        label_text_char_counts[batch_id] = current_label_text_chars
        current = []
        current_chars = 0
        current_label_text_chars = 0

    for source in rows:
        row = dict(source)
        row_chars = _batch_upload_chars(row)
        row_label_text_chars = len(str(row.get("label_text", "")))
        # A single row is always emitted; label_text is normally far below the
        # limit because it is capped to 128 tokens.  Keeping this branch makes
        # the behaviour deterministic even for malformed metadata with a very
        # large company/title value.
        if current and (
            len(current) >= effective
            or current_chars + row_chars > MAX_BATCH_CHARS
        ):
            flush()
        current.append(row)
        current_chars += row_chars
        current_label_text_chars += row_label_text_chars
    flush()
    flattened = [row for batch in grouped.values() for row in batch]
    return flattened, grouped, label_char_counts, label_text_char_counts


def export_external_label_package(
    recovery_root: Path | str = DEFAULT_RECOVERY_ROOT,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    years: Sequence[int] = tuple(range(2015, 2026)),
    extra_files: Sequence[Path | str] | None = None,
    batch_size: int = MAX_BATCH_ROWS,
) -> dict[str, Any]:
    """Create deduplicated input and blank output JSONL for external labelling."""

    recovery_path = Path(recovery_root).resolve()
    default_extras = (
        DEFAULT_PILOT_RECOVERY_FILES
        if recovery_path == DEFAULT_RECOVERY_ROOT.resolve()
        else ()
    )
    files = discover_recovery_files(
        recovery_root,
        years=years,
        extra_files=default_extras if extra_files is None else extra_files,
    )
    raw_rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for path in files:
        for row in _read_jsonl(path):
            accepted, reason = _eligible(row)
            if accepted:
                raw_rows.append(_input_record(row))
            else:
                rejected[reason] = rejected.get(reason, 0) + 1

    frame = pd.DataFrame(raw_rows)
    if not frame.empty:
        frame["_published"] = pd.to_datetime(
            frame["published_at"], utc=True, errors="coerce"
        )
        frame = frame.dropna(subset=["_published"])
        frame = frame.loc[
            frame["_published"].dt.year.isin([int(value) for value in years])
        ].copy()
        frame = frame.sort_values(
            ["_published", "ticker", "article_id"], kind="stable"
        )
        frame = frame.drop_duplicates(["ticker", "source_url"], keep="first")
        frame = frame.drop_duplicates(
            ["ticker", "content_sha256"], keep="first"
        ).reset_index(drop=True)
        frame = frame.drop(columns=["_published"])
    source_rows = frame.to_dict(orient="records") if not frame.empty else []
    rows, grouped, batch_label_chars, batch_label_text_chars = _assign_batches(
        source_rows, requested_batch_size=batch_size
    )
    effective_batch_size = min(max(1, int(batch_size)), MAX_BATCH_ROWS)
    batch_label_text_sha256 = {
        batch_id: hashlib.sha256(
            "\n".join(str(row.get("label_text", "")) for row in batch_rows).encode(
                "utf-8"
            )
        ).hexdigest()
        for batch_id, batch_rows in grouped.items()
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "external-label-input.jsonl"
    _write_jsonl(input_path, rows)
    input_sha256 = _sha256_file(input_path)
    prompt_version = LABEL_PROMPT_VERSION
    prompt_sha256 = hashlib.sha256(_chatgpt_prompt().encode("utf-8")).hexdigest()
    batch_paths: list[str] = []
    batch_file_hashes: dict[str, str] = {}
    for batch_id, batch_rows in grouped.items():
        batch_path = output / "batches" / f"{batch_id}.jsonl"
        # Upload only the bounded label view.  Full article text remains in the
        # immutable snapshot/review pack and is intentionally not duplicated in
        # chat-facing files.
        _write_jsonl(
            batch_path,
            (_batch_upload_record(row) for row in batch_rows),
        )
        batch_paths.append(str(batch_path))
        batch_file_hashes[batch_id] = _sha256_file(batch_path)
    template_rows = [
        {
            "article_id": row["article_id"],
            "content_sha256": row["content_sha256"],
            "label_text_sha256": row["label_text_sha256"],
            "batch_id": row.get("batch_id", ""),
            "batch_sha256": batch_file_hashes.get(str(row.get("batch_id", "")), ""),
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "entity_relevant": None,
            "sentiment": None,
            "event_category": "",
            "language": "",
            "confidence": None,
            "evidence": [],
            "rationale": "",
            "label_status": "",
            "labelled_by": "",
            "labelled_at": "",
        }
        for row in rows
    ]
    template_path = output / "external-label-output-template.jsonl"
    _write_jsonl(template_path, template_rows)
    schema_path = output / "external-label-output.schema.json"
    _atomic_write_text(
        schema_path,
        json.dumps(_label_schema(), indent=2, ensure_ascii=False),
    )
    csv_path = output / "external-label-index.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        columns = [
            "batch_id",
            "article_id",
            "ticker",
            "company_name",
            "published_at",
            "source_domain",
            "text_source",
            "is_pseudo_document",
            "source_url",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    # A ready-to-review CSV is kept separate from the compact index.  It
    # carries the immutable identity/provenance fields, the bounded label view
    # and full article context, plus blank annotation columns for user-led
    # labelling workflows.
    review_path = output / "review-pack.csv"
    review_columns = [
        "article_id",
        "ticker",
        "company_name",
        "published_at",
        "title",
        "source_domain",
        "source_url",
        "content_sha256",
        "label_text_sha256",
        "batch_id",
        "batch_sha256",
        "prompt_version",
        "prompt_sha256",
        "text_source",
        "is_pseudo_document",
        "publisher_full_text_available",
        "recovery_status",
        "recovered_at",
        "reconstruction_quality_tier",
        "label_text",
        "article_text",
        "entity_relevant",
        "sentiment",
        "event_category",
        "language",
        "confidence",
        "evidence",
        "rationale",
        "label_status",
        "labelled_by",
        "labelled_at",
    ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_columns)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in review_columns}
            payload["batch_sha256"] = batch_file_hashes.get(
                str(row.get("batch_id", "")), ""
            )
            payload["prompt_version"] = prompt_version
            payload["prompt_sha256"] = prompt_sha256
            for key in review_columns[review_columns.index("entity_relevant") :]:
                payload[key] = ""
            writer.writerow(payload)
    by_year: dict[str, int] = {}
    by_ticker: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for row in rows:
        year = str(pd.Timestamp(row["published_at"]).year)
        by_year[year] = by_year.get(year, 0) + 1
        ticker = str(row["ticker"])
        by_ticker[ticker] = by_ticker.get(ticker, 0) + 1
        source = str(row["text_source"])
        by_source[source] = by_source.get(source, 0) + 1
    guide_path = output / "LABELLING_GUIDE.md"
    _atomic_write_text(
        guide_path,
        _guide(
            len(rows),
            tuple(int(value) for value in years),
            tuple(sorted(int(value) for value in by_year)),
        ),
    )
    prompt_path = output / "CHATGPT_LABELLING_PROMPT.md"
    _atomic_write_text(prompt_path, _chatgpt_prompt())

    # Freeze source/recovery provenance next to the immutable input snapshot.
    # Source paths and their content hashes make it possible to detect a
    # regenerated package that silently used a different recovery checkpoint.
    recovery_file_records: list[dict[str, Any]] = []
    for path in files:
        try:
            stat = path.stat()
            digest = _sha256_file(path)
            recovery_file_records.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "bytes": int(stat.st_size),
                }
            )
        except OSError:
            # ``discover_recovery_files`` only returns existing paths, but a
            # concurrent cleanup should remain visible rather than aborting a
            # package that has already been written.
            recovery_file_records.append(
                {"path": str(path), "sha256": "missing", "bytes": None}
            )
    provenance_basis = {
        "years": [int(value) for value in years],
        "recovery_files": recovery_file_records,
        "raw_eligible_rows": len(raw_rows),
        "deduplicated_label_rows": len(rows),
    }
    source_manifest_sha256 = _sha256_bytes(
        json.dumps(
            recovery_file_records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    provenance_sha256 = _sha256_bytes(
        json.dumps(
            provenance_basis, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    snapshot_id = f"external-news-{provenance_sha256[:16]}"

    # Emit a sidecar for each batch so a reviewer can return one labelled batch
    # with its own immutable identity, without relying on the package summary.
    batch_sidecar_paths: dict[str, str] = {}
    for batch_id, batch_rows in grouped.items():
        batch_path = output / "batches" / f"{batch_id}.jsonl"
        sidecar_path = output / "batches" / f"{batch_id}.manifest.json"
        sidecar = {
            "schema_version": "external-news-labelling-batch-v1",
            "snapshot_id": snapshot_id,
            "run_id": snapshot_id,
            "label_run_id": snapshot_id,
            "snapshot_sha256": input_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "batch_id": batch_id,
            "batch_path": str(batch_path),
            "batch_sha256": _sha256_file(batch_path),
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "row_count": len(batch_rows),
            "label_payload_chars": int(batch_label_chars.get(batch_id, 0)),
            "label_text_chars": int(batch_label_text_chars.get(batch_id, 0)),
            "character_count": int(batch_label_chars.get(batch_id, 0)),
            "label_text_sha256": batch_label_text_sha256.get(batch_id, ""),
            "max_rows": MAX_BATCH_ROWS,
            "max_label_chars": MAX_BATCH_CHARS,
            "production_changes": False,
            "llm_calls": 0,
        }
        _atomic_write_text(
            sidecar_path,
            json.dumps(sidecar, indent=2, ensure_ascii=False),
        )
        batch_sidecar_paths[batch_id] = str(sidecar_path)

    # Hash package artifacts before writing the manifest.  The manifest itself
    # and the summary are intentionally excluded to avoid a self-referential
    # hash cycle; both hashes are returned in the summary for verification.
    artifact_paths: list[Path] = [
        input_path,
        template_path,
        schema_path,
        csv_path,
        review_path,
        guide_path,
        prompt_path,
        *(output / "batches" / f"{batch_id}.jsonl" for batch_id in grouped),
        *(Path(path) for path in batch_sidecar_paths.values()),
    ]
    artifact_hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
        for path in artifact_paths
        if path.exists()
    }
    batch_manifest = [
        {
            "batch_id": batch_id,
            "path": str(output / "batches" / f"{batch_id}.jsonl"),
            "rows": len(grouped[batch_id]),
            "label_chars": int(batch_label_chars.get(batch_id, 0)),
            "label_payload_chars": int(batch_label_chars.get(batch_id, 0)),
            "label_text_chars": int(batch_label_text_chars.get(batch_id, 0)),
            "characters": int(batch_label_chars.get(batch_id, 0)),
            "character_count": int(batch_label_chars.get(batch_id, 0)),
            "label_text_sha256": batch_label_text_sha256.get(batch_id, ""),
            "sha256": artifact_hashes.get(
                f"batches/{batch_id}.jsonl", ""
            ),
            "batch_sha256": artifact_hashes.get(
                f"batches/{batch_id}.jsonl", ""
            ),
            "sidecar_manifest_path": batch_sidecar_paths.get(batch_id, ""),
            "sidecar_manifest_sha256": artifact_hashes.get(
                f"batches/{batch_id}.manifest.json", ""
            ),
        }
        for batch_id in grouped
    ]
    manifest = {
        "schema_version": "external-news-labelling-snapshot-v1",
        "snapshot_id": snapshot_id,
        "run_id": snapshot_id,
        "label_run_id": snapshot_id,
        "snapshot_path": str(input_path),
        "snapshot_sha256": input_sha256,
        "input_snapshot_sha256": input_sha256,
        "input_file_sha256": input_sha256,
        "snapshot_hash": input_sha256,
        "provenance_sha256": provenance_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "years": [int(value) for value in years],
        "row_count": len(rows),
        "raw_eligible_rows": len(raw_rows),
        "recovery_files": recovery_file_records,
        "provenance": {
            "source_manifest_sha256": source_manifest_sha256,
            "recovery_files": recovery_file_records,
            "generator": "pipeline.external_news_label_export",
        },
        "artifact_hashes": artifact_hashes,
        "batch_limits": {
            "max_rows": MAX_BATCH_ROWS,
            "max_label_chars": MAX_BATCH_CHARS,
            "requested_batch_size": max(1, int(batch_size)),
            "effective_batch_size": effective_batch_size,
        },
        "label_text_policy": {
            "tokenizer": "deterministic_whitespace_v1",
            "max_tokens": LABEL_TEXT_MAX_TOKENS,
            "max_chars": LABEL_TEXT_MAX_CHARS,
            "fields": ["title", "article_text"],
        },
        "batches": batch_manifest,
        "production_changes": False,
        "llm_calls": 0,
    }
    manifest_path = output / "external-label-snapshot-manifest.json"
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False),
    )
    manifest_sha256 = _sha256_file(manifest_path)
    summary = {
        # Keep the package schema identifier stable; all additions below are
        # optional sidecar/provenance metadata for existing consumers.
        "schema_version": "external-news-labelling-package-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "run_id": snapshot_id,
        "label_run_id": snapshot_id,
        "snapshot_sha256": input_sha256,
        "input_snapshot_sha256": input_sha256,
        "snapshot_hash": input_sha256,
        "provenance_sha256": provenance_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "provenance": {
            "source_manifest_sha256": source_manifest_sha256,
            "recovery_files": recovery_file_records,
            "generator": "pipeline.external_news_label_export",
        },
        "snapshot_manifest_path": str(manifest_path),
        "snapshot_manifest": str(manifest_path),
        "snapshot_manifest_sha256": manifest_sha256,
        "snapshot_manifest_hash": manifest_sha256,
        "provenance_manifest_path": str(manifest_path),
        "provenance_manifest": str(manifest_path),
        "provenance_manifest_sha256": manifest_sha256,
        # ``manifest_path``/``manifest_sha256`` are concise compatibility
        # aliases used by generic dataset tooling.
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_hash": manifest_sha256,
        "years": [int(value) for value in years],
        "recovery_files": [str(path) for path in files],
        "recovery_file_hashes": {
            record["path"]: record["sha256"] for record in recovery_file_records
        },
        "raw_eligible_rows": len(raw_rows),
        "deduplicated_label_rows": len(rows),
        "rejected_rows": rejected,
        "rows_by_year": by_year,
        "rows_by_ticker": by_ticker,
        "rows_by_text_source": by_source,
        "requested_batch_size": max(1, int(batch_size)),
        "batch_size": effective_batch_size,
        "batch_max_rows": MAX_BATCH_ROWS,
        "batch_max_chars": MAX_BATCH_CHARS,
        "batch_count": len(grouped),
        "batch_row_counts": {
            batch_id: len(batch_rows) for batch_id, batch_rows in grouped.items()
        },
        "batch_label_chars": batch_label_chars,
        "batch_label_text_chars": batch_label_text_chars,
        "batch_label_text_sha256": batch_label_text_sha256,
        "batch_sha256": batch_file_hashes,
        "batch_char_counts": batch_label_chars,
        "batch_character_counts": batch_label_chars,
        "batch_manifests": batch_manifest,
        "batch_paths": batch_paths,
        "batch_sidecar_paths": batch_sidecar_paths,
        "batch_manifest_paths": batch_sidecar_paths,
        "label_text_policy": {
            "tokenizer": "deterministic_whitespace_v1",
            "max_tokens": LABEL_TEXT_MAX_TOKENS,
            "max_chars": LABEL_TEXT_MAX_CHARS,
            "fields": ["title", "article_text"],
        },
        "artifact_hashes": artifact_hashes,
        "input_path": str(input_path),
        "snapshot_path": str(input_path),
        "input_file_sha256": input_sha256,
        "template_path": str(template_path),
        "schema_path": str(schema_path),
        "guide_path": str(guide_path),
        "prompt_path": str(prompt_path),
        "review_pack_path": str(review_path),
        "production_changes": False,
        "llm_calls": 0,
    }
    summary_path = output / "package-summary.json"
    _atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False),
    )
    return {**summary, "summary_path": str(summary_path)}
