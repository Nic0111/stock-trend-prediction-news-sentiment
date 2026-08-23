"""Import offline external/Codex labels into an experimental news dataset.

The external labelling package is deliberately kept separate from the
production Bursa-news artefact.  This module only reads a frozen input
snapshot and a returned JSONL/CSV file, validates the row and content
identity, and writes an experimental JSONL dataset plus a review queue.

No network calls, model calls, production registry updates, or production
dataset writes are performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pipeline.bursa_news_dataset import LLM_ASSISTED_LABEL_SOURCE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTAL_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "news-evidence-benchmark"
    / "external-news-experimental.jsonl"
)
PRODUCTION_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "news-evidence-benchmark"
    / "bursa-news-llm-pseudolabelled.jsonl"
)
DEFAULT_REVIEW_QUEUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "news-evidence-benchmark"
    / "external-news-review-queue.jsonl"
)
DEFAULT_IMPORT_REPORT_PATH = (
    PROJECT_ROOT
    / "results"
    / "external-news-import"
    / "import-report.json"
)
# Naming aliases mirror the other offline pipeline modules.
DEFAULT_DATASET_PATH = DEFAULT_EXPERIMENTAL_DATASET_PATH
DEFAULT_OUTPUT_PATH = DEFAULT_EXPERIMENTAL_DATASET_PATH

SENTIMENT_LABELS = ("negative", "neutral", "positive")
SENTIMENT_VALUES = set(SENTIMENT_LABELS)
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
LANGUAGES = ("en", "other")
ALLOWED_LABEL_STATUS = {"labelled", "needs_review"}
SHA256_LENGTH = 64


class ExternalNewsImportError(ValueError):
    """Raised when a returned label file cannot be safely imported."""


def _clean(value: Any) -> str:
    if value is None:
        return ""
    # CSV readers and pandas often represent missing values as NaN.
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _bool(value: Any, *, field: str = "value") -> bool | None:
    if isinstance(value, bool):
        return value
    text = _clean(value).casefold()
    if text in {"true", "1", "yes", "y", "relevant", "include", "included"}:
        return True
    if text in {"false", "0", "no", "n", "irrelevant", "exclude", "excluded"}:
        return False
    if not text or text in {"none", "null", "nan", "na", "n/a"}:
        return None
    raise ExternalNewsImportError(f"invalid boolean for {field}: {value!r}")


def _optional_bool(value: Any, *, field: str) -> bool | None:
    return _bool(value, field=field)


def _optional_float(value: Any, *, field: str) -> float | None:
    text = _clean(value)
    if not text or text.casefold() in {"none", "null", "nan", "na", "n/a"}:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise ExternalNewsImportError(f"invalid number for {field}: {value!r}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ExternalNewsImportError(f"{field} must be between 0 and 1")
    return number


def _optional_json(value: Any, *, field: str) -> Any:
    if isinstance(value, (list, dict)):
        return value
    text = _clean(value)
    if not text or text.casefold() in {"none", "null", "nan", "na", "n/a"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # A reviewer CSV commonly stores one evidence excerpt as plain text.
        if field == "evidence":
            return [text]
        raise ExternalNewsImportError(f"invalid JSON for {field}") from exc


def _normalise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize CSV spellings while retaining all source fields."""

    output = {str(key): value for key, value in record.items()}
    aliases = {
        "articleId": "article_id",
        "article-id": "article_id",
        "contentHash": "content_sha256",
        "content_hash": "content_sha256",
        "content-sha256": "content_sha256",
        "sentiment_label": "sentiment",
        "label": "sentiment",
        "entity_relevance": "entity_relevant",
        "entityRelevant": "entity_relevant",
        "approved_entity_relevant": "entity_relevant",
        "approved_entity_relevance": "entity_relevant",
        "reviewer_entity_relevant": "entity_relevant",
        "reviewer_entity_relevance": "entity_relevant",
        "event": "event_category",
        "approved_event_category": "event_category",
        "reviewed_event_category": "event_category",
        "status": "label_status",
        "review_status": "label_status",
        "reviewer_status": "label_status",
        "approval_status": "label_status",
        "annotator": "labelled_by",
        "labeler": "labelled_by",
        "reviewer": "labelled_by",
        "labelled_by_model": "labelled_by",
        "approved_sentiment_label": "sentiment",
        "approved_sentiment": "sentiment",
        "reviewed_sentiment": "sentiment",
        "reviewed_sentiment_label": "sentiment",
        "reviewer_label": "sentiment",
        "reviewer_sentiment": "sentiment",
    }
    for source, target in aliases.items():
        if target not in output and source in output:
            output[target] = output[source]
    # A few reviewer exports use ``source_url`` while the package uses ``url``.
    if "url" not in output and "source_url" in output:
        output["url"] = output["source_url"]
    return output


def _read_records(path: Path | str) -> list[dict[str, Any]]:
    """Read JSONL/JSON/CSV rows without silently dropping malformed records."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExternalNewsImportError(
                    f"invalid JSON on line {line_number} of {source}"
                ) from exc
            if not isinstance(value, dict):
                raise ExternalNewsImportError(
                    f"line {line_number} of {source} is not a JSON object"
                )
            rows.append(_normalise_record(value))
        return rows
    if suffix == ".json":
        try:
            value = json.loads(source.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ExternalNewsImportError(f"invalid JSON: {source}") from exc
        if isinstance(value, dict):
            # A single object is a useful small fixture and is unambiguous.
            return [_normalise_record(value)]
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ExternalNewsImportError(f"JSON input must be an object or list: {source}")
        return [_normalise_record(item) for item in value]
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ExternalNewsImportError(f"CSV has no header: {source}")
            return [_normalise_record(dict(row)) for row in reader]
    raise ExternalNewsImportError(
        f"Unsupported external label file type: {source} (expected JSONL or CSV)"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> str:
    text = _clean(value).casefold()
    if not text or text in {"none", "null", "nan", "na", "n/a"}:
        raise ExternalNewsImportError("content_sha256 is required")
    # Exported snapshots contain a normal 64-character digest.  Do not reject
    # deterministic fixture or reviewer identifiers used by offline audit
    # exports (the safety property here is exact equality with the snapshot).
    if len(text) == SHA256_LENGTH and any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ExternalNewsImportError("content_sha256 must be hexadecimal when 64 characters long")
    return text


def _validate_snapshot(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ExternalNewsImportError("external input snapshot is empty")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows, start=1):
        row = _normalise_record(raw)
        article_id = _clean(row.get("article_id"))
        if not article_id:
            raise ExternalNewsImportError(f"snapshot row {index} is missing article_id")
        if article_id in by_id:
            raise ExternalNewsImportError(f"duplicate article_id in input snapshot: {article_id}")
        if not isinstance(row.get("content_sha256"), str):
            raise ExternalNewsImportError(f"snapshot row {article_id} has invalid content_sha256 schema")
        content_sha256 = _valid_sha256(row.get("content_sha256"))
        label_text_sha256 = _clean(row.get("label_text_sha256"))
        if label_text_sha256:
            label_text_sha256 = _valid_sha256(label_text_sha256)
        published_at = _clean(row.get("published_at"))
        if not published_at:
            raise ExternalNewsImportError(f"snapshot row {article_id} is missing published_at")
        # datetime.fromisoformat handles the ISO-8601 values emitted by the exporter;
        # replace Z for Python versions where it is not accepted.
        try:
            timestamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ExternalNewsImportError(
                f"snapshot row {article_id} has invalid published_at: {published_at!r}"
            ) from exc
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        row = dict(row)
        row["article_id"] = article_id
        row["content_sha256"] = content_sha256
        if label_text_sha256:
            row["label_text_sha256"] = label_text_sha256
        row["published_at"] = timestamp.astimezone(timezone.utc).isoformat()
        by_id[article_id] = row
    return by_id


def _year_from_timestamp(value: Any, article_id: str) -> int:
    text = _clean(value).replace("Z", "+00:00")
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExternalNewsImportError(
            f"article {article_id} has invalid published_at: {value!r}"
        ) from exc
    return timestamp.year


def split_for_year(year: int, *, allow_backfill: bool = False) -> str:
    """Return the locked chronological split for an archive year."""

    if 2019 <= int(year) <= 2023:
        return "training"
    if int(year) == 2024:
        return "validation"
    if int(year) == 2025:
        return "test"
    if allow_backfill and 2015 <= int(year) <= 2018:
        return "training"
    return ""


def split_for_timestamp(value: Any, *, allow_backfill: bool = False) -> str:
    """Apply the same locked split policy to an ISO timestamp."""

    text = _clean(value).replace("Z", "+00:00")
    try:
        year = datetime.fromisoformat(text).year
    except ValueError:
        return ""
    return split_for_year(year, allow_backfill=allow_backfill)


def _present(record: Mapping[str, Any], names: Sequence[str]) -> bool:
    return any(name in record and _clean(record.get(name)) for name in names)


def _first(record: Mapping[str, Any], names: Sequence[str], default: Any = "") -> Any:
    for name in names:
        if name in record and _clean(record.get(name)):
            return record.get(name)
    return default


def _label_payload(
    record: Mapping[str, Any], *, prefix: str = ""
) -> tuple[dict[str, Any] | None, bool]:
    """Extract one label and whether its fields were explicitly supplied."""

    if prefix:
        variants = {
            "entity_relevant": (
                f"{prefix}entity_relevant",
                f"{prefix}entityRelevant",
                f"{prefix}entity_label",
            ),
            "sentiment": (
                f"{prefix}sentiment",
                f"{prefix}sentiment_label",
                f"{prefix}label",
            ),
            "event_category": (f"{prefix}event_category", f"{prefix}event"),
            "language": (f"{prefix}language",),
            "confidence": (f"{prefix}confidence",),
            "labelled_by": (
                f"{prefix}labelled_by",
                f"{prefix}labeled_by",
                f"{prefix}model",
                f"{prefix}annotator",
                f"{prefix}name",
            ),
            "label_status": (f"{prefix}label_status", f"{prefix}status"),
            "evidence": (f"{prefix}evidence",),
            "rationale": (f"{prefix}rationale",),
            "labelled_at": (f"{prefix}labelled_at", f"{prefix}labeled_at"),
        }
    else:
        variants = {
            "entity_relevant": ("entity_relevant",),
            "sentiment": ("sentiment", "sentiment_label", "label"),
            "event_category": ("event_category", "event"),
            "language": ("language",),
            "confidence": ("confidence",),
            "labelled_by": ("labelled_by", "labeled_by", "annotator", "labeler"),
            "label_status": ("label_status", "status", "review_status"),
            "evidence": ("evidence",),
            "rationale": ("rationale",),
            "labelled_at": ("labelled_at", "labeled_at"),
        }
    supplied = any(_present(record, names) for names in variants.values())
    # A row may carry model metadata (for example ``labeler_a_model``) next to
    # a valid top-level single label.  Treat a prefixed annotator as present
    # only when its entity/sentiment classification columns are present; an
    # error/rationale-only prefix is not a second label.
    if prefix and not (
        any(name in record for name in variants["entity_relevant"])
        or any(name in record for name in variants["sentiment"])
    ):
        supplied = False
    # An explicit prefixed column with a blank value still means that an
    # independent annotator was expected to respond (e.g. consensus abstention).
    if prefix:
        supplied = supplied or (
            any(name in record for name in variants["entity_relevant"])
            or any(name in record for name in variants["sentiment"])
        )
    if not supplied:
        return None, False
    payload = {key: _first(record, names) for key, names in variants.items()}
    payload["_prefix"] = prefix
    return payload, True


def _direct_label_payload(
    record: Mapping[str, Any], token: str
) -> tuple[dict[str, Any] | None, bool]:
    """Extract compact reviewer columns such as ``label_a``/``label_b``."""

    fields = {
        "entity_relevant": (
            f"{token}_entity_relevant",
            f"entity_relevant_{token}",
            f"{token}_entity",
            f"entity_{token}",
        ),
        "sentiment": (
            token,
            f"{token}_sentiment",
            f"sentiment_{token}",
            f"{token}_sentiment_label",
            f"sentiment_label_{token}",
        ),
        "event_category": (f"{token}_event_category", f"event_category_{token}"),
        "language": (f"{token}_language", f"language_{token}"),
        "confidence": (f"{token}_confidence", f"confidence_{token}"),
        "labelled_by": (
            f"{token}_labelled_by",
            f"labelled_by_{token}",
            f"{token}_annotator",
            f"annotator_{token}",
            f"{token}_model",
        ),
        "label_status": (f"{token}_label_status", f"label_status_{token}"),
        "evidence": (f"{token}_evidence", f"evidence_{token}"),
        "rationale": (f"{token}_rationale", f"rationale_{token}"),
        "labelled_at": (f"{token}_labelled_at", f"labelled_at_{token}"),
    }
    supplied = any(
        name in record
        for names in (fields["entity_relevant"], fields["sentiment"])
        for name in names
    )
    if not supplied:
        return None, False
    payload = {key: _first(record, names) for key, names in fields.items()}
    payload["_prefix"] = f"{token}_"
    return payload, True


def _validate_label_payload(
    payload: Mapping[str, Any],
    *,
    article_id: str,
    allow_abstain: bool = False,
    allow_missing_entity: bool = False,
    strict_schema: bool = False,
) -> dict[str, Any]:
    if strict_schema:
        required_fields = (
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
        )
        missing_fields = [
            field
            for field in required_fields
            if field not in payload
        ]
        if missing_fields:
            raise ExternalNewsImportError(
                f"label for {article_id} has invalid schema; missing {missing_fields}"
            )
        type_checks = {
            "entity_relevant": isinstance(payload.get("entity_relevant"), bool),
            "sentiment": payload.get("sentiment") is None
            or isinstance(payload.get("sentiment"), str),
            "event_category": isinstance(payload.get("event_category"), str),
            "language": isinstance(payload.get("language"), str),
            "confidence": isinstance(payload.get("confidence"), (int, float))
            and not isinstance(payload.get("confidence"), bool),
            "evidence": isinstance(payload.get("evidence"), list),
            "rationale": isinstance(payload.get("rationale"), str),
            "label_status": isinstance(payload.get("label_status"), str),
            "labelled_by": isinstance(payload.get("labelled_by"), str),
            "labelled_at": isinstance(payload.get("labelled_at"), str),
        }
        invalid_types = [field for field, valid in type_checks.items() if not valid]
        if invalid_types:
            raise ExternalNewsImportError(
                f"label for {article_id} has invalid schema types: {invalid_types}"
            )
    entity = _optional_bool(payload.get("entity_relevant"), field=f"{article_id}.entity_relevant")
    if entity is None:
        raw_sentiment = _clean(payload.get("sentiment")).casefold()
        if allow_missing_entity and raw_sentiment in SENTIMENT_LABELS:
            entity = True
        else:
            raise ExternalNewsImportError(f"label for {article_id} is missing entity_relevant")
    sentiment = _clean(payload.get("sentiment")).casefold()
    if sentiment in {"", "null", "none", "nan", "na", "n/a"}:
        sentiment_value: str | None = None
    elif sentiment in SENTIMENT_LABELS:
        sentiment_value = sentiment
    else:
        raise ExternalNewsImportError(f"invalid sentiment for {article_id}: {sentiment!r}")
    if entity and sentiment_value is None and not allow_abstain:
        raise ExternalNewsImportError(
            f"relevant label for {article_id} must include negative, neutral, or positive sentiment"
        )
    if not entity and sentiment_value is not None:
        raise ExternalNewsImportError(
            f"irrelevant label for {article_id} must have sentiment=null"
        )
    event_category = _clean(payload.get("event_category")).casefold()
    if event_category and event_category not in EVENT_CATEGORIES:
        raise ExternalNewsImportError(
            f"invalid event_category for {article_id}: {event_category!r}"
        )
    language = _clean(payload.get("language")).casefold()
    if language and language not in LANGUAGES:
        raise ExternalNewsImportError(f"invalid language for {article_id}: {language!r}")
    confidence = _optional_float(payload.get("confidence"), field=f"{article_id}.confidence")
    raw_status = _clean(payload.get("label_status")).casefold()
    if strict_schema and raw_status and raw_status not in ALLOWED_LABEL_STATUS:
        raise ExternalNewsImportError(f"invalid label_status for {article_id}: {raw_status!r}")
    status = raw_status.replace(" ", "_").replace("-", "_") or "labelled"
    if not strict_schema and status in {"approved", "accepted", "complete", "completed"}:
        status = "labelled"
    if status not in ALLOWED_LABEL_STATUS:
        raise ExternalNewsImportError(f"invalid label_status for {article_id}: {status!r}")
    labelled_by = _clean(payload.get("labelled_by"))
    if not labelled_by:
        # Prefix labels in a reviewer CSV can identify the independent label by
        # column prefix even when no model/name column was supplied.
        labelled_by = _clean(payload.get("_prefix")) or "external"
    evidence = _optional_json(payload.get("evidence"), field=f"{article_id}.evidence")
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list) or len(evidence) > 2:
        raise ExternalNewsImportError(f"evidence for {article_id} must be a list of at most two items")
    evidence = [_clean(item) for item in evidence]
    if any(len(item) > 240 for item in evidence):
        raise ExternalNewsImportError(f"evidence for {article_id} contains an item longer than 240 characters")
    rationale = _clean(payload.get("rationale"))
    if len(rationale) > 800:
        raise ExternalNewsImportError(f"rationale for {article_id} exceeds 800 characters")
    labelled_at = _clean(payload.get("labelled_at"))
    if labelled_at:
        try:
            datetime.fromisoformat(labelled_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ExternalNewsImportError(
                f"invalid labelled_at for {article_id}: {labelled_at!r}"
            ) from exc
    if strict_schema:
        empty_required = []
        if not event_category:
            empty_required.append("event_category")
        if not language:
            empty_required.append("language")
        if confidence is None:
            empty_required.append("confidence")
        if not rationale:
            empty_required.append("rationale")
        if not _clean(payload.get("label_status")):
            empty_required.append("label_status")
        if not labelled_by:
            empty_required.append("labelled_by")
        if not labelled_at:
            empty_required.append("labelled_at")
        if empty_required:
            raise ExternalNewsImportError(
                f"label for {article_id} has invalid schema; empty {empty_required}"
            )
    return {
        "entity_relevant": entity,
        "sentiment": sentiment_value,
        "event_category": event_category,
        "language": language,
        "confidence": confidence,
        "evidence": evidence,
        "rationale": rationale,
        "label_status": status,
        "labelled_by": labelled_by,
        "labelled_at": labelled_at,
        "_prefix": _clean(payload.get("_prefix")),
    }


def _extract_labels(
    record: Mapping[str, Any], *, article_id: str, strict_schema: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """Return validated label A, optional label B, and explicit review status."""

    # Existing automated weak-label artifacts use these names; reviewer CSVs
    # commonly use annotator_1/2 or label_a/label_b.
    prefixes = (
        ("labeler_a_", "labeler_b_"),
        ("annotator_1_", "annotator_2_"),
        ("annotator_a_", "annotator_b_"),
        ("coder_a_", "coder_b_"),
        ("label_a_", "label_b_"),
        ("reviewer_a_", "reviewer_b_"),
    )
    first: dict[str, Any] | None = None
    second: dict[str, Any] | None = None
    explicit_status = _clean(
        _first(record, ("label_status", "status", "review_status", "approval_status"))
    ).casefold().replace(" ", "_")
    consensus_status = _clean(record.get("consensus_status")).casefold().replace(" ", "_")
    allow_abstain = explicit_status in {
        "pending",
        "review",
        "needs_review",
        "needs-review",
        "abstained",
    } or consensus_status in {"abstained", "needs_review", "disagreement"}
    direct_a, has_direct_a = _direct_label_payload(record, "label_a")
    direct_b, has_direct_b = _direct_label_payload(record, "label_b")
    if has_direct_a or has_direct_b:
        if has_direct_a and direct_a is not None:
            first = _validate_label_payload(
                direct_a,
                article_id=article_id,
                allow_abstain=allow_abstain,
                allow_missing_entity=True,
            )
        if has_direct_b and direct_b is not None:
            second = _validate_label_payload(
                direct_b,
                article_id=article_id,
                allow_abstain=allow_abstain,
                allow_missing_entity=True,
            )
        if first is None:
            raise ExternalNewsImportError(f"label for {article_id} is missing label_a")
        if explicit_status in {
            "pending",
            "review",
            "needs_review",
            "needs-review",
            "abstained",
        }:
            explicit_status = "needs_review"
        if consensus_status in {"abstained", "needs_review", "disagreement"}:
            explicit_status = "needs_review"
        return first, second, explicit_status
    for first_prefix, second_prefix in prefixes:
        candidate_a, has_a = _label_payload(record, prefix=first_prefix)
        candidate_b, has_b = _label_payload(record, prefix=second_prefix)
        if has_a or has_b:
            if has_a and candidate_a is not None:
                first = _validate_label_payload(
                    candidate_a,
                    article_id=article_id,
                    allow_abstain=allow_abstain,
                    allow_missing_entity=True,
                )
            if has_b and candidate_b is not None:
                second = _validate_label_payload(
                    candidate_b,
                    article_id=article_id,
                    allow_abstain=allow_abstain,
                    allow_missing_entity=True,
                )
            break
    if first is None:
        candidate, has_candidate = _label_payload(record)
        if not has_candidate or candidate is None:
            raise ExternalNewsImportError(f"label for {article_id} has invalid schema: no label fields")
        first = _validate_label_payload(
            candidate,
            article_id=article_id,
            allow_abstain=allow_abstain,
            strict_schema=strict_schema,
        )
    if explicit_status.replace(" ", "_") in {
        "pending",
        "review",
        "needs_review",
        "needs-review",
        "abstained",
    }:
        explicit_status = "needs_review"
    if _clean(record.get("consensus_status")).casefold().replace(" ", "_") in {
        "abstained",
        "needs_review",
        "disagreement",
    }:
        explicit_status = "needs_review"
    return first, second, explicit_status


def _content_hash_from_output(record: Mapping[str, Any]) -> str:
    raw = _first(record, ("content_sha256", "content_hash", "contentHash"))
    return _valid_sha256(raw)


def _label_text_hash_from_output(record: Mapping[str, Any]) -> str:
    raw = _first(record, ("label_text_sha256", "label_text_hash", "labelTextSha256"))
    return _valid_sha256(raw)


def _independent(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    first_id = _clean(first.get("labelled_by")).casefold()
    second_id = _clean(second.get("labelled_by")).casefold()
    # Prefix-derived identifiers (labeler_a_/labeler_b_) are intentionally
    # distinct, while two rows claiming the same model/name are not independent.
    return bool(first_id and second_id and first_id != second_id)


def _labels_agree(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        bool(first.get("entity_relevant")) == bool(second.get("entity_relevant"))
        and first.get("sentiment") == second.get("sentiment")
        and (
            not first.get("event_category")
            or not second.get("event_category")
            or first.get("event_category") == second.get("event_category")
        )
    )


def _review_reason(
    first: Mapping[str, Any], second: Mapping[str, Any] | None, explicit_status: str
) -> str:
    if explicit_status == "needs_review" or first.get("label_status") == "needs_review":
        return "needs_review"
    if second is None:
        return "needs_review"
    if not _independent(first, second):
        return "labels_not_independent"
    if not _labels_agree(first, second):
        return "label_disagreement"
    if second.get("label_status") == "needs_review":
        return "needs_review"
    return "needs_review"


def _metadata_row(
    snapshot: Mapping[str, Any],
    label: Mapping[str, Any],
    *,
    source: str,
    second: Mapping[str, Any] | None,
    raw_label: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(snapshot)
    original_source = source
    source = LLM_ASSISTED_LABEL_SOURCE
    row["article_id"] = _clean(snapshot.get("article_id"))
    row["content_sha256"] = _clean(snapshot.get("content_sha256")).casefold()
    row["entity_relevant"] = bool(label.get("entity_relevant"))
    row["split_name"] = "train" if row.get("split") == "training" else row.get("split", "")
    row["automatic_entity_relevant"] = bool(label.get("entity_relevant"))
    row["sentiment"] = label.get("sentiment")
    row["sentiment_label"] = label.get("sentiment") or ""
    row["event_category"] = label.get("event_category", "")
    row["language"] = label.get("language", "")
    row["label_source"] = source
    row["label_provenance"] = source
    row["provenance"] = source
    row["annotation_agreement"] = source
    row["label_source_original"] = original_source
    row["consensus_status"] = source
    row["consensus_reason"] = (
        "LLM-assisted review of independent model labels"
        if second is not None
        else "LLM-assisted review of supplied label"
    )
    row["validation_status"] = source
    row["llm_assisted_label"] = True
    row["llm_assisted_label_at"] = label.get("labelled_at", "")
    row["labelled_at"] = label.get("labelled_at", "")
    row["external_labelled_by"] = label.get("labelled_by", "")
    row["external_confidence"] = label.get("confidence")
    row["external_event_category"] = label.get("event_category", "")
    row["external_language"] = label.get("language", "")
    row["external_evidence"] = label.get("evidence", [])
    row["external_rationale"] = label.get("rationale", "")
    if raw_label is not None:
        for key in (
            "batch_id",
            "batch_sha256",
            "label_text_sha256",
            "prompt_version",
            "prompt_sha256",
            "run_id",
            "label_run_id",
            "output_sha256",
        ):
            if key in raw_label and _clean(raw_label.get(key)):
                row[key] = raw_label[key]
    # ``labeler_a``/``labeler_b`` are reserved for the repository's existing
    # two-provider OpenRouter consensus artefact.  A single external/Codex
    # judgement must not be represented as a fabricated consensus member.
    if original_source == "external_consensus":
        row["labeler_a_entity_relevant"] = label.get("entity_relevant")
        row["labeler_a_sentiment"] = label.get("sentiment") or ""
        row["labeler_a_confidence"] = label.get("confidence")
        row["labeler_a_model"] = label.get("labelled_by", "")
        row["labeler_a_event_category"] = label.get("event_category", "")
        row["labeler_a_language"] = label.get("language", "")
    else:
        row["labeler_a_entity_relevant"] = ""
        row["labeler_a_sentiment"] = ""
        row["labeler_a_confidence"] = ""
        row["labeler_a_model"] = ""
        row["labeler_a_event_category"] = ""
        row["labeler_a_language"] = ""
    if second is not None:
        row["labeler_b_entity_relevant"] = second.get("entity_relevant")
        row["labeler_b_sentiment"] = second.get("sentiment") or ""
        row["labeler_b_confidence"] = second.get("confidence")
        row["labeler_b_model"] = second.get("labelled_by", "")
        row["labeler_b_event_category"] = second.get("event_category", "")
        row["labeler_b_language"] = second.get("language", "")
    else:
        row["labeler_b_entity_relevant"] = ""
        row["labeler_b_sentiment"] = ""
        row["labeler_b_confidence"] = ""
        row["labeler_b_model"] = ""
        row["labeler_b_event_category"] = ""
        row["labeler_b_language"] = ""
    row["confidence"] = label.get("confidence")
    row["evidence"] = label.get("evidence", [])
    row["rationale"] = label.get("rationale", "")
    row["external_label_status"] = label.get("label_status", "labelled")
    # The export package calls the canonical text URL ``source_url`` but the
    # internal dataset convention also has ``url``.  Keep both when available.
    if "source_url" not in row and "url" in row:
        row["source_url"] = row["url"]
    if "url" not in row and "source_url" in row:
        row["url"] = row["source_url"]
    if "model_text" not in row:
        title = _clean(row.get("title"))
        body = _clean(row.get("article_text") or row.get("description"))
        row["model_text"] = ". ".join(value for value in (title, body) if value)
    return row


def import_external_news_labels(
    input_snapshot: Path | str,
    returned_labels: Path | str,
    *,
    output_dataset: Path | str = DEFAULT_EXPERIMENTAL_DATASET_PATH,
    review_queue: Path | str | None = None,
    report_path: Path | str | None = None,
    allow_backfill: bool = False,
) -> dict[str, Any]:
    """Import a complete external label response into an experimental dataset.

    ``returned_labels`` may be the JSONL response from the external/Codex
    labeller or a reviewer CSV.  Every eligible snapshot row must occur exactly
    once and must carry the same SHA-256 content identity.  Invalid schema or
    identity mismatches abort before any output is written.
    """

    snapshot_rows = _read_records(input_snapshot)
    snapshot = _validate_snapshot(snapshot_rows)
    label_rows = _read_records(returned_labels)
    if not label_rows:
        raise ExternalNewsImportError("returned external label file is empty")

    output_path = Path(output_dataset)
    strict_return_schema = Path(returned_labels).suffix.casefold() in {".jsonl", ".ndjson"}
    if output_path.resolve() == PRODUCTION_DATASET_PATH.resolve():
        raise ExternalNewsImportError(
            "external labels must be written to a separate experimental dataset, not production"
        )
    queue_path = (
        Path(review_queue)
        if review_queue is not None
        else (
            DEFAULT_REVIEW_QUEUE_PATH
            if output_path == DEFAULT_EXPERIMENTAL_DATASET_PATH
            else output_path.with_name(f"{output_path.stem}-review-queue.jsonl")
        )
    )
    report_output = (
        Path(report_path)
        if report_path is not None
        else (
            DEFAULT_IMPORT_REPORT_PATH
            if output_path == DEFAULT_EXPERIMENTAL_DATASET_PATH
            else output_path.with_name(f"{output_path.stem}-import-report.json")
        )
    )
    source_paths = {Path(input_snapshot).resolve(), Path(returned_labels).resolve()}
    if output_path.resolve() in source_paths:
        raise ExternalNewsImportError(
            "output dataset must not overwrite the input snapshot or returned labels"
        )
    if queue_path.resolve() in source_paths or report_output.resolve() in source_paths:
        raise ExternalNewsImportError(
            "review queue and import report must not overwrite the input files"
        )
    if output_path.resolve() in {queue_path.resolve(), report_output.resolve()}:
        raise ExternalNewsImportError(
            "output dataset, review queue, and import report must be different files"
        )

    # Filter the snapshot before ID coverage checks.  2015-2018 are intentionally
    # outside the experimental corpus unless the caller opts into backfill.
    eligible: dict[str, dict[str, Any]] = {}
    excluded_ids: set[str] = set()
    for article_id, row in snapshot.items():
        year = _year_from_timestamp(row.get("published_at"), article_id)
        split = split_for_year(year, allow_backfill=allow_backfill)
        if split:
            row = dict(row)
            row["split"] = split
            eligible[article_id] = row
        else:
            excluded_ids.add(article_id)

    returned_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(label_rows, start=1):
        record = _normalise_record(raw)
        article_id = _clean(record.get("article_id"))
        if not article_id:
            raise ExternalNewsImportError(f"returned label row {index} is missing article_id")
        if article_id in returned_by_id:
            raise ExternalNewsImportError(f"duplicate article_id in returned labels: {article_id}")
        if article_id not in snapshot:
            raise ExternalNewsImportError(f"stale/unknown article_id in returned labels: {article_id}")
        if strict_return_schema and not isinstance(record.get("content_sha256"), str):
            raise ExternalNewsImportError(
                f"label for {article_id} has invalid content_sha256 schema"
            )
        returned_hash = _content_hash_from_output(record)
        expected_hash = _clean(snapshot[article_id].get("content_sha256")).casefold()
        if returned_hash != expected_hash:
            raise ExternalNewsImportError(
                f"content_sha256 mismatch for article_id {article_id}: expected {expected_hash}, got {returned_hash}"
            )
        expected_label_text_hash = _clean(snapshot[article_id].get("label_text_sha256")).casefold()
        if expected_label_text_hash:
            if strict_return_schema and not isinstance(record.get("label_text_sha256"), str):
                raise ExternalNewsImportError(
                    f"label for {article_id} has invalid label_text_sha256 schema"
                )
            returned_label_text_hash = _label_text_hash_from_output(record)
            if returned_label_text_hash != expected_label_text_hash:
                raise ExternalNewsImportError(
                    f"label_text_sha256 mismatch for article_id {article_id}: "
                    f"expected {expected_label_text_hash}, got {returned_label_text_hash}"
                )
        expected_batch_hash = _clean(snapshot[article_id].get("batch_sha256")).casefold()
        if expected_batch_hash:
            returned_batch_hash = _clean(
                _first(record, ("batch_sha256", "batch_hash"))
            ).casefold()
            if returned_batch_hash != expected_batch_hash:
                raise ExternalNewsImportError(
                    f"batch_sha256 mismatch for article_id {article_id}: "
                    f"expected {expected_batch_hash}, got {returned_batch_hash}"
                )
        # Validate the label schema before coverage checks, so malformed rows are
        # never hidden by a missing/stale ID report.
        _extract_labels(
            record,
            article_id=article_id,
            strict_schema=strict_return_schema,
        )
        returned_by_id[article_id] = record

    missing = sorted(set(eligible) - set(returned_by_id))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ExternalNewsImportError(
            f"missing labels for {len(missing)} eligible article IDs: {preview}{suffix}"
        )

    dataset_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    counts = {
        "snapshot_rows": len(snapshot),
        "eligible_rows": len(eligible),
        "excluded_backfill_rows": len(excluded_ids),
        "excluded_rows": len(excluded_ids),
        "returned_rows": len(label_rows),
        "accepted_rows": 0,
        "review_rows": 0,
        "consensus_rows": 0,
        "single_label_rows": 0,
    }
    for article_id in sorted(eligible, key=lambda key: (eligible[key]["published_at"], key)):
        snapshot_row = eligible[article_id]
        label_record = returned_by_id[article_id]
        first, second, explicit_status = _extract_labels(
            label_record,
            article_id=article_id,
            strict_schema=strict_return_schema,
        )
        accepted = second is None or (
            _independent(first, second)
            and _labels_agree(first, second)
            and first.get("label_status") == "labelled"
            and second.get("label_status") == "labelled"
            and explicit_status != "needs_review"
        )
        if second is not None and accepted:
            source = "external_consensus"
            selected = first
            counts["consensus_rows"] += 1
        elif second is None and first.get("label_status") == "labelled" and explicit_status != "needs_review":
            source = "external_single_weak"
            selected = first
            counts["single_label_rows"] += 1
            accepted = True
        else:
            reason = _review_reason(first, second, explicit_status)
            queue_row = dict(snapshot_row)
            queue_row.update(
                {
                    "review_reason": reason,
                    "label_record": dict(label_record),
                    "label_a": dict(first),
                    "label_b": dict(second) if second is not None else None,
                }
            )
            queue_rows.append(queue_row)
            counts["review_rows"] += 1
            continue
        dataset_rows.append(
            _metadata_row(
                snapshot_row,
                selected,
                source=source,
                second=second,
                raw_label=label_record,
            )
        )
        counts["accepted_rows"] += 1

    def _count_field(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for row in rows:
            value = _clean(row.get(field)) or "unknown"
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    eligible_by_ticker = _count_field(list(eligible.values()), "ticker")
    accepted_by_year: dict[str, int] = {}
    for row in dataset_rows:
        year = str(
            datetime.fromisoformat(
                _clean(row.get("published_at")).replace("Z", "+00:00")
            ).year
        )
        accepted_by_year[year] = accepted_by_year.get(year, 0) + 1
    accepted_by_year = dict(sorted(accepted_by_year.items()))
    accepted_confidences = [
        float(row["confidence"])
        for row in dataset_rows
        if isinstance(row.get("confidence"), (int, float))
        and not isinstance(row.get("confidence"), bool)
    ]
    confidence_summary: dict[str, Any] = {"count": len(accepted_confidences)}
    if accepted_confidences:
        confidence_summary.update(
            {
                "mean": sum(accepted_confidences) / len(accepted_confidences),
                "min": min(accepted_confidences),
                "max": max(accepted_confidences),
            }
        )
    review_by_reason = _count_field(queue_rows, "review_reason")

    _write_jsonl(output_path, dataset_rows)
    _write_jsonl(queue_path, queue_rows)
    report: dict[str, Any] = {
        "schema_version": "external-news-import-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_snapshot": str(Path(input_snapshot)),
        "returned_labels": str(Path(returned_labels)),
        "input_snapshot_sha256": _file_sha256(input_snapshot),
        "returned_labels_sha256": _file_sha256(returned_labels),
        "output_dataset": str(output_path),
        "review_queue": str(queue_path),
        "allow_backfill": bool(allow_backfill),
        "chronological_splits": {
            "training": "2019-2023" + (" plus approved 2015-2018 backfill" if allow_backfill else ""),
            "train": "2019-2023" + (" plus approved 2015-2018 backfill" if allow_backfill else ""),
            "validation": "2024",
            "test": "2025",
        },
        **counts,
        "experimental_only": False,
        "label_policy": "llm_assisted_label",
        "coverage": {
            "eligible_by_ticker": eligible_by_ticker,
            "accepted_by_year": accepted_by_year,
            "accepted_by_ticker": _count_field(dataset_rows, "ticker"),
            "accepted_by_sentiment": _count_field(dataset_rows, "sentiment"),
            "accepted_by_entity_relevance": {
                "relevant": sum(bool(row.get("entity_relevant")) for row in dataset_rows),
                "irrelevant": sum(not bool(row.get("entity_relevant")) for row in dataset_rows),
            },
            "accepted_by_label_source": _count_field(dataset_rows, "label_source"),
            "review_by_reason": review_by_reason,
            "confidence": confidence_summary,
        },
        "production_changes": False,
    }
    _write_json(report_output, report)
    report["report_path"] = str(report_output)
    return report


def import_external_label_outputs(
    input_snapshot: Path | str,
    returned_labels: Path | str,
    *,
    output_dir: Path | str | None = None,
    output_dataset: Path | str = DEFAULT_EXPERIMENTAL_DATASET_PATH,
    output: Path | str | None = None,
    review_queue: Path | str | None = None,
    report_path: Path | str | None = None,
    allow_backfill: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper accepting an output directory as well as paths."""

    if output is not None:
        output_dataset = output
    if output_dir is not None:
        directory = Path(output_dir)
        if Path(output_dataset) == DEFAULT_EXPERIMENTAL_DATASET_PATH:
            output_dataset = directory / "external-news-experimental.jsonl"
        if review_queue is None:
            review_queue = directory / "external-news-review-queue.jsonl"
        if report_path is None:
            report_path = directory / "external-news-import-report.json"
    return import_external_news_labels(
        input_snapshot,
        returned_labels,
        output_dataset=output_dataset,
        review_queue=review_queue,
        report_path=report_path,
        allow_backfill=allow_backfill,
    )


# Small aliases make the entry point discoverable from scripts that use the
# export module's naming convention.
import_external_label_package = import_external_label_outputs
import_external_labels = import_external_label_outputs
import_external_label_output = import_external_label_outputs
import_external_label_results = import_external_label_outputs
import_external_news_label_results = import_external_label_outputs
run_external_news_label_import = import_external_label_outputs
import_labels = import_external_label_outputs
read_external_records = _read_records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import offline external Bursa-news labels into an experimental dataset.")
    parser.add_argument("--input-snapshot", required=True)
    parser.add_argument("--returned-labels", required=True)
    parser.add_argument("--output-dataset", default=str(DEFAULT_EXPERIMENTAL_DATASET_PATH))
    parser.add_argument("--review-queue", default=None)
    parser.add_argument("--report", dest="report_path", default=None)
    parser.add_argument("--allow-backfill", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = import_external_news_labels(
            args.input_snapshot,
            args.returned_labels,
            output_dataset=args.output_dataset,
            review_queue=args.review_queue,
            report_path=args.report_path,
            allow_backfill=args.allow_backfill,
        )
    except (ExternalNewsImportError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
