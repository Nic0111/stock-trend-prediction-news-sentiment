"""Create auditable Codex-reviewed Bursa-news CSV outputs.

This offline utility reads a frozen review package and a separately stored
decision file, fills only the annotation columns, then combines the reviewed
rows with an existing validated CSV. It never touches the canonical dataset,
models, registry, or runtime configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANNOTATION_COLUMNS = (
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def compact_title(title: str) -> str:
    text = " ".join(title.split())
    return text[:240] if text else "Article text reviewed."


def rationale(company: str, *, relevant: bool, sentiment: str | None) -> str:
    if not relevant:
        return (
            "Codex review: the article is not materially about "
            f"{company}; the retrieved name is incidental, a different entity, "
            "or broad market context."
        )
    wording = {
        "positive": "a likely favourable financial or operating implication",
        "negative": "a likely adverse financial or operating implication",
        "neutral": "no clearly directional financial implication",
    }[str(sentiment)]
    return (
        "Codex review: the article directly describes a company-specific "
        f"development with {wording}."
    )


def apply_decisions(
    rows: list[dict[str, str]], decisions: dict[str, Any], labelled_at: str
) -> list[dict[str, str]]:
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_decisions, dict):
        raise ValueError("decision file is missing its decisions object")
    expected = {str(index) for index in range(1, len(rows) + 1)}
    actual = set(raw_decisions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"decision coverage mismatch; missing={missing[:5]} extra={extra[:5]}")
    reviewed: list[dict[str, str]] = []
    for index, original in enumerate(rows, start=1):
        row = dict(original)
        decision = raw_decisions[str(index)]
        relevant = decision.get("entity_relevant")
        sentiment = decision.get("sentiment")
        category = decision.get("event_category")
        confidence = decision.get("confidence")
        if not isinstance(relevant, bool):
            raise ValueError(f"decision {index} has no boolean entity_relevant")
        if relevant and sentiment not in {"negative", "neutral", "positive"}:
            raise ValueError(f"decision {index} has invalid relevant sentiment")
        if not relevant and sentiment is not None:
            raise ValueError(f"decision {index} must use null sentiment when irrelevant")
        if not isinstance(category, str) or not category:
            raise ValueError(f"decision {index} has no event category")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"decision {index} has invalid confidence")
        row.update(
            {
                "entity_relevant": str(relevant).lower(),
                "sentiment": sentiment or "",
                "event_category": category,
                "language": "en",
                "confidence": f"{float(confidence):.2f}",
                "evidence": json.dumps([compact_title(row.get("title", ""))]),
                "rationale": rationale(
                    row.get("company_name", "the named company"),
                    relevant=relevant,
                    sentiment=sentiment,
                ),
                "label_status": "labelled",
                "labelled_by": "codex_full_revalidation",
                "labelled_at": labelled_at,
            }
        )
        reviewed.append(row)
    return reviewed


def assert_unique(rows: list[dict[str, str]], *, label: str) -> None:
    seen_ids: set[str] = set()
    seen_hashes: set[tuple[str, str]] = set()
    for row in rows:
        article_id = row.get("article_id", "").strip()
        if not article_id or article_id in seen_ids:
            raise ValueError(f"{label} contains duplicate or missing article_id: {article_id}")
        seen_ids.add(article_id)
        content_hash = row.get("content_sha256", "").strip().casefold()
        ticker = row.get("ticker", "").strip().upper()
        if content_hash:
            key = (ticker, content_hash)
            if key in seen_hashes:
                raise ValueError(f"{label} contains duplicate ticker/content hash: {key}")
            seen_hashes.add(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--validated-pack", required=True, type=Path)
    parser.add_argument("--labelled-output", required=True, type=Path)
    parser.add_argument("--combined-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    headers, review_rows = read_csv(args.review_pack)
    validated_headers, validated_rows = read_csv(args.validated_pack)
    if headers != validated_headers:
        raise ValueError("input CSV headers differ; refusing to combine mismatched schemas")
    if any(column not in headers for column in ANNOTATION_COLUMNS):
        raise ValueError("review-pack does not contain the required annotation columns")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    expected_hash = str(decisions.get("source_sha256", "")).casefold()
    observed_hash = sha256_file(args.review_pack)
    if observed_hash != expected_hash:
        raise ValueError(
            "review-pack checksum differs from the reviewed snapshot; "
            "regenerate decisions for the current input"
        )
    labelled_at = datetime.now(timezone.utc).isoformat()
    reviewed = apply_decisions(review_rows, decisions, labelled_at)
    assert_unique(validated_rows, label="validated pack")
    assert_unique(reviewed, label="Codex-reviewed pack")
    overlap = {row["article_id"] for row in validated_rows}.intersection(
        row["article_id"] for row in reviewed
    )
    if overlap:
        raise ValueError(f"input packs overlap by article_id: {sorted(overlap)[:3]}")
    combined = sorted(
        [*validated_rows, *reviewed],
        key=lambda row: (row.get("published_at", ""), row.get("ticker", ""), row.get("article_id", "")),
    )
    assert_unique(combined, label="combined pack")
    write_csv(args.labelled_output, headers, reviewed)
    write_csv(args.combined_output, headers, combined)
    relevant = [row for row in reviewed if row["entity_relevant"] == "true"]
    report = {
        "schema_version": "stocksense-codex-review-output-v1",
        "created_at": labelled_at,
        "review_pack": str(args.review_pack),
        "review_pack_sha256": observed_hash,
        "decisions": str(args.decisions),
        "validated_pack": str(args.validated_pack),
        "reviewed_rows": len(reviewed),
        "reviewed_relevant_rows": len(relevant),
        "reviewed_irrelevant_rows": len(reviewed) - len(relevant),
        "reviewed_sentiment": dict(Counter(row["sentiment"] for row in relevant)),
        "validated_rows": len(validated_rows),
        "combined_rows": len(combined),
        "production_changes": False,
        "provenance_note": decisions.get("provenance_note", ""),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
