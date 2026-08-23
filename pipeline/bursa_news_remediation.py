"""Audit Bursa sentiment labels and export a safe expansion queue.

This module is intentionally offline and non-production.  It never edits the
canonical dataset and never assigns a sentiment label.  It reports class and
company coverage, identifies provenance/quality gaps, and prepares recovered
but unlabeled records for an external LLM-assisted label workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MASTER_DATASET = PROJECT_ROOT / "data" / "news-evidence-benchmark" / "bursa-news-llm-assisted-label.jsonl"
DEFAULT_REVIEW_PACKS = (
    PROJECT_ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2" / "codex-label-package" / "cycle-001" / "review-pack.csv",
    PROJECT_ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2-retry-unrecovered-v1" / "codex-label-package" / "review-pack-appended.csv",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "news-evidence-benchmark" / "remediation"
VALID_SENTIMENTS = ("negative", "neutral", "positive")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _normalise(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _truthy(value: Any) -> bool:
    return _normalise(value).casefold() in {"1", "true", "yes", "y", "t"}


def _label(row: Mapping[str, Any]) -> str:
    for name in ("sentiment_label", "sentiment", "external_sentiment"):
        value = _normalise(row.get(name)).casefold()
        if value in VALID_SENTIMENTS:
            return value
    return ""


def _key(row: Mapping[str, Any]) -> str:
    return _normalise(row.get("article_id")) or _normalise(row.get("content_sha256")) or _normalise(row.get("source_url"))


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    values = frame[column].fillna("").astype(str).str.strip()
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).to_dict().items()}


def _class_table(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {label: 0 for label in VALID_SENTIMENTS}
    values = frame["_label"]
    return {label: int((values == label).sum()) for label in VALID_SENTIMENTS}


def audit_bursa_news_labels(
    master_path: Path = DEFAULT_MASTER_DATASET,
    *,
    review_packs: Sequence[Path] = DEFAULT_REVIEW_PACKS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Write deterministic label-quality and company-coverage diagnostics."""

    master = _read_table(Path(master_path)).copy()
    master["_label"] = master.apply(_label, axis=1)
    relevant = master[master.get("entity_relevant", True).map(_truthy) if "entity_relevant" in master else pd.Series(True, index=master.index)]
    labelled = relevant[relevant["_label"].isin(VALID_SENTIMENTS)]
    master["_year"] = pd.to_datetime(master.get("published_at"), utc=True, errors="coerce").dt.year

    company_rows: dict[str, Any] = {}
    for ticker, group in labelled.groupby(labelled["ticker"].fillna("").astype(str)):
        classes = _class_table(group)
        company_rows[str(ticker)] = {
            "rows": int(len(group)),
            "classes": classes,
            "minimum_class_count": int(min(classes.values()) if classes else 0),
            "neutral_share": float(classes["neutral"] / len(group)) if len(group) else 0.0,
            "years": _counts(group, "_year"),
        }

    provenance = {}
    for column in ("validation_status", "label_source", "label_provenance", "text_source", "reconstruction_quality_tier"):
        provenance[column] = _counts(labelled, column)
    missing = {
        "article_text": int(master.get("article_text", pd.Series("", index=master.index)).map(_normalise).eq("").sum()),
        "model_text": int(master.get("model_text", pd.Series("", index=master.index)).map(_normalise).eq("").sum()),
        "confidence": int(master.get("confidence", pd.Series("", index=master.index)).map(_normalise).eq("").sum()),
        "evidence": int(master.get("evidence", pd.Series("", index=master.index)).map(_normalise).eq("").sum()),
        "rationale": int(master.get("rationale", pd.Series("", index=master.index)).map(_normalise).eq("").sum()),
    }

    review_summary: list[dict[str, Any]] = []
    seen_keys = {_key(row) for row in master.to_dict("records") if _key(row)}
    unlabelled_keys: set[str] = set()
    for path in review_packs:
        source = Path(path)
        if not source.exists():
            review_summary.append({"path": str(source), "exists": False})
            continue
        frame = _read_table(source).copy()
        frame["_label"] = frame.apply(_label, axis=1)
        keys = frame.apply(_key, axis=1)
        labelled_rows = frame[frame["_label"].isin(VALID_SENTIMENTS)]
        unlabelled_rows = frame[~frame["_label"].isin(VALID_SENTIMENTS)]
        unlabelled_keys.update(str(value) for value in unlabelled_rows.apply(_key, axis=1) if str(value))
        review_summary.append(
            {
                "path": str(source),
                "exists": True,
                "rows": int(len(frame)),
                "labelled_rows": int(len(labelled_rows)),
                "unlabelled_rows": int(len(unlabelled_rows)),
                "duplicate_keys_in_master": int(sum(str(value) in seen_keys for value in keys if str(value))),
                "class_counts": _class_table(labelled_rows),
                "tickers": _counts(frame, "ticker"),
            }
        )

    report = {
        "schema_version": "bursa-news-label-remediation-v1",
        "master_path": str(Path(master_path)),
        "master_rows": int(len(master)),
        "relevant_rows": int(len(relevant)),
        "relevant_labelled_rows": int(len(labelled)),
        "class_counts": _class_table(labelled),
        "class_shares": {
            label: float(_class_table(labelled)[label] / len(labelled)) if len(labelled) else 0.0
            for label in VALID_SENTIMENTS
        },
        "year_counts": _counts(labelled, "_year"),
        "company_coverage": company_rows,
        "provenance_counts": provenance,
        "missing_quality_fields": missing,
        "review_packs": review_summary,
        "unlabelled_review_keys": int(len(unlabelled_keys)),
        "recommendations": [
            "Prioritise companies with the lowest minimum class count and neutral share below 0.20.",
            "Label only entity-relevant, English, recovered articles; keep unrelated rows out of training.",
            "Keep independent/prospective rows outside the 2015-2025 retrospective test until they are sealed and reviewed.",
            "Do not treat automatic model predictions as ground-truth labels.",
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label-quality-audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def export_label_expansion_queue(
    review_packs: Sequence[Path] = DEFAULT_REVIEW_PACKS,
    *,
    master_path: Path = DEFAULT_MASTER_DATASET,
    output_path: Path = DEFAULT_OUTPUT_DIR / "label-expansion-queue.csv",
) -> dict[str, Any]:
    """Export unlabeled recovered records without changing canonical data."""

    master = _read_table(Path(master_path))
    existing = {_key(row) for row in master.to_dict("records") if _key(row)}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in review_packs:
        source = Path(path)
        if not source.exists():
            continue
        frame = _read_table(source)
        for row in frame.to_dict("records"):
            key = _key(row)
            if not key or key in existing or key in seen or _label(row) in VALID_SENTIMENTS:
                continue
            # Keep a compact, review-friendly contract and preserve recovery
            # provenance for the eventual import step.
            rows.append(
                {
                    "article_id": _normalise(row.get("article_id")),
                    "ticker": _normalise(row.get("ticker")),
                    "company_name": _normalise(row.get("company_name")),
                    "published_at": _normalise(row.get("published_at")),
                    "title": _normalise(row.get("title")),
                    "source_domain": _normalise(row.get("source_domain")),
                    "source_url": _normalise(row.get("source_url")),
                    "text_source": _normalise(row.get("text_source")),
                    "is_pseudo_document": _truthy(row.get("is_pseudo_document")),
                    "recovery_status": _normalise(row.get("recovery_status")),
                    "reconstruction_quality_tier": _normalise(row.get("reconstruction_quality_tier")),
                    "article_text": _normalise(row.get("article_text")) or _normalise(row.get("label_text")),
                    "reviewer_entity_relevant": "",
                    "reviewer_sentiment": "",
                    "reviewer_confidence": "",
                    "reviewer_event_category": "",
                    "reviewer_rationale": "",
                    "review_status": "pending_llm_assisted_label",
                }
            )
            seen.add(key)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return {"output_path": str(output_path), "rows": int(len(rows)), "canonical_unchanged": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--master", default=str(DEFAULT_MASTER_DATASET))
    audit.add_argument("--review-pack", action="append", default=None)
    audit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    queue = subparsers.add_parser("export-queue")
    queue.add_argument("--master", default=str(DEFAULT_MASTER_DATASET))
    queue.add_argument("--review-pack", action="append", default=None)
    queue.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "label-expansion-queue.csv"))
    args = parser.parse_args(argv)
    packs = tuple(Path(value) for value in args.review_pack) if args.review_pack else DEFAULT_REVIEW_PACKS
    if args.command == "audit":
        print(json.dumps(audit_bursa_news_labels(Path(args.master), review_packs=packs, output_dir=Path(args.output_dir)), indent=2, default=str))
    else:
        print(json.dumps(export_label_expansion_queue(packs, master_path=Path(args.master), output_path=Path(args.output)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
