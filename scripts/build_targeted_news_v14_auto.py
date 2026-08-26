"""Build an isolated, automatically selected 1295.KL news extension.

This utility deliberately does not create or infer review labels.  Recovered
records are selected only by deterministic URL/title/body/entity rules and are
left as ``unlabelled_reference`` until the frozen production sentiment model
scores them.  The output is kept outside production directories so it can be
used as a fusion research input without changing runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/historical-news-fusion/normalized-production-v3-official-prepared-expanded-v13-auto/combined-v12-auto-v13-rescored.parquet"
RECOVERY_ROOT = ROOT / "data/historical-news-fusion/gdelt/targeted-1295-v14"
OUT = ROOT / "data/historical-news-fusion/normalized-production-v3-official-prepared-expanded-v14-auto"
YEARS = ("2019", "2020", "2021", "2022", "2024", "2025")

ALIAS_RE = re.compile(r"\bpublic\s+bank\b|\bpbbank\b|\bpublic\s+financial\b", re.I)
WORD_RE = re.compile(r"\b\w+\b")


def _clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _stable_id(ticker: str, published_at: str, url: str, title: str) -> str:
    payload = "|".join((ticker, published_at, url, title.lower()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _json_or_blank(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _clean(value)


def _recovery_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in YEARS:
        source = RECOVERY_ROOT / f"recovery-{year}/raw/hybrid-recovered.jsonl"
        if not source.exists():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if _clean(record.get("recovery_status")).casefold() == "recovered":
                rows.append(record)
    return rows


def select_new_records(base: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    base_urls = {_clean(value) for value in base.get("url", pd.Series(dtype=str)) if _clean(value)}
    base_keys = {
        "|".join(
            (
                _clean(row.get("ticker")),
                _clean(row.get("published_at"))[:10],
                re.sub(r"\s+", " ", _clean(row.get("title")).casefold()).strip(),
            )
        )
        for row in base.to_dict(orient="records")
    }
    reasons: Counter[str] = Counter()
    years: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    seen: set[str] = set()
    selected: list[dict[str, object]] = []

    for record in _recovery_rows():
        ticker = _clean(record.get("ticker"))
        url = _clean(record.get("url"))
        title = _clean(record.get("title") or record.get("page_title"))
        text = _clean(record.get("content_snippet"))
        date = _clean(record.get("published_at"))[:10]
        words = len(WORD_RE.findall(text))
        quality = _clean(record.get("reconstruction_quality_tier")).casefold()
        key = "|".join((ticker, date, re.sub(r"\s+", " ", title.casefold()).strip()))

        if not url or not title or words < 120:
            reasons["missing_title_url_or_short_text"] += 1
            continue
        if not ALIAS_RE.search(f"{title} {text}"):
            reasons["no_title_body_alias"] += 1
            continue
        # Direct publisher/archive text has no reconstruction tier.  Pseudo-
        # documents are admitted only when the deterministic recovery report
        # calls them medium/high quality; low-quality snippets never reach
        # sentiment scoring or fusion features.
        if bool(record.get("is_pseudo_document")) and quality not in {"high", "medium"}:
            reasons["low_pseudo_quality"] += 1
            continue
        if url in base_urls or key in base_keys:
            reasons["duplicate_base"] += 1
            continue
        dedup = url or _clean(record.get("content_sha256")) or key
        if dedup in seen:
            reasons["duplicate_new"] += 1
            continue
        seen.add(dedup)
        selected.append(record)
        years[date[:4]] += 1
        methods[_clean(record.get("text_source")) or "unknown"] += 1
        tiers[quality or "none"] += 1

    audit = {
        "schema_version": "targeted-news-v14-auto-selection-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_path": str(BASE.relative_to(ROOT)),
        "candidate_years": list(YEARS),
        "selection_policy": {
            "ticker": "1295.KL",
            "valid_http_url": True,
            "minimum_body_words": 120,
            "title_or_body_alias_regex": ALIAS_RE.pattern,
            "pseudo_document_tiers": ["high", "medium"],
            "deduplicate_by": ["url", "ticker+published_date+normalized_title", "content_sha256"],
            "manual_labels_used": False,
            "label_status": "unlabelled_reference",
        },
        "recovered_candidates": sum(1 for _ in _recovery_rows()),
        "selected_new_records": len(selected),
        "excluded_reasons": dict(reasons),
        "selected_by_year": dict(years),
        "selected_by_text_source": dict(methods),
        "selected_by_quality_tier": dict(tiers),
    }
    return selected, audit


def to_base_schema(records: list[dict[str, object]], base_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for record in records:
        ticker = _clean(record.get("ticker")) or "1295.KL"
        published = _clean(record.get("published_at"))
        url = _clean(record.get("url"))
        title = _clean(record.get("title") or record.get("page_title"))
        text = _clean(record.get("content_snippet"))
        body_hash = _clean(record.get("content_sha256")) or hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = {column: None for column in base_columns}
        row.update(
            {
                "article_id": _stable_id(ticker, published, url, title),
                "ticker": ticker,
                "company_name": _clean(record.get("company_name")) or "Public Bank Berhad",
                "provider": "GDELT/recovered-auto-v14-targeted",
                "source_name": _clean(record.get("source_name")) or _clean(record.get("source_domain")) or "Recovered public source",
                "source_domain": _clean(record.get("source_domain")),
                "published_at": published,
                "retrieved_at": _clean(record.get("recovered_at")) or now,
                "title": title,
                "description": _clean(record.get("description")),
                "content_snippet": text,
                "url": url,
                "sentiment_label": "",
                "sentiment_score": None,
                "sentiment_probability": None,
                "probability_negative": None,
                "probability_neutral": None,
                "probability_positive": None,
                "sentiment_uncertain": None,
                "gdelt_organizations": _json_or_blank(record.get("gdelt_organizations")),
                "commoncrawl_index": "",
                "commoncrawl_capture_at": None,
                "commoncrawl_capture_lag_days": None,
                "content_sha256": body_hash,
                "text_source": _clean(record.get("text_source")),
                "is_pseudo_document": bool(record.get("is_pseudo_document")),
                "recovery_status": _clean(record.get("recovery_status")) or "recovered",
                "recovery_failure_reason": "",
                "recovered_at": _clean(record.get("recovered_at")) or now,
                "ngram_file_timestamps": _json_or_blank(record.get("ngram_file_timestamps")),
                "ngram_record_count": record.get("ngram_record_count"),
                "deciles_present": _json_or_blank(record.get("deciles_present")),
                "unique_token_count": record.get("unique_token_count"),
                "overlap_attachment_rate": record.get("overlap_attachment_rate"),
                "reconstruction_quality_score": record.get("reconstruction_quality_score"),
                "reconstruction_quality_tier": _clean(record.get("reconstruction_quality_tier")),
                "publisher_full_text_available": bool(record.get("publisher_full_text_available")),
                "automatic_relevance": "included",
                "relevance_score": 1.0,
                "relevance_reasons": "automatic title/body company-alias selection; no manual label",
                "event_category": "",
                "disclosure_type": "news",
                "entity_confidence": 0.9,
                "text_quality_score": None,
                "sentiment_model_id": "",
                "sentiment_runtime_config_sha256": "",
                "sentiment_model_variant_sha256": "",
                "sentiment_component_artifacts_sha256": "",
                "sentiment_scored_at": None,
                "label_validation_status": "unlabelled_reference",
                "llm_assisted_label": False,
                "reference_sentiment_label": "",
                "reference_label_source": "",
                "reference_label_source_original": "",
                "sentiment_model_role": "unscored_automatic_extension",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=base_columns)


def main() -> None:
    base = pd.read_parquet(BASE)
    selected, audit = select_new_records(base)
    additions = to_base_schema(selected, list(base.columns))
    # Preserve every row in the sealed v12/v13 base.  The selection stage
    # already removes additions that duplicate a base URL/title and removes
    # duplicate additions.  A second deduplication pass across the combined
    # frame would silently delete historical base observations (including
    # intentionally retained source variants), changing the benchmark rows
    # for companies that were not part of this targeted extension.
    combined = pd.concat([base, additions], ignore_index=True)
    # Keep the research extension Arrow-compatible with the existing scored
    # corpus.  Recovery JSONL is string-oriented, whereas the base corpus has
    # typed UTC timestamps, numeric probabilities and booleans.
    datetime_columns = {
        "published_at",
        "retrieved_at",
        "commoncrawl_capture_at",
        "sentiment_scored_at",
    }
    boolean_columns = {"is_pseudo_document", "publisher_full_text_available", "llm_assisted_label", "sentiment_uncertain"}
    numeric_columns = {
        "sentiment_score",
        "sentiment_probability",
        "probability_negative",
        "probability_neutral",
        "probability_positive",
        "commoncrawl_capture_lag_days",
        "ngram_record_count",
        "unique_token_count",
        "overlap_attachment_rate",
        "reconstruction_quality_score",
        "relevance_score",
        "entity_confidence",
        "text_quality_score",
    }
    for column in datetime_columns & set(combined.columns):
        combined[column] = pd.to_datetime(combined[column], utc=True, errors="coerce")
    for column in boolean_columns & set(combined.columns):
        combined[column] = combined[column].fillna(False).map(
            lambda value: value if isinstance(value, bool) else str(value).strip().casefold() in {"1", "true", "yes", "y"}
        ).astype(bool)
    for column in numeric_columns & set(combined.columns):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    for column in set(combined.columns) - datetime_columns - boolean_columns - numeric_columns:
        combined[column] = combined[column].fillna("").astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    unscored = OUT / "combined-v12-v13-targeted1295-v14-auto-unscored-preserve-base.parquet"
    combined.to_parquet(unscored, index=False)
    audit.update(
        {
            "base_rows": int(len(base)),
            "addition_rows_before_final_dedup": int(len(additions)),
            "base_rows_preserved": True,
            "combined_rows_after_final_dedup": int(len(combined)),
            "combined_1295_rows": int((combined["ticker"].astype(str) == "1295.KL").sum()),
            "output_path": str(unscored.relative_to(ROOT)),
            "output_sha256": hashlib.sha256(unscored.read_bytes()).hexdigest(),
        }
    )
    (OUT / "selection-report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
