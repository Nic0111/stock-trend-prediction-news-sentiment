"""Prepare public Bursa/issuer disclosures for offline fusion evidence.

The module does not scrape websites or bypass publisher controls.  It accepts
lawfully exported CSV/JSON/Parquet metadata and applies a deterministic,
auditable material-event taxonomy before the normal historical-news importer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


EVENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("financial_results_guidance", (r"\bquarter(?:ly)?\b", r"\bfinancial results?\b", r"\bprofit\b", r"\bpat\b", r"\blat\b", r"\bnet profit\b", r"\bearnings?\b", r"\brevenue\b", r"\bguidance\b", r"\boperating performance\b", r"\bstrong performance\b", r"\bperformance in\b")),
    ("dividends_distributions", (r"\bdividend\b", r"\bdistribution\b", r"\bentitlement date\b")),
    ("contracts_order_wins", (r"\bcontract\b", r"\border book\b", r"\baward(?:ed)?\b", r"\btender\b", r"\bprovision of\b.*\bservices?\b", r"\bstrategic (?:partnership|alliances?)\b", r"\bpartnerships?\b", r"\balliances?\b", r"\bcollaborat(?:e|es|ing|ion)s?\b", r"\bjointly\b", r"\bdeal\b", r"\bmemorandum of understanding\b", r"\bmou\b", r"\bagreements?\b", r"\binterconnection\b", r"\bpartners?\b")),
    ("acquisitions_disposals_investments", (r"\bacquisitions?\b", r"\bacquir(?:e|es|ed|ing)\b", r"\btakeover\b", r"\bdisposals?\b", r"\binvestments?\b", r"\bequity interest\b", r"\bjoint venture\b")),
    ("financing_capital_changes", (r"\bfinancing\b", r"\bloan\b", r"\bcredit\b", r"\bbase rate\b", r"\breference rate\b", r"\brepayment\b", r"\bmoratorium\b", r"\bfinancial relief\b", r"\bplacement\b", r"\brights? issue\b", r"\bshare (?:issue|buy-back)\b", r"\bsukuk\b", r"\bchanges? in (?:sub\.?\s*s-?hldr|substantial shareholder)\b")),
    ("regulatory_legal_governance", (r"\blitigation\b", r"\bregulator", r"\bcompliance\b", r"\bgovernance\b", r"\binvestigation\b", r"\bbudget\b", r"\bcovid\b", r"\bfraud\b", r"\bscam\b", r"\bwinding[- ]?up\b")),
    ("management_changes", (r"\bappointment\b", r"\bresignation\b", r"\bchief executive\b", r"\bdirector\b", r"\bchanges? in (?:boardroom|audit committee|nomination(?: and remuneration)? committee|remuneration committee)\b")),
    ("operational_product_strategy", (r"\blaunch\b", r"\boperations?\b", r"\bstrategy\b", r"\bgrowth\b", r"\bexpansion\b", r"\bproduce\b", r"\bproduction\b", r"\bconstruct(?:ion)?\b", r"\bplant\b", r"\bfacilit(?:y|ies)\b", r"\bproject\b", r"\benergy\b", r"\bsolar\b", r"\brenewable\b", r"\bdigital(?:isation|ization)?\b", r"\bplatform\b", r"\bpayments?\b", r"\bcapacity\b", r"\bportfolio\b", r"\brecycling\b", r"\bwaste\b", r"\bspecialt(?:y|ies) chemicals?\b")),
)

ROUTINE_PATTERNS = (
    r"\bnotice of meeting\b",
    r"\bminutes of meeting\b",
    r"\bclosed period\b",
    r"\bchange in registered address\b",
    r"\bmonthly shareholding statistics\b",
    r"\bapplication for quotation\b",
)

ALLOWED_SOURCE_KINDS = frozenset(
    {"bursa_announcement", "issuer_ir", "press_release"}
)


def classify_material_event(title: Any, description: Any = "") -> str:
    text = f"{title or ''} {description or ''}".casefold()
    if any(re.search(pattern, text) for pattern in ROUTINE_PATTERNS):
        return "routine_administrative"
    for category, patterns in EVENT_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            return category
    return "unclassified"


def prepare_official_disclosures(frame: pd.DataFrame, *, source_kind: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    source_kind = str(source_kind).strip().casefold()
    if source_kind not in ALLOWED_SOURCE_KINDS:
        raise ValueError(
            f"source_kind must be one of {sorted(ALLOWED_SOURCE_KINDS)}, got {source_kind!r}"
        )
    output = frame.copy()
    title = output.get("title", output.get("headline", pd.Series("", index=output.index)))
    description = output.get("description", output.get("summary", pd.Series("", index=output.index)))
    classified = [
        classify_material_event(headline, summary)
        for headline, summary in zip(title, description)
    ]
    # Curated issuer supplements may carry an explicit category backed by the
    # primary page.  Preserve it through preparation; otherwise the title-only
    # classifier would silently erase the auditable override and under-count
    # material events.  All ordinary exports remain deterministically
    # classified from title/description.
    existing_category = output.get("event_category", pd.Series("", index=output.index))
    curated = output.get("legacy_metadata_override", pd.Series(False, index=output.index)).fillna(False).astype(bool)
    output["event_category"] = [
        str(category).strip() if bool(is_curated) and str(category).strip() else inferred
        for category, is_curated, inferred in zip(existing_category, curated, classified)
    ]
    output = output.loc[output["event_category"].ne("routine_administrative")].copy()
    output["source_kind"] = source_kind
    output["disclosure_type"] = source_kind
    output["provider"] = "bursa_official" if source_kind == "bursa_announcement" else "issuer_official"
    output["entity_confidence"] = 1.0
    # A source export is already scoped to the issuer/company selected by the
    # register.  Preserve that deterministic entity anchor for downstream
    # scoring; otherwise concatenating it with reviewed news can introduce
    # missing values and the scorer's entity filter would silently discard the
    # official rows.
    output["entity_relevant"] = True
    output["text_source"] = output.get("text_source", pd.Series("official_export", index=output.index)).replace("", "official_export")
    output["publisher_full_text_available"] = output.get(
        "publisher_full_text_available", pd.Series(False, index=output.index)
    ).fillna(False)
    output["is_pseudo_document"] = output.get(
        "is_pseudo_document", pd.Series(False, index=output.index)
    ).fillna(False)
    body = output.get("article_text", output.get("content", description)).astype(str)
    output["text_quality_score"] = body.str.split().str.len().clip(upper=300) / 300.0
    output["text_quality_score"] = output["text_quality_score"].clip(lower=0.35, upper=1.0)
    return output.reset_index(drop=True)


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported disclosure export: {path}")


def run_prepare_official_disclosures(
    sources: Sequence[str], *, output_path: str, source_kind: str
) -> dict[str, Any]:
    frames = [prepare_official_disclosures(_read(Path(path)), source_kind=source_kind) for path in sources]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.casefold() == ".csv":
        result.to_csv(target, index=False)
    elif target.suffix.casefold() in {".jsonl", ".ndjson"}:
        result.to_json(target, orient="records", lines=True, date_format="iso")
    else:
        result.to_parquet(target, index=False)
    payload = {
        "schema_version": "official-disclosure-evidence-v1",
        "source_kind": source_kind,
        "rows": len(result),
        "event_categories": result.get("event_category", pd.Series(dtype=str)).value_counts().to_dict(),
        "output_path": str(target),
        "next_step": "score-news, then import-news into the v3 historical corpus",
    }
    print(json.dumps(payload, indent=2))
    return payload
