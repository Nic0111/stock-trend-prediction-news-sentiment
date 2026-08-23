"""Offline GDELT + Common Crawl acquisition for the fusion research dataset.

GDELT is used only as an historical URL/entity index.  Article text is
recovered from Common Crawl and then passed through StockSense's own
entity-disambiguation and promoted FinBERT classifier.  This module is never
called by the live FastAPI request path.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

try:  # Optional for offline SQL/report operations.
    import httpx
except ImportError:  # pragma: no cover - exercised only in minimal environments
    httpx = None  # type: ignore[assignment]

from pipeline.download_financial_news_data import (
    add_sentiment_scores,
    load_sentiment_model,
    load_stock_metadata,
)
from pipeline.bursa_news_dataset import load_policy
from pipeline.historical_news_fusion_data import (
    DEFAULT_FUSION_DATA_DIR,
    import_historical_news,
)
from pipeline.news_relevance import (
    COMPANY_ENTITY_PROFILES,
    classify_entity_relevance,
    classify_non_news_source,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GDELT_DIR = PROJECT_ROOT / "data" / "historical-news-fusion" / "gdelt"
DEFAULT_GDELT_RETRIEVAL_PROFILE_PATH = (
    PROJECT_ROOT / "config" / "gdelt_retrieval_profiles.json"
)
DEFAULT_GDELT_QUERY_PROFILE = "high-recall-v4"
GDELT_TABLE = "`gdelt-bq.gdeltv2.gkg_partitioned`"
COMMON_CRAWL_COLLECTIONS_URL = "https://index.commoncrawl.org/collinfo.json"
COMMON_CRAWL_DATA_URL = "https://data.commoncrawl.org"
DEFAULT_CAPTURE_LAG_DAYS = 30
DEFAULT_REQUEST_INTERVAL_SECONDS = 1.0
DEFAULT_PILOT_YEAR = 2023
DEFAULT_PILOT_CANDIDATES_PER_TICKER = 150
DEFAULT_MONTHLY_QUERY_GIB = 75.0
DEFAULT_MONTHLY_QUERY_BUDGET_GIB = 900.0
DEFAULT_COMMON_CRAWL_RETRIES = 5
DEFAULT_INDEXED_MIN_URLS_PER_TICKER = 50
DEFAULT_INDEXED_MIN_URLS_BY_TICKER = {"8869.KL": 30}
DEFAULT_INDEXED_MIN_DOMAINS_PER_TICKER = 2
DEFAULT_INDEXED_MIN_MONTHS_PER_TICKER = 6
DEFAULT_CANDIDATES_PER_TICKER_MONTH = 500
DEFAULT_REMAINING_COLLECTION_YEARS = (2019, 2020, 2021, 2022, 2024, 2025)
DEFAULT_EXTERNAL_COLLECTION_YEARS = tuple(range(2015, 2026))
DEFAULT_REUSABLE_2023_EXPORT = (
    DEFAULT_GDELT_DIR / "pilot-2023-high-recall-v3b" / "exports"
)


class CommonCrawlRateLimitError(RuntimeError):
    """Raised when Common Crawl asks the client to stop and retry later."""

    retry_after_hours = 24


class BigQueryRunnerError(RuntimeError):
    """Raised when an authenticated GDELT query cannot be safely executed."""


def _json_safe(value: Any) -> Any:
    """Convert BigQuery scalar values into JSONL-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_heartbeat(path: Path | str | None, payload: Mapping[str, Any]) -> None:
    """Atomically expose long-running offline job progress."""

    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


@dataclass(frozen=True)
class GDELTPilotResult:
    report_path: Path
    indexed_rows: int
    recovered_rows: int
    accepted_rows: int
    ready_for_full_collection: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["report_path"] = str(self.report_path)
        return payload


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _sql_array(values: Sequence[str]) -> str:
    return "[" + ", ".join(_sql_literal(value) for value in values) + "]"


def load_gdelt_retrieval_profile(
    query_profile: str = DEFAULT_GDELT_QUERY_PROFILE,
    *,
    path: Path = DEFAULT_GDELT_RETRIEVAL_PROFILE_PATH,
) -> Mapping[str, Mapping[str, Sequence[str]]]:
    """Load the versioned, recall-oriented candidate retrieval aliases.

    These aliases affect only the offline GDELT candidate search. They do not
    alter the strict production entity classifier or its acceptance thresholds.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    if query_profile not in profiles:
        raise ValueError(
            f"Unsupported GDELT query profile {query_profile!r}; "
            f"available profiles: {sorted(profiles)}"
        )
    profile = profiles[query_profile]
    if not isinstance(profile, Mapping):
        raise ValueError(f"GDELT query profile {query_profile!r} is malformed")
    return profile


def _profile_terms(
    profile: Mapping[str, Sequence[str]], key: str
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in profile.get(key, ())
            if str(value).strip()
        )
    )


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"Expected an ISO date (YYYY-MM-DD), got {value!r}") from exc


def month_date_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Return inclusive monthly date ranges, including both boundary dates."""

    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    ranges: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        last_day = date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1])
        ranges.append((cursor.isoformat(), min(last_day, end).isoformat()))
        cursor = last_day + timedelta(days=1)
    return ranges


def build_gdelt_bigquery_sql(
    *,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    tickers: Sequence[str] | None = None,
    query_profile: str = DEFAULT_GDELT_QUERY_PROFILE,
    max_candidates_per_ticker_month: int = DEFAULT_CANDIDATES_PER_TICKER_MONTH,
) -> str:
    """Build a partition-pruned, multi-channel GKG candidate export.

    Organization and page-title matches increase recall, but every exported row
    remains only a candidate. Recovered text still has to pass StockSense's
    strict entity-disambiguation layer before FinBERT can score it.
    """

    if max_candidates_per_ticker_month <= 0:
        raise ValueError("max_candidates_per_ticker_month must be positive")
    metadata = load_stock_metadata()
    selected = [ticker.upper() for ticker in (tickers or metadata.keys())]
    missing = [ticker for ticker in selected if ticker not in metadata]
    if missing:
        raise ValueError(f"Unsupported GDELT fusion tickers: {missing}")
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    exclusive_end = end + timedelta(days=1)
    retrieval_profiles = load_gdelt_retrieval_profile(query_profile)
    company_rows: list[str] = []
    for ticker in selected:
        company_name = metadata[ticker]
        profile = retrieval_profiles.get(ticker)
        if not profile:
            raise ValueError(
                f"GDELT retrieval profile {query_profile!r} has no entry for {ticker}"
            )
        company_rows.append(
            "SELECT "
            f"{_sql_literal(ticker)} AS ticker, "
            f"{_sql_literal(company_name)} AS company_name, "
            f"{_sql_array(_profile_terms(profile, 'distinctive'))} AS distinctive_aliases, "
            f"{_sql_array(_profile_terms(profile, 'ambiguous'))} AS ambiguous_aliases, "
            f"{_sql_array(_profile_terms(profile, 'anchors'))} AS anchors"
        )
    return (
        f"-- StockSense historical fusion candidate export ({query_profile}).\n"
        "-- GDELT tone is intentionally not selected; StockSense FinBERT scores recovered text.\n"
        "-- Query matches are retrieval evidence only and never bypass entity disambiguation.\n"
        "-- The end date is inclusive; the partition predicate ends at the next day.\n"
        "WITH company_aliases AS (\n"
        + "\nUNION ALL\n".join(company_rows)
        + "\n), gkg AS (\n"
        "  SELECT SourceCollectionIdentifier, DocumentIdentifier, DATE,\n"
        "    V2Organizations,\n"
        "    REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS page_title,\n"
        "    REGEXP_REPLACE(\n"
        "      REGEXP_REPLACE(LOWER(REGEXP_REPLACE(COALESCE(V2Organizations, ''), r',\\d+', '')),\n"
        "        r'[^a-z0-9;]+', ' '), r'\\s*;\\s*', ';') AS organization_token_blob,\n"
        "    REGEXP_REPLACE(LOWER(REGEXP_REPLACE(COALESCE(V2Organizations, ''), r',\\d+', '')),\n"
        "      r'[^a-z0-9]+', ' ') AS organization_phrase_blob\n"
        f"  FROM {GDELT_TABLE}\n"
        f"  WHERE _PARTITIONTIME >= TIMESTAMP({_sql_literal(start.isoformat())})\n"
        f"    AND _PARTITIONTIME < TIMESTAMP({_sql_literal(exclusive_end.isoformat())})\n"
        "    AND SourceCollectionIdentifier = 1\n"
        "    AND DocumentIdentifier IS NOT NULL\n"
        "), matched AS (\n"
        "SELECT\n"
        "  ca.ticker,\n"
        "  ca.company_name,\n"
        "  'GDELT' AS provider,\n"
        "  NET.HOST(g.DocumentIdentifier) AS source_name,\n"
        "  g.DocumentIdentifier AS url,\n"
        "  PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(g.DATE AS STRING)) AS published_at,\n"
        "  g.page_title,\n"
        "  g.V2Organizations AS gdelt_organizations,\n"
        "  ARRAY(SELECT alias FROM UNNEST(ca.distinctive_aliases) alias\n"
        "    WHERE STRPOS(CONCAT(';', g.organization_token_blob, ';'),\n"
        "      CONCAT(';', REGEXP_REPLACE(LOWER(alias), r'[^a-z0-9]+', ' '), ';')) > 0) AS exact_org_aliases,\n"
        "  ARRAY(SELECT alias FROM UNNEST(ca.distinctive_aliases) alias\n"
        "    WHERE STRPOS(CONCAT(' ', g.organization_phrase_blob, ' '),\n"
        "      CONCAT(' ', REGEXP_REPLACE(LOWER(alias), r'[^a-z0-9]+', ' '), ' ')) > 0) AS contained_org_aliases,\n"
        "  ARRAY(SELECT alias FROM UNNEST(ca.distinctive_aliases) alias\n"
        "    WHERE STRPOS(CONCAT(' ', REGEXP_REPLACE(LOWER(COALESCE(g.page_title, '')), r'[^a-z0-9]+', ' '), ' '),\n"
        "      CONCAT(' ', REGEXP_REPLACE(LOWER(alias), r'[^a-z0-9]+', ' '), ' ')) > 0) AS title_aliases,\n"
        "  ARRAY(SELECT alias FROM UNNEST(ca.ambiguous_aliases) alias\n"
        "    WHERE STRPOS(CONCAT(' ', g.organization_phrase_blob, ' '),\n"
        "      CONCAT(' ', REGEXP_REPLACE(LOWER(alias), r'[^a-z0-9]+', ' '), ' ')) > 0) AS ambiguous_org_aliases,\n"
        "  ARRAY(SELECT alias FROM UNNEST(ca.ambiguous_aliases) alias\n"
        "    WHERE STRPOS(CONCAT(' ', REGEXP_REPLACE(LOWER(COALESCE(g.page_title, '')), r'[^a-z0-9]+', ' '), ' '),\n"
        "      CONCAT(' ', REGEXP_REPLACE(LOWER(alias), r'[^a-z0-9]+', ' '), ' ')) > 0) AS ambiguous_title_aliases,\n"
        "  ARRAY(SELECT anchor FROM UNNEST(ca.anchors) anchor\n"
        "    WHERE STRPOS(CONCAT(' ', REGEXP_REPLACE(LOWER(CONCAT(COALESCE(g.page_title, ''), ' ',\n"
        "      COALESCE(g.V2Organizations, ''), ' ', COALESCE(g.DocumentIdentifier, ''))), r'[^a-z0-9]+', ' '), ' '),\n"
        "      CONCAT(' ', REGEXP_REPLACE(LOWER(anchor), r'[^a-z0-9]+', ' '), ' ')) > 0) AS matched_anchors\n"
        "FROM gkg AS g\n"
        "CROSS JOIN company_aliases AS ca\n"
        "), candidates AS (\n"
        "SELECT *,\n"
        "  CASE\n"
        "    WHEN ARRAY_LENGTH(exact_org_aliases) > 0 THEN 'organization_exact'\n"
        "    WHEN ARRAY_LENGTH(contained_org_aliases) > 0 THEN 'organization_distinctive'\n"
        "    WHEN ARRAY_LENGTH(title_aliases) > 0 THEN 'title_distinctive'\n"
        "    WHEN ARRAY_LENGTH(ambiguous_org_aliases) > 0 AND ARRAY_LENGTH(matched_anchors) > 0 THEN 'organization_ambiguous_anchored'\n"
        "    ELSE 'title_ambiguous_anchored'\n"
        "  END AS match_channel,\n"
        "  CASE\n"
        "    WHEN ARRAY_LENGTH(exact_org_aliases) > 0 THEN 5\n"
        "    WHEN ARRAY_LENGTH(contained_org_aliases) > 0 THEN 4\n"
        "    WHEN ARRAY_LENGTH(title_aliases) > 0 THEN 3\n"
        "    WHEN ARRAY_LENGTH(ambiguous_org_aliases) > 0 AND ARRAY_LENGTH(matched_anchors) > 0 THEN 2\n"
        "    ELSE 1\n"
        "  END AS candidate_priority,\n"
        "  ARRAY(SELECT DISTINCT value FROM UNNEST(ARRAY_CONCAT(\n"
        "    exact_org_aliases, contained_org_aliases, title_aliases,\n"
        "    ambiguous_org_aliases, ambiguous_title_aliases)) value ORDER BY value) AS matched_aliases\n"
        "FROM matched\n"
        "WHERE ARRAY_LENGTH(exact_org_aliases) > 0\n"
        "   OR ARRAY_LENGTH(contained_org_aliases) > 0\n"
        "   OR ARRAY_LENGTH(title_aliases) > 0\n"
        "   OR (ARRAY_LENGTH(ambiguous_org_aliases) > 0 AND ARRAY_LENGTH(matched_anchors) > 0)\n"
        "   OR (ARRAY_LENGTH(ambiguous_title_aliases) > 0 AND ARRAY_LENGTH(matched_anchors) > 0)\n"
        "), deduplicated AS (\n"
        "SELECT *, ROW_NUMBER() OVER (\n"
        "  PARTITION BY ticker, url ORDER BY candidate_priority DESC, published_at DESC\n"
        ") AS url_rank\n"
        "FROM candidates\n"
        "), ranked AS (\n"
        "SELECT *, ROW_NUMBER() OVER (\n"
        "  PARTITION BY ticker, FORMAT_TIMESTAMP('%Y-%m', published_at)\n"
        "  ORDER BY candidate_priority DESC, FARM_FINGERPRINT(url)\n"
        ") AS monthly_rank\n"
        "FROM deduplicated WHERE url_rank = 1\n"
        ")\n"
        "SELECT ticker, company_name, provider, source_name, url, published_at,\n"
        "  page_title, gdelt_organizations, match_channel, candidate_priority,\n"
        "  matched_aliases, ARRAY_TO_STRING(matched_aliases, '; ') AS matched_alias,\n"
        "  matched_anchors,\n"
        "  match_channel IN ('organization_exact', 'organization_distinctive') AS organization_match,\n"
        "  match_channel IN ('title_distinctive', 'title_ambiguous_anchored') AS title_match\n"
        "FROM ranked\n"
        f"WHERE monthly_rank <= {int(max_candidates_per_ticker_month)}\n"
        "ORDER BY ticker, published_at, candidate_priority DESC, url;\n"
    )


def write_gdelt_query_bundle(
    output_dir: Path = DEFAULT_GDELT_DIR,
    *,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    tickers: Sequence[str] | None = None,
    shard: str = "monthly",
    max_shard_gib: float = DEFAULT_MONTHLY_QUERY_GIB,
    monthly_budget_gib: float = DEFAULT_MONTHLY_QUERY_BUDGET_GIB,
    pilot_year: int | None = None,
    query_profile: str = DEFAULT_GDELT_QUERY_PROFILE,
    max_candidates_per_ticker_month: int = DEFAULT_CANDIDATES_PER_TICKER_MONTH,
) -> dict[str, Any]:
    """Write SQL shards plus a free-quota reproducibility manifest.

    This function never contacts BigQuery and never enables billing. The caller
    runs the generated SQL manually in BigQuery Sandbox.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if shard not in {"single", "monthly"}:
        raise ValueError("shard must be 'single' or 'monthly'")
    if max_shard_gib <= 0 or monthly_budget_gib <= 0:
        raise ValueError("max_shard_gib and monthly_budget_gib must be positive")
    ranges = (
        month_date_ranges(start_date, end_date)
        if shard == "monthly"
        else [(start_date, end_date)]
    )
    estimated_budget = len(ranges) * float(max_shard_gib)
    if estimated_budget > float(monthly_budget_gib):
        raise ValueError(
            "The requested SQL bundle exceeds the free monthly query budget: "
            f"{estimated_budget:.1f} GiB estimated versus {monthly_budget_gib:.1f} GiB. "
            "Run one year (or fewer shards) at a time and perform a dry run before execution."
        )
    shard_rows: list[dict[str, Any]] = []
    query_paths: list[str] = []
    for shard_start, shard_end in ranges:
        sql = build_gdelt_bigquery_sql(
            start_date=shard_start,
            end_date=shard_end,
            tickers=tickers,
            query_profile=query_profile,
            max_candidates_per_ticker_month=max_candidates_per_ticker_month,
        )
        if shard == "monthly":
            query_path = output_dir / "queries" / f"{shard_start[:7]}.sql"
        else:
            query_path = output_dir / "gdelt-gkg-export.sql"
        query_path.parent.mkdir(parents=True, exist_ok=True)
        query_path.write_text(sql, encoding="utf-8")
        digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        shard_rows.append(
            {
                "start_date": shard_start,
                "end_date": shard_end,
                "query_path": str(query_path),
                "query_sha256": digest,
                "maximum_bytes_billed_gib": float(max_shard_gib),
            }
        )
        query_paths.append(str(query_path))
    manifest = {
        "schema_version": "gdelt-commoncrawl-acquisition-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "GDELT 2.1 GKG BigQuery",
        "table": GDELT_TABLE.strip("`"),
        "start_date": start_date,
        "end_date": end_date,
        "tickers": list(tickers or load_stock_metadata().keys()),
        "shard": shard,
        "query_paths": query_paths,
        "query_shards": shard_rows,
        "free_only": True,
        "billing_enabled": False,
        "monthly_query_budget_gib": float(monthly_budget_gib),
        "max_shard_gib": float(max_shard_gib),
        "estimated_maximum_bytes_billed_gib": estimated_budget,
        "pilot_year": int(pilot_year) if pilot_year is not None else None,
        "query_profile": query_profile,
        "retrieval_profile_path": str(DEFAULT_GDELT_RETRIEVAL_PROFILE_PATH),
        "max_candidates_per_ticker_month": int(
            max_candidates_per_ticker_month
        ),
        "notes": [
            "Run this query against the public GDELT BigQuery dataset in BigQuery Sandbox with billing disabled.",
            "Perform a dry run before execution and stop when the free monthly allowance is reached.",
            "Export each shard locally as CSV, JSONL or Parquet; do not create paid cloud storage.",
            "No GDELT tone field is used.",
            "A GDELT match is candidate provenance only and cannot bypass strict entity disambiguation.",
            "The exported URLs must be enriched with Common Crawl before FinBERT scoring.",
        ],
    }
    manifest_path = output_dir / "acquisition-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _append_jsonl_record(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, default=str) + "\n")


def run_authenticated_gdelt_queries(
    project_id: str,
    *,
    query_dir: Path | str = DEFAULT_GDELT_DIR / "queries",
    export_dir: Path | str = DEFAULT_GDELT_DIR / "exports",
    state_path: Path | str | None = None,
    maximum_bytes_billed_gib: float = DEFAULT_MONTHLY_QUERY_GIB,
    monthly_budget_gib: float = DEFAULT_MONTHLY_QUERY_BUDGET_GIB,
    location: str = "US",
    client: Any | None = None,
    dry_run_only: bool = False,
) -> dict[str, Any]:
    """Run generated GDELT SQL shards through authenticated BigQuery.

    The caller must first configure Google Application Default Credentials
    locally (``gcloud auth application-default login``).  Queries are executed
    sequentially, dry-run checked, capped with ``maximum_bytes_billed`` and
    streamed to local JSONL files.  The function never enables billing and
    resumes from ``state_path`` after interruption.

    ``client`` is injectable for tests.  The production path imports the
    optional BigQuery client lazily so the live application does not require the
    cloud SDK.
    """

    project_id = str(project_id or "").strip()
    injected_client = client is not None
    if not project_id:
        raise BigQueryRunnerError("A Google Cloud project_id is required")
    if maximum_bytes_billed_gib <= 0 or monthly_budget_gib <= 0:
        raise BigQueryRunnerError("BigQuery byte budgets must be positive")
    if maximum_bytes_billed_gib > monthly_budget_gib:
        raise BigQueryRunnerError(
            "maximum_bytes_billed_gib cannot exceed monthly_budget_gib"
        )

    query_root = Path(query_dir)
    output_root = Path(export_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    state_file = Path(state_path) if state_path else output_root / "query-state.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl_records(state_file):
        shard = str(record.get("shard", ""))
        if shard:
            completed[shard] = record

    query_files = sorted(query_root.glob("*.sql"))
    if not query_files:
        raise BigQueryRunnerError(f"No SQL shards found in {query_root}")

    if client is None:
        try:
            from google.cloud import bigquery  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BigQueryRunnerError(
                "Authenticated GDELT runs require google-cloud-bigquery. "
                "Install requirements-gdelt.txt after configuring Google Cloud."
            ) from exc
        try:
            import google.auth  # type: ignore

            credentials, _ = google.auth.default(quota_project_id=project_id)
            client = bigquery.Client(project=project_id, credentials=credentials)
        except Exception:
            # A signed-in gcloud CLI is sufficient for the bq command but is
            # not automatically visible as Application Default Credentials.
            # Reuse that local user session without persisting or printing its
            # access token. This keeps the one-time Google consent in gcloud.
            try:
                from google.auth.credentials import Credentials  # type: ignore

                gcloud_command = shutil.which("gcloud") or shutil.which("gcloud.cmd")
                if not gcloud_command:
                    candidates = [
                        Path(os.environ.get("LOCALAPPDATA", ""))
                        / "Google"
                        / "Cloud SDK"
                        / "google-cloud-sdk"
                        / "bin"
                        / "gcloud.cmd",
                        Path(os.environ.get("ProgramFiles", ""))
                        / "Google"
                        / "Cloud SDK"
                        / "google-cloud-sdk"
                        / "bin"
                        / "gcloud.cmd",
                    ]
                    gcloud_command = next(
                        (str(path) for path in candidates if path.is_file()), None
                    )
                if not gcloud_command:
                    raise FileNotFoundError("gcloud CLI was not found")

                class _GcloudCredentials(Credentials):
                    def refresh(self, request: Any) -> None:
                        completed = subprocess.run(
                            [str(gcloud_command), "auth", "print-access-token"],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        token = completed.stdout.strip()
                        if not token:
                            raise RuntimeError("gcloud returned an empty access token")
                        self.token = token
                        self.expiry = datetime.now(timezone.utc) + timedelta(minutes=50)

                credentials = _GcloudCredentials()
                # The signed-in gcloud profile can carry a quota project that
                # differs from the explicit BigQuery job project.  Bind both
                # quota and service usage to the caller-supplied project so a
                # stale local default cannot disable or bill the wrong project.
                credentials._quota_project_id = project_id
                credentials.refresh(None)
                client = bigquery.Client(project=project_id, credentials=credentials)
            except Exception as exc:  # pragma: no cover - environment dependent
                raise BigQueryRunnerError(
                    "BigQuery authentication failed. Run "
                    "gcloud auth application-default login and retry."
                ) from exc
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError:  # pragma: no cover - only used when an injected fake is supplied
        bigquery = None  # type: ignore[assignment]

    byte_cap = int(float(maximum_bytes_billed_gib) * (1024**3))
    monthly_cap = int(float(monthly_budget_gib) * (1024**3))
    prior_bytes = sum(
        int(record.get("bytes_processed", 0) or 0)
        for record in completed.values()
        if record.get("status") == "completed"
    )
    project_month_bytes = 0
    if not injected_client and bigquery is not None:
        region = str(location or "US").strip().lower()
        usage_sql = (
            "SELECT COALESCE(SUM(total_bytes_billed), 0) AS total_bytes_billed "
            f"FROM `region-{region}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT "
            "WHERE creation_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) "
            "AND job_type = 'QUERY' AND state = 'DONE' AND error_result IS NULL"
        )
        try:
            usage_job = client.query(
                usage_sql,
                job_config=bigquery.QueryJobConfig(use_query_cache=False),
                location=location,
            )
            usage_rows = list(usage_job.result())
            if usage_rows:
                usage_row = usage_rows[0]
                if hasattr(usage_row, "items"):
                    usage_payload = dict(usage_row.items())
                    project_month_bytes = int(
                        usage_payload.get("total_bytes_billed", 0) or 0
                    )
                else:
                    project_month_bytes = int(
                        getattr(usage_row, "total_bytes_billed", 0) or 0
                    )
        except Exception as exc:
            raise BigQueryRunnerError(
                "Could not read current-month BigQuery usage; refusing to run "
                f"without a project-wide quota guard: {exc}"
            ) from exc

    pending: list[tuple[Path, str, str, Path]] = []
    dry_run_bytes: dict[str, int] = {}
    for query_path in query_files:
        shard = query_path.stem
        previous = completed.get(shard, {})
        export_path = output_root / f"{shard}.jsonl"
        if previous.get("status") == "completed" and export_path.exists():
            continue
        sql = query_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        if previous.get("query_sha256") and previous["query_sha256"] != digest:
            raise BigQueryRunnerError(
                f"Query shard {shard} changed since its checkpoint; use a new output directory."
            )
        if bigquery is not None:
            dry_config = bigquery.QueryJobConfig(
                dry_run=True,
                use_query_cache=False,
                maximum_bytes_billed=byte_cap,
            )
            dry_job = client.query(sql, job_config=dry_config, location=location)
        else:
            dry_job = client.query(sql, job_config=None, location=location)
        bytes_processed = int(
            getattr(dry_job, "total_bytes_processed", 0) or 0
        )
        if bytes_processed > byte_cap:
            raise BigQueryRunnerError(
                f"Dry run for {shard} estimates {bytes_processed / 1024**3:.2f} GiB, "
                f"above the {maximum_bytes_billed_gib:.2f} GiB shard cap."
            )
        dry_run_bytes[shard] = bytes_processed
        pending.append((query_path, sql, digest, export_path))

    baseline_budget_bytes = (
        project_month_bytes if not injected_client else prior_bytes
    )
    planned_bytes = sum(dry_run_bytes.values())
    if baseline_budget_bytes + planned_bytes > monthly_cap:
        raise BigQueryRunnerError(
            "The complete query bundle would exceed the project-wide monthly "
            f"safety budget: {(baseline_budget_bytes + planned_bytes) / 1024**3:.1f} GiB "
            f"projected versus {monthly_budget_gib:.1f} GiB allowed."
        )

    if dry_run_only:
        return {
            "project_id": project_id,
            "query_dir": str(query_root),
            "export_dir": str(output_root),
            "state_path": str(state_file),
            "dry_run_only": True,
            "completed_shards": len(
                [
                    record
                    for record in completed.values()
                    if record.get("status") == "completed"
                ]
            ),
            "pending_shards": len(pending),
            "dry_run_bytes_by_shard": dry_run_bytes,
            "project_month_bytes_before_run": project_month_bytes,
            "project_month_gib_before_run": round(
                project_month_bytes / (1024**3), 4
            ),
            "planned_bytes": planned_bytes,
            "planned_gib": round(planned_bytes / (1024**3), 4),
            "projected_month_gib": round(
                (baseline_budget_bytes + planned_bytes) / (1024**3), 4
            ),
            "monthly_budget_gib": float(monthly_budget_gib),
        }

    records: list[dict[str, Any]] = []
    stopped_reason: str | None = None
    for query_path in query_files:
        shard = query_path.stem
        previous = completed.get(shard, {})
        export_path = output_root / f"{shard}.jsonl"
        if previous.get("status") == "completed" and export_path.exists():
            records.append(previous)
            continue
        sql = query_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        try:
            bytes_processed = dry_run_bytes[shard]
            if bigquery is not None:
                run_config = bigquery.QueryJobConfig(
                    dry_run=False,
                    use_query_cache=False,
                    maximum_bytes_billed=byte_cap,
                )
            else:
                run_config = None
            job = client.query(sql, job_config=run_config, location=location)
            result = job.result()
            rows = 0
            with export_path.open("w", encoding="utf-8", newline="") as handle:
                for row in result:
                    payload = dict(row.items()) if hasattr(row, "items") else dict(row)
                    handle.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")
                    rows += 1
            record = {
                "shard": shard,
                "query_path": str(query_path),
                "query_sha256": digest,
                "status": "completed",
                "job_id": str(getattr(job, "job_id", "")),
                "bytes_processed": bytes_processed,
                "rows_exported": rows,
                "export_path": str(export_path),
                "project_id": project_id,
                "location": location,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _append_jsonl_record(state_file, record)
            records.append(record)
            completed[shard] = record
            prior_bytes += bytes_processed
        except BigQueryRunnerError:
            raise
        except Exception as exc:
            failed = {
                "shard": shard,
                "query_path": str(query_path),
                "query_sha256": digest,
                "status": "failed",
                "error": str(exc),
                "project_id": project_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _append_jsonl_record(state_file, failed)
            raise BigQueryRunnerError(
                f"BigQuery shard {shard} failed: {exc}"
            ) from exc
    return {
        "project_id": project_id,
        "query_dir": str(query_root),
        "export_dir": str(output_root),
        "state_path": str(state_file),
        "completed_shards": len([r for r in records if r.get("status") == "completed"]),
        "records": records,
        "bytes_processed": prior_bytes,
        "bytes_processed_gib": round(prior_bytes / (1024**3), 4),
        "project_month_bytes_before_run": project_month_bytes,
        "project_month_gib_before_run": round(
            project_month_bytes / (1024**3), 4
        ),
        "planned_bytes": planned_bytes,
        "planned_gib": round(planned_bytes / (1024**3), 4),
        "monthly_budget_gib": float(monthly_budget_gib),
        "stopped_reason": stopped_reason,
        "dry_run_only": False,
    }


def collect_remaining_gdelt_years(
    project_id: str,
    *,
    pilot_report_path: Path | str,
    output_dir: Path | str = DEFAULT_GDELT_DIR,
    years: Sequence[int] = DEFAULT_REMAINING_COLLECTION_YEARS,
    maximum_bytes_billed_gib: float = DEFAULT_MONTHLY_QUERY_GIB,
    monthly_budget_gib: float = DEFAULT_MONTHLY_QUERY_BUDGET_GIB,
    location: str = "US",
    execute: bool = False,
    query_profile: str = DEFAULT_GDELT_QUERY_PROFILE,
    max_candidates_per_ticker_month: int = DEFAULT_CANDIDATES_PER_TICKER_MONTH,
) -> dict[str, Any]:
    """Plan all remaining years first, then optionally execute sequentially.

    No query is executed unless the 2023 pilot report passes and the sum of all
    remaining dry-run estimates fits beneath the project-wide monthly ceiling.
    """

    report_path = Path(pilot_report_path)
    if not report_path.exists():
        raise BigQueryRunnerError(f"Pilot report does not exist: {report_path}")
    pilot = json.loads(report_path.read_text(encoding="utf-8"))
    if int(pilot.get("pilot_year", 0)) != DEFAULT_PILOT_YEAR:
        raise BigQueryRunnerError("Remaining-year collection requires the 2023 pilot")
    if not bool(pilot.get("ready_for_full_collection")):
        raise BigQueryRunnerError(
            "The 2023 Common Crawl and dual-LLM pilot has not passed; "
            "remaining-year queries are blocked."
        )
    normalized_years = tuple(dict.fromkeys(int(year) for year in years))
    invalid = [
        year
        for year in normalized_years
        if year < 2019 or year > 2025 or year == DEFAULT_PILOT_YEAR
    ]
    if invalid:
        raise BigQueryRunnerError(f"Unsupported remaining collection years: {invalid}")

    root = Path(output_dir)
    plans: list[dict[str, Any]] = []
    for year in normalized_years:
        year_root = root / str(year)
        query_root = year_root / "queries"
        export_root = year_root / "exports"
        write_gdelt_query_bundle(
            year_root,
            start_date=f"{year:04d}-01-01",
            end_date=f"{year:04d}-12-31",
            shard="monthly",
            max_shard_gib=maximum_bytes_billed_gib,
            monthly_budget_gib=monthly_budget_gib,
            query_profile=query_profile,
            max_candidates_per_ticker_month=max_candidates_per_ticker_month,
        )
        dry_run = run_authenticated_gdelt_queries(
            project_id,
            query_dir=query_root,
            export_dir=export_root,
            maximum_bytes_billed_gib=maximum_bytes_billed_gib,
            monthly_budget_gib=monthly_budget_gib,
            location=location,
            dry_run_only=True,
        )
        plans.append({"year": year, **dry_run})

    current_bytes = max(
        (int(plan.get("project_month_bytes_before_run", 0)) for plan in plans),
        default=0,
    )
    planned_bytes = sum(int(plan.get("planned_bytes", 0)) for plan in plans)
    budget_bytes = int(float(monthly_budget_gib) * (1024**3))
    if current_bytes + planned_bytes > budget_bytes:
        raise BigQueryRunnerError(
            "All remaining years were dry-run together and exceed the monthly "
            f"safety ceiling: {(current_bytes + planned_bytes) / 1024**3:.2f} GiB "
            f"projected versus {monthly_budget_gib:.2f} GiB allowed. "
            "No remaining-year query was executed."
        )

    executions: list[dict[str, Any]] = []
    if execute:
        for plan in plans:
            year = int(plan["year"])
            year_root = root / str(year)
            executions.append(
                {
                    "year": year,
                    **run_authenticated_gdelt_queries(
                        project_id,
                        query_dir=year_root / "queries",
                        export_dir=year_root / "exports",
                        maximum_bytes_billed_gib=maximum_bytes_billed_gib,
                        monthly_budget_gib=monthly_budget_gib,
                        location=location,
                        dry_run_only=False,
                    ),
                }
            )
    summary = {
        "schema_version": "gdelt-remaining-years-guard-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_report": str(report_path),
        "pilot_passed": True,
        "years": list(normalized_years),
        "execute": bool(execute),
        "current_month_gib": round(current_bytes / (1024**3), 4),
        "all_years_planned_gib": round(planned_bytes / (1024**3), 4),
        "projected_month_gib": round(
            (current_bytes + planned_bytes) / (1024**3), 4
        ),
        "monthly_budget_gib": float(monthly_budget_gib),
        "plans": plans,
        "executions": executions,
    }
    summary_path = root / "remaining-years-plan.json"
    _write_heartbeat(summary_path, summary)
    return {**summary, "summary_path": str(summary_path)}


def collect_external_label_candidate_years(
    project_id: str,
    *,
    output_dir: Path | str = DEFAULT_GDELT_DIR / "external-corpus-2015-2025",
    years: Sequence[int] = DEFAULT_EXTERNAL_COLLECTION_YEARS,
    reusable_2023_export: Path | str | None = DEFAULT_REUSABLE_2023_EXPORT,
    maximum_bytes_billed_gib: float = 20.0,
    monthly_budget_gib: float = DEFAULT_MONTHLY_QUERY_BUDGET_GIB,
    location: str = "US",
    execute: bool = False,
    query_profile: str = DEFAULT_GDELT_QUERY_PROFILE,
    max_candidates_per_ticker_month: int = DEFAULT_CANDIDATES_PER_TICKER_MONTH,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Plan or collect 2015-2025 candidates for external labelling.

    This path intentionally does not require the 2023 model-readiness pilot:
    discovery candidates are not production evidence and the caller will label
    them outside StockSense.  It remains fail-closed on cost.  Every unfinished
    monthly shard is dry-run first, all years are summed with current project
    usage, and no query executes unless the complete pending bundle fits below
    ``monthly_budget_gib``.

    The already completed 2023 high-recall export is reused by default so the
    same public-data query is not billed twice.
    """

    project_id = str(project_id or "").strip()
    if not project_id:
        raise BigQueryRunnerError("A Google Cloud project_id is required")
    normalized_years = tuple(dict.fromkeys(int(year) for year in years))
    invalid = [
        year for year in normalized_years if year < 2015 or year > 2025
    ]
    if invalid:
        raise BigQueryRunnerError(
            f"External-labelling collection supports 2015-2025 only: {invalid}"
        )
    if not normalized_years:
        raise BigQueryRunnerError("At least one collection year is required")

    run_queries = runner or run_authenticated_gdelt_queries
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reusable_path = (
        Path(reusable_2023_export) if reusable_2023_export is not None else None
    )
    plans: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []

    for year in normalized_years:
        if year == 2023 and reusable_path is not None:
            state_path = reusable_path / "query-state.jsonl"
            completed = {
                str(record.get("shard", "")): record
                for record in _read_jsonl_records(state_path)
                if record.get("status") == "completed"
            }
            exports = sorted(reusable_path.glob("2023-??.jsonl"))
            if len(completed) >= 12 and len(exports) >= 12:
                reused.append(
                    {
                        "year": 2023,
                        "export_dir": str(reusable_path),
                        "completed_shards": len(completed),
                        "rows_exported": int(
                            sum(
                                int(record.get("rows_exported", 0) or 0)
                                for record in completed.values()
                            )
                        ),
                        "bytes_processed": int(
                            sum(
                                int(record.get("bytes_processed", 0) or 0)
                                for record in completed.values()
                            )
                        ),
                    }
                )
                continue

        year_root = root / str(year)
        query_root = year_root / "queries"
        export_root = year_root / "exports"
        write_gdelt_query_bundle(
            year_root,
            start_date=f"{year:04d}-01-01",
            end_date=f"{year:04d}-12-31",
            shard="monthly",
            max_shard_gib=maximum_bytes_billed_gib,
            monthly_budget_gib=max(
                float(monthly_budget_gib),
                float(maximum_bytes_billed_gib) * 12.0,
            ),
            query_profile=query_profile,
            max_candidates_per_ticker_month=max_candidates_per_ticker_month,
        )
        dry_run = run_queries(
            project_id,
            query_dir=query_root,
            export_dir=export_root,
            maximum_bytes_billed_gib=maximum_bytes_billed_gib,
            monthly_budget_gib=monthly_budget_gib,
            location=location,
            dry_run_only=True,
        )
        plans.append({"year": year, **dry_run})

    current_bytes = max(
        (int(plan.get("project_month_bytes_before_run", 0)) for plan in plans),
        default=0,
    )
    planned_bytes = sum(int(plan.get("planned_bytes", 0)) for plan in plans)
    budget_bytes = int(float(monthly_budget_gib) * (1024**3))
    ready = current_bytes + planned_bytes <= budget_bytes
    executions: list[dict[str, Any]] = []

    summary = {
        "schema_version": "gdelt-external-label-corpus-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "years": list(normalized_years),
        "query_profile": query_profile,
        "max_candidates_per_ticker_month": int(
            max_candidates_per_ticker_month
        ),
        "execute_requested": bool(execute),
        "free_only": True,
        "production_pipeline_modified": False,
        "llm_labelling_requested": False,
        "reused_exports": reused,
        "current_month_gib": round(current_bytes / (1024**3), 4),
        "all_pending_years_planned_gib": round(
            planned_bytes / (1024**3), 4
        ),
        "projected_month_gib": round(
            (current_bytes + planned_bytes) / (1024**3), 4
        ),
        "monthly_budget_gib": float(monthly_budget_gib),
        "ready_for_execution": bool(ready),
        "blocked_reason": (
            ""
            if ready
            else (
                "The aggregate dry run plus current-month project usage exceeds "
                "the configured free-only ceiling. No query was executed."
            )
        ),
        "plans": plans,
        "executions": executions,
    }
    summary_path = root / "external-corpus-plan.json"
    _write_heartbeat(summary_path, summary)

    if execute and ready:
        for plan in plans:
            year = int(plan["year"])
            year_root = root / str(year)
            executions.append(
                {
                    "year": year,
                    **run_queries(
                        project_id,
                        query_dir=year_root / "queries",
                        export_dir=year_root / "exports",
                        maximum_bytes_billed_gib=maximum_bytes_billed_gib,
                        monthly_budget_gib=monthly_budget_gib,
                        location=location,
                        dry_run_only=False,
                    ),
                }
            )
        summary["executions"] = executions
        summary["completed"] = True
        _write_heartbeat(summary_path, summary)
    else:
        summary["completed"] = False
        _write_heartbeat(summary_path, summary)

    return {**summary, "summary_path": str(summary_path)}


def load_gdelt_export(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        files = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.casefold() in {".csv", ".json", ".jsonl", ".parquet", ".pq"}
        )
        if not files:
            raise ValueError("GDELT export directory contains no CSV, JSONL, or Parquet files")
    else:
        files = [path]

    frames: list[pd.DataFrame] = []
    for file_path in files:
        suffix = file_path.suffix.casefold()
        if suffix == ".csv":
            frames.append(pd.read_csv(file_path))
        elif suffix in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(file_path))
        elif suffix in {".json", ".jsonl"}:
            frames.append(pd.read_json(file_path, lines=suffix == ".jsonl"))
        else:
            raise ValueError("GDELT export must be CSV, JSON, JSONL, or Parquet")
    frame = pd.concat(frames, ignore_index=True)
    required = {"ticker", "company_name", "url", "published_at"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"GDELT export is missing columns: {missing}")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["published_at"])
    frame["provider"] = "GDELT/CommonCrawl"
    frame["source_domain"] = frame["url"].map(
        lambda value: (urlparse(str(value)).hostname or "").casefold().removeprefix("www.")
    )
    return frame.drop_duplicates(["ticker", "url"], keep="last").reset_index(drop=True)


def evaluate_indexed_candidate_gate(
    indexed: pd.DataFrame,
    *,
    minimum_urls_per_ticker: int = DEFAULT_INDEXED_MIN_URLS_PER_TICKER,
    minimum_urls_by_ticker: Mapping[str, int] | None = None,
    minimum_domains_per_ticker: int = DEFAULT_INDEXED_MIN_DOMAINS_PER_TICKER,
    minimum_months_per_ticker: int = DEFAULT_INDEXED_MIN_MONTHS_PER_TICKER,
    required_tickers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the cheap, pre-recovery coverage gate for GDELT candidates."""

    url_overrides = {
        str(ticker).upper(): int(value)
        for ticker, value in (
            DEFAULT_INDEXED_MIN_URLS_BY_TICKER
            if minimum_urls_by_ticker is None
            else minimum_urls_by_ticker
        ).items()
    }
    if min(
        minimum_urls_per_ticker,
        minimum_domains_per_ticker,
        minimum_months_per_ticker,
        *(url_overrides.values() or [minimum_urls_per_ticker]),
    ) <= 0:
        raise ValueError("Indexed candidate gate thresholds must be positive")
    tickers = tuple(
        str(ticker).upper()
        for ticker in (required_tickers or load_stock_metadata().keys())
    )
    frame = indexed.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=["ticker", "url", "published_at", "source_domain"])
    frame["ticker"] = frame.get("ticker", pd.Series(dtype=str)).astype(str).str.upper()
    frame["published_at"] = pd.to_datetime(
        frame.get("published_at", pd.Series(dtype=str)),
        utc=True,
        errors="coerce",
    )
    if "source_domain" not in frame:
        frame["source_domain"] = frame.get("url", pd.Series(dtype=str)).map(
            lambda value: (urlparse(str(value)).hostname or "")
            .casefold()
            .removeprefix("www.")
        )
    frame["_month"] = frame["published_at"].dt.strftime("%Y-%m")
    channel_counts = (
        frame.groupby(["ticker", "match_channel"]).size().unstack(fill_value=0)
        if "match_channel" in frame and not frame.empty
        else pd.DataFrame()
    )
    acceptable_channels = {
        "organization_exact",
        "organization_distinctive",
        "title_distinctive",
        "organization_ambiguous_anchored",
        "title_ambiguous_anchored",
    }
    companies: dict[str, dict[str, Any]] = {}
    ready = True
    for ticker in tickers:
        rows = frame.loc[frame["ticker"] == ticker]
        required_url_count = int(
            url_overrides.get(ticker, minimum_urls_per_ticker)
        )
        url_count = int(rows.get("url", pd.Series(dtype=str)).astype(str).nunique())
        domain_count = int(
            rows.get("source_domain", pd.Series(dtype=str))
            .astype(str)
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )
        month_count = int(rows["_month"].dropna().nunique())
        channels = (
            {
                str(channel): int(count)
                for channel, count in channel_counts.loc[ticker].items()
                if int(count) > 0
            }
            if ticker in channel_counts.index
            else {}
        )
        has_supported_channel = any(
            channel in acceptable_channels and count > 0
            for channel, count in channels.items()
        )
        failures: list[str] = []
        if url_count < required_url_count:
            failures.append(
                f"{url_count} URLs < {required_url_count} required"
            )
        if domain_count < minimum_domains_per_ticker:
            failures.append(
                f"{domain_count} domains < {minimum_domains_per_ticker} required"
            )
        if month_count < minimum_months_per_ticker:
            failures.append(
                f"{month_count} months < {minimum_months_per_ticker} required"
            )
        if not has_supported_channel:
            failures.append("no supported candidate match channel")
        passed = not failures
        ready = ready and passed
        companies[ticker] = {
            "passed": passed,
            "distinct_urls": url_count,
            "required_distinct_urls": required_url_count,
            "source_domains": domain_count,
            "months": month_count,
            "match_channels": channels,
            "failures": failures,
        }
    return {
        "ready_for_recovery": bool(ready),
        "thresholds": {
            "minimum_urls_per_ticker": int(minimum_urls_per_ticker),
            "minimum_urls_by_ticker": url_overrides,
            "minimum_domains_per_ticker": int(minimum_domains_per_ticker),
            "minimum_months_per_ticker": int(minimum_months_per_ticker),
        },
        "required_tickers": list(tickers),
        "companies": companies,
    }


def select_pilot_candidates(
    frame: pd.DataFrame,
    *,
    per_ticker: int = DEFAULT_PILOT_CANDIDATES_PER_TICKER,
) -> pd.DataFrame:
    """Select a deterministic month/domain-balanced pilot sample."""

    if per_ticker <= 0:
        raise ValueError("per_ticker must be positive")
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper().str.strip()
    working["published_at"] = pd.to_datetime(
        working["published_at"], utc=True, errors="coerce"
    )
    working = working.dropna(subset=["published_at", "url"]).copy()
    working["source_domain"] = working.get(
        "source_domain",
        working["url"].map(
            lambda value: (urlparse(str(value)).hostname or "")
            .casefold()
            .removeprefix("www.")
        ),
    )
    # ``strftime`` preserves the UTC month without pandas' timezone-dropping
    # PeriodArray warning and remains deterministic across pandas versions.
    working["_month"] = working["published_at"].dt.strftime("%Y-%m")
    working["_selection_key"] = working.apply(
        lambda row: hashlib.sha256(
            f"{row['ticker']}|{row['_month']}|{row['source_domain']}|{row['url']}".encode(
                "utf-8"
            )
        ).hexdigest(),
        axis=1,
    )
    working = working.sort_values(
        ["ticker", "_month", "source_domain", "_selection_key", "url"]
    )
    selected_by_ticker: list[list[Any]] = []
    for ticker, ticker_frame in working.groupby("ticker", sort=True):
        groups = [
            group
            for _, group in ticker_frame.groupby(
                ["_month", "source_domain"], sort=True
            )
        ]
        positions = [0] * len(groups)
        ticker_indices: list[Any] = []
        ticker_selected = 0
        while ticker_selected < per_ticker:
            progressed = False
            for group_index, group in enumerate(groups):
                position = positions[group_index]
                if position >= len(group):
                    continue
                ticker_indices.append(group.index[position])
                positions[group_index] += 1
                ticker_selected += 1
                progressed = True
                if ticker_selected >= per_ticker:
                    break
            if ticker_selected >= per_ticker or not progressed:
                break
        selected_by_ticker.append(ticker_indices)

    # Recovery is deliberately safety-bounded and resumable.  Returning one
    # complete ticker before the next would let every short pass concentrate
    # on whichever symbol sorts first, producing a badly skewed partial
    # corpus.  Interleave the already month/domain-balanced ticker samples so
    # each pass advances every company while preserving the exact same
    # selected rows and deterministic order.
    selected_indices: list[Any] = []
    longest = max((len(indices) for indices in selected_by_ticker), default=0)
    for position in range(longest):
        for indices in selected_by_ticker:
            if position < len(indices):
                selected_indices.append(indices[position])
    selected = working.loc[selected_indices].copy()
    return selected.drop(columns=["_month", "_selection_key"], errors="ignore").reset_index(
        drop=True
    )


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        if tag.casefold() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1
        if tag.casefold() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)


def extract_html_text(html: str) -> tuple[str, str]:
    parser = _ReadableHTML()
    parser.feed(str(html or ""))
    title = " ".join(parser.title_parts).strip()
    body = " ".join(parser.text_parts).strip()
    return title, body


def parse_common_crawl_record(payload: bytes) -> tuple[str, str]:
    """Extract title/body from a gzip-compressed WARC range response."""

    try:
        decoded = gzip.decompress(payload)
    except OSError:
        decoded = payload
    # Skip WARC headers and embedded HTTP headers.
    first = decoded.find(b"\r\n\r\n")
    second = decoded.find(b"\r\n\r\n", first + 4) if first >= 0 else -1
    body = decoded[(second + 4 if second >= 0 else first + 4 if first >= 0 else 0) :]
    html = body.decode("utf-8", errors="replace")
    return extract_html_text(html)


class CommonCrawlClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
        max_retries: int = DEFAULT_COMMON_CRAWL_RETRIES,
        user_agent: str | None = None,
        transport: Any | None = None,
    ) -> None:
        if httpx is None:
            raise RuntimeError(
                "Common Crawl recovery requires httpx. Install the project requirements first."
            )
        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
            headers={
                "User-Agent": user_agent
                or os.getenv(
                    "STOCKSENSE_CRAWL_USER_AGENT",
                    "StockSense academic research/1.0",
                )
            },
        )
        self.request_interval_seconds = max(
            DEFAULT_REQUEST_INTERVAL_SECONDS, float(request_interval_seconds)
        )
        self.max_retries = max(1, int(max_retries))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CommonCrawlClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Make a bounded request and stop on sustained Common Crawl throttling."""

        for attempt in range(self.max_retries):
            try:
                response = self._client.request(method, url, **kwargs)
            except Exception as exc:
                if httpx is not None and isinstance(exc, httpx.HTTPError):
                    if attempt + 1 >= self.max_retries:
                        raise
                    time.sleep(min(60.0, 2.0**attempt))
                    continue
                raise
            if response.status_code in {429, 502, 503, 504}:
                if attempt + 1 >= self.max_retries:
                    raise CommonCrawlRateLimitError(
                        f"Common Crawl remained unavailable with HTTP "
                        f"{response.status_code} after {self.max_retries} attempts; "
                        "the checkpoint is safe to resume later."
                    )
                time.sleep(min(60.0, 2.0**attempt))
                continue
            return response
        raise CommonCrawlRateLimitError("Common Crawl request retry budget exhausted")

    def collections(self) -> list[dict[str, Any]]:
        response = self._request("GET", COMMON_CRAWL_COLLECTIONS_URL)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Common Crawl collection response is not a list")
        return [dict(item) for item in payload]

    @staticmethod
    def candidate_indexes(
        collections: Sequence[Mapping[str, Any]],
        publication_year: int,
        publication_at: pd.Timestamp | None = None,
        limit: int = 3,
    ) -> list[str]:
        candidates: list[tuple[float, str]] = []
        publication = pd.Timestamp(publication_at) if publication_at is not None else None
        if publication is not None and publication.tzinfo is None:
            publication = publication.tz_localize("UTC")
        for item in collections:
            identifier = str(item.get("id", ""))
            match = re.match(r"CC-MAIN-(\d{4})-(\d{2})$", identifier)
            if not match or int(match.group(1)) not in {
                int(publication_year),
                int(publication_year) + 1,
            }:
                continue
            if publication is None:
                distance = 0.0
            else:
                approximate_start = pd.Timestamp(
                    date(int(match.group(1)), 1, 1), tz="UTC"
                ) + pd.Timedelta(weeks=max(0, int(match.group(2)) - 1))
                distance = abs((approximate_start - publication).total_seconds())
            candidates.append((distance, identifier))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [identifier for _, identifier in candidates[: max(1, int(limit))]]

    def lookup(self, index_id: str, url: str) -> list[dict[str, Any]]:
        endpoint = f"https://index.commoncrawl.org/{index_id}-index"
        response = self._request(
            "GET",
            endpoint, params={"url": url, "output": "json", "filter": "status:200"}
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        records: list[dict[str, Any]] = []
        for line in response.text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def fetch_record(self, record: Mapping[str, Any]) -> tuple[str, str]:
        filename = str(record["filename"])
        offset = int(record["offset"])
        length = int(record["length"])
        response = self._request(
            "GET",
            f"{COMMON_CRAWL_DATA_URL}/{filename}",
            headers={"Range": f"bytes={offset}-{offset + length - 1}"},
        )
        response.raise_for_status()
        return parse_common_crawl_record(response.content)

    def recover(
        self,
        url: str,
        published_at: pd.Timestamp,
        *,
        collections: Sequence[Mapping[str, Any]],
        max_capture_lag_days: int = DEFAULT_CAPTURE_LAG_DAYS,
    ) -> dict[str, Any] | None:
        publication = pd.Timestamp(published_at)
        if publication.tzinfo is None:
            publication = publication.tz_localize("UTC")
        candidates: list[
            tuple[float, str, dict[str, Any], pd.Timestamp]
        ] = []
        for index_id in self.candidate_indexes(
            collections, publication.year, publication_at=publication, limit=3
        ):
            time.sleep(self.request_interval_seconds)
            records = self.lookup(index_id, url)
            for record in records:
                captured = pd.to_datetime(
                    str(record.get("timestamp", "")), format="%Y%m%d%H%M%S", utc=True, errors="coerce"
                )
                if pd.isna(captured):
                    continue
                lag = (captured - publication).total_seconds() / 86400.0
                if 0 <= lag <= max_capture_lag_days:
                    candidates.append((lag, index_id, record, captured))
        for lag, index_id, record, captured in sorted(
            candidates, key=lambda item: item[0]
        ):
            time.sleep(self.request_interval_seconds)
            try:
                title, body = self.fetch_record(record)
            except CommonCrawlRateLimitError:
                raise
            except (OSError, ValueError, KeyError):
                continue
            if not title and not body:
                continue
            if len(re.findall(r"\b\w+\b", body)) < 100:
                continue
            return {
                "title": title,
                "description": body[:4000],
                "content_snippet": body[:12000],
                "commoncrawl_index": index_id,
                "commoncrawl_capture_at": captured.isoformat(),
                "commoncrawl_capture_lag_days": round(float(lag), 3),
                "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            }
        return None


def recover_common_crawl_content(
    frame: pd.DataFrame,
    *,
    client: CommonCrawlClient | None = None,
    max_capture_lag_days: int = DEFAULT_CAPTURE_LAG_DAYS,
    max_articles: int | None = None,
    checkpoint_path: Path | str | None = None,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    max_runtime_minutes: float | None = 45.0,
    heartbeat_path: Path | str | None = None,
) -> pd.DataFrame:
    """Recover article text with bounded requests, caching and resume support."""

    owned_client = client is None
    crawler = client or CommonCrawlClient(
        request_interval_seconds=request_interval_seconds
    )
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("key"):
                completed[str(record["key"])] = record

    def row_key(item: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            f"{str(item.get('ticker', '')).upper()}|{str(item.get('url', '')).strip()}".encode(
                "utf-8"
            )
        ).hexdigest()

    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def save_state(value: Mapping[str, Any]) -> None:
        if not checkpoint:
            return
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(dict(value), ensure_ascii=False, default=str) + "\n"
            )
    started = time.monotonic()
    attempted = 0
    stop_reason: str | None = None
    try:
        collections = crawler.collections()
        rows: list[dict[str, Any]] = []
        source = frame.head(max_articles) if max_articles is not None else frame
        for item in source.to_dict(orient="records"):
            if (
                max_runtime_minutes is not None
                and (time.monotonic() - started) / 60.0 >= max_runtime_minutes
            ):
                stop_reason = "maximum runtime reached"
                break
            key = row_key(item)
            previous = completed.get(key)
            if previous and previous.get("status") == "recovered":
                payload = previous.get("payload")
                if isinstance(payload, dict):
                    rows.append(payload)
                continue
            if previous and previous.get("status") in {"no_capture", "invalid"}:
                continue
            attempted += 1
            try:
                recovered = crawler.recover(
                    str(item["url"]),
                    pd.Timestamp(item["published_at"]),
                    collections=collections,
                    max_capture_lag_days=max_capture_lag_days,
                )
            except CommonCrawlRateLimitError:
                # Persist the boundary before stopping. A later invocation can
                # inspect the checkpoint and resume deliberately, rather than
                # retrying the same URL in an uncontrolled loop.
                save_state({"key": key, "status": "rate_limited"})
                raise
            if recovered:
                payload = {**item, **recovered}
                rows.append(payload)
                save_state({"key": key, "status": "recovered", "payload": payload})
            else:
                save_state({"key": key, "status": "no_capture"})
            _write_heartbeat(
                heartbeat_path,
                {
                    "status": "running",
                    "input_rows": int(len(source)),
                    "attempted_this_run": attempted,
                    "recovered_rows": len(rows),
                    "last_key": key,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        _write_heartbeat(
            heartbeat_path,
            {
                "status": "paused" if stop_reason else "completed",
                "stop_reason": stop_reason,
                "input_rows": int(len(source)),
                "attempted_this_run": attempted,
                "recovered_rows": len(rows),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return pd.DataFrame(rows)
    except CommonCrawlRateLimitError:
        # The already-written checkpoint is the resume boundary. Do not sleep
        # for 24 hours or retry in a loop that could worsen the block.
        _write_heartbeat(
            heartbeat_path,
            {
                "status": "rate_limited",
                "attempted_this_run": attempted,
                "stop_reason": "Common Crawl rate limit",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        if owned_client:
            crawler.close()


def filter_and_score_historical_articles(
    frame: pd.DataFrame,
    *,
    score_sentiment: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply production relevance rules before optional local FinBERT scoring."""

    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for article in frame.to_dict(orient="records"):
        hard_reason = classify_non_news_source(
            article.get("url"), article.get("source_domain")
        )
        if hard_reason:
            article["automatic_relevance"] = "hard_excluded"
            article["relevance_reasons"] = [hard_reason]
            article["automatic_action"] = "exclude"
            article["requires_user_approval"] = False
            review.append(article)
            continue
        decision = classify_entity_relevance(
            str(article.get("ticker", "")),
            str(article.get("title", "")),
            str(article.get("description", "")),
        )
        article.update(decision.to_dict())
        if decision.status == "included":
            article["automatic_action"] = "include"
            article["requires_user_approval"] = False
            accepted.append(article)
        else:
            article["automatic_action"] = (
                "needs_approval" if decision.status == "review" else "exclude"
            )
            article["requires_user_approval"] = decision.status == "review"
            review.append(article)
    if accepted and score_sentiment:
        add_sentiment_scores(accepted, load_sentiment_model())
    return pd.DataFrame(accepted), pd.DataFrame(review)


def build_entity_approval_queue(excluded: pd.DataFrame) -> pd.DataFrame:
    """Return only borderline entity matches that require user approval.

    High-confidence company matches are accepted automatically and clear
    collisions/package registries are excluded automatically.  The approval
    queue is intentionally limited to the classifier's ``review`` band so a
    user never has to inspect every recovered article.
    """

    if excluded.empty:
        return pd.DataFrame(
            columns=[
                "article_id",
                "ticker",
                "company_name",
                "title",
                "url",
                "source_domain",
                "automatic_relevance",
                "relevance_score",
                "relevance_reasons",
                "matched_entity_terms",
                "collision_signals",
                "automatic_action",
                "requires_user_approval",
                "approval_status",
            ]
        )
    queue = excluded.loc[
        excluded.get("automatic_relevance", pd.Series(dtype=str)).astype(str).eq("review")
    ].copy()
    if queue.empty:
        queue["approval_status"] = pd.Series(dtype=str)
    else:
        queue["approval_status"] = "pending"
    return queue.reindex(
        columns=[
            "article_id",
            "ticker",
            "company_name",
            "title",
            "url",
            "source_domain",
            "automatic_relevance",
            "relevance_score",
            "relevance_reasons",
            "matched_entity_terms",
            "collision_signals",
            "automatic_action",
            "requires_user_approval",
            "approval_status",
        ]
    )


def apply_entity_approvals(
    accepted: pd.DataFrame,
    excluded: pd.DataFrame,
    approvals: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply decisions for the borderline queue without reopening auto decisions."""

    if approvals is None or approvals.empty or excluded.empty:
        return accepted, excluded
    required = {"ticker", "url", "approval_status"}
    if not required.issubset(approvals.columns):
        raise ValueError(
            "Approval file must contain ticker, url and approval_status columns"
        )
    decisions: dict[tuple[str, str], str] = {}
    for row in approvals.to_dict(orient="records"):
        status = str(row.get("approval_status", "")).strip().casefold()
        if status not in {"approved", "include", "included", "rejected", "exclude", "excluded"}:
            continue
        key = (str(row.get("ticker", "")).upper().strip(), str(row.get("url", "")).strip())
        decisions[key] = status
    if not decisions:
        return accepted, excluded
    accepted_rows = accepted.to_dict(orient="records")
    remaining: list[dict[str, Any]] = []
    for row in excluded.to_dict(orient="records"):
        key = (str(row.get("ticker", "")).upper().strip(), str(row.get("url", "")).strip())
        status = decisions.get(key)
        if status in {"approved", "include", "included"} and row.get("automatic_relevance") == "review":
            row["automatic_action"] = "include_by_user"
            row["requires_user_approval"] = False
            row["approval_status"] = "approved"
            accepted_rows.append(row)
        elif status in {"rejected", "exclude", "excluded"} and row.get("automatic_relevance") == "review":
            row["automatic_action"] = "exclude_by_user"
            row["requires_user_approval"] = False
            row["approval_status"] = "rejected"
            remaining.append(row)
        else:
            remaining.append(row)
    return pd.DataFrame(accepted_rows), pd.DataFrame(remaining)


def load_entity_approvals(path: Path | str) -> pd.DataFrame:
    """Load user decisions exported from the automatic approval queue."""

    value = Path(path)
    if value.suffix.casefold() == ".csv":
        return pd.read_csv(value)
    if value.suffix.casefold() in {".json", ".jsonl"}:
        return pd.read_json(value, lines=value.suffix.casefold() == ".jsonl")
    raise ValueError("Entity approvals must be CSV, JSON or JSONL")


def build_manual_entity_review_template(
    accepted: pd.DataFrame,
    excluded: pd.DataFrame,
    *,
    per_ticker: int = 20,
) -> pd.DataFrame:
    """Build an included/excluded audit sample for precision and recall."""

    samples: list[pd.DataFrame] = []
    for ticker in sorted(
        set(accepted.get("ticker", pd.Series(dtype=str)).astype(str))
        | set(excluded.get("ticker", pd.Series(dtype=str)).astype(str))
    ):
        included = accepted.loc[
            accepted.get("ticker", pd.Series(dtype=str)).astype(str) == ticker
        ].copy()
        excluded_rows = excluded.loc[
            excluded.get("ticker", pd.Series(dtype=str)).astype(str) == ticker
        ].copy()
        for group, automatic in ((included, True), (excluded_rows, False)):
            if group.empty:
                continue
            if len(group) > per_ticker:
                group = group.sample(n=per_ticker, random_state=42)
            group = group.copy()
            group["automatic_included"] = automatic
            group["actually_related"] = ""
            group["review_notes"] = ""
            samples.append(group)
    if not samples:
        return pd.DataFrame(
            columns=[
                "article_id",
                "ticker",
                "title",
                "url",
                "source_domain",
                "automatic_relevance",
                "relevance_score",
                "automatic_included",
                "actually_related",
                "review_notes",
            ]
        )
    return pd.concat(samples, ignore_index=True).reindex(
        columns=[
            "article_id",
            "ticker",
            "title",
            "url",
            "source_domain",
            "automatic_relevance",
            "relevance_score",
            "automatic_included",
            "actually_related",
            "review_notes",
        ]
    )


def build_gdelt_pilot_report(
    indexed: pd.DataFrame,
    recovered: pd.DataFrame,
    accepted: pd.DataFrame,
    *,
    output_dir: Path = DEFAULT_GDELT_DIR,
    manually_reviewed: pd.DataFrame | None = None,
    llm_labelled: pd.DataFrame | None = None,
    feature_frame: pd.DataFrame | None = None,
    pilot_year: int = DEFAULT_PILOT_YEAR,
    pilot_policy: Mapping[str, Any] | None = None,
) -> GDELTPilotResult:
    """Write the free-only go/no-go report before full collection."""

    metadata = load_stock_metadata()
    tickers = set(metadata)
    gates = {
        "minimum_indexed_urls_per_ticker": DEFAULT_INDEXED_MIN_URLS_PER_TICKER,
        "minimum_indexed_urls_by_ticker": DEFAULT_INDEXED_MIN_URLS_BY_TICKER,
        "minimum_indexed_domains_per_ticker": DEFAULT_INDEXED_MIN_DOMAINS_PER_TICKER,
        "minimum_indexed_months_per_ticker": DEFAULT_INDEXED_MIN_MONTHS_PER_TICKER,
        "minimum_text_recovery_rate": 0.60,
        "minimum_entity_precision": 0.90,
        "minimum_entity_recall": 0.75,
        "minimum_timestamp_completeness": 0.95,
        "minimum_source_domains_per_ticker": 2,
        "minimum_accepted_articles": 200,
        "minimum_accepted_per_ticker": 15,
    }
    gates.update(load_policy().get("year_quality_gates", {}))
    if pilot_policy:
        gates.update(dict(pilot_policy))
    indexed_gate = evaluate_indexed_candidate_gate(
        indexed,
        minimum_urls_per_ticker=int(gates["minimum_indexed_urls_per_ticker"]),
        minimum_urls_by_ticker=dict(
            gates.get(
                "minimum_indexed_urls_by_ticker",
                DEFAULT_INDEXED_MIN_URLS_BY_TICKER,
            )
        ),
        minimum_domains_per_ticker=int(
            gates["minimum_indexed_domains_per_ticker"]
        ),
        minimum_months_per_ticker=int(gates["minimum_indexed_months_per_ticker"]),
        required_tickers=sorted(tickers),
    )
    evaluation_articles = accepted.copy()
    llm_metrics: dict[str, Any] | None = None
    reference_type = "human_review" if manually_reviewed is not None else "none"
    if llm_labelled is not None:
        from pipeline.llm_bursa_news_labeler import consensus_metrics

        labels = llm_labelled.copy()
        completed = labels.get(
            "labelling_completed", pd.Series(False, index=labels.index)
        ).map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"1", "true", "yes", "y"}
        )
        consensus = (
            labels.get("consensus_status", pd.Series("", index=labels.index))
            .astype(str)
            .str.casefold()
            .eq("accepted")
        )
        entity_truth = labels.get(
            "entity_relevant", pd.Series(False, index=labels.index)
        ).map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"1", "true", "yes", "y"}
        )
        evaluation_articles = labels.loc[completed & consensus & entity_truth].copy()
        llm_metrics = consensus_metrics(labels)
        reference_type = "dual_llm_consensus_pseudolabel"
    accepted_tickers = set(
        evaluation_articles.get("ticker", pd.Series(dtype=str)).astype(str)
    )
    source_counts = (
        evaluation_articles.assign(
            _domain=evaluation_articles.get(
                "source_domain", pd.Series("", index=evaluation_articles.index)
            )
        )
        .groupby("ticker")["_domain"]
        .nunique()
        .to_dict()
        if not evaluation_articles.empty
        else {}
    )
    precision: float | None = None
    recall: float | None = None
    reviewed_rows = 0
    if manually_reviewed is not None and {
        "automatic_included",
        "actually_related",
    }.issubset(manually_reviewed.columns):
        def boolean_column(values: pd.Series) -> pd.Series:
            return values.map(
                lambda value: value
                if isinstance(value, bool)
                else str(value).strip().casefold() in {"1", "true", "yes", "y"}
            )

        review = manually_reviewed.copy()
        review = review.loc[review["actually_related"].astype(str).str.strip().ne("")]
        predicted = boolean_column(review["automatic_included"])
        truth = boolean_column(review["actually_related"])
        reviewed_rows = len(review)
        selected = int(predicted.sum())
        actual_positive = int(truth.sum())
        precision = float((predicted & truth).sum() / selected) if selected else 0.0
        recall = float((predicted & truth).sum() / actual_positive) if actual_positive else 0.0
    elif llm_labelled is not None:
        review = llm_labelled.copy()
        completed = review.get(
            "labelling_completed", pd.Series(False, index=review.index)
        ).map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"1", "true", "yes", "y"}
        )
        consensus = (
            review.get("consensus_status", pd.Series("", index=review.index))
            .astype(str)
            .str.casefold()
            .eq("accepted")
        )
        review = review.loc[completed & consensus].copy()
        predicted = review.get(
            "automatic_entity_relevant", pd.Series(False, index=review.index)
        ).map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"1", "true", "yes", "y"}
        )
        truth = review.get(
            "entity_relevant", pd.Series(False, index=review.index)
        ).map(
            lambda value: value
            if isinstance(value, bool)
            else str(value).strip().casefold() in {"1", "true", "yes", "y"}
        )
        reviewed_rows = len(review)
        selected = int(predicted.sum())
        actual_positive = int(truth.sum())
        precision = float((predicted & truth).sum() / selected) if selected else 0.0
        recall = (
            float((predicted & truth).sum() / actual_positive)
            if actual_positive
            else 0.0
        )
    recovered_rate = float(len(recovered) / len(indexed)) if len(indexed) else 0.0
    timestamp_values = pd.to_datetime(
        evaluation_articles.get("published_at", pd.Series(dtype=str)),
        utc=True,
        errors="coerce",
    )
    timestamp_completeness = (
        float(timestamp_values.notna().mean()) if len(evaluation_articles) else 0.0
    )
    coverage_rates: dict[str, float] = {}
    if feature_frame is not None and not feature_frame.empty:
        features = feature_frame.copy()
        ticker_column = "ticker" if "ticker" in features else "Ticker"
        horizon_column = "horizon_days" if "horizon_days" in features else "horizon"
        if "news_coverage_score" in features:
            eligible = pd.to_numeric(
                features["news_coverage_score"], errors="coerce"
            ).ge(0.5)
        elif "coverage_confidence" in features:
            eligible = features["coverage_confidence"].astype(str).str.casefold().isin(
                {"medium", "high"}
            )
        else:
            eligible = pd.Series(False, index=features.index)
        features["_medium_high"] = eligible
        grouped = features.groupby([ticker_column, horizon_column])["_medium_high"].mean()
        coverage_rates = {
            f"{str(ticker).upper()}:{int(horizon)}d": float(rate)
            for (ticker, horizon), rate in grouped.items()
        }
    article_coverage = {
        ticker: int(
            (
                evaluation_articles.get("ticker", pd.Series(dtype=str)) == ticker
            ).sum()
        )
        for ticker in sorted(tickers)
    }
    source_diversity_passed = all(source_counts.get(ticker, 0) >= 2 for ticker in tickers)
    all_companies_present = tickers.issubset(accepted_tickers)
    per_ticker_volume_passed = all(
        article_coverage[ticker] >= int(gates["minimum_accepted_per_ticker"])
        for ticker in tickers
    )
    ready = bool(
        indexed_gate["ready_for_recovery"]
        and all_companies_present
        and source_diversity_passed
        and recovered_rate >= float(gates["minimum_text_recovery_rate"])
        and timestamp_completeness >= float(gates["minimum_timestamp_completeness"])
        and len(evaluation_articles) >= int(gates["minimum_accepted_articles"])
        and per_ticker_volume_passed
        and precision is not None
        and precision >= float(gates["minimum_entity_precision"])
        and recall is not None
        and recall >= float(gates["minimum_entity_recall"])
        and (
            llm_metrics is None
            or (
                llm_metrics["parse_success_rate"]
                >= float(
                    load_policy()
                    .get("labeling", {})
                    .get("minimum_parse_success_rate", 0.95)
                )
                and llm_metrics["consensus_coverage"]
                >= float(
                    load_policy()
                    .get("labeling", {})
                    .get("minimum_consensus_coverage", 0.80)
                )
                and llm_metrics.get("entity_kappa") is not None
                and llm_metrics["entity_kappa"]
                >= float(
                    load_policy()
                    .get("labeling", {})
                    .get("minimum_entity_kappa", 0.75)
                )
                and llm_metrics.get("sentiment_kappa") is not None
                and llm_metrics["sentiment_kappa"]
                >= float(
                    load_policy()
                    .get("labeling", {})
                    .get("minimum_sentiment_kappa", 0.70)
                )
            )
        )
    )
    report = {
        "schema_version": "gdelt-commoncrawl-pilot-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_year": int(pilot_year),
        "indexed_rows": int(len(indexed)),
        "indexed_candidate_gate": indexed_gate,
        "recovered_rows": int(len(recovered)),
        "accepted_rows": int(len(evaluation_articles)),
        "text_recovery_rate": recovered_rate,
        "entity_precision": precision,
        "entity_recall": recall,
        "reviewed_rows": reviewed_rows,
        "entity_reference_type": reference_type,
        "llm_consensus_metrics": llm_metrics,
        "timestamp_completeness": timestamp_completeness,
        "entity_precision_requirement": float(gates["minimum_entity_precision"]),
        "entity_recall_requirement": float(gates["minimum_entity_recall"]),
        "text_recovery_requirement": float(gates["minimum_text_recovery_rate"]),
        "timestamp_completeness_requirement": float(gates["minimum_timestamp_completeness"]),
        "minimum_accepted_articles": int(gates["minimum_accepted_articles"]),
        "minimum_accepted_per_ticker": int(gates["minimum_accepted_per_ticker"]),
        "medium_high_coverage_by_ticker_horizon": coverage_rates,
        "medium_high_coverage_is_informational": True,
        "all_companies_present": all_companies_present,
        "source_diversity_passed": source_diversity_passed,
        "per_ticker_volume_passed": per_ticker_volume_passed,
        "accepted_articles_by_ticker": article_coverage,
        "unique_sources_by_ticker": {
            ticker: int(source_counts.get(ticker, 0)) for ticker in sorted(tickers)
        },
        "ready_for_full_collection": ready,
        "decision_notes": (
            []
            if ready
            else [
                "Do not train or promote fusion until all pilot gates pass.",
                "Do not start additional collection years until the indexed and full pilot gates pass for all ten companies.",
                "Entity precision and recall require accepted dual-LLM consensus labels for included and excluded samples.",
                "The pilot uses 2023 so 2024 validation and 2025 test remain untouched.",
                "Feature coverage is reported separately after the complete historical archive exists.",
                "No paid provider or billing-enabled cloud query may be used to bypass a failed gate.",
            ]
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"pilot-{int(pilot_year)}-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return GDELTPilotResult(
        report_path,
        len(indexed),
        len(recovered),
        len(evaluation_articles),
        ready,
    )


def run_gdelt_enrichment(
    export_path: str,
    *,
    output_dir: str | None = None,
    max_articles: int | None = None,
    pilot_per_ticker: int | None = None,
    checkpoint_path: str | None = None,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    capture_lag_days: int = DEFAULT_CAPTURE_LAG_DAYS,
    score_sentiment: bool = False,
    approvals_path: str | None = None,
    enforce_indexed_gate: bool = False,
    max_runtime_minutes: float | None = 45.0,
    heartbeat_path: str | None = None,
) -> dict[str, Any]:
    """CLI orchestration for bounded, resumable recovery and filtering."""

    target = Path(output_dir) if output_dir else DEFAULT_GDELT_DIR
    target.mkdir(parents=True, exist_ok=True)
    raw_dir = target / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    indexed = load_gdelt_export(Path(export_path))
    indexed_gate = evaluate_indexed_candidate_gate(indexed)
    indexed_gate_path = raw_dir / "indexed-candidate-gate.json"
    indexed_gate_path.write_text(
        json.dumps(indexed_gate, indent=2), encoding="utf-8"
    )
    if enforce_indexed_gate and not indexed_gate["ready_for_recovery"]:
        payload = {
            "blocked": True,
            "reason": "Indexed candidate gate failed; Common Crawl was not started.",
            "indexed_rows": int(len(indexed)),
            "indexed_candidate_gate": indexed_gate,
            "indexed_candidate_gate_path": str(indexed_gate_path),
        }
        print(json.dumps(payload, indent=2))
        return payload
    selection = indexed
    if pilot_per_ticker is not None:
        selection = select_pilot_candidates(indexed, per_ticker=pilot_per_ticker)
    if max_articles is not None:
        selection = selection.head(max_articles)
    state_path = Path(checkpoint_path) if checkpoint_path else target / "raw" / "recovery-state.jsonl"
    recovery_heartbeat = (
        Path(heartbeat_path)
        if heartbeat_path
        else target / "raw" / "recovery-heartbeat.json"
    )
    recovered = (
        recover_common_crawl_content(
            selection,
            max_capture_lag_days=capture_lag_days,
            checkpoint_path=state_path,
            request_interval_seconds=request_interval_seconds,
            max_runtime_minutes=max_runtime_minutes,
            heartbeat_path=recovery_heartbeat,
        )
        if not selection.empty
        else selection.copy()
    )
    accepted, review = filter_and_score_historical_articles(
        recovered, score_sentiment=score_sentiment
    )
    entity_candidates = pd.concat([accepted, review], ignore_index=True, sort=False)
    approvals = load_entity_approvals(approvals_path) if approvals_path else None
    accepted, review = apply_entity_approvals(accepted, review, approvals)
    selected_path = raw_dir / "pilot-indexed.csv"
    selection.to_csv(selected_path, index=False)
    recovered_path = raw_dir / "commoncrawl-recovered.jsonl"
    candidate_path = raw_dir / "entity-candidates.jsonl"
    review_path = raw_dir / "entity-review.jsonl"
    recovered.to_json(recovered_path, orient="records", lines=True, date_format="iso")
    entity_candidates.to_json(
        candidate_path, orient="records", lines=True, date_format="iso"
    )
    review.to_json(review_path, orient="records", lines=True, date_format="iso")
    approval_queue = build_entity_approval_queue(review)
    approval_queue_path = raw_dir / "entity-approval-queue.jsonl"
    approval_queue.to_json(
        approval_queue_path, orient="records", lines=True, date_format="iso"
    )
    accepted_path = raw_dir / "accepted-scored.jsonl"
    accepted.to_json(accepted_path, orient="records", lines=True, date_format="iso")
    review_template = build_manual_entity_review_template(accepted, review)
    manual_review_path = raw_dir / "manual-entity-review-template.csv"
    review_template.to_csv(manual_review_path, index=False)
    imported = (
        import_historical_news(
            accepted_path,
            output_dir=Path(output_dir).parent
            if output_dir
            else DEFAULT_FUSION_DATA_DIR,
            provider="GDELT/CommonCrawl",
        )
        if not accepted.empty
        else None
    )
    payload = {
        "indexed_rows": int(len(indexed)),
        "blocked": False,
        "indexed_candidate_gate": indexed_gate,
        "indexed_candidate_gate_path": str(indexed_gate_path),
        "selected_rows": int(len(selection)),
        "recovered_rows": int(len(recovered)),
        "accepted_rows": int(len(accepted)),
        "review_rows": int(len(review)),
        "import": imported.to_dict() if imported is not None else None,
        "raw_files": [
            str(selected_path),
            str(recovered_path),
            str(candidate_path),
            str(review_path),
            str(approval_queue_path),
            str(accepted_path),
        ],
        "automatic_entity_decisions": {
            "accepted_without_approval": int(len(accepted)),
            "excluded_without_approval": int(
                review.get("requires_user_approval", pd.Series(dtype=bool))
                .astype(bool)
                .eq(False)
                .sum()
            ),
            "approval_required": int(len(approval_queue)),
        },
        "entity_approval_queue": str(approval_queue_path),
        "approvals_applied": str(approvals_path) if approvals_path else None,
        "manual_review_template": str(manual_review_path),
        "checkpoint": str(state_path),
        "heartbeat": str(recovery_heartbeat),
        "max_runtime_minutes": max_runtime_minutes,
        "capture_lag_days": int(capture_lag_days),
        "request_interval_seconds": float(request_interval_seconds),
        "sentiment_scored": bool(score_sentiment),
    }
    print(json.dumps(payload, indent=2))
    return payload
