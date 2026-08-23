"""Quality-first preparation and validation of a Bursa news sentiment dataset.

This module is offline-only. It converts recovered historical articles into a
chronological template, prevents duplicate events from crossing split
boundaries, and validates either single labels or optional independent
LLM-consensus annotations. Dataset volume and agreement requirements are
controlled by the selected policy rather than hard-coded in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score

from pipeline.download_financial_news_data import clean_text, load_stock_metadata
from pipeline.historical_news_fusion_data import normalize_historical_news_frame


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "bursa_news_dataset_policy.json"
DEFAULT_EXTERNAL_SINGLE_WEAK_POLICY_PATH = (
    PROJECT_ROOT / "config" / "bursa_news_dataset_policy_external_single_weak.json"
)
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "news-evidence-benchmark"
    / "bursa-news-llm-assisted-label.jsonl"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "results" / "bursa-news-dataset" / "readiness-report.json"
)
DATASET_SCHEMA_VERSION = "bursa-news-llm-assisted-label-v1"
SENTIMENT_LABELS = ("negative", "neutral", "positive")
# Canonical provenance for the active Bursa corpus.  These labels were
# reviewed with an LLM workflow; they are useful weak supervision, not
# independently annotated human ground truth.
LLM_ASSISTED_LABEL_SOURCE = "llm_assisted_label"
LLM_ASSISTED_LABEL_DISPLAY = "LLM-assisted label"

# Historical workflow identifiers are migration input only. New datasets and
# every active writer emit ``llm_assisted_label``.
MIGRATION_LABEL_SOURCES = frozenset(
    {
        "external_single_weak",
        "external_consensus",
        "user_or_codex_single_label",
        "user_adjudicated_after_llm_abstention",
    }
)
EVENT_SIMILARITY_THRESHOLD = 0.88
EVENT_WINDOW_DAYS = 3
OPTIONAL_RECOVERY_PROVENANCE_COLUMNS = (
    "text_source",
    "is_pseudo_document",
    "recovery_status",
    "recovery_failure_reason",
    "recovered_at",
    "ngram_file_timestamps",
    "ngram_record_count",
    "deciles_present",
    "unique_token_count",
    "overlap_attachment_rate",
    "reconstruction_quality_score",
    "reconstruction_quality_tier",
    "publisher_full_text_available",
)


@dataclass(frozen=True)
class BursaNewsReadiness:
    ready: bool
    policy_version: str
    dataset_path: str
    labelled_relevant_rows: int
    split_counts: dict[str, int]
    ticker_counts: dict[str, int]
    test_ticker_counts: dict[str, int]
    class_counts: dict[str, int]
    inter_annotator_kappa: float | None
    entity_precision: float | None
    entity_recall: float | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def reason(self) -> str:
        return "ready" if self.ready else "; ".join(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "reason": self.reason}


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_table(path: Path | str) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(source, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported Bursa news file type: {source}")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    elif suffix == ".jsonl":
        frame.to_json(path, orient="records", lines=True, date_format="iso")
    else:
        raise ValueError("Output must be CSV, Parquet, PQ, or JSONL")


def _normalise_url(value: Any) -> str:
    raw = clean_text(value)
    if not raw:
        return ""
    parsed = urlparse(raw)
    query = "&".join(
        part
        for part in parsed.query.split("&")
        if part
        and not part.casefold().startswith(("utm_", "fbclid=", "gclid=", "mc_cid="))
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


def _normalise_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).casefold()).strip()


def canonical_label_source(value: Any) -> str:
    """Map legacy label provenance to the active LLM-assisted name.

    ``external_single_weak`` was an experiment-time classification.  It is
    accepted only as migration input and is never emitted by the active
    Bursa-labelled corpus.
    """

    source = clean_text(value).casefold().replace("-", "_").replace(" ", "_")
    if source in {
        "llm_assisted_label",
        "llm_assisted",
        "llm_label",
        "codex_reviewed",
        "codex_single_review",
    }:
        return LLM_ASSISTED_LABEL_SOURCE
    if source in MIGRATION_LABEL_SOURCES or source in {"single", "single_label"}:
        return LLM_ASSISTED_LABEL_SOURCE
    return source


def _content_hash(row: Mapping[str, Any]) -> str:
    body = clean_text(row.get("content_snippet")) or clean_text(row.get("description"))
    basis = "|".join(
        (
            clean_text(row.get("ticker")).upper(),
            _normalise_title(row.get("title")),
            re.sub(r"\s+", " ", body.casefold()).strip(),
        )
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _article_id(row: Mapping[str, Any]) -> str:
    existing = clean_text(row.get("article_id"))
    if existing:
        return existing
    basis = "|".join(
        (
            clean_text(row.get("ticker")).upper(),
            clean_text(row.get("published_at")),
            _normalise_title(row.get("title")),
            _normalise_url(row.get("url")),
        )
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def split_for_timestamp(value: Any) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return ""
    if timestamp.year <= 2023:
        return "training"
    if timestamp.year == 2024:
        return "validation"
    if timestamp.year == 2025:
        return "test"
    return ""


def _derive_event_ids(frame: pd.DataFrame) -> pd.Series:
    """Greedily cluster near-date, highly similar titles within each ticker."""
    event_ids = pd.Series("", index=frame.index, dtype="string")
    for ticker, ticker_frame in frame.groupby("ticker", sort=True):
        recent: list[tuple[pd.Timestamp, str, str]] = []
        ordered = ticker_frame.sort_values(["published_at", "article_id"])
        for index, row in ordered.iterrows():
            published = pd.Timestamp(row["published_at"])
            title = _normalise_title(row["title"]) or _normalise_title(row["model_text"])
            recent = [
                item
                for item in recent
                if abs((published - item[0]).total_seconds()) <= EVENT_WINDOW_DAYS * 86400
            ]
            matched = ""
            best_similarity = 0.0
            for _, prior_title, event_id in recent:
                similarity = SequenceMatcher(None, title, prior_title).ratio()
                if similarity >= EVENT_SIMILARITY_THRESHOLD and similarity > best_similarity:
                    matched = event_id
                    best_similarity = similarity
            if not matched:
                basis = f"{ticker}|{published.date().isoformat()}|{title}"
                matched = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]
            event_ids.at[index] = matched
            recent.append((published, title, matched))
    return event_ids


def _keep_event_in_latest_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep a duplicated event only in its latest chronological split.

    Reassigning an older article to a later split makes its declared split
    disagree with its publication year.  Dropping the older duplicate keeps
    both the time split and the event-isolation invariant true.
    """

    order = {"training": 0, "validation": 1, "test": 2}
    output = frame.copy()
    keep = pd.Series(True, index=output.index, dtype=bool)
    for _, indices in output.groupby("event_id").groups.items():
        chosen = max(
            (str(output.at[index, "split"]) for index in indices),
            key=lambda value: order.get(value, -1),
        )
        for index in indices:
            if str(output.at[index, "split"]) != chosen:
                keep.at[index] = False
    return output.loc[keep].copy()


def materialize_llm_assisted_dataset(source: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical Bursa schema from labelled recovery records.

    Legacy source values are retained only in ``label_source_original`` for
    auditability.  Active output uses ``llm_assisted_label`` throughout and
    explicitly does not claim independent human ground truth.
    """

    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if source.empty:
        return pd.DataFrame(columns=list(load_policy().get("required_columns", [])))

    output = source.copy()
    if "url" not in output and "source_url" in output:
        output["url"] = output["source_url"]
    if "source_name" not in output:
        output["source_name"] = output.get("source_domain", "")
    if "description" not in output:
        for candidate in ("article_text", "content_snippet", "label_text"):
            if candidate in output:
                output["description"] = output[candidate]
                break
        else:
            output["description"] = ""
    if "model_text" not in output:
        output["model_text"] = output.apply(
            lambda row: ". ".join(
                value
                for value in (
                    clean_text(row.get("title")),
                    clean_text(row.get("description")),
                )
                if value
            ),
            axis=1,
        )
    if "sentiment_label" not in output and "sentiment" in output:
        output["sentiment_label"] = output["sentiment"]

    required_identity = (
        "article_id",
        "ticker",
        "published_at",
        "title",
        "content_sha256",
        "entity_relevant",
        "sentiment_label",
    )
    missing_identity = [column for column in required_identity if column not in output]
    if missing_identity:
        raise ValueError(
            f"External labelled dataset is missing identity/label columns: {missing_identity}"
        )

    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["published_at"] = pd.to_datetime(
        output["published_at"], utc=True, errors="coerce"
    )
    if output["published_at"].isna().any():
        raise ValueError("External labelled dataset contains invalid published_at values")
    output["split"] = output["published_at"].map(split_for_timestamp)
    output = output.loc[output["split"].ne("")].copy()
    output["url"] = output.get("url", "").map(_normalise_url)
    output["entity_relevant"] = _coerce_bool(output["entity_relevant"])
    if "automatic_entity_relevant" in output:
        output["automatic_entity_relevant"] = _coerce_bool(
            output["automatic_entity_relevant"]
        )
    else:
        output["automatic_entity_relevant"] = False
    output["sentiment_label"] = (
        output["sentiment_label"].fillna("").astype(str).str.strip().str.casefold()
    )
    invalid_sentiment = output.loc[
        output["entity_relevant"]
        & ~output["sentiment_label"].isin(SENTIMENT_LABELS)
    ]
    if not invalid_sentiment.empty:
        raise ValueError(
            f"{len(invalid_sentiment)} relevant external rows lack a valid sentiment label"
        )
    output.loc[~output["entity_relevant"], "sentiment_label"] = ""
    output["label_source_original"] = output.get(
        "label_source", pd.Series("", index=output.index)
    ).map(clean_text)
    output["label_source"] = LLM_ASSISTED_LABEL_SOURCE
    output["label_provenance"] = LLM_ASSISTED_LABEL_SOURCE
    output["provenance"] = LLM_ASSISTED_LABEL_SOURCE
    output["annotation_agreement"] = LLM_ASSISTED_LABEL_SOURCE
    output["consensus_status"] = LLM_ASSISTED_LABEL_SOURCE
    output["consensus_reason"] = (
        "LLM-assisted label recorded under the active dataset policy; "
        "not independent human ground truth"
    )
    output["validation_status"] = LLM_ASSISTED_LABEL_SOURCE
    output["llm_assisted_label"] = True
    output["llm_assisted_label_at"] = output.get(
        "labelled_at", pd.Series("", index=output.index)
    )
    output["event_id"] = _derive_event_ids(output)
    output = _keep_event_in_latest_split(output)
    output = output.drop_duplicates("article_id", keep="last")
    output = output.drop_duplicates(["ticker", "content_sha256"], keep="last")

    required_columns = list(load_policy().get("required_columns", []))
    for column in required_columns:
        if column not in output:
            output[column] = ""
    for column in (
        "label_source_original",
        "label_provenance",
        "provenance",
        "validation_status",
        "llm_assisted_label",
        "llm_assisted_label_at",
    ):
        if column not in output:
            output[column] = ""
    provenance = [
        column
        for column in OPTIONAL_RECOVERY_PROVENANCE_COLUMNS
        if column in output and column not in required_columns
    ]
    validation_audit_columns = [
        column
        for column in (
            "label_source_original",
            "label_provenance",
            "provenance",
            "consensus_status",
            "consensus_reason",
            "validation_status",
            "llm_assisted_label",
            "llm_assisted_label_at",
        )
        if column in output
        and column not in required_columns
        and column not in provenance
    ]
    audit_columns = [
        column
        for column in (
            "confidence",
            "evidence",
            "rationale",
            "external_labelled_by",
            "external_label_status",
        )
        if column in output
        and column not in required_columns
        and column not in provenance
        and column not in validation_audit_columns
    ]
    return output[
        required_columns + provenance + validation_audit_columns + audit_columns
    ].sort_values(
        ["split", "ticker", "published_at", "article_id"]
    ).reset_index(drop=True)


def materialize_external_single_weak_dataset(source: pd.DataFrame) -> pd.DataFrame:
    """Deprecated compatibility alias for the LLM-assisted migration."""

    return materialize_llm_assisted_dataset(source)


def build_annotation_template(
    source: pd.DataFrame,
    *,
    recommended_start_year: int = 2019,
    approved_backfill_years: Sequence[int] = (),
) -> pd.DataFrame:
    """Prepare unique, chronologically split rows without creating labels."""
    try:
        metadata = load_stock_metadata()
    except FileNotFoundError:
        metadata = {}
    normalized = normalize_historical_news_frame(
        source, default_company_names=metadata
    )
    if normalized.empty:
        return pd.DataFrame()
    output = normalized.copy()
    output["published_at"] = pd.to_datetime(
        output["published_at"], utc=True, errors="coerce"
    )
    approved_years = {int(year) for year in approved_backfill_years}
    invalid_backfill = approved_years - {2015, 2016, 2017, 2018}
    if invalid_backfill:
        raise ValueError(
            f"Optional backfill years must be within 2015-2018: {sorted(invalid_backfill)}"
        )
    years = output["published_at"].dt.year
    output = output.loc[
        years.between(recommended_start_year, 2025) | years.isin(approved_years)
    ].copy()
    output["url"] = output["url"].map(_normalise_url)
    output["model_text"] = output.apply(
        lambda row: ". ".join(
            value
            for value in (
                clean_text(row.get("title")),
                clean_text(row.get("description"))
                or clean_text(row.get("content_snippet")),
            )
            if value
        ),
        axis=1,
    )
    output = output.loc[output["model_text"].str.strip().ne("")].copy()
    output["article_id"] = [
        _article_id(row) for row in output.to_dict(orient="records")
    ]
    output["content_sha256"] = [
        _content_hash(row) for row in output.to_dict(orient="records")
    ]
    output = output.drop_duplicates(["ticker", "url"], keep="first")
    output = output.drop_duplicates(["ticker", "content_sha256"], keep="first")
    # Historical archives do not provide a consistently trustworthy language
    # field. Annotators must confirm English instead of the pipeline silently
    # assuming it.
    output["language"] = ""
    output["automatic_entity_relevant"] = output["automatic_relevance"].eq("included")
    output["entity_relevant"] = ""
    output["sentiment_label"] = ""
    output["event_category"] = ""
    output["label_source"] = ""
    output["labeler_a_model"] = ""
    output["labeler_a_entity_relevant"] = ""
    output["labeler_a_sentiment"] = ""
    output["labeler_a_confidence"] = ""
    output["labeler_b_model"] = ""
    output["labeler_b_entity_relevant"] = ""
    output["labeler_b_sentiment"] = ""
    output["labeler_b_confidence"] = ""
    output["consensus_status"] = ""
    output["consensus_reason"] = ""
    output["prompt_version"] = ""
    output["prompt_sha256"] = ""
    output["labelled_at"] = ""
    output["annotator_1_label"] = ""
    output["annotator_2_label"] = ""
    output["adjudicated_label"] = ""
    output["annotation_agreement"] = "unlabelled"
    output["split"] = output["published_at"].map(split_for_timestamp)
    output = output.loc[output["split"].ne("")].copy()
    output["event_id"] = _derive_event_ids(output)
    output = _keep_event_in_latest_split(output)
    template_policy = load_policy()
    columns = list(template_policy.get("required_columns", []))
    columns.extend(
        column
        for column in template_policy.get("optional_columns", [])
        if column not in columns
    )
    for column in columns:
        if column not in output:
            output[column] = ""
    columns.extend(
        column
        for column in OPTIONAL_RECOVERY_PROVENANCE_COLUMNS
        if column in output and column not in columns
    )
    return output[columns].sort_values(
        ["split", "ticker", "published_at", "article_id"]
    ).reset_index(drop=True)


def resolve_independent_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in (
        "annotator_1_label",
        "annotator_2_label",
        "adjudicated_label",
        "sentiment_label",
    ):
        value = output.get(column, pd.Series("", index=output.index))
        output[column] = value.astype("string").str.strip().str.casefold()
    first = output["annotator_1_label"]
    second = output["annotator_2_label"]
    adjudicated = output["adjudicated_label"]
    double = first.isin(SENTIMENT_LABELS) & second.isin(SENTIMENT_LABELS)
    agreed = double & first.eq(second)
    disagreed = double & first.ne(second)
    single = first.isin(SENTIMENT_LABELS) & ~second.isin(SENTIMENT_LABELS)
    output.loc[agreed, "sentiment_label"] = first[agreed]
    output.loc[disagreed & adjudicated.isin(SENTIMENT_LABELS), "sentiment_label"] = (
        adjudicated[disagreed & adjudicated.isin(SENTIMENT_LABELS)]
    )
    output.loc[single, "sentiment_label"] = first[single]
    output["annotation_agreement"] = "unlabelled"
    output.loc[single, "annotation_agreement"] = "single"
    output.loc[agreed, "annotation_agreement"] = "agreed"
    output.loc[disagreed, "annotation_agreement"] = "disagreed"
    output.loc[disagreed & adjudicated.isin(SENTIMENT_LABELS), "annotation_agreement"] = (
        "adjudicated"
    )
    return output


def resolve_llm_consensus_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize accepted LLM consensus while leaving abstentions unlabelled."""

    output = frame.copy()
    status = (
        output.get("consensus_status", pd.Series("", index=output.index))
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    source = (
        output.get("label_source", pd.Series("", index=output.index))
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    sentiment = (
        output.get("sentiment_label", pd.Series("", index=output.index))
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    accepted = status.eq("accepted") & source.isin(
        {"llm_consensus", "user_adjudicated_after_llm_abstention"}
    )
    output["sentiment_label"] = ""
    output.loc[accepted & sentiment.isin(SENTIMENT_LABELS), "sentiment_label"] = (
        sentiment[accepted & sentiment.isin(SENTIMENT_LABELS)]
    )
    output.loc[~accepted, "entity_relevant"] = False
    output["annotation_agreement"] = "unlabelled"
    output.loc[accepted, "annotation_agreement"] = "llm_consensus"
    output.loc[status.eq("abstained"), "annotation_agreement"] = "llm_abstained"
    return output


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().casefold() in {"1", "true", "yes", "y"}
    )


def validate_bursa_news_dataset(
    frame: pd.DataFrame,
    *,
    policy: Mapping[str, Any] | None = None,
    dataset_path: str = "",
) -> BursaNewsReadiness:
    policy = dict(policy or load_policy())
    errors: list[str] = []
    warnings: list[str] = []
    required = [str(value) for value in policy.get("required_columns", [])]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        return BursaNewsReadiness(
            False,
            str(policy.get("policy_version", "unknown")),
            dataset_path,
            0,
            {},
            {},
            {},
            {},
            None,
            None,
            None,
            tuple(errors),
            tuple(warnings),
        )
    labeling_mode = str(
        policy.get("labeling", {}).get("mode", "llm_assisted_label")
    ).strip().casefold()
    if labeling_mode == "llm_consensus":
        data = resolve_llm_consensus_labels(frame)
    elif labeling_mode in {
        "llm_assisted_label",
        "llm_assisted",
        "external_single_weak",
        "single_label",
        "single",
        "codex_reviewed",
    }:
        data = frame.copy()
        data["sentiment_label"] = (
            data.get("sentiment_label", pd.Series("", index=data.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        data["label_source"] = data.get(
            "label_source", pd.Series("", index=data.index)
        ).map(canonical_label_source)
        if labeling_mode in {"codex_reviewed", "external_single_weak"}:
            warnings.append(
                "Legacy label-policy mode detected; active output is recorded as "
                "LLM-assisted label and is not independent human ground truth."
            )
        else:
            data["annotation_agreement"] = LLM_ASSISTED_LABEL_SOURCE
            warnings.append(
                "Labels are recorded as LLM-assisted labels; independent "
                "human ground truth is not claimed."
            )
    else:
        raise ValueError(f"Unsupported Bursa-news labelling mode: {labeling_mode}")
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["split"] = data["split"].astype(str).str.casefold().str.strip()
    data["published_at"] = pd.to_datetime(
        data["published_at"], utc=True, errors="coerce"
    )
    data["entity_relevant"] = _coerce_bool(data["entity_relevant"])
    data["automatic_entity_relevant"] = _coerce_bool(
        data["automatic_entity_relevant"]
    )
    if data["published_at"].isna().any():
        errors.append("one or more rows have invalid publication timestamps")
    outside_supported_period = ~data["published_at"].dt.year.between(2015, 2025)
    if outside_supported_period.any():
        errors.append(
            f"{int(outside_supported_period.sum())} rows are outside the supported 2015-2025 archive"
        )
    expected_split = data["published_at"].map(split_for_timestamp)
    mismatch = expected_split.ne(data["split"]) & expected_split.ne("")
    if mismatch.any():
        errors.append(f"{int(mismatch.sum())} rows do not match chronological split rules")
    event_split_counts = data.groupby("event_id")["split"].nunique()
    crossing = int(event_split_counts.gt(1).sum())
    if crossing:
        errors.append(f"{crossing} event clusters cross chronological splits")
    duplicate_content = int(
        data.loc[data["entity_relevant"]]
        .duplicated(["ticker", "content_sha256"])
        .sum()
    )
    if duplicate_content:
        errors.append(f"{duplicate_content} duplicate ticker/content rows remain")
    relevant = data.loc[
        data["entity_relevant"] & data["sentiment_label"].isin(SENTIMENT_LABELS)
    ].copy()
    require_llm_assisted_label = bool(
        policy.get("labeling", {}).get(
            "require_llm_assisted_label",
            labeling_mode in {"llm_assisted_label", "llm_assisted"},
        )
    )
    if require_llm_assisted_label:
        sources = (
            relevant.get("label_source", pd.Series("", index=relevant.index)).map(
                canonical_label_source
            )
        )
        invalid_sources = ~sources.eq(LLM_ASSISTED_LABEL_SOURCE)
        if invalid_sources.any():
            errors.append(
                f"{int(invalid_sources.sum())} relevant rows are not marked "
                "llm_assisted_label"
            )
    language = relevant.get("language", pd.Series("", index=relevant.index))
    non_english = ~language.astype(str).str.casefold().str.strip().eq("en")
    if non_english.any():
        errors.append(
            f"{int(non_english.sum())} relevant labelled rows are not confirmed as English"
        )
    blank_model_text = relevant["model_text"].astype(str).str.strip().eq("")
    if blank_model_text.any():
        errors.append(f"{int(blank_model_text.sum())} relevant rows have empty model_text")
    readiness_gates = policy.get("readiness_gates", {})
    enforce_dataset_size = bool(
        readiness_gates.get("enforce_dataset_size", True)
    )
    if enforce_dataset_size:
        minimum_rows = int(policy.get("minimum_rows", 0))
        if len(relevant) < minimum_rows:
            errors.append(
                f"labelled relevant rows={len(relevant)}; requires {minimum_rows}"
            )
    else:
        warnings.append(
            "Dataset-volume readiness gates are disabled; row counts are reported "
            "descriptively and are not a production-readiness requirement."
        )
    split_counts = relevant["split"].value_counts().to_dict()
    if enforce_dataset_size:
        for split, required_rows in policy.get("minimum_split_rows", {}).items():
            if int(split_counts.get(split, 0)) < int(required_rows):
                errors.append(
                    f"{split} rows={int(split_counts.get(split, 0))}; requires {int(required_rows)}"
                )
    elif bool(readiness_gates.get("require_nonempty_splits", True)):
        for split in policy.get("split_periods", {}):
            if int(split_counts.get(split, 0)) == 0:
                errors.append(f"required split is empty: {split}")
    ticker_counts = relevant["ticker"].value_counts().to_dict()
    test_ticker_counts = (
        relevant.loc[relevant["split"].eq("test"), "ticker"].value_counts().to_dict()
    )
    for ticker in policy.get("required_tickers", []):
        count = int(ticker_counts.get(ticker, 0))
        test_count = int(test_ticker_counts.get(ticker, 0))
        if enforce_dataset_size:
            if count < int(policy.get("minimum_rows_per_ticker", 0)):
                errors.append(
                    f"{ticker} rows={count}; "
                    f"requires {int(policy.get('minimum_rows_per_ticker', 0))}"
                )
            if test_count < int(policy.get("minimum_test_rows_per_ticker", 0)):
                errors.append(
                    f"{ticker} test rows={test_count}; "
                    f"requires {int(policy.get('minimum_test_rows_per_ticker', 0))}"
                )
        elif bool(readiness_gates.get("require_all_tickers", True)) and count == 0:
            errors.append(f"required ticker has no labelled relevant rows: {ticker}")
    class_counts = relevant["sentiment_label"].value_counts().to_dict()
    absent_classes = [label for label in SENTIMENT_LABELS if not class_counts.get(label)]
    if absent_classes:
        errors.append(f"sentiment classes absent: {absent_classes}")
    if labeling_mode == "llm_consensus":
        first_labels = (
            relevant["labeler_a_sentiment"].astype(str).str.casefold().str.strip()
        )
        second_labels = (
            relevant["labeler_b_sentiment"].astype(str).str.casefold().str.strip()
        )
    elif labeling_mode in {
        "llm_assisted_label",
        "llm_assisted",
        "external_single_weak",
        "single_label",
        "single",
    }:
        first_labels = relevant["sentiment_label"]
        second_labels = pd.Series("", index=relevant.index, dtype="string")
    else:
        first_labels = relevant.get(
            "annotator_1_label", relevant["sentiment_label"]
        )
        second_labels = relevant.get(
            "annotator_2_label", pd.Series("", index=relevant.index)
        )
    first = first_labels.isin(SENTIMENT_LABELS)
    second = second_labels.isin(SENTIMENT_LABELS)
    adjudicated_llm = (
        relevant.get("label_source", pd.Series("", index=relevant.index))
        .astype(str)
        .str.casefold()
        .eq("user_adjudicated_after_llm_abstention")
    )
    double = (first & second) | adjudicated_llm
    require_double_labels = bool(
        policy.get("labeling", {}).get("require_independent_double_labels", True)
    ) and bool(readiness_gates.get("enforce_independent_agreement", True))
    if not require_double_labels:
        warnings.append(
            "Independent double-labelling and agreement measurement are disabled "
            "by policy; no kappa-based readiness claim is made."
        )
    if require_double_labels:
        for split, minimum_fraction in policy.get(
            "minimum_double_label_fraction", {}
        ).items():
            mask = relevant["split"].eq(split)
            fraction = float(double[mask].mean()) if mask.any() else 0.0
            if fraction + 1e-12 < float(minimum_fraction):
                errors.append(
                    f"{split} double-label fraction={fraction:.3f}; "
                    f"requires {float(minimum_fraction):.3f}"
                )
    kappa: float | None = None
    kappa_rows = first & second
    if require_double_labels and kappa_rows.any():
        kappa = float(
            cohen_kappa_score(
                first_labels[kappa_rows],
                second_labels[kappa_rows],
                labels=list(SENTIMENT_LABELS),
            )
        )
        if pd.isna(kappa):
            kappa = None
        minimum_kappa = policy.get("minimum_inter_annotator_kappa")
        if minimum_kappa is not None and (
            kappa is None or kappa < float(minimum_kappa)
        ):
            errors.append(
                f"inter-annotator kappa={kappa}; requires "
                f"{float(minimum_kappa):.2f}"
            )
    elif require_double_labels:
        errors.append("no double-labelled rows are available")
    entity_precision: float | None = None
    entity_recall: float | None = None
    enforce_entity_metrics = bool(
        policy.get("labeling", {}).get("enforce_entity_filter_metrics", True)
    )
    if data["entity_relevant"].nunique() > 1:
        entity_precision = float(
            precision_score(
                data["entity_relevant"],
                data["automatic_entity_relevant"],
                zero_division=0,
            )
        )
        entity_recall = float(
            recall_score(
                data["entity_relevant"],
                data["automatic_entity_relevant"],
                zero_division=0,
            )
        )
        if enforce_entity_metrics:
            entity_gates = policy.get("year_quality_gates", {})
            if entity_precision < float(entity_gates.get("minimum_entity_precision", 0)):
                errors.append(
                    f"entity precision={entity_precision:.3f}; requires "
                    f"{float(entity_gates.get('minimum_entity_precision', 0)):.3f}"
                )
            if entity_recall < float(entity_gates.get("minimum_entity_recall", 0)):
                errors.append(
                    f"entity recall={entity_recall:.3f}; requires "
                    f"{float(entity_gates.get('minimum_entity_recall', 0)):.3f}"
                )
        else:
            warnings.append(
                "Automatic entity-filter precision/recall are descriptive only for "
                "this experimental weak-label policy."
            )
    elif enforce_entity_metrics:
        errors.append(
            "entity precision/recall require reviewed relevant and irrelevant rows"
        )
    return BursaNewsReadiness(
        not errors,
        str(policy.get("policy_version", "unknown")),
        dataset_path,
        int(len(relevant)),
        {str(key): int(value) for key, value in split_counts.items()},
        {str(key): int(value) for key, value in ticker_counts.items()},
        {str(key): int(value) for key, value in test_ticker_counts.items()},
        {str(key): int(value) for key, value in class_counts.items()},
        kappa,
        entity_precision,
        entity_recall,
        tuple(errors),
        tuple(warnings),
    )


def build_year_quality_report(
    year: int,
    indexed: pd.DataFrame,
    recovered: pd.DataFrame,
    accepted: pd.DataFrame,
    reviewed: pd.DataFrame,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = dict(policy or load_policy())
    gates = policy.get("year_quality_gates", {})
    required_tickers = {str(value) for value in policy.get("required_tickers", [])}
    recovered_rate = float(len(recovered) / len(indexed)) if len(indexed) else 0.0
    timestamps = pd.to_datetime(
        accepted.get("published_at", pd.Series(dtype=str)), utc=True, errors="coerce"
    )
    timestamp_completeness = float(timestamps.notna().mean()) if len(accepted) else 0.0
    accepted_tickers = set(
        accepted.get("ticker", pd.Series(dtype=str)).astype(str).str.upper()
    )
    domains = (
        accepted.groupby("ticker")["source_domain"].nunique().to_dict()
        if {"ticker", "source_domain"}.issubset(accepted.columns)
        else {}
    )
    accepted_counts = (
        accepted.groupby("ticker").size().to_dict()
        if "ticker" in accepted.columns
        else {}
    )
    predicted = _coerce_bool(
        reviewed.get("automatic_included", pd.Series(False, index=reviewed.index))
    )
    actual = _coerce_bool(
        reviewed.get("actually_related", pd.Series(False, index=reviewed.index))
    )
    precision = (
        float(precision_score(actual, predicted, zero_division=0))
        if len(reviewed)
        else None
    )
    recall = (
        float(recall_score(actual, predicted, zero_division=0))
        if len(reviewed)
        else None
    )
    checks = {
        "text_recovery": recovered_rate
        >= float(gates.get("minimum_text_recovery_rate", 0)),
        "entity_precision": precision is not None
        and precision >= float(gates.get("minimum_entity_precision", 0)),
        "entity_recall": recall is not None
        and recall >= float(gates.get("minimum_entity_recall", 0)),
        "timestamp_completeness": timestamp_completeness
        >= float(gates.get("minimum_timestamp_completeness", 0)),
        "all_companies": required_tickers.issubset(accepted_tickers),
        "source_diversity": all(
            int(domains.get(ticker, 0))
            >= int(gates.get("minimum_source_domains_per_ticker", 0))
            for ticker in required_tickers
        ),
        "accepted_volume": len(accepted)
        >= int(gates.get("minimum_accepted_articles", 0)),
        "per_ticker_volume": all(
            int(accepted_counts.get(ticker, 0))
            >= int(gates.get("minimum_accepted_per_ticker", 0))
            for ticker in required_tickers
        ),
    }
    return {
        "schema_version": "bursa-news-year-quality-v1",
        "year": int(year),
        "indexed_rows": int(len(indexed)),
        "recovered_rows": int(len(recovered)),
        "accepted_rows": int(len(accepted)),
        "text_recovery_rate": recovered_rate,
        "entity_precision": precision,
        "entity_recall": recall,
        "timestamp_completeness": timestamp_completeness,
        "accepted_tickers": sorted(accepted_tickers),
        "source_domains_per_ticker": {
            str(key): int(value) for key, value in domains.items()
        },
        "accepted_articles_by_ticker": {
            str(key): int(value) for key, value in accepted_counts.items()
        },
        "checks": checks,
        "ready": all(checks.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and validate the LLM-assisted-label Bursa FinBERT dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", default=str(DEFAULT_DATASET_PATH))
    prepare.add_argument(
        "--backfill-report",
        action="append",
        default=[],
        help="A passing year-quality JSON report for one optional 2015-2018 year",
    )
    materialize = subparsers.add_parser(
        "materialize-external-single-weak",
        help=(
            "Legacy alias: migrate labelled records into the LLM-assisted-label "
            "canonical Bursa schema."
        ),
    )
    materialize.add_argument("--source", required=True)
    materialize.add_argument("--output", required=True)
    migrate = subparsers.add_parser(
        "migrate-llm-assisted-label",
        help=(
            "Rewrite legacy Bursa label provenance to llm_assisted_label while "
            "retaining the original source in an audit column."
        ),
    )
    migrate.add_argument("--source", required=True)
    migrate.add_argument("--output", default=str(DEFAULT_DATASET_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    validate.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    validate.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    year_report = subparsers.add_parser("year-report")
    year_report.add_argument("--year", type=int, required=True)
    year_report.add_argument("--indexed", required=True)
    year_report.add_argument("--recovered", required=True)
    year_report.add_argument("--accepted", required=True)
    year_report.add_argument("--reviewed", required=True)
    year_report.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        approved_backfill_years: list[int] = []
        for report_value in args.backfill_report:
            report = json.loads(Path(report_value).read_text(encoding="utf-8"))
            year = int(report.get("year", 0))
            if not report.get("ready") or year not in {2015, 2016, 2017, 2018}:
                raise ValueError(
                    f"Backfill report is not an approved 2015-2018 year: {report_value}"
                )
            approved_backfill_years.append(year)
        frame = build_annotation_template(
            read_table(args.source),
            approved_backfill_years=approved_backfill_years,
        )
        _write_table(frame, Path(args.output))
        print(json.dumps({"rows": len(frame), "output": args.output}, indent=2))
        return 0
    if args.command in {
        "materialize-external-single-weak",
        "migrate-llm-assisted-label",
    }:
        frame = materialize_llm_assisted_dataset(read_table(args.source))
        _write_table(frame, Path(args.output))
        print(
            json.dumps(
                {
                    "rows": len(frame),
                    "relevant_labelled_rows": int(
                        (
                            _coerce_bool(frame["entity_relevant"])
                            & frame["sentiment_label"].isin(SENTIMENT_LABELS)
                        ).sum()
                    ),
                    "output": args.output,
                    "label_source": LLM_ASSISTED_LABEL_SOURCE,
                    "experimental_only": False,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "validate":
        report = validate_bursa_news_dataset(
            read_table(args.dataset),
            policy=load_policy(Path(args.policy)),
            dataset_path=args.dataset,
        )
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ready else 2
    if args.command == "year-report":
        report = build_year_quality_report(
            args.year,
            read_table(args.indexed),
            read_table(args.recovered),
            read_table(args.accepted),
            read_table(args.reviewed),
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 2
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
