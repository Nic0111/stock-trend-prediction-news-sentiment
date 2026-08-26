"""Fusion v4: compact, target-specific, point-in-time news correction.

This module is deliberately isolated from the active v3 registry.  It turns
the v4 design into an executable candidate pipeline while keeping the
stock-led runtime unchanged until every horizon passes the frozen gates.

The event-impact model is trained only on earlier target-complete rows.  Its
probability is then supplied to an anchored residual fusion model; no-news
rows are returned exactly as the stock input.  Public disclosure metadata may
be joined later through the existing official-disclosure exporter.  Missing
optional event columns are explicit readiness gaps, never silently invented.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.fusion_evaluation_v2 import (
    _block_bootstrap,
    _calibration_curve,
    _calibration_regression,
    _clark_west,
    _holm_adjust,
    _paired_details,
    _purged_development_splits,
    _read_json,
    _sha256_payload,
)
from pipeline.fusion_evaluation_v3 import (
    FULL_FEATURES as V3_FULL_FEATURES,
    _capacity_matched_frames,
    _development_oof_loss_difference,
    _enhance_features,
    _metrics,
    _number,
    _placebo_frame,
    _project_reference,
    _readiness,
    _select_stock_meta_configuration,
    _select_configuration,
    _select_variant,
    _stock_meta_oof,
    _variants,
    _shared_threshold,
    _identity,
    scientific_result_sha256,
)
from pipeline.fusion_model import (
    DEFAULT_FUSION_DATASET,
    dataset_manifest_hash,
    load_fusion_dataset,
)
from pipeline.historical_news_fusion_data import _sentiment_runtime_identity
from pipeline.news_relevance import stable_event_id
from pipeline.incremental_news_fusion import IncrementalNewsFusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = PROJECT_ROOT / "config" / "fusion_evaluation_v4.json"
DEFAULT_REPORT = PROJECT_ROOT / "results" / "fusion" / "fusion-evaluation-v4.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "models" / "fusion" / "staged-fusion-v4-registry.json"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "fusion" / "staged-v4"

# The compact contract intentionally separates occurrence, tone, reliability
# and target-specific event impact.  It is small enough to be estimable with
# sparse news rows and is the only contract eligible for promotion.
OCCURRENCE_FEATURES = (
    "v4_log_event_count",
    "v4_attention_surprise",
)
TONE_FEATURES = (
    "v4_positive_impact",
    "v4_negative_impact",
    "v4_sentiment_surprise",
)
RELIABILITY_FEATURES = (
    "v4_event_reliability",
    "v4_official_share",
    "v4_source_diversity",
)
EVENT_IMPACT_FEATURE = "v4_event_impact_logit"
V4_FEATURES = (*OCCURRENCE_FEATURES, *TONE_FEATURES, *RELIABILITY_FEATURES, EVENT_IMPACT_FEATURE)
V4_ARMS = {
    "occurrence_only": OCCURRENCE_FEATURES,
    "sentiment_only": TONE_FEATURES,
    "event_impact_only": (EVENT_IMPACT_FEATURE,),
    "compact_full": V4_FEATURES,
}
IMPACT_INPUTS = (
    "v4_positive_impact",
    "v4_negative_impact",
    "v4_sentiment_surprise",
    "v4_log_event_count",
    "v4_attention_surprise",
    "v4_event_reliability",
    "v4_official_share",
    "v4_source_diversity",
)


class FusionV4ReadinessError(ValueError):
    """Raised when the v4 evidence contract is not ready for full evaluation."""


def rebind_v4_sentiment_config(
    source_path: Path,
    output_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Rebind an isolated feature table to the active sentiment config.

    Historical feature tables can carry a stale *top-level* runtime-config
    hash after a metadata-only edit.  Rebinding is allowed only when the
    active variant, variant hash and component-artifact hash all match the
    source rows, and article-level identity is explicitly verified.  No
    probabilities or feature values are recomputed; the output is an
    isolated copy with a provenance record explaining the metadata-only
    repair.  The source and production registries are never modified.
    """

    if source_path.resolve() == output_path.resolve():
        raise ValueError("v4 sentiment rebind requires a different output path")
    policy, policy_hash = load_policy(policy_path)
    source = load_fusion_dataset(source_path).reset_index(drop=True)
    active = _sentiment_runtime_identity(
        PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
    )
    if not active.get("config_sha256"):
        raise FusionV4ReadinessError("Active sentiment runtime config is unavailable")

    def _unique(column: str) -> set[str]:
        return {
            str(value)
            for value in source.get(column, pd.Series(dtype=object)).dropna().astype(str)
            if str(value).strip()
        }

    variant_values = _unique("sentiment_model_variant")
    variant_hash_values = _unique("sentiment_model_variant_sha256")
    component_values = _unique("sentiment_component_artifacts_sha256")
    config_values = _unique("sentiment_runtime_config_sha256")
    checks = {
        "source_has_single_config_hash": len(config_values) == 1,
        "source_config_differs_from_active": config_values
        and config_values != {str(active["config_sha256"])},
        "variant_matches_active": variant_values == {str(active["variant"])},
        "variant_hash_matches_active": variant_hash_values
        == {str(active["variant_sha256"])},
        "component_hash_matches_active": component_values
        == {str(active["component_artifacts_sha256"])},
        "article_identity_verified": set(
            source.get(
                "article_sentiment_identity_verified", pd.Series(dtype=bool)
            )
            .fillna(False)
            .astype(bool)
        )
        == {True},
    }
    if not all(bool(value) for value in checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise FusionV4ReadinessError(
            "Refusing v4 sentiment rebind; component identity is not proven: "
            + ", ".join(failed)
        )

    # Hash all feature/prediction columns except the metadata field that is
    # intentionally changed.  This catches accidental value edits in the
    # rebind operation and makes the parity claim auditable.
    parity_columns = [
        column
        for column in source.columns
        if column != "sentiment_runtime_config_sha256"
    ]
    before_hash = hashlib.sha256(
        pd.util.hash_pandas_object(source[parity_columns], index=True)
        .to_numpy(dtype="uint64")
        .tobytes()
    ).hexdigest()
    rebound = source.copy()
    rebound["sentiment_runtime_config_sha256"] = str(active["config_sha256"])
    after_hash = hashlib.sha256(
        pd.util.hash_pandas_object(rebound[parity_columns], index=True)
        .to_numpy(dtype="uint64")
        .tobytes()
    ).hexdigest()
    if before_hash != after_hash:
        raise FusionV4ReadinessError(
            "v4 sentiment rebind parity check failed; feature values changed"
        )

    _write_parquet_atomic(rebound, output_path)
    output_hash = dataset_manifest_hash(output_path)
    manifest = {
        "schema_version": "fusion-v4-sentiment-rebind-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": "Retrospective evidence, 2019-2025",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "source_dataset_sha256": dataset_manifest_hash(source_path),
        "output_dataset_sha256": output_hash,
        "policy_sha256": policy_hash,
        "rows": int(len(rebound)),
        "old_sentiment_runtime_config_sha256": sorted(config_values),
        "new_sentiment_runtime_config_sha256": active["config_sha256"],
        "sentiment_model_variant": active["variant"],
        "sentiment_model_variant_sha256": active["variant_sha256"],
        "sentiment_component_artifacts_sha256": active[
            "component_artifacts_sha256"
        ],
        "checks": checks,
        "feature_value_parity_excluding_config_hash": before_hash == after_hash,
        "production_unchanged": True,
        "rebind_scope": "metadata-only active-config hash repair; no rescoring",
        "policy": policy.get("schema_version"),
    }
    target_manifest = manifest_path or output_path.with_suffix(".manifest.json")
    _atomic_json(target_manifest, manifest)
    return manifest


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a sealed feature table without exposing a partial parquet file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def seal_v4_dataset(
    source_path: Path,
    output_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Seal an identity-checked v4 feature table for an isolated benchmark.

    This step is intentionally a *copy and contract check*: it does not score
    articles, retrain either model, or mutate the production feature table.
    It prevents a long evaluation from accidentally using a stale candidate by
    persisting the policy, active artifact identities and content hashes beside
    the exact parquet file that will be evaluated.
    """

    policy, policy_hash = load_policy(policy_path)
    source = load_fusion_dataset(source_path)
    prepared, impact_provenance = prepare_v4_features(
        source, horizons=policy["supported_horizons"]
    )
    provenance = _identity(prepared, source_path, policy_hash)
    if not provenance["strict_identity_passed"]:
        failed = [key for key, passed in provenance["checks"].items() if not passed]
        raise FusionV4ReadinessError(
            "Cannot seal Fusion v4 data with stale or incomplete artifact identity: "
            + ", ".join(failed)
        )
    # Keep the source rows and values intact, but normalize the index so a
    # downstream read cannot inherit an accidental pandas index column.
    _write_parquet_atomic(prepared.reset_index(drop=True), output_path)
    sealed_hash = dataset_manifest_hash(output_path)
    manifest = {
        "schema_version": "fusion-v4-dataset-seal-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": "Retrospective evidence, 2019-2025",
        "source_path": str(source_path),
        "sealed_dataset_path": str(output_path),
        "source_dataset_sha256": provenance["dataset_sha256"],
        "sealed_dataset_sha256": sealed_hash,
        "policy_sha256": policy_hash,
        "rows": int(len(prepared)),
        "tickers": sorted(prepared["ticker"].astype(str).unique().tolist()),
        "horizons": sorted(
            int(value) for value in prepared["horizon"].dropna().unique()
        ),
        "observation_min": str(pd.to_datetime(prepared["observation_date"]).min().date()),
        "observation_max": str(pd.to_datetime(prepared["observation_date"]).max().date()),
        "provenance": provenance,
        "event_impact_provenance": impact_provenance,
        "production_unchanged": True,
    }
    target_manifest = manifest_path or output_path.with_suffix(".manifest.json")
    _atomic_json(target_manifest, manifest)
    return manifest


def _parse_event_ids(value: Any) -> list[str]:
    """Decode the compact JSON event-id fields used by fusion rows."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if isinstance(decoded, list):
        return [str(item) for item in decoded if str(item).strip()]
    return []


def _blind_audit_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Choose a deterministic company/year/event-stratified audit sample.

    The sample is deliberately selected from article-level records and does
    not expose the production relevance or sentiment fields.  A stable hash
    rather than pandas' process-dependent sampling is used so rerunning the
    command yields the same queue and can be resumed safely.
    """

    if frame.empty or sample_size <= 0:
        return frame.iloc[0:0].copy()
    working = frame.copy().reset_index(drop=True)
    working["_year"] = pd.to_datetime(working["published_at"], errors="coerce").dt.year.fillna(0).astype(int)
    working["_category"] = working.get("event_category", "").fillna("").astype(str).replace("nan", "")
    working["_stratum"] = (
        working["ticker"].astype(str)
        + "|"
        + working["_year"].astype(str)
        + "|"
        + working["_category"]
    )
    working["_sort_key"] = working["article_id"].astype(str).map(
        lambda value: hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest()
    )
    tickers = sorted(working["ticker"].astype(str).unique())
    per_ticker = max(1, int(math.ceil(int(sample_size) / max(1, len(tickers)))))
    selected_indexes: list[int] = []
    for ticker in tickers:
        group = working.loc[working["ticker"].astype(str).eq(ticker)].sort_values(
            ["_stratum", "_sort_key"]
        )
        target = min(per_ticker, len(group))
        if target <= 0:
            continue
        # Round-robin strata gives each company a spread over years and event
        # categories before the deterministic fill phase below.
        strata = [part for _, part in group.groupby("_stratum", sort=True)]
        picks: list[int] = []
        position = 0
        while len(picks) < target and strata:
            made_progress = False
            for part in strata:
                if position < len(part):
                    picks.append(int(part.index[position]))
                    made_progress = True
                    if len(picks) >= target:
                        break
            if not made_progress:
                break
            position += 1
        selected_indexes.extend(picks)
    selected = working.loc[sorted(set(selected_indexes))]
    if len(selected) < int(sample_size):
        remaining = working.drop(index=selected.index).sort_values(["_stratum", "_sort_key"])
        selected = pd.concat([selected, remaining.head(int(sample_size) - len(selected))], ignore_index=False)
    return selected.sort_values(["ticker", "published_at", "event_id"]).head(int(sample_size)).copy()


def export_v4_blind_audit_queue(
    dataset_path: Path,
    article_root: Path,
    output_path: Path,
    *,
    sample_size: int = 200,
    seed: int = 20260825,
    policy_path: Path = DEFAULT_POLICY,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Export an independent, blinded article-review queue for Fusion v4.

    This command never marks an article as audited and never changes a feature
    table.  It links the article corpus back to event IDs present in the sealed
    feature table, hides existing automatic relevance/sentiment labels, and
    leaves reviewer fields blank.  The returned CSV is the input for an
    independent reviewer; it cannot satisfy the readiness gate by itself.
    """

    if int(sample_size) <= 0:
        raise ValueError("sample_size must be positive")
    policy, policy_hash = load_policy(policy_path)
    feature = load_fusion_dataset(dataset_path)
    eligible_events: set[str] = set()
    for column in ("news_material_event_ids", "news_event_ids"):
        for value in feature.get(column, pd.Series(dtype=object)):
            eligible_events.update(_parse_event_ids(value))
    if not eligible_events:
        raise FusionV4ReadinessError("No event IDs are available in the v4 feature table for audit sampling")

    required_columns = {
        "article_id", "ticker", "company_name", "published_at", "title",
        "description", "content_snippet", "url", "event_category",
    }
    article_frames: list[pd.DataFrame] = []
    for path in sorted(article_root.rglob("*.parquet")):
        part = pd.read_parquet(path)
        for column in required_columns:
            if column not in part.columns:
                part[column] = ""
        article_frames.append(part[list(required_columns)])
    if not article_frames:
        raise FileNotFoundError(f"No article parquet files found under {article_root}")
    articles = pd.concat(article_frames, ignore_index=True)
    articles["ticker"] = articles["ticker"].astype(str)
    articles["published_at"] = pd.to_datetime(articles["published_at"], errors="coerce", utc=True)
    articles = articles.loc[articles["published_at"].dt.year.between(2019, 2025)].copy()
    articles["event_id"] = articles.apply(
        lambda row: str(row.get("event_id") or stable_event_id(row.to_dict())), axis=1
    )
    articles = articles.loc[articles["event_id"].isin(eligible_events)].copy()
    articles = articles.sort_values(["event_id", "published_at", "article_id"]).drop_duplicates(
        "event_id", keep="first"
    )
    selected = _blind_audit_sample(articles, int(sample_size), int(seed))
    if selected.empty:
        raise FusionV4ReadinessError("The v4 article corpus has no event records matching the feature table")

    queue = pd.DataFrame(
        {
            "audit_id": [f"v4-blind-{index:04d}" for index in range(1, len(selected) + 1)],
            "article_id": selected["article_id"].astype(str).to_numpy(),
            "event_id": selected["event_id"].astype(str).to_numpy(),
            "ticker": selected["ticker"].astype(str).to_numpy(),
            "company_name": selected["company_name"].astype(str).to_numpy(),
            "published_at": selected["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
            "title": selected["title"].fillna("").astype(str).to_numpy(),
            "description": selected["description"].fillna("").astype(str).to_numpy(),
            "content_snippet": selected["content_snippet"].fillna("").astype(str).to_numpy(),
            "url": selected["url"].fillna("").astype(str).to_numpy(),
            "audit_entity_relevant": "",
            "audit_event_material": "",
            "audit_reviewer_id": "",
            "audit_reviewed_at": "",
            "audit_notes": "",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    queue.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output_path)
    queue_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "fusion-v4-blind-audit-queue-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": "Retrospective evidence, 2019-2025",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_manifest_hash(dataset_path),
        "article_root": str(article_root),
        "output_path": str(output_path),
        "queue_sha256": queue_sha,
        "policy_sha256": policy_hash,
        "seed": int(seed),
        "requested_rows": int(sample_size),
        "exported_rows": int(len(queue)),
        "eligible_event_ids_in_features": int(len(eligible_events)),
        "eligible_article_events": int(len(articles)),
        "companies": sorted(queue["ticker"].unique().tolist()),
        "unreviewed_by_design": True,
        "production_unchanged": True,
        "review_fields": [
            "audit_entity_relevant",
            "audit_event_material",
            "audit_reviewer_id",
            "audit_reviewed_at",
            "audit_notes",
        ],
        "note": "This queue is an input for independent review; it does not satisfy the blind-audit gate until reviewed and explicitly imported.",
    }
    _atomic_json(manifest_path or output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def run_fusion_v4_preflight(
    dataset_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
    report_path: Path,
    horizons: Sequence[int] | None = None,
    selection_frame: str = "raw",
) -> dict[str, Any]:
    """Run selection-aware v4 readiness checks before bootstrap/placebos.

    A failed preflight is a useful, explicit result rather than a reason to
    relax the evidence gates.  The report records the exact missing evidence so
    disclosure recovery or audit work can be targeted before the full run.
    """

    policy, policy_hash = load_policy(policy_path)
    source = load_fusion_dataset(dataset_path)
    raw, impact_provenance = prepare_v4_features(
        source, horizons=policy["supported_horizons"]
    )
    provenance = _identity(raw, dataset_path, policy_hash)
    requested = tuple(int(value) for value in (horizons or policy["supported_horizons"]))
    payload: dict[str, Any] = {
        "schema_version": "fusion-v4-readiness-preflight-v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": "Retrospective evidence, 2019-2025",
        "policy_sha256": policy_hash,
        "dataset_sha256": provenance["dataset_sha256"],
        "provenance": provenance,
        "event_impact_provenance": impact_provenance,
        "horizons": {},
        "selection_frame": str(selection_frame).strip().casefold().replace("-", "_"),
        "production_unchanged": True,
    }
    for horizon in requested:
        result = _v4_preflight(
            raw,
            horizon,
            policy,
            provenance,
            selection_frame=selection_frame,
        )
        payload["horizons"][str(horizon)] = result
    payload["overall_ready"] = bool(
        provenance["strict_identity_passed"]
        and all(item["readiness"]["passed"] for item in payload["horizons"].values())
    )
    payload["ready_horizons"] = [
        int(key)
        for key, item in payload["horizons"].items()
        if item["readiness"]["passed"]
    ]
    _atomic_json(report_path, payload)
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def load_policy(path: Path = DEFAULT_POLICY) -> tuple[dict[str, Any], str]:
    policy = _read_json(path)
    if policy.get("schema_version") != "fusion-evaluation-v4":
        raise ValueError(f"Invalid Fusion v4 policy: {path}")
    return policy, _sha256_payload(policy)


def _safe_num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return _number(frame, column, default).replace([np.inf, -np.inf], np.nan).fillna(default)


def _compact_base(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive compact v4 event features from the frozen point-in-time table."""

    # ``_enhance_features`` is shared with the v3 diagnostics and deliberately
    # preserves its historical columns.  Rebuild the v4 inputs here from the
    # normalized frame so the compact contract does not inherit an index
    # alignment mistake from an older surprise calculation after sorting.
    result = frame.copy().reset_index(drop=True)
    available = _safe_num(result, "news_available").gt(0).astype(float)
    count = _safe_num(result, "news_independent_event_count")
    articles = _safe_num(result, "news_usable_article_count")
    weighted = _safe_num(result, "news_weighted_sentiment")
    positive_share = _safe_num(result, "news_positive_share").clip(0.0, 1.0)
    negative_share = _safe_num(result, "news_negative_share").clip(0.0, 1.0)
    reliability = _safe_num(result, "news_event_reliability_mean")
    if "news_event_reliability_mean" not in result or not reliability.gt(0).any():
        # Older tables do not carry event reliability.  This is an explicit,
        # deterministic proxy, not a claim that the row passed the v4 audit.
        reliability = (
            _safe_num(result, "news_mean_confidence", 0.0)
            * _safe_num(result, "news_mean_entity_confidence", 1.0)
            * _safe_num(result, "news_mean_text_quality", 1.0)
            * _safe_num(result, "news_coverage_score", 0.0).clip(0.0, 1.0)
        )
    positive = _safe_num(result, "news_positive_event_impact")
    negative = _safe_num(result, "news_negative_event_impact")
    if "news_positive_event_impact" not in result or not positive.gt(0).any():
        positive = positive_share * reliability
    if "news_negative_event_impact" not in result or not negative.gt(0).any():
        negative = negative_share * reliability
    official = _safe_num(result, "news_official_disclosure_share")
    source = np.log1p(_safe_num(result, "news_unique_source_count").clip(lower=0.0))
    result["news_available"] = available
    result["v4_log_event_count"] = available * np.log1p(count.clip(lower=0.0))
    # Calculate surprises after the frame has been normalized and sorted.  A
    # prior v3 implementation computed a Series before sorting and could align
    # it to the wrong observation when the index was reset.
    result["v4_log_article_count"] = available * np.log1p(articles.clip(lower=0.0))
    group_keys = ["ticker", "horizon_days"]
    for name in ("news_window_days", "news_half_life_days"):
        if name in result:
            group_keys.append(name)
    if all(key in result for key in group_keys + ["observation_date"]):
        ordered = result.sort_values([*group_keys, "observation_date"], kind="stable").copy()
        # ``weighted`` is already in the same row order; reindex it to the
        # temporary sorted frame explicitly before computing a trailing value.
        ordered["_v4_weighted"] = weighted.reindex(ordered.index).to_numpy()
        trailing_volume = ordered.groupby(group_keys, dropna=False, sort=False)["v4_log_event_count"].transform(
            lambda values: values.shift(1).rolling(252, min_periods=20).mean()
        )
        trailing_articles = ordered.groupby(group_keys, dropna=False, sort=False)["v4_log_article_count"].transform(
            lambda values: values.shift(1).rolling(252, min_periods=20).mean()
        )
        trailing_tone = ordered.groupby(group_keys, dropna=False, sort=False)["_v4_weighted"].transform(
            lambda values: values.shift(1).rolling(252, min_periods=20).mean()
        )
        ordered["_v4_attention_surprise"] = (
            ordered["v4_log_event_count"] - trailing_volume
        ).fillna(ordered["v4_log_article_count"] - trailing_articles).fillna(0.0)
        ordered["_v4_sentiment_surprise"] = (
            ordered["_v4_weighted"] - trailing_tone
        ).fillna(ordered["_v4_weighted"]).fillna(0.0)
        result = ordered.sort_index(kind="stable")
        attention_surprise = result["_v4_attention_surprise"]
        tone_surprise = result["_v4_sentiment_surprise"]
    else:
        attention_surprise = np.log1p(count.clip(lower=0.0)).where(count.gt(0), np.log1p(articles.clip(lower=0.0)))
        tone_surprise = weighted
    result["v4_attention_surprise"] = available * pd.Series(attention_surprise, index=result.index).fillna(0.0)
    result["v4_positive_impact"] = available * positive.clip(lower=0.0, upper=1.0)
    result["v4_negative_impact"] = available * negative.clip(lower=0.0, upper=1.0)
    result["v4_sentiment_surprise"] = available * pd.Series(tone_surprise, index=result.index).fillna(0.0)
    result["v4_event_reliability"] = available * reliability.clip(lower=0.0, upper=1.0)
    result["v4_official_share"] = available * official.clip(lower=0.0, upper=1.0)
    result["v4_source_diversity"] = available * source.clip(lower=0.0)
    for feature in V4_FEATURES:
        if feature not in result:
            result[feature] = 0.0
    result["v4_event_impact_logit"] = 0.0
    result["v4_event_impact_probability_up"] = 0.5
    result["v4_event_impact_ready"] = False
    # This is an automated provenance/identity audit, not a semantic label
    # review.  It confirms that a news-bearing feature row has an event key,
    # verified article-level sentiment identity and an aggregation-policy
    # identity.  It deliberately does not infer relevance or sentiment truth
    # and must never be described as human-labelled evidence.
    event_ids = result.get("news_event_ids", pd.Series("", index=result.index)).astype(str).str.strip()
    identity = result.get(
        "article_sentiment_identity_verified",
        pd.Series(False, index=result.index),
    ).fillna(False).astype(bool)
    sentiment_hash = result.get(
        "sentiment_runtime_config_sha256",
        pd.Series("", index=result.index),
    ).astype(str).str.strip()
    aggregation_hash = result.get(
        "news_aggregation_policy_sha256",
        pd.Series("", index=result.index),
    ).astype(str).str.strip()
    result["v4_automated_audit_passed"] = (
        available.gt(0)
        & event_ids.ne("")
        & identity
        & sentiment_hash.ne("")
        & aggregation_hash.ne("")
    )
    return result


class TargetSpecificEventImpact:
    """Small regularised model estimating direction after a material event."""

    def __init__(self, *, C: float = 0.5) -> None:
        self.C = float(C)
        self.scaler: StandardScaler | None = None
        self.model: LogisticRegression | None = None
        self.training_rows = 0
        self.feature_names = tuple(IMPACT_INPUTS)

    def _matrix(self, frame: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        values = frame.reindex(columns=self.feature_names, fill_value=0.0).apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        if fit:
            self.scaler = StandardScaler().fit(values)
        if self.scaler is None:
            raise RuntimeError("TargetSpecificEventImpact has not been fitted")
        return self.scaler.transform(values)

    def fit(self, frame: pd.DataFrame) -> "TargetSpecificEventImpact":
        eligible = frame.loc[
            _safe_num(frame, "news_available").gt(0)
            & _safe_num(frame, "target_up", np.nan).isin([0, 1])
        ].copy()
        target = _safe_num(eligible, "target_up", np.nan).astype(int)
        self.training_rows = int(len(eligible))
        if len(eligible) < 20 or target.nunique() < 2:
            self.scaler = None
            self.model = None
            return self
        matrix = self._matrix(eligible, fit=True)
        self.model = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        ).fit(matrix, target.to_numpy(dtype=int))
        return self

    @property
    def ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def predict_proba_up(self, frame: pd.DataFrame) -> np.ndarray:
        available = _safe_num(frame, "news_available").gt(0).to_numpy()
        output = np.full(len(frame), 0.5, dtype=float)
        if self.ready:
            output[available] = self.model.predict_proba(self._matrix(frame.iloc[np.flatnonzero(available)]))[:, 1]
        return np.clip(output, 1e-6, 1 - 1e-6)

    def provenance(self) -> dict[str, Any]:
        return {
            "module": "TargetSpecificEventImpact",
            "feature_names": list(self.feature_names),
            "C": self.C,
            "training_rows": self.training_rows,
            "ready": self.ready,
            "target": "subsequent target_up within the selected horizon",
            "fit_scope": "earlier target-complete development rows only",
        }


def _event_impact_oof(development: pd.DataFrame) -> np.ndarray:
    """Return leakage-safe event-impact probabilities for development rows."""

    values = np.full(len(development), 0.5, dtype=float)
    try:
        splits = _purged_development_splits(development)
    except ValueError:
        return values
    for train_index, validation_index in splits:
        model = TargetSpecificEventImpact().fit(development.iloc[train_index])
        values[validation_index] = model.predict_proba_up(development.iloc[validation_index])
    return values


def prepare_v4_features(
    frame: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 5, 10),
    development_end: int = 2023,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add event-impact features using only earlier data for each variant."""

    result = _compact_base(_enhance_features(frame))
    dates = pd.to_datetime(result["observation_date"], errors="coerce")
    impact_provenance: dict[str, Any] = {}
    for horizon in horizons:
        mask_horizon = _safe_num(result, "horizon_days").eq(int(horizon))
        variants = ["news_window_days", "news_half_life_days"]
        subset = result.loc[mask_horizon]
        if subset.empty:
            continue
        for variant_key, indexes in subset.groupby(variants, dropna=False, sort=False).groups.items():
            indexes = np.asarray(list(indexes), dtype=int)
            dev_mask = (
                dates.iloc[indexes].dt.year.between(2019, int(development_end))
                & pd.to_datetime(result.iloc[indexes]["target_date"], errors="coerce").lt(
                    pd.Timestamp(f"{int(development_end) + 1}-01-01")
                )
            )
            dev_indexes = indexes[np.asarray(dev_mask, dtype=bool)]
            development = result.iloc[dev_indexes].reset_index(drop=True)
            if len(development):
                oof = _event_impact_oof(development)
                result.loc[dev_indexes, "v4_event_impact_probability_up"] = oof
                result.loc[dev_indexes, "v4_event_impact_ready"] = True
                result.loc[dev_indexes, "v4_event_impact_logit"] = logit(np.clip(oof, 1e-6, 1 - 1e-6))
                model = TargetSpecificEventImpact().fit(development)
                later_indexes = indexes[~np.asarray(dev_mask, dtype=bool)]
                if len(later_indexes):
                    later = result.iloc[later_indexes]
                    later_p = model.predict_proba_up(later)
                    result.loc[later_indexes, "v4_event_impact_probability_up"] = later_p
                    result.loc[later_indexes, "v4_event_impact_logit"] = logit(later_p)
                    result.loc[later_indexes, "v4_event_impact_ready"] = model.ready
                impact_provenance[f"{horizon}:{variant_key}"] = model.provenance()
    return result, impact_provenance


_BLIND_AUDIT_COLUMNS = (
    "news_blind_audit_passed",
    "article_blind_audit_passed",
    "entity_disambiguation_audit_passed",
    "news_entity_audit_passed",
    "article_entity_audit_passed",
)

_AUTOMATED_AUDIT_COLUMNS = (
    "v4_automated_audit_passed",
    "news_automated_provenance_audit_passed",
)


def _count_blind_audit_rows(frame: pd.DataFrame) -> tuple[int, str | None]:
    """Count explicit, auditable entity-review rows without guessing.

    Historical feature tables created before the v4 evidence contract do not
    contain a blind-audit field.  In that case the count is deliberately zero;
    a missing audit column must not be interpreted as a passed review.
    """

    for column in _BLIND_AUDIT_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            return int(values.fillna(False).sum()), column
        normalized = values.astype(str).str.strip().str.casefold()
        passed = normalized.isin({"1", "true", "yes", "pass", "passed", "reviewed"})
        return int(passed.sum()), column
    return 0, None


def _count_automated_audit_rows(frame: pd.DataFrame) -> tuple[int, str | None]:
    """Count deterministic provenance checks without inventing human review.

    Automated rows prove that the point-in-time event and model identities are
    present.  They do not prove article relevance, sentiment correctness or
    independent annotation agreement.
    """

    for column in _AUTOMATED_AUDIT_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            return int(values.fillna(False).sum()), column
        normalized = values.astype(str).str.strip().str.casefold()
        passed = normalized.isin({"1", "true", "yes", "pass", "passed"})
        return int(passed.sum()), column
    return 0, None


def _select_v4_variant(
    raw: pd.DataFrame,
    horizon: int,
    policy: Mapping[str, Any],
    features: Sequence[str] = V4_FEATURES,
    *,
    development_end: int = 2023,
    fit_intercept_candidates: Sequence[bool] = (True,),
    selection_frame: str = "raw",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Select a v4 window after evidence-sufficiency screening.

    Window and model performance are selected from purged development rows.
    A candidate is eligible only when its *calibration-year* evidence can
    satisfy the pre-registered company/news sufficiency contract.  This
    prevents an under-covered current-session variant from winning a tiny
    development Brier difference and then failing the locked benchmark gate.
    Calibration-year availability is used only as a non-performance data
    sufficiency check; benchmark labels are never read by this selector.
    """

    selection_frame = str(selection_frame).strip().casefold().replace("-", "_")
    if selection_frame not in {"raw", "capacity_matched"}:
        raise ValueError("selection_frame must be 'raw' or 'capacity_matched'")
    gates = dict(policy.get("readiness_gates") or {})
    minimum_company = int(gates.get("minimum_news_present_per_company_horizon", 0))
    minimum_rows = int(gates.get("minimum_news_present_benchmark_rows", 0))
    calibration_year = int(development_end) + 1
    trials: list[dict[str, Any]] = []
    candidates: list[
        tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[dict[str, Any]]]
    ] = []
    rejected: list[dict[str, Any]] = []
    for variant, rows in _variants(raw, horizon):
        dates = pd.to_datetime(rows["observation_date"], errors="coerce")
        targets = pd.to_datetime(rows["target_date"], errors="coerce")
        development = rows.loc[
            dates.dt.year.between(2019, int(development_end))
            & targets.lt(pd.Timestamp(f"{int(development_end) + 1}-01-01"))
        ].reset_index(drop=True)
        calibration = rows.loc[
            dates.dt.year.eq(calibration_year)
            & targets.lt(pd.Timestamp(f"{calibration_year + 1}-01-01"))
        ]
        if development.empty:
            continue
        selection_development = development
        stock_meta_configuration: dict[str, Any] | None = None
        if selection_frame == "capacity_matched":
            # The headline comparison uses a capacity-matched stock-only
            # meta-baseline.  Select the anchored news correction on that same
            # probability frame; selecting on raw stock probabilities would
            # optimize a different baseline and can especially distort the
            # sparse 10-day result.
            try:
                stock_meta_configuration, _ = _select_stock_meta_configuration(
                    development, policy
                )
                selection_development = development.copy()
                selection_development["stock_probability_raw"] = (
                    selection_development["stock_probability_up"]
                )
                selection_development["stock_probability_up"] = _stock_meta_oof(
                    development, stock_meta_configuration
                )
            except (ValueError, RuntimeError):
                continue
        try:
            selected, tuning = _select_configuration(
                selection_development,
                features,
                policy,
                fit_intercept_candidates=fit_intercept_candidates,
            )
        except ValueError:
            continue
        calibration_news = calibration.loc[_safe_num(calibration, "news_available").gt(0)]
        company_counts = calibration_news.groupby("ticker").size()
        company_gap = {
            str(ticker): max(0, minimum_company - int(count))
            for ticker, count in company_counts.items()
            if int(count) < minimum_company
        }
        row = {
            **variant,
            **selected,
            "selection_calibration_year": calibration_year,
            "selection_calibration_news_rows": int(len(calibration_news)),
            "selection_calibration_company_gaps": company_gap,
            "selection_frame": selection_frame,
            "selection_stock_meta_configuration": stock_meta_configuration,
        }
        eligible = (
            len(calibration_news) >= minimum_rows
            and not company_gap
        )
        row["selection_evidence_sufficient"] = bool(eligible)
        trials.append({**row, "candidate_trials": tuning})
        if eligible:
            candidates.append((rows, row, selected, tuning))
        else:
            rejected.append(row)
    if not candidates:
        raise FusionV4ReadinessError(
            f"No evidence-sufficient v4 window for {horizon}D; "
            f"calibration-year gaps: {rejected}"
        )
    rows, selected_row, selected_model, selected_tuning = min(
        candidates,
        key=lambda value: (
            -value[2]["development_brier_improvement"],
            0 if value[2]["family"] == "ridge" else 1,
            value[0]["news_window_days"].iloc[0]
            if "news_window_days" in value[0]
            else 0,
        ),
    )
    selected_row = {
        **selected_row,
        "selection_rule": (
            "purged-development-performance-with-calibration-evidence-sufficiency"
            f"-{selection_frame}"
        ),
        "selected_configuration_trials": selected_tuning,
    }
    return rows, selected_row, selected_model, trials


def _v4_readiness(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    policy: Mapping[str, Any],
    development_loss_difference: np.ndarray,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the complete v4 evidence-readiness contract.

    The v3 readiness helper supplies the per-company news, material-event and
    development-power checks.  v4 adds explicit event-impact training,
    deterministic automated provenance checks and strict artifact identity.
    Manual semantic labels are not inferred or fabricated; every missing
    identity field remains a failed gate rather than being silently imputed.
    """

    readiness = _readiness(
        development,
        calibration,
        benchmark,
        policy,
        development_loss_difference,
    )
    gates = dict(policy["readiness_gates"])
    training_rows = int(_safe_num(development, "news_available").gt(0).sum())
    audit_rows, audit_column = _count_blind_audit_rows(
        pd.concat([development, calibration, benchmark], ignore_index=True)
    )
    automated_audit_rows, automated_audit_column = _count_automated_audit_rows(
        pd.concat([development, calibration, benchmark], ignore_index=True)
    )
    audit_policy = dict(policy.get("evidence_audit") or {})
    audit_mode = str(audit_policy.get("mode", "blind_human"))
    manual_review_required = bool(audit_policy.get("manual_review_required", audit_mode == "blind_human"))
    automated_audit_required = int(
        gates.get(
            "minimum_automated_audit_rows",
            audit_policy.get("minimum_rows", 0),
        )
    )
    checks = dict(readiness.get("checks") or {})
    checks["minimum_event_impact_training_rows"] = training_rows >= int(
        gates["minimum_event_impact_training_rows"]
    )
    checks["minimum_blind_audit_rows"] = audit_rows >= int(
        gates.get("minimum_blind_audit_rows", 0)
    )
    checks["minimum_automated_audit_rows"] = automated_audit_rows >= automated_audit_required
    checks["strict_artifact_identity"] = bool(
        provenance.get("strict_identity_passed")
    )
    requirements = dict(readiness.get("requirements") or {})
    requirements.update(
        {
            "event_impact_training_rows": int(gates["minimum_event_impact_training_rows"]),
            "blind_audit_rows": int(gates.get("minimum_blind_audit_rows", 0)),
            "automated_audit_rows": automated_audit_required,
            "strict_artifact_identity": True,
        }
    )
    gaps = dict(readiness.get("evidence_gaps") or {})
    gaps.update(
        {
            "event_impact_training_rows": max(
                0,
                int(gates["minimum_event_impact_training_rows"]) - training_rows,
            ),
            "blind_audit_rows": max(
                0,
                int(gates.get("minimum_blind_audit_rows", 0)) - audit_rows,
            ),
            "automated_audit_rows": max(
                0,
                automated_audit_required - automated_audit_rows,
            ),
            # Keep the gap representation numerically interpretable.  The
            # inherited v3 helper may carry a stale boolean here even when
            # the v4 identity check has passed; recompute it from the actual
            # v4 provenance rather than exposing contradictory diagnostics.
            "strict_artifact_identity": 0
            if bool(provenance.get("strict_identity_passed"))
            else 1,
        }
    )
    readiness["checks"] = checks
    readiness["passed"] = all(bool(value) for value in checks.values())
    readiness["requirements"] = requirements
    readiness["evidence_gaps"] = gaps
    readiness["event_impact_training_rows"] = training_rows
    readiness["blind_audit_rows"] = audit_rows
    readiness["blind_audit_column"] = audit_column
    readiness["automated_audit_rows"] = automated_audit_rows
    readiness["automated_audit_column"] = automated_audit_column
    readiness["audit_mode"] = audit_mode
    readiness["manual_review_required"] = manual_review_required
    readiness["strict_artifact_identity"] = bool(
        provenance.get("strict_identity_passed")
    )
    return readiness


def _v4_preflight(
    raw: pd.DataFrame,
    horizon: int,
    policy: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    fit_intercept_candidates: Sequence[bool] = (True,),
    selection_frame: str = "raw",
) -> dict[str, Any]:
    """Run selection-aware readiness checks before expensive resampling."""

    selected_rows, selected, candidate, window_trials = _select_v4_variant(
        raw,
        horizon,
        policy,
        features=V4_FEATURES,
        fit_intercept_candidates=fit_intercept_candidates,
        selection_frame=selection_frame,
    )
    dates = pd.to_datetime(selected_rows["observation_date"], errors="coerce")
    targets = pd.to_datetime(selected_rows["target_date"], errors="coerce")
    development = selected_rows.loc[
        dates.dt.year.between(2019, 2023)
        & targets.lt(pd.Timestamp("2024-01-01"))
    ].reset_index(drop=True)
    calibration = selected_rows.loc[
        dates.dt.year.eq(2024) & targets.lt(pd.Timestamp("2025-01-01"))
    ].reset_index(drop=True)
    benchmark = selected_rows.loc[dates.dt.year.eq(2025)].reset_index(drop=True)
    if min(len(development), len(calibration), len(benchmark)) == 0:
        raise ValueError(f"Horizon {horizon} lacks a required v4 evaluation period")
    loss_difference = _development_oof_loss_difference(
        development,
        V4_FEATURES,
        candidate,
    )
    readiness = _v4_readiness(
        development,
        calibration,
        benchmark,
        policy,
        loss_difference,
        provenance,
    )
    return {
        "selected_window_days": float(selected.get("news_window_days", 0.0)),
        "selected_half_life_days": float(selected.get("news_half_life_days", 0.0)),
        "selected_configuration": candidate,
        "selection_frame": str(selection.get("selection_frame") or selection_frame),
        "selection_stock_meta_configuration": selection.get(
            "selection_stock_meta_configuration"
        ),
        "development_rows": int(len(development)),
        "calibration_rows": int(len(calibration)),
        "benchmark_rows": int(len(benchmark)),
        "window_trial_count": int(len(window_trials)),
        "readiness": readiness,
    }


def _v4_placebo_frame(frame: pd.DataFrame, *, mode: str, seed: int) -> pd.DataFrame:
    """Shuffle tone/impact while preserving ticker-month availability."""

    result = frame.copy()
    date = pd.to_datetime(result["observation_date"], errors="coerce")
    groups = [result["ticker"].astype(str), date.dt.year, date.dt.month]
    content = [*TONE_FEATURES, EVENT_IMPACT_FEATURE]
    columns = ["news_available", *V4_FEATURES] if mode == "complete_event" else content
    generator = np.random.default_rng(seed)
    for _, index in result.groupby(groups, sort=False, dropna=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        if len(positions) > 1:
            shuffled = generator.permutation(positions)
            result.loc[positions, columns] = result.loc[shuffled, columns].to_numpy()
    return result


def _v4_placebos(
    benchmark: pd.DataFrame,
    model: IncrementalNewsFusion,
    stock_probability: np.ndarray,
    target: np.ndarray,
    actual: float,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    outputs: dict[str, list[float]] = {"content_only": [], "complete_event": []}
    for mode_index, mode in enumerate(outputs):
        for iteration in range(int(iterations)):
            placebo = _v4_placebo_frame(
                benchmark, mode=mode, seed=int(seed) + mode_index * 1_000_000 + iteration
            )
            probability = model.predict(placebo).probability_up
            outputs[mode].append(
                float(np.mean((target - stock_probability) ** 2) - np.mean((target - probability) ** 2))
            )
    return {
        mode: {
            "iterations": len(values),
            "mean": float(np.mean(values)) if values else 0.0,
            "p95": float(np.quantile(values, 0.95)) if values else 0.0,
            "actual_percentile": float(np.mean(np.asarray(values) < actual)) if values else 0.0,
            "distribution": values,
        }
        for mode, values in outputs.items()
    }


def _v4_rolling(
    raw: pd.DataFrame,
    horizon: int,
    policy: Mapping[str, Any],
    *,
    fit_intercept_candidates: Sequence[bool] = (True,),
    selection_frame: str = "raw",
) -> list[dict[str, Any]]:
    selection_frame = str(selection_frame).strip().casefold().replace("-", "_")
    if selection_frame not in {"raw", "capacity_matched"}:
        raise ValueError("selection_frame must be 'raw' or 'capacity_matched'")
    rows: list[dict[str, Any]] = []
    for origin in policy["rolling_origins"]:
        prepared, _ = prepare_v4_features(raw, horizons=(horizon,), development_end=int(origin["train_end"]))
        try:
            selected_rows, selected, selected_candidate, _ = _select_v4_variant(
                prepared,
                horizon,
                policy,
                features=V4_FEATURES,
                development_end=int(origin["train_end"]),
                fit_intercept_candidates=fit_intercept_candidates,
                selection_frame=selection_frame,
            )
        except ValueError:
            continue
        dates = pd.to_datetime(selected_rows["observation_date"])
        targets = pd.to_datetime(selected_rows["target_date"], errors="coerce")
        train_end = int(origin["train_end"])
        calibration_year = int(origin["calibration"])
        evaluation_year = int(origin["evaluation"])
        development_raw = selected_rows.loc[dates.dt.year.between(2019, train_end) & targets.lt(pd.Timestamp(f"{calibration_year}-01-01"))].reset_index(drop=True)
        calibration_raw = selected_rows.loc[dates.dt.year.eq(calibration_year) & targets.lt(pd.Timestamp(f"{evaluation_year}-01-01"))].reset_index(drop=True)
        evaluation_raw = selected_rows.loc[dates.dt.year.eq(evaluation_year)].reset_index(drop=True)
        if min(len(development_raw), len(calibration_raw), len(evaluation_raw)) == 0:
            continue
        development, calibration, evaluation, _, _ = _capacity_matched_frames(development_raw, calibration_raw, evaluation_raw, policy)
        if selection_frame == "capacity_matched":
            # ``_select_v4_variant`` already selected the compact correction on
            # the same capacity-matched stock frame.  Re-selecting here would
            # make the rolling-origin protocol disagree with the main fit.
            candidate = selected_candidate
        else:
            candidate, _ = _select_configuration(
                development,
                V4_FEATURES,
                policy,
                fit_intercept_candidates=fit_intercept_candidates,
            )
        model = IncrementalNewsFusion(
            V4_FEATURES,
            penalty_strength=float(candidate["penalty_strength"]),
            l1_ratio=float(candidate["l1_ratio"]),
            adjustment_cap=float(candidate["adjustment_cap"]),
            random_seed=42,
            fit_intercept=bool(candidate.get("fit_intercept", True)),
        ).fit(development, development["target_up"].astype(int))
        probability = model.predict(evaluation).probability_up
        stock = evaluation["stock_probability_up"].to_numpy(dtype=float)
        target = evaluation["target_up"].astype(int).to_numpy()
        threshold = _shared_threshold(calibration)
        rows.append({
            "evaluation_year": evaluation_year,
            "rows": int(len(evaluation)),
            "news_present_rows": int(evaluation["news_available"].sum()),
            "brier_improvement": float(np.mean((target - stock) ** 2) - np.mean((target - probability) ** 2)),
            "balanced_accuracy_gain": float(_metrics(target, probability, threshold)["balanced_accuracy"] - _metrics(target, stock, threshold)["balanced_accuracy"]),
            "selected_window_days": selected.get("news_window_days"),
            "selected_half_life_days": selected.get("news_half_life_days"),
            "selection_frame": selection_frame,
        })
    return rows


def _v4_loco(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    benchmark: pd.DataFrame,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ticker in sorted(benchmark["ticker"].astype(str).unique()):
        train = development.loc[development["ticker"].astype(str).ne(ticker)].reset_index(drop=True)
        calibrate = calibration.loc[calibration["ticker"].astype(str).ne(ticker)].reset_index(drop=True)
        evaluate = benchmark.loc[benchmark["ticker"].astype(str).eq(ticker)].reset_index(drop=True)
        if min(len(train), len(calibrate), len(evaluate)) == 0 or int(train["news_available"].sum()) < len(V4_FEATURES) + 2:
            continue
        model = IncrementalNewsFusion(
            V4_FEATURES,
            penalty_strength=float(candidate["penalty_strength"]),
            l1_ratio=float(candidate["l1_ratio"]),
            adjustment_cap=float(candidate["adjustment_cap"]),
            random_seed=42,
            fit_intercept=bool(candidate.get("fit_intercept", True)),
        ).fit(train, train["target_up"].astype(int))
        fused = model.predict(evaluate).probability_up
        stock = evaluate["stock_probability_up"].to_numpy(dtype=float)
        target = evaluate["target_up"].astype(int).to_numpy()
        threshold = _shared_threshold(calibrate)
        result[ticker] = {
            "rows": int(len(evaluate)),
            "news_present_rows": int(evaluate["news_available"].sum()),
            "brier_improvement": float(np.mean((target - stock) ** 2) - np.mean((target - fused) ** 2)),
            "balanced_accuracy_gain": float(_metrics(target, fused, threshold)["balanced_accuracy"] - _metrics(target, stock, threshold)["balanced_accuracy"]),
        }
    return result


def _write_v4_registry(report: Mapping[str, Any], registry_path: Path) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for key, value in (report.get("horizons") or {}).items():
        artifact = PROJECT_ROOT / str(value["artifact_path"])
        horizons[str(key)] = {
            "accepted": bool(value.get("accepted")),
            "artifact_path": value["artifact_path"],
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.is_file() else None,
            "selected_window_days": value.get("selected_window_days"),
            "selected_half_life_days": value.get("selected_half_life_days"),
            "decision_threshold": value.get("shared_decision_threshold"),
            "runtime_label": "Fusion-supported" if value.get("accepted") else "Stock-led fallback",
        }
    accepted = [int(key) for key, value in horizons.items() if value["accepted"]]
    all_accepted = bool(horizons) and len(accepted) == len(horizons)
    # The v4 estimator uses a compact ``v4_*`` feature contract and a
    # target-specific event-impact model.  Runtime materialisation is kept
    # explicitly disabled until that point-in-time state is persisted and a
    # parity test proves that the live frame matches the sealed offline frame.
    runtime_feature_contract_ready = False
    runtime_enabled = all_accepted and runtime_feature_contract_ready
    registry = {
        "schema_version": "fusion-registry-v6",
        "status": "accepted" if all_accepted else "research_only",
        "overall_status": "all_horizon_fusion" if all_accepted else "stock_led_fallback",
        "overall_all_horizon_accepted": all_accepted,
        "runtime_action": "enable_fusion" if runtime_enabled else "stock_led_fallback",
        "runtime_feature_contract_ready": runtime_feature_contract_ready,
        "runtime_note": (
            "Scientific v4 acceptance is isolated from production until the "
            "point-in-time runtime feature contract and event-impact state pass parity checks."
        ),
        "claim_scope": report["claim_scope"],
        "policy_sha256": report["policy_sha256"],
        "dataset_sha256": report["dataset_sha256"],
        "sentiment_identity": report.get("provenance", {}).get("sentiment_variant"),
        "provenance": report.get("provenance"),
        "accepted_horizons": accepted,
        "horizons": horizons,
        "report_path": _project_reference(Path(str(report["_report_path"]))),
        "promotion_note": "v4 candidate only; active production is unchanged until reviewed",
    }
    _atomic_json(registry_path, registry)
    return registry


def run_fusion_evaluation_v4(
    dataset_path: Path = DEFAULT_FUSION_DATASET,
    *,
    policy_path: Path = DEFAULT_POLICY,
    report_path: Path = DEFAULT_REPORT,
    registry_path: Path = DEFAULT_REGISTRY,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    permutation_iterations: int | None = None,
    bootstrap_iterations: int | None = None,
    horizons: Sequence[int] | None = None,
    include_rolling_origins: bool = True,
    enforce_readiness: bool = True,
    fit_intercept_candidates: Sequence[bool] = (True,),
    selection_frame: str = "raw",
) -> dict[str, Any]:
    """Run the isolated v4 candidate evaluation on the frozen feature table."""

    policy, policy_hash = load_policy(policy_path)
    source = load_fusion_dataset(dataset_path)
    raw, impact_provenance = prepare_v4_features(source, horizons=policy["supported_horizons"])
    provenance = _identity(raw, dataset_path, policy_hash)
    requested = tuple(int(value) for value in (horizons or policy["supported_horizons"]))
    # Readiness is selection-aware and includes the complete evidence
    # contract (per-company coverage, material events, power, automated
    # provenance audit and strict artifact identity).  It runs before expensive bootstrap/placebo
    # resampling and is retained in the diagnostic report when
    # ``--allow-unready`` is used.
    preflight: dict[str, Any] = {}
    for horizon in requested:
        preflight[str(horizon)] = _v4_preflight(
            raw,
            horizon,
            policy,
            provenance,
            fit_intercept_candidates=fit_intercept_candidates,
        )
    readiness_summary = {
        key: value["readiness"] for key, value in preflight.items()
    }
    failed_readiness = {
        horizon: value["readiness"].get("evidence_gaps", {})
        for horizon, value in preflight.items()
        if not value["readiness"].get("passed")
    }
    if enforce_readiness and failed_readiness:
        raise FusionV4ReadinessError(
            "Fusion v4 requires expanded, point-in-time news evidence and matching "
            "model provenance before the full benchmark; "
            f"readiness gaps={failed_readiness}"
        )
    report: dict[str, Any] = {
        "schema_version": "stocksense-fusion-evaluation-v4",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": "Retrospective evidence, 2019-2025",
        "periods": policy["periods"],
        "policy_sha256": policy_hash,
        "dataset_sha256": provenance["dataset_sha256"],
        "provenance": provenance,
        "event_impact_provenance": impact_provenance,
        "protocol": {
            "schema": "fusion-evaluation-v4",
            "stock_active_registry_frozen": True,
            "sentiment_artifact_frozen": True,
            "primary_fusion_features": list(V4_FEATURES),
            "no_news_exact_fallback": True,
            "public_disclosure_join_required_for_readiness": True,
        },
        "readiness": readiness_summary,
        "readiness_preflight": {
            key: {
                "selected_window_days": value["selected_window_days"],
                "selected_half_life_days": value["selected_half_life_days"],
                "selected_configuration": value["selected_configuration"],
                "development_rows": value["development_rows"],
                "calibration_rows": value["calibration_rows"],
                "benchmark_rows": value["benchmark_rows"],
                "window_trial_count": value["window_trial_count"],
            }
            for key, value in preflight.items()
        },
        "arms": ["stock_only", "stock_only_meta", *V4_ARMS, "v3_full_diagnostic", "permuted_news_placebos"],
        "horizons": {},
        "selection_frame": str(selection_frame).strip().casefold().replace("-", "_"),
    }
    effective_permutations = int(permutation_iterations if permutation_iterations is not None else policy["permutation_iterations"])
    effective_bootstraps = int(bootstrap_iterations if bootstrap_iterations is not None else policy["bootstrap_iterations"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_p_values: dict[int, float] = {}
    gates = policy["acceptance_gates"]
    for horizon in requested:
        selected_rows, selected, selected_candidate, window_trials = _select_v4_variant(
            raw,
            horizon,
            policy,
            features=V4_FEATURES,
            fit_intercept_candidates=fit_intercept_candidates,
        )
        dates = pd.to_datetime(selected_rows["observation_date"])
        targets = pd.to_datetime(selected_rows["target_date"], errors="coerce")
        development_raw = selected_rows.loc[dates.dt.year.between(2019, 2023) & targets.lt(pd.Timestamp("2024-01-01"))].reset_index(drop=True)
        calibration_raw = selected_rows.loc[dates.dt.year.eq(2024) & targets.lt(pd.Timestamp("2025-01-01"))].reset_index(drop=True)
        benchmark_raw = selected_rows.loc[dates.dt.year.eq(2025)].reset_index(drop=True)
        if min(len(development_raw), len(calibration_raw), len(benchmark_raw)) == 0:
            raise ValueError(f"Horizon {horizon} lacks complete v4 periods")
        development, calibration, benchmark, stock_meta_config, stock_meta_trials = _capacity_matched_frames(development_raw, calibration_raw, benchmark_raw, policy)
        threshold = _shared_threshold(calibration)
        target = benchmark["target_up"].astype(int).to_numpy()
        stock_probability = benchmark["stock_probability_up"].to_numpy(dtype=float)
        stock_prediction = (stock_probability >= threshold).astype(int)
        stock_metrics = _metrics(target, stock_probability, threshold)
        raw_stock_probability = benchmark["stock_probability_raw"].to_numpy(dtype=float)
        arms: dict[str, Any] = {
            "stock_only": {"metrics": _metrics(target, raw_stock_probability, threshold), "diagnostic_only": True},
            "stock_only_meta": {"metrics": stock_metrics, "selected_configuration": stock_meta_config, "development_trials": stock_meta_trials, "capacity_matched": True},
        }
        models: dict[str, IncrementalNewsFusion] = {}
        for arm_name, features in V4_ARMS.items():
            arm_selected, arm_trials = _select_configuration(
                development,
                features,
                policy,
                fit_intercept_candidates=fit_intercept_candidates,
            )
            model = IncrementalNewsFusion(
                features,
                penalty_strength=float(arm_selected["penalty_strength"]),
                l1_ratio=float(arm_selected["l1_ratio"]),
                adjustment_cap=float(arm_selected["adjustment_cap"]),
                random_seed=42,
                fit_intercept=bool(arm_selected.get("fit_intercept", True)),
            ).fit(development, development["target_up"].astype(int))
            probability = model.predict(benchmark).probability_up
            arms[arm_name] = {"metrics": _metrics(target, probability, threshold), "selected_configuration": arm_selected, "development_trials": arm_trials, "model_provenance": model.provenance()}
            models[arm_name] = model
        # Retain the v3 stack only as a named diagnostic; v4's compact model is
        # the sole candidate eligible for the acceptance decision.
        full_model = models["compact_full"]
        prediction = full_model.predict(benchmark)
        fusion_probability = prediction.probability_up
        fusion_decision = (fusion_probability >= threshold).astype(int)
        unavailable = ~prediction.eligible_news
        if not np.array_equal(fusion_probability[unavailable], stock_probability[unavailable]) or not np.array_equal(fusion_decision[unavailable], stock_prediction[unavailable]):
            raise AssertionError("v4 no-news rows must preserve the stock probability and direction exactly")
        fusion_metrics = arms["compact_full"]["metrics"]
        brier_improvement = float(np.mean((target - stock_probability) ** 2) - np.mean((target - fusion_probability) ** 2))
        ba_gain = float(fusion_metrics["balanced_accuracy"] - stock_metrics["balanced_accuracy"])
        bootstrap = _block_bootstrap(benchmark, target, stock_probability, fusion_probability, stock_prediction, fusion_decision, block_size=int(policy["bootstrap_blocks"][str(horizon)]), iterations=effective_bootstraps, seed=int(policy["random_seed"]) + horizon)
        placebo = _v4_placebos(benchmark, full_model, stock_probability, target, brier_improvement, iterations=effective_permutations, seed=int(policy["random_seed"]) + horizon * 100_000)
        clark_west = _clark_west(target, stock_probability, fusion_probability, benchmark["observation_date"])
        raw_p_values[horizon] = float(clark_west["one_sided_p_value"])
        paired = _paired_details(benchmark, target, stock_prediction, fusion_decision)
        gains = [float(row["gain"]) for row in paired["per_company"].values()]
        rolling = (
            _v4_rolling(
                source,
                horizon,
                policy,
                fit_intercept_candidates=fit_intercept_candidates,
            )
            if include_rolling_origins
            else []
        )
        loco = _v4_loco(development, calibration, benchmark, arms["compact_full"]["selected_configuration"])
        # Re-evaluate readiness on the exact selected split.  The event-impact
        # readiness gate is about training evidence, not the number of rows
        # that happened to receive a fitted score in the benchmark.
        development_loss_difference = _development_oof_loss_difference(
            development,
            V4_FEATURES,
            arms["compact_full"]["selected_configuration"],
        )
        readiness = _v4_readiness(
            development,
            calibration,
            benchmark,
            policy,
            development_loss_difference,
            provenance,
        )
        impact_rows = int(readiness["event_impact_training_rows"])
        checks = {
            "readiness": bool(readiness["passed"]),
            "minimum_rows": len(benchmark) >= int(gates["minimum_benchmark_rows"]),
            "all_tickers": benchmark["ticker"].nunique() >= int(gates["minimum_tickers"]),
            "event_impact_rows": bool(
                readiness["checks"]["minimum_event_impact_training_rows"]
            ),
            "brier_improves": brier_improvement > 0,
            "brier_bootstrap_lower_above_zero": bootstrap["brier_improvement"]["lower"] > 0,
            "balanced_accuracy_gain": ba_gain >= float(gates["minimum_balanced_accuracy_gain"]),
            "balanced_accuracy_lower_non_negative": bootstrap["balanced_accuracy_gain"]["lower"] >= 0,
            "macro_f1_not_materially_worse": fusion_metrics["macro_f1"] >= stock_metrics["macro_f1"] - float(gates["maximum_macro_f1_loss"]),
            "ece_not_materially_worse": fusion_metrics["ece"] <= stock_metrics["ece"] + float(gates["maximum_ece_increase"]),
            "content_placebo": placebo["content_only"]["actual_percentile"] >= float(gates["minimum_placebo_percentile"]),
            "complete_event_placebo": placebo["complete_event"]["actual_percentile"] >= float(gates["minimum_placebo_percentile"]),
            "rolling_year_consistency": sum(item["brier_improvement"] > 0 for item in rolling) >= int(gates["minimum_positive_rolling_years"]),
            "rolling_complete": include_rolling_origins and len(rolling) == len(policy["rolling_origins"]),
            "median_company_gain_non_negative": bool(gains) and float(np.median(gains)) >= 0,
            "company_non_deterioration": sum(value >= 0 for value in gains) >= int(gates["minimum_non_deteriorating_companies"]),
            "no_large_company_loss": bool(gains) and min(gains) >= -float(gates["maximum_company_balanced_accuracy_loss"]),
            "strict_artifact_identity": provenance["strict_identity_passed"],
        }
        artifact_path = artifact_dir / f"fusion-v4-{provenance['dataset_sha256'][:12]}-{policy_hash[:12]}-{horizon}d.joblib"
        joblib.dump({"schema_version": "fusion-model-artifact-v4", "horizon_days": horizon, "model": full_model, "shared_decision_threshold": threshold, "selected_configuration": arms["compact_full"]["selected_configuration"], "selected_window_days": float(selected.get("news_window_days", 0)), "selected_half_life_days": float(selected.get("news_half_life_days", 0)), "policy_sha256": policy_hash, "dataset_sha256": provenance["dataset_sha256"], "feature_names": list(V4_FEATURES), "event_impact_provenance": impact_provenance}, artifact_path)
        report["horizons"][str(horizon)] = {
            "benchmark_rows": int(len(benchmark)),
            "news_present_rows": int(benchmark["news_available"].sum()),
            "event_impact_ready_rows": impact_rows,
            "ticker_count": int(benchmark["ticker"].nunique()),
            "selected_window_days": float(selected.get("news_window_days", 0)),
            "selected_half_life_days": float(selected.get("news_half_life_days", 0)),
            "shared_decision_threshold": threshold,
            "selected_configuration": arms["compact_full"]["selected_configuration"],
            "window_and_model_trials": window_trials,
            "arms": arms,
            "brier_improvement": brier_improvement,
            "balanced_accuracy_gain": ba_gain,
            "bootstrap": bootstrap,
            "clark_west": clark_west,
            "placebo": placebo,
            "calibration": {"stock_only": {"curve": _calibration_curve(target, stock_probability), **_calibration_regression(target, stock_probability)}, "compact_full": {"curve": _calibration_curve(target, fusion_probability), **_calibration_regression(target, fusion_probability)}},
            "error_transitions": paired["error_transitions"],
            "per_company": paired["per_company"],
            "rolling_origins": rolling,
            "leave_one_company_out": loco,
            "readiness": readiness,
            "acceptance_checks": checks,
            "artifact_path": _project_reference(artifact_path),
            "runtime_label": "Stock-led fallback",
            "accepted": False,
        }
    adjusted = _holm_adjust(raw_p_values)
    for horizon, value in adjusted.items():
        entry = report["horizons"][str(horizon)]
        entry["holm_adjusted_clark_west_p_value"] = value
        entry["acceptance_checks"]["holm_adjusted_significance"] = value < float(gates["holm_alpha"])
        entry["accepted"] = all(bool(item) for item in entry["acceptance_checks"].values())
        entry["runtime_label"] = "Fusion-supported" if entry["accepted"] else "Stock-led fallback"
    report["accepted_horizons"] = [int(key) for key, value in report["horizons"].items() if value["accepted"]]
    report["overall_status"] = "all_horizon_acceptance" if len(report["accepted_horizons"]) == len(requested) else "stock_led_fallback"
    report["overall_all_horizon_accepted"] = len(report["accepted_horizons"]) == len(requested)
    report["scientific_result_sha256"] = scientific_result_sha256(report)
    report["_report_path"] = str(report_path)
    report["registry_path"] = _project_reference(registry_path)
    registry = _write_v4_registry(report, registry_path)
    report["registry_status"] = registry["status"]
    _atomic_json(report_path, {key: value for key, value in report.items() if key != "_report_path"})
    report.pop("_report_path", None)
    return report
