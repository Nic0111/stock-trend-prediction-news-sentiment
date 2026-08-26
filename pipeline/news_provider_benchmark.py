"""Metrics-only NewsAPI versus Marketaux benchmark.

The benchmark is intentionally separate from the dashboard runtime.  It keeps
the same ten tickers, cutoff convention, FinBERT backend and context windows
for both providers, and never promotes a provider automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from pipeline.download_financial_news_data import (
    NEWS_CONTEXT_CONFIG_VERSION,
    NewsContextResult,
    build_news_context_metadata,
    clean_text,
    load_stock_metadata,
    retrieve_financial_news,
)
from pipeline.marketaux_news_provider import (
    MarketauxConfigurationError,
    MarketauxRequestError,
    _record_marketaux_response,
    _reserve_marketaux_request,
    get_marketaux_api_token,
    probe_marketaux_entities,
    retrieve_marketaux_financial_news,
)


BENCHMARK_ROOT = Path(__file__).resolve().parent.parent / "results" / "news-provider-marketaux-benchmark"
PROVIDERS = ("newsapi", "marketaux")
DEFAULT_CUTOFF_LAG_DAYS = 1
DEFAULT_STUDY_DAYS = 14


@dataclass
class ProviderRecord:
    provider: str
    ticker: str
    as_of_date: str
    status: str
    elapsed_seconds: float
    error: str | None
    result: Dict[str, Any] | None
    run_id: str = "legacy"
    collection_date: str | None = None
    cutoff_lag_days: int = DEFAULT_CUTOFF_LAG_DAYS
    sentiment_backend: str = "unknown"
    sentiment_model: str = "unknown"
    provider_metadata: Dict[str, Any] = field(default_factory=dict)
    context_config_version: int = 1


def get_sentiment_runtime_descriptor() -> Dict[str, str]:
    config = Path(__file__).resolve().parent.parent / "models" / "sentiment-runtime-config.json"
    loaded: Dict[str, Any] = {}
    if config.exists():
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                loaded = payload
        except (OSError, json.JSONDecodeError):
            loaded = {}
    variant = (
        os.getenv("STOCKSENSE_SENTIMENT_VARIANT", "").strip()
        or clean_text(loaded.get("active_variant"))
        or "finbert_phrasebank_v1"
    )
    challenger = (loaded.get("challengers") or {}).get(variant)
    if variant != "finbert_phrasebank_v1" and isinstance(challenger, dict):
        return {
            "backend": clean_text(challenger.get("backend")) or "unknown",
            "model": variant,
            "variant": variant,
        }
    return {
        "backend": clean_text(loaded.get("default_backend"))
        or clean_text(loaded.get("backend"))
        or "unknown",
        "model": clean_text(loaded.get("model_directory")) or "unknown",
        "variant": variant,
    }


def _safe_provider_status(exc: Exception) -> str:
    if isinstance(exc, MarketauxConfigurationError):
        return "not_configured"
    if getattr(exc, "status", None) == 429:
        return "rate_limited"
    if getattr(exc, "status", None) == 402:
        return "usage_limited"
    return "error"


def _snapshot_path(run_dir: Path, ticker: str, provider: str) -> Path:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in ticker).strip("-")
    return run_dir / f"{slug}-{provider}.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_entity_coverage(root: Path) -> Dict[str, Dict[str, Any]]:
    path = root / "entity-coverage.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _ensure_entity_coverage(
    root: Path,
    tickers: Sequence[str],
) -> tuple[Dict[str, Dict[str, Any]], str | None]:
    existing = _load_entity_coverage(root)
    if all(ticker in existing for ticker in tickers):
        return existing, None
    try:
        coverage = probe_marketaux_entities(tickers, token=get_marketaux_api_token())
    except MarketauxRequestError as exc:
        return existing, str(exc)
    merged = {**existing, **coverage}
    _write_json(root / "entity-coverage.json", merged)
    return merged, None


def _retrieve(
    ticker: str,
    provider: str,
    as_of_date: date,
    entity_coverage: Mapping[str, Mapping[str, Any]],
) -> NewsContextResult:
    if provider == "newsapi":
        return retrieve_financial_news(
            ticker,
            as_of_date=as_of_date,
            persist_outputs=False,
        )
    if provider == "marketaux":
        _reserve_marketaux_request()
        try:
            result = retrieve_marketaux_financial_news(
                ticker,
                as_of_date=as_of_date,
                entity_mapping=entity_coverage.get(ticker),
                persist_outputs=False,
                force_search=True,
            )
        except Exception as exc:
            _record_marketaux_response(
                getattr(exc, "headers", {}) or {},
                f"benchmark_error_{getattr(exc, 'status', None) or 'request'}",
            )
            raise
        result.provider_metadata["marketaux_usage"] = _record_marketaux_response(
            (result.provider_metadata or {}).get("marketaux_headers") or {},
            "benchmark_ok",
        )
        return result
    raise ValueError(f"Unsupported provider: {provider}")


def collect_news_provider_benchmark(
    as_of_date: date | str | None = None,
    tickers: Sequence[str] | None = None,
    providers: Sequence[str] = PROVIDERS,
    benchmark_root: Path = BENCHMARK_ROOT,
    run_id: str | None = None,
    cutoff_lag_days: int = DEFAULT_CUTOFF_LAG_DAYS,
) -> List[ProviderRecord]:
    """Collect one immutable NewsAPI/Marketaux snapshot."""
    selected_providers = tuple(providers)
    invalid = set(selected_providers) - set(PROVIDERS)
    if invalid:
        raise ValueError(f"Unsupported providers: {', '.join(sorted(invalid))}")
    collection_date = pd.Timestamp(as_of_date).date() if as_of_date is not None else date.today()
    if cutoff_lag_days < 0:
        raise ValueError("cutoff_lag_days cannot be negative")
    effective_date = collection_date - timedelta(days=cutoff_lag_days)
    metadata = load_stock_metadata()
    selected_tickers = [clean_text(item).upper() for item in (tickers or metadata.keys())]
    for ticker in selected_tickers:
        if ticker not in metadata:
            raise ValueError(f"Unsupported ticker: {ticker}")
    selected_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    benchmark_root.mkdir(parents=True, exist_ok=True)
    entity_coverage, coverage_error = _ensure_entity_coverage(benchmark_root, selected_tickers)
    sentiment = get_sentiment_runtime_descriptor()
    run_dir = benchmark_root / "snapshots" / collection_date.isoformat() / selected_run_id
    records: List[ProviderRecord] = []
    for ticker in selected_tickers:
        for provider in selected_providers:
            started = time.monotonic()
            provider_metadata: Dict[str, Any] = {}
            if provider == "marketaux":
                provider_metadata["entity_coverage"] = entity_coverage.get(ticker, {})
                if coverage_error:
                    provider_metadata["entity_probe_error"] = coverage_error
            try:
                result = _retrieve(ticker, provider, effective_date, entity_coverage)
                provider_metadata.update(result.provider_metadata)
                record = ProviderRecord(
                    provider=provider,
                    ticker=ticker,
                    as_of_date=effective_date.isoformat(),
                    status="ok",
                    elapsed_seconds=time.monotonic() - started,
                    error=None,
                    result=result.to_dict(),
                    run_id=selected_run_id,
                    collection_date=collection_date.isoformat(),
                    cutoff_lag_days=cutoff_lag_days,
                    sentiment_backend=sentiment["backend"],
                    sentiment_model=sentiment["model"],
                    provider_metadata=provider_metadata,
                    context_config_version=NEWS_CONTEXT_CONFIG_VERSION,
                )
            except Exception as exc:
                provider_metadata.update(getattr(exc, "headers", {}) or {})
                record = ProviderRecord(
                    provider=provider,
                    ticker=ticker,
                    as_of_date=effective_date.isoformat(),
                    status=_safe_provider_status(exc),
                    elapsed_seconds=time.monotonic() - started,
                    error=str(exc),
                    result=None,
                    run_id=selected_run_id,
                    collection_date=collection_date.isoformat(),
                    cutoff_lag_days=cutoff_lag_days,
                    sentiment_backend=sentiment["backend"],
                    sentiment_model=sentiment["model"],
                    provider_metadata=provider_metadata,
                )
            records.append(record)
            _write_json(_snapshot_path(run_dir, ticker, provider), asdict(record))
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": selected_run_id,
            "collection_date": collection_date.isoformat(),
            "cutoff_date": effective_date.isoformat(),
            "cutoff_lag_days": cutoff_lag_days,
            "providers": list(selected_providers),
            "tickers": selected_tickers,
            "context_config": build_news_context_metadata(),
            "entity_coverage_error": coverage_error,
            "records": [asdict(record) for record in records],
        },
    )
    return records


def _load_records(benchmark_root: Path) -> List[ProviderRecord]:
    records: List[ProviderRecord] = []
    snapshot_root = benchmark_root / "snapshots"
    if not snapshot_root.exists():
        return records
    for path in sorted(snapshot_root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("provider") in PROVIDERS:
                records.append(ProviderRecord(**payload))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return records


def _provider_metrics(records: Sequence[ProviderRecord], provider: str) -> Dict[str, Any]:
    selected = [record for record in records if record.provider == provider]
    successful = [record for record in selected if record.status == "ok" and record.result]
    contexts = [context for record in successful for context in (record.result or {}).get("contexts", {}).values()]
    usable = [int(context.get("usable_article_count", 0)) for context in contexts]
    coverage = [str(context.get("coverage_confidence", "Unavailable")) for context in contexts]
    articles = [article for record in successful for article in (record.result or {}).get("articles", [])]
    return {
        "snapshots": len(selected),
        "successful_snapshots": len(successful),
        "success_rate": len(successful) / len(selected) if selected else 0.0,
        "error_rate": 1.0 - (len(successful) / len(selected) if selected else 0.0),
        "rate_limited_snapshots": sum(record.status == "rate_limited" for record in selected),
        "usage_limited_snapshots": sum(record.status == "usage_limited" for record in selected),
        "not_configured_snapshots": sum(record.status == "not_configured" for record in selected),
        "median_latency_seconds": float(pd.Series([record.elapsed_seconds for record in selected]).median()) if selected else None,
        "p95_latency_seconds": float(pd.Series([record.elapsed_seconds for record in selected]).quantile(0.95)) if selected else None,
        "relevant_articles": len(articles),
        "median_usable_articles": float(pd.Series(usable).median()) if usable else 0.0,
        "unavailable_or_low_rate": sum(item in {"Unavailable", "Low"} for item in coverage) / len(coverage) if coverage else 1.0,
        "high_or_medium_rate": sum(item in {"High", "Medium"} for item in coverage) / len(coverage) if coverage else 0.0,
        "unique_sources": len({clean_text(article.get("source_name")).casefold() for article in articles if clean_text(article.get("source_name"))}),
        "snippet_rate": sum(bool(clean_text(article.get("description"))) for article in articles) / len(articles) if articles else 0.0,
        "entity_supported_snapshots": sum(
            1
            for record in selected
            if record.provider_metadata.get("entity_coverage", {}).get("status") == "supported"
        ),
    }


def _article_identity(article: Mapping[str, Any]) -> str:
    url = clean_text(article.get("url"))
    if url:
        return url.casefold().rstrip("/")
    return clean_text(article.get("title")).casefold()


def _union_article_metrics(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, int]:
    merged: Dict[str, set[str]] = {}
    for provider, payload in (("newsapi", first), ("marketaux", second)):
        for article in payload.get("articles", []):
            if not isinstance(article, Mapping):
                continue
            identity = _article_identity(article)
            if identity:
                merged.setdefault(identity, set()).add(provider)
    return {
        "merged_relevant_articles": len(merged),
        "newsapi_exclusive_articles": sum(values == {"newsapi"} for values in merged.values()),
        "marketaux_exclusive_articles": sum(values == {"marketaux"} for values in merged.values()),
        "shared_articles": sum(len(values) == 2 for values in merged.values()),
    }


def _payload_has_usable_articles(payload: Mapping[str, Any] | None) -> bool:
    return bool(payload) and any(
        int(context.get("usable_article_count", 0)) > 0
        for context in (payload or {}).get("contexts", {}).values()
        if isinstance(context, Mapping)
    )


def _policy_metrics(records: Sequence[ProviderRecord]) -> Dict[str, Any]:
    grouped: Dict[tuple[str, str], Dict[str, ProviderRecord]] = {}
    for record in records:
        grouped.setdefault((record.collection_date or record.as_of_date, record.ticker), {})[record.provider] = record
    fallback: List[str] = []
    marketaux_recoveries = 0
    union_rows: List[Dict[str, int]] = []
    for providers in grouped.values():
        newsapi = providers.get("newsapi")
        marketaux = providers.get("marketaux")
        newsapi_ok = bool(newsapi and newsapi.status == "ok" and newsapi.result)
        marketaux_ok = bool(marketaux and marketaux.status == "ok" and marketaux.result)
        newsapi_usable = newsapi_ok and _payload_has_usable_articles(newsapi.result)
        marketaux_usable = marketaux_ok and _payload_has_usable_articles(marketaux.result)
        if newsapi_usable:
            fallback.append("newsapi")
        elif marketaux_usable:
            fallback.append("marketaux")
            marketaux_recoveries += 1
        else:
            fallback.append("unavailable")
        if newsapi_ok and marketaux_ok:
            union_rows.append(_union_article_metrics(newsapi.result or {}, marketaux.result or {}))
    return {
        "fallback": {
            "observations": len(fallback),
            "newsapi_selected": fallback.count("newsapi"),
            "marketaux_selected": fallback.count("marketaux"),
            "unavailable": fallback.count("unavailable"),
            "marketaux_recovery_rate": marketaux_recoveries / len(fallback) if fallback else 0.0,
        },
        "union": {
            "paired_observations": len(union_rows),
            "merged_relevant_articles": sum(row["merged_relevant_articles"] for row in union_rows),
            "marketaux_exclusive_articles": sum(row["marketaux_exclusive_articles"] for row in union_rows),
            "newsapi_exclusive_articles": sum(row["newsapi_exclusive_articles"] for row in union_rows),
            "shared_articles": sum(row["shared_articles"] for row in union_rows),
        },
    }


def _article_rows_from_records(records: Sequence[ProviderRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if not record.result:
            continue
        for article in record.result.get("articles", []):
            row = dict(article)
            row["snapshot_provider"] = record.provider
            row["snapshot_ticker"] = record.ticker
            rows.append(row)
    return rows


def _write_annotation_template(records: Sequence[ProviderRecord], path: Path) -> None:
    rows = _article_rows_from_records(records)
    rows.sort(key=lambda row: (str(row.get("snapshot_provider")), str(row.get("snapshot_ticker")), str(row.get("published_at"))), reverse=True)
    selected: List[Dict[str, Any]] = []
    for provider in PROVIDERS:
        provider_rows = [row for row in rows if row.get("snapshot_provider") == provider]
        selected.extend(provider_rows[:100])
    fields = [
        "snapshot_provider",
        "snapshot_ticker",
        "provider_sources",
        "title",
        "description",
        "text_completeness",
        "url",
        "published_at",
        "sentiment_label",
        "sentiment_probability",
        "relevant",
        "human_sentiment",
        "evidence_clarity_1_to_5",
        "ambiguous",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)


def _annotation_metrics(annotation_path: Path | None) -> Dict[str, Any]:
    if annotation_path is None or not annotation_path.exists():
        return {"labelled_rows": 0, "relevance_precision": None, "sentiment_accuracy": None, "mean_clarity": None}
    frame = pd.read_csv(annotation_path)
    required = {"relevant", "human_sentiment", "evidence_clarity_1_to_5"}
    if not required.issubset(frame.columns):
        return {"labelled_rows": int(len(frame)), "relevance_precision": None, "sentiment_accuracy": None, "mean_clarity": None}
    relevant = frame[frame["relevant"].astype(str).str.casefold().isin({"1", "true", "yes", "y"})]
    precision = float(len(relevant) / len(frame)) if len(frame) else None
    labelled_sentiment = relevant[relevant["human_sentiment"].astype(str).str.strip() != ""]
    accuracy = None
    if len(labelled_sentiment) and "sentiment_label" in labelled_sentiment:
        accuracy = float(
            (labelled_sentiment["sentiment_label"].astype(str).str.casefold() == labelled_sentiment["human_sentiment"].astype(str).str.casefold()).mean()
        )
    clarity = pd.to_numeric(relevant["evidence_clarity_1_to_5"], errors="coerce") if len(relevant) else pd.Series(dtype=float)
    return {
        "labelled_rows": int(len(frame)),
        "relevance_precision": precision,
        "sentiment_accuracy": accuracy,
        "mean_clarity": float(clarity.mean()) if clarity.notna().any() else None,
    }


def _annotation_metrics_by_provider(annotation_path: Path | None) -> Dict[str, Dict[str, Any]]:
    if annotation_path is None or not annotation_path.exists():
        return {}
    frame = pd.read_csv(annotation_path)
    if "snapshot_provider" not in frame.columns:
        return {"all": _annotation_metrics(annotation_path)}
    metrics: Dict[str, Dict[str, Any]] = {}
    for provider, provider_frame in frame.groupby(frame["snapshot_provider"].astype(str).str.casefold()):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as temporary:
            provider_frame.to_csv(temporary.name, index=False)
            temporary_path = Path(temporary.name)
        try:
            metrics[provider] = _annotation_metrics(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return metrics


def apply_provider_decision_rules(report: Dict[str, Any]) -> Dict[str, Any]:
    """Apply optional gates without changing runtime configuration."""
    metrics = report["metrics"]
    labels = report["annotation_metrics"]
    labels_by_provider = report["annotation_metrics_by_provider"]
    baseline = metrics["newsapi"]
    challenger = metrics["marketaux"]
    candidate_labels = labels_by_provider.get("marketaux", labels)
    newsapi_labels = labels_by_provider.get("newsapi", {})
    minimum_evidence = (
        report["completed_collection_days"] >= DEFAULT_STUDY_DAYS - 2
        and labels.get("labelled_rows", 0) >= 200
        and candidate_labels.get("labelled_rows", 0) >= 100
        and report.get("sentiment_evidence_valid", False)
    )
    candidate_precision = candidate_labels.get("relevance_precision")
    baseline_precision = newsapi_labels.get("relevance_precision")
    precision_non_inferior = (
        candidate_precision is not None
        and baseline_precision is not None
        and candidate_precision >= 0.80
        and candidate_precision >= baseline_precision - 0.03
    )
    sentiment_non_inferior = (
        candidate_labels.get("sentiment_accuracy") is not None
        and newsapi_labels.get("sentiment_accuracy") is not None
        and candidate_labels["sentiment_accuracy"] >= newsapi_labels["sentiment_accuracy"] - 0.03
    )
    coverage_gain = baseline["unavailable_or_low_rate"] - challenger["unavailable_or_low_rate"]
    fallback_eligible = bool(
        minimum_evidence
        and challenger["success_rate"] >= 0.90
        and candidate_precision is not None
        and candidate_precision >= 0.75
        and candidate_precision >= (baseline_precision or candidate_precision) - 0.05
        and report["policy_metrics"]["fallback"]["marketaux_recovery_rate"] >= 0.50
    )
    complement_eligible = bool(
        minimum_evidence
        and precision_non_inferior
        and report["policy_metrics"]["union"]["paired_observations"] > 0
        and report["policy_metrics"]["union"]["marketaux_exclusive_articles"] >= max(1, int(0.20 * max(1, baseline["relevant_articles"])))
        and coverage_gain >= 0.15
    )
    replacement_eligible = bool(
        complement_eligible
        and sentiment_non_inferior
        and challenger["success_rate"] >= 0.95
        and (candidate_labels.get("mean_clarity") or 0) >= 3.0
    )
    if replacement_eligible:
        recommendation = "marketaux_replacement"
    elif complement_eligible:
        recommendation = "newsapi_marketaux_complement"
    elif fallback_eligible:
        recommendation = "marketaux_fallback"
    else:
        recommendation = "retain_newsapi"
    report["decision_status"] = "rules_applied"
    report["decision_eligibility"] = {
        "replacement": replacement_eligible,
        "complementary": complement_eligible,
        "fallback": fallback_eligible,
    }
    report["recommendation"] = recommendation
    return report


def evaluate_news_provider_benchmark(
    benchmark_root: Path = BENCHMARK_ROOT,
    annotation_path: Path | None = None,
    apply_decision_rules: bool = False,
) -> Dict[str, Any]:
    all_records = _load_records(benchmark_root)
    current_records = [
        record for record in all_records
        if record.context_config_version == NEWS_CONTEXT_CONFIG_VERSION
    ]
    records = current_records or all_records
    excluded_legacy_records = len(all_records) - len(records)
    selected_context_config_version = (
        NEWS_CONTEXT_CONFIG_VERSION
        if current_records
        else (all_records[0].context_config_version if all_records else NEWS_CONTEXT_CONFIG_VERSION)
    )
    annotation_path = annotation_path or benchmark_root / "annotations.csv"
    _write_annotation_template(records, benchmark_root / "annotation-template.csv")
    metrics = {provider: _provider_metrics(records, provider) for provider in PROVIDERS}
    labels = _annotation_metrics(annotation_path)
    labels_by_provider = _annotation_metrics_by_provider(annotation_path)
    policy_metrics = _policy_metrics(records)
    completed_days = len({record.collection_date or record.as_of_date for record in records})
    backends = sorted({record.sentiment_backend for record in records})
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": len(records),
        "news_context_config_version": selected_context_config_version,
        "excluded_legacy_records": excluded_legacy_records,
        "completed_collection_days": completed_days,
        "sentiment_backends": backends,
        "sentiment_evidence_valid": backends == ["finbert"],
        "metrics": metrics,
        "policy_metrics": policy_metrics,
        "annotation_metrics": labels,
        "annotation_metrics_by_provider": labels_by_provider,
        "decision_status": "awaiting_user_approval",
        "recommendation": None,
        "promotion_gates": {
            "minimum_collection_days": DEFAULT_STUDY_DAYS - 2,
            "minimum_labelled_rows": 200,
            "marketaux_relevance_precision_minimum": 0.70,
            "fallback_success_rate_minimum": 0.90,
        },
    }
    if apply_decision_rules:
        report = apply_provider_decision_rules(report)
    _write_json(benchmark_root / "evaluation.json", report)
    lines = [
        "# News provider benchmark: NewsAPI versus Marketaux",
        "",
        f"Context configuration version: **{report['news_context_config_version']}**",
        f"Excluded legacy records: **{report['excluded_legacy_records']}**",
        "",
        f"Decision status: **{report['decision_status']}**",
        f"Recommendation: **{report.get('recommendation') or 'pending explicit approval'}**",
        "",
        "| Provider | Snapshots | Success | Relevant articles | Low/Unavailable | Median usable | Snippet rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in PROVIDERS:
        item = metrics[provider]
        lines.append(
            f"| {provider} | {item['snapshots']} | {item['success_rate']:.1%} | {item['relevant_articles']} | {item['unavailable_or_low_rate']:.1%} | {item['median_usable_articles']:.1f} | {item['snippet_rate']:.1%} |"
        )
    lines.extend(["", "## Annotation status", "", json.dumps(labels, indent=2)])
    lines.extend(["", "## Policy metrics", "", json.dumps(policy_metrics, indent=2)])
    (benchmark_root / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NewsAPI with the experimental Marketaux provider")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--as-of")
    collect.add_argument("--ticker", action="append")
    collect.add_argument("--provider", action="append", choices=list(PROVIDERS), default=None)
    collect.add_argument("--run-id")
    collect.add_argument("--cutoff-lag-days", type=int, default=DEFAULT_CUTOFF_LAG_DAYS)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--annotations")
    evaluate.add_argument("--apply-decision-rules", action="store_true")
    args = parser.parse_args()
    if args.command == "collect":
        records = collect_news_provider_benchmark(
            as_of_date=args.as_of,
            tickers=args.ticker,
            providers=args.provider or PROVIDERS,
            run_id=args.run_id,
            cutoff_lag_days=args.cutoff_lag_days,
        )
        print(json.dumps([asdict(record) for record in records], indent=2))
    else:
        report = evaluate_news_provider_benchmark(
            annotation_path=Path(args.annotations) if args.annotations else None,
            apply_decision_rules=args.apply_decision_rules,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
