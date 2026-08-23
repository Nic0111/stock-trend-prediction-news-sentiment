"""Resumable Bursa sentiment evidence remediation orchestration.

This module deliberately stops after producing a Codex review package.  It
does not call a labelling API, retrain a model, alter the sentiment runtime, or
promote a fusion registry.  A later explicit resume can import reviewed labels
through the existing append workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pipeline.bursa_news_append import (
    BursaNewsAppendError,
    export_bursa_news_append_labels,
    prepare_bursa_news_append,
    recover_bursa_news_append,
    triage_bursa_news_append,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "news-evidence-benchmark" / "bursa-news-llm-assisted-label.jsonl"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "data" / "historical-news-fusion" / "gdelt" / "external-corpus-2015-2025"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "data" / "historical-news-fusion" / "gdelt" / "sentiment-remediation"
DEFAULT_BENCHMARK = PROJECT_ROOT / "results" / "finbert-ensemble-defensible" / "production-ready-20260807-v1" / "report.json"
DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
VALID_SENTIMENTS = ("negative", "neutral", "positive")
MINIMUM_ACTIONABLE_ARTICLES = 90
MINIMUM_ACTIONABLE_CLASS_ROWS = 12
CLASS_FAILURE_F1 = 0.05
MAXIMUM_MACRO_F1_LOSS = -0.05


class SentimentRemediationError(RuntimeError):
    """Raised when remediation cannot continue without violating its contract."""


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SentimentRemediationError(f"Unable to read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SentimentRemediationError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SentimentRemediationError(
                f"Invalid JSONL at {path}:{line_number}"
            ) from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _recovery_row_counts(path: Path) -> tuple[int, int]:
    """Return (recovered, unavailable) counts from a hybrid recovery file."""

    recovered = unavailable = 0
    if not path.exists():
        return recovered, unavailable
    for row in _read_jsonl(path):
        status = str(row.get("recovery_status") or "").strip().casefold()
        if status in {"recovered", "recovered_not_label_eligible"}:
            recovered += 1
        elif status == "unavailable":
            unavailable += 1
    return recovered, unavailable


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def _label(row: Mapping[str, Any]) -> str:
    return str(row.get("sentiment_label") or "").strip().casefold()


def _split(row: Mapping[str, Any]) -> str:
    return str(row.get("split") or "").strip().casefold()


def _eligible(row: Mapping[str, Any]) -> bool:
    language = str(row.get("language") or "").strip().casefold()
    return (
        _truthy(row.get("entity_relevant"))
        and (not language or language.startswith("en"))
        and _label(row) in VALID_SENTIMENTS
    )


def _benchmark_company_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    report = _read_json(path)
    values = ((report.get("final_2025") or {}).get("per_company") or {})
    return {
        str(ticker).upper(): dict(record)
        for ticker, record in values.items()
        if isinstance(record, Mapping)
    }


def audit_sentiment_coverage(
    dataset_path: Path | str = DEFAULT_DATASET,
    *,
    benchmark_report: Path | str = DEFAULT_BENCHMARK,
) -> dict[str, Any]:
    """Describe ticker/class/split deficits and identify remediation priorities."""

    dataset = Path(dataset_path).resolve()
    rows = _read_jsonl(dataset)
    eligible = [row for row in rows if _eligible(row)]
    benchmark = _benchmark_company_metrics(Path(benchmark_report).resolve())
    tickers = sorted({str(row.get("ticker") or "").upper() for row in rows if row.get("ticker")})
    companies: dict[str, Any] = {}
    priorities: list[str] = []
    for ticker in tickers:
        group = [row for row in eligible if str(row.get("ticker") or "").upper() == ticker]
        classes = Counter(_label(row) for row in group)
        splits: dict[str, dict[str, int]] = {}
        for split in ("training", "validation", "test"):
            counts = Counter(_label(row) for row in group if _split(row) == split)
            splits[split] = {label: int(counts.get(label, 0)) for label in VALID_SENTIMENTS}
        metrics = benchmark.get(ticker, {})
        ensemble = metrics.get("ensemble") if isinstance(metrics.get("ensemble"), Mapping) else {}
        phrasebank = metrics.get("phrasebank") if isinstance(metrics.get("phrasebank"), Mapping) else {}
        macro_f1 = float(ensemble.get("macro_f1", 0.0) or 0.0) if ensemble else None
        neutral_f1 = float(ensemble.get("neutral_f1", 0.0) or 0.0) if ensemble else None
        delta = metrics.get("macro_f1_delta")
        macro_delta = float(delta) if delta is not None else None
        minimum_class_count = min((classes.get(label, 0) for label in VALID_SENTIMENTS), default=0)
        reasons: list[str] = []
        if len(group) < MINIMUM_ACTIONABLE_ARTICLES:
            reasons.append(f"eligible_rows_below_{MINIMUM_ACTIONABLE_ARTICLES}")
        if minimum_class_count < MINIMUM_ACTIONABLE_CLASS_ROWS:
            reasons.append(f"minority_class_below_{MINIMUM_ACTIONABLE_CLASS_ROWS}")
        if neutral_f1 is not None and neutral_f1 <= CLASS_FAILURE_F1:
            reasons.append(f"neutral_f1_at_or_below_{CLASS_FAILURE_F1:.2f}")
        if macro_delta is not None and macro_delta < MAXIMUM_MACRO_F1_LOSS:
            reasons.append("macro_f1_loss_vs_phrasebank_over_0.05")
        if reasons:
            priorities.append(ticker)
        companies[ticker] = {
            "eligible_rows": len(group),
            "class_counts": {label: int(classes.get(label, 0)) for label in VALID_SENTIMENTS},
            "minimum_class_count": int(minimum_class_count),
            "split_class_counts": splits,
            "retrospective_rows": int(metrics.get("rows", 0) or 0) if metrics else 0,
            "ensemble_macro_f1": macro_f1,
            "ensemble_neutral_f1": neutral_f1,
            "macro_f1_delta_vs_phrasebank": macro_delta,
            "phrasebank_macro_f1": (
                float(phrasebank.get("macro_f1", 0.0) or 0.0) if phrasebank else None
            ),
            "priority_reasons": reasons,
        }
    return {
        "schema_version": "sentiment-remediation-coverage-v1",
        "created_at": _utc_now(),
        "dataset_path": str(dataset),
        "dataset_sha256": _sha256_file(dataset),
        "raw_rows": len(rows),
        "eligible_rows": len(eligible),
        "priority_thresholds": {
            "minimum_articles": MINIMUM_ACTIONABLE_ARTICLES,
            "minimum_rows_per_class": MINIMUM_ACTIONABLE_CLASS_ROWS,
            "class_failure_f1": CLASS_FAILURE_F1,
            "maximum_macro_f1_loss_vs_phrasebank": MAXIMUM_MACRO_F1_LOSS,
        },
        "priority_tickers": priorities,
        "control_tickers": [ticker for ticker in tickers if ticker not in priorities],
        "companies": companies,
    }


def repair_codex_provenance(
    dataset_path: Path | str,
    *,
    backup_dir: Path | str,
) -> dict[str, Any]:
    """Truthfully mark historical Codex-labelled rows without changing labels."""

    dataset = Path(dataset_path).resolve()
    rows = _read_jsonl(dataset)
    before = _sha256_file(dataset)
    changed = 0
    for row in rows:
        original = str(row.get("label_source_original") or "").casefold()
        labelled_by = str(row.get("external_labelled_by") or "").casefold()
        status = str(row.get("validation_status") or "").casefold()
        is_codex = "codex" in original or labelled_by == "codex" or status.startswith("codex_")
        if not is_codex:
            continue
        row["label_source"] = "llm_assisted_label"
        row["label_provenance"] = "llm_assisted_label"
        row["provenance"] = "llm_assisted_label"
        row["validation_status"] = "llm_single_pass"
        row["annotation_agreement"] = "llm_single_review"
        row["llm_assisted_label"] = True
        row["llm_assisted_label_at"] = row.get("labelled_at", "")
        changed += 1
    backup = Path(backup_dir).resolve() / f"{dataset.stem}-before-codex-provenance-{int(time.time())}{dataset.suffix}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset, backup)
    if changed:
        _write_jsonl(dataset, rows)
    return {
        "dataset_path": str(dataset),
        "backup_path": str(backup),
        "rows": len(rows),
        "changed_rows": changed,
        "before_sha256": before,
        "after_sha256": _sha256_file(dataset),
        "labels_changed": False,
    }


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _recovery_progress(work_dir: Path) -> dict[str, Any]:
    manifest_path = work_dir / "append-manifest.json"
    if not manifest_path.exists():
        return {"prepared": False, "complete": False, "years": {}}
    append_manifest = _read_json(manifest_path)
    recovery_root = Path(
        append_manifest.get("recovery_root_dir") or (work_dir / "recovery-root")
    )
    years: dict[str, Any] = {}
    complete = True
    for year_text, record in sorted((append_manifest.get("candidate_files") or {}).items()):
        candidate_rows = int(record.get("rows", 0) or 0)
        year = int(year_text)
        report_path = recovery_root / str(year) / "recovery" / f"pilot-{year}-ngram-usability-report.json"
        report = _read_json(report_path) if report_path.exists() else {}
        completed = report.get("completed") is True
        complete = complete and completed
        raw_dir = recovery_root / str(year) / "recovery" / "raw"
        raw_path = raw_dir / "hybrid-recovered.jsonl"
        recovered_rows, unavailable_rows = _recovery_row_counts(raw_path)
        checkpoint = report.get("checkpoint") if isinstance(report.get("checkpoint"), Mapping) else {}
        reported_processed = int(
            report.get("processed_candidates")
            or checkpoint.get("processed_candidates", 0)
            or 0
        )
        heartbeat_path = raw_dir / "hybrid-recovery-heartbeat.json"
        heartbeat = _read_json(heartbeat_path) if heartbeat_path.exists() else {}
        publisher_path = raw_dir / "publisher-state.jsonl"
        publisher_attempted = (
            len(_read_jsonl(publisher_path)) if publisher_path.exists() else 0
        )
        live_completed = int(heartbeat.get("completed", 0) or 0)
        if str(heartbeat.get("stage") or "") != "publisher":
            live_completed = 0
        processed = max(reported_processed, publisher_attempted, live_completed)
        years[year_text] = {
            "candidates": candidate_rows,
            "processed": min(candidate_rows, processed) if processed else recovered_rows,
            "publisher_attempted": min(candidate_rows, publisher_attempted),
            "recovered": recovered_rows,
            "unavailable": unavailable_rows,
            "remaining": max(0, candidate_rows - (processed or recovered_rows)),
            "completed": completed,
            "stage": str(heartbeat.get("stage") or ("complete" if completed else "queued")),
            "checkpoint_updated_at": heartbeat.get("updated_at"),
            "report_path": str(report_path),
        }
    return {
        "prepared": True,
        "complete": bool(years) and complete,
        "selected_rows": int(append_manifest.get("selected_rows", 0) or 0),
        "years": years,
    }


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "remediation-manifest.json"


def remediation_status(
    *,
    run_id: str,
    run_root: Path | str = DEFAULT_RUN_ROOT,
) -> dict[str, Any]:
    run_dir = Path(run_root).resolve() / run_id
    manifest_file = _manifest_path(run_dir)
    if not manifest_file.exists():
        return {"run_id": run_id, "status": "not_started", "run_dir": str(run_dir)}
    manifest = _read_json(manifest_file)
    progress = _recovery_progress(run_dir / "historical")
    return {
        **manifest,
        "recovery_progress": progress,
        "free_space_gib": round(_free_gib(run_dir), 2),
    }


def _update_manifest(run_dir: Path, **changes: Any) -> dict[str, Any]:
    path = _manifest_path(run_dir)
    payload = _read_json(path) if path.exists() else {}
    payload.update(changes)
    payload["updated_at"] = _utc_now()
    _write_json(path, payload)
    return payload


def run_sentiment_remediation(
    *,
    run_id: str,
    run_root: Path | str = DEFAULT_RUN_ROOT,
    dataset_path: Path | str = DEFAULT_DATASET,
    candidate_root: Path | str = DEFAULT_CANDIDATE_ROOT,
    benchmark_report: Path | str = DEFAULT_BENCHMARK,
    max_runtime_minutes: float = 45.0,
    max_download_gib: float = 5.0,
    request_interval_seconds: float = 1.0,
    minimum_free_gib: float = 20.0,
    max_passes: int = 50,
    repair_provenance: bool = True,
    skip_archives: bool = False,
) -> dict[str, Any]:
    """Prepare, recover, triage and export a low-coverage Codex review pack."""

    if not run_id or any(character in run_id for character in "\\/:*?\"<>|"):
        raise ValueError("run_id must be a non-empty filesystem-safe identifier")
    run_dir = Path(run_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    historical = run_dir / "historical"
    manifest_file = _manifest_path(run_dir)
    dataset = Path(dataset_path).resolve()

    if not manifest_file.exists():
        provenance = (
            repair_codex_provenance(dataset, backup_dir=run_dir / "backups")
            if repair_provenance
            else {"skipped": True, "dataset_sha256": _sha256_file(dataset)}
        )
        audit = audit_sentiment_coverage(dataset, benchmark_report=benchmark_report)
        _write_json(run_dir / "coverage-before.json", audit)
        priorities = list(audit["priority_tickers"])
        if not priorities:
            payload = {
                "schema_version": "sentiment-remediation-run-v1",
                "run_id": run_id,
                "status": "historical_targets_satisfied",
                "created_at": _utc_now(),
                "dataset_path": str(dataset),
                "production_changes": False,
                "provenance_repair": provenance,
                "next_action": "Prepare the separately sealed 2026 prospective cohort.",
            }
            _write_json(manifest_file, payload)
            return remediation_status(run_id=run_id, run_root=run_root)
        prepare = prepare_bursa_news_append(
            dataset_path=dataset,
            candidate_root=Path(candidate_root),
            work_dir=historical,
            years=tuple(range(2015, 2026)),
            target_per_class_per_ticker=40,
            training_target_per_class=28,
            validation_target_per_class=6,
            test_target_per_class=6,
            recovery_multiplier=8,
            max_candidates_per_cycle=1000,
            tickers=priorities,
        )
        _write_json(
            manifest_file,
            {
                "schema_version": "sentiment-remediation-run-v1",
                "run_id": run_id,
                "status": "retrieving_historical",
                "created_at": _utc_now(),
                "dataset_path": str(dataset),
                "dataset_sha256": _sha256_file(dataset),
                "candidate_root": str(Path(candidate_root).resolve()),
                "benchmark_report": str(Path(benchmark_report).resolve()),
                "priority_tickers": priorities,
                "control_tickers": audit["control_tickers"],
                "historical_work_dir": str(historical),
                "selected_candidates": int(prepare.get("selected_rows", 0) or 0),
                "label_policy": "llm_assisted_label_weak_supervision",
                "historical_period": ["2015-01-01", "2025-12-31"],
                "prospective_year": 2026,
                "provenance_repair": provenance,
                "active_runtime_unchanged": True,
                "production_changes": False,
            },
        )

    for pass_number in range(1, max(1, int(max_passes)) + 1):
        free = _free_gib(run_dir)
        if free < float(minimum_free_gib):
            _update_manifest(
                run_dir,
                status="paused_storage",
                pause_reason=f"free space {free:.2f} GiB is below {minimum_free_gib:.2f} GiB",
                recovery_pass=pass_number,
            )
            return remediation_status(run_id=run_id, run_root=run_root)
        progress = _recovery_progress(historical)
        if progress.get("complete"):
            break
        try:
            recover_bursa_news_append(
                work_dir=historical,
                max_runtime_minutes=max_runtime_minutes,
                max_download_gib=max_download_gib,
                request_interval_seconds=request_interval_seconds,
                skip_archives=skip_archives,
            )
        except Exception as exc:  # preserve checkpoints for network/provider failures
            _update_manifest(
                run_dir,
                status="paused_network",
                pause_reason=f"{type(exc).__name__}: {exc}",
                recovery_pass=pass_number,
            )
            return remediation_status(run_id=run_id, run_root=run_root)
        _update_manifest(
            run_dir,
            status="retrieving_historical",
            recovery_pass=pass_number,
            free_space_gib=round(free, 2),
        )
    else:
        _update_manifest(
            run_dir,
            status="paused_safety_limit",
            pause_reason=f"maximum recovery passes reached: {max_passes}",
        )
        return remediation_status(run_id=run_id, run_root=run_root)

    try:
        triage = triage_bursa_news_append(work_dir=historical, random_control_share=0.10)
        package = export_bursa_news_append_labels(work_dir=historical, batch_size=25)
    except BursaNewsAppendError as exc:
        _update_manifest(
            run_dir,
            status="recovery_complete_no_label_package",
            pause_reason=str(exc),
        )
        return remediation_status(run_id=run_id, run_root=run_root)
    label_package = package.get("label_package", {})
    package_dir = label_package.get("output_dir") or label_package.get("package_dir")
    if not package_dir and label_package.get("review_pack_path"):
        package_dir = str(Path(label_package["review_pack_path"]).parent)
    _update_manifest(
        run_dir,
        status="awaiting_codex_labels",
        triage=triage,
        label_package=package,
        label_package_dir=package_dir,
        next_action=(
            "Review the exported package with Codex, import the returned labels, "
            "then resume the chronological GPU benchmark."
        ),
    )
    return remediation_status(run_id=run_id, run_root=run_root)


def resume_sentiment_remediation(**kwargs: Any) -> dict[str, Any]:
    """Resume the same immutable recovery workspace from its checkpoints."""

    return run_sentiment_remediation(**kwargs)
