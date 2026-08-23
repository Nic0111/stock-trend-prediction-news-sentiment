"""Repeatable end-to-end integrity audit for the React/FastAPI StockSense runtime."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.runtime_provenance import (
    PROJECT_ROOT,
    registry_payload,
    registry_signature,
    sha256_file,
    sentiment_artifact_hashes,
    stock_artifact_hashes,
)
from pipeline.fusion_model import DEFAULT_FUSION_DATASET, check_fusion_dataset_readiness
from pipeline.news_correction_audit import correction_audit_summary
from pipeline.market_calendar import calendar_precision
from pipeline.news_evidence_readiness import check_news_evidence_readiness
from pipeline.runtime_validation import sentiment_integrity


RESULTS_ROOT = PROJECT_ROOT / "results" / "system-audit"


def _check(name: str, passed: bool, detail: Any = None, critical: bool = True) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "critical": critical, "detail": detail}


def repair_registry_provenance(registry: dict[str, Any] | None = None) -> bool:
    """Explicitly backfill hashes into an old registry.

    This is intentionally separate from ``run_system_audit`` so an integrity
    check can never repair the object that it is supposed to inspect.
    """
    registry = dict(registry or registry_payload())
    changed = False
    hashes = stock_artifact_hashes(registry)
    for horizon, artifact_hash in hashes.items():
        entry = registry.setdefault("horizons", {}).setdefault(str(horizon), {})
        if entry.get("artifact_sha256") != artifact_hash:
            entry["artifact_sha256"] = artifact_hash
            changed = True
    run_id = str(registry.get("run_id") or time.strftime("%Y%m%dT%H%M%S"))
    benchmark_context = registry.setdefault("benchmark_context", {})
    if benchmark_context.get("evaluation_periods_path") != "config/evaluation_periods.json":
        benchmark_context["evaluation_periods_path"] = "config/evaluation_periods.json"
        changed = True
    bundle_dir = PROJECT_ROOT / "results" / "training-runs" / run_id
    report_sources = [
        PROJECT_ROOT / "results" / "model-comparison-summary.csv",
        PROJECT_ROOT / "results" / "stock-model-candidates.csv",
        PROJECT_ROOT / "results" / "stock-model-baselines.csv",
        PROJECT_ROOT / "results" / "stock-model-per-ticker-metrics.csv",
    ]
    if not registry.get("report_bundle_path") and all(path.exists() for path in report_sources):
        bundle_dir.mkdir(parents=True, exist_ok=True)
        for source in report_sources:
            shutil.copy2(source, bundle_dir / source.name)
        registry["report_bundle_path"] = str(bundle_dir.relative_to(PROJECT_ROOT))
        registry["provenance"] = {
            "dataset_sha256": registry.get("dataset_sha256"),
            "artifact_sha256": {str(h): value for h, value in hashes.items()},
            "report_sha256": {source.name: sha256_file(bundle_dir / source.name) for source in report_sources},
            "publication_status": "migrated",
        }
        (bundle_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "migration": True}, indent=2), encoding="utf-8")
        changed = True
    if changed:
        registry.setdefault("provenance", {})["migrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        temporary = PROJECT_ROOT / "models" / "model-registry.json.tmp"
        temporary.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(PROJECT_ROOT / "models" / "model-registry.json")
    return changed


def _date_coverage(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        frame = pd.read_csv(path, usecols=["Date"])
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        return {"exists": True, "rows": int(len(frame)), "start": str(dates.min().date()), "end": str(dates.max().date())}
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def _run_command(args: list[str], cwd: Path, timeout: int = 900) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def run_system_audit(level: str = "full", live_providers: bool = False) -> dict[str, Any]:
    if level not in {"quick", "full"}:
        raise ValueError("level must be quick or full")
    run_id = time.strftime("%Y%m%dT%H%M%S")
    output_dir = RESULTS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    checks.append(_check("python", sys.version_info >= (3, 12), sys.version))
    environment_ok = Path(sys.executable).is_file()
    try:
        import fastapi  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        environment_ok = False
    checks.append(
        _check(
            "environment",
            environment_ok,
            {"python_executable": sys.executable},
        )
    )
    precision = calendar_precision()
    checks.append(
        _check(
            "market_calendar_precision",
            precision == "bursa_exchange_calendar",
            {
                "precision": precision,
                "action": (
                    "Install the pinned exchange-calendars dependency for Bursa holiday awareness."
                    if precision != "bursa_exchange_calendar"
                    else None
                ),
            },
            critical=False,
        )
    )

    registry_error = None
    registry: dict[str, Any] = {}
    try:
        registry = registry_payload()
    except Exception as exc:
        registry_error = str(exc)
    checks.append(_check("registry_loadable", registry_error is None, registry_error))
    profile = str(registry.get("search_profile", "unknown"))
    publication_status = str(registry.get("publication_status", "unknown"))
    manual_override = registry.get("manual_override")
    manual_override_active = bool(
        isinstance(manual_override, dict) and manual_override.get("active")
    )
    registry_governance_detail = {
        "search_profile": profile,
        "publication_status": publication_status,
        "manual_override_active": manual_override_active,
        "manual_override_reason": (
            str(manual_override.get("reason"))
            if manual_override_active and isinstance(manual_override, dict) and manual_override.get("reason")
            else None
        ),
    }
    checks.append(
        _check(
            "registry_publication_policy",
            publication_status != "validated" or profile == "official",
            registry_governance_detail,
        )
    )
    checks.append(
        _check(
            "registry_officially_validated",
            profile == "official" and publication_status == "validated" and not manual_override_active,
            registry_governance_detail,
            critical=False,
        )
    )
    report_bundle = PROJECT_ROOT / str(registry.get("report_bundle_path", ""))
    report_manifest_ok = bool(report_bundle.exists() and (report_bundle / "manifest.json").exists())
    checks.append(_check("report_bundle_consistency", report_manifest_ok, str(report_bundle), critical=False))

    artifact_details: dict[str, Any] = {}
    for horizon in (1, 5, 10):
        entry = registry.get("horizons", {}).get(str(horizon), {})
        path = PROJECT_ROOT / str(entry.get("artifact_path", ""))
        actual = sha256_file(path) if path.exists() else None
        expected = entry.get("artifact_sha256")
        artifact_details[str(horizon)] = {"path": str(path), "exists": path.exists(), "expected": expected, "actual": actual}
        checks.append(
            _check(
                f"stock_artifact_{horizon}d",
                path.exists() and bool(expected) and expected == actual,
                artifact_details[str(horizon)],
            )
        )
        observed_acceptance = entry.get("observed_acceptance") or {}
        # The registry deliberately keeps the observed-period and shadow
        # statuses alongside their conservative effective status.  Do not
        # treat an observed-period acceptance as a production acceptance when
        # the shadow period (or publication policy) has already downgraded it.
        effective_status = str(
            entry.get("effective_status")
            or entry.get("global_status")
            or observed_acceptance.get("status")
            or "experimental"
        ).casefold()
        if effective_status == "accepted":
            development_checks = (
                (entry.get("official_selection") or {}).get("development_checks")
                or observed_acceptance.get("official_development_checks")
                or {}
            )
            shadow_status = str(
                (entry.get("shadow_acceptance") or {}).get("status")
                or entry.get("shadow_evidence_status")
                or ""
            ).casefold()
            acceptance_ok = bool(
                str(observed_acceptance.get("status", "")).casefold() == "accepted"
                and shadow_status == "accepted"
                and development_checks
                and all(bool(value) for value in development_checks.values())
            )
            checks.append(
                _check(
                    f"accepted_horizon_provenance_{horizon}d",
                    acceptance_ok,
                    {
                        "benchmark_passed": entry.get("benchmark_passed"),
                        "shadow_evidence_status": shadow_status,
                        "official_development_checks": development_checks,
                    },
                )
            )

    sentiment_hashes = sentiment_artifact_hashes()
    finbert_model_key = str((PROJECT_ROOT / "models" / "experimental" / "finbert-sentiment" / "model.safetensors").relative_to(PROJECT_ROOT))
    checks.append(_check("finbert_artifact", finbert_model_key in sentiment_hashes, sentiment_hashes))
    runtime_config_path = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
    try:
        runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
        configured_hashes = runtime_config.get("artifact_hashes", {})
        runtime_hashes_ok = sentiment_integrity()["ok"]
    except (OSError, ValueError, TypeError):
        runtime_hashes_ok = False
        configured_hashes = {}
    checks.append(_check("finbert_runtime_hashes", runtime_hashes_ok, {"configured": configured_hashes, "actual": sentiment_hashes}))
    checks.append(_check("stock_metadata", (PROJECT_ROOT / "data" / "stock-metadata.csv").exists()))

    raw_files = list((PROJECT_ROOT / "data" / "raw-stock-dataset").glob("*.csv"))
    raw_coverage = {str(path.name): _date_coverage(path) for path in raw_files}
    valid_raw_coverage = [
        item
        for item in raw_coverage.values()
        if item.get("exists") and not item.get("error")
    ]
    raw_period_ok = len(valid_raw_coverage) >= 10 and all(
        str(item.get("start", "9999"))[:4] <= "2015"
        and str(item.get("end", "0000"))[:4] >= "2025"
        for item in valid_raw_coverage
    )
    checks.append(_check("raw_stock_coverage_2015_2025", raw_period_ok, raw_coverage))
    klci = PROJECT_ROOT / "data" / "raw-market-context-dataset" / "ftse-bursa-malaysia-klci-raw.csv"
    klci_coverage = _date_coverage(klci)
    klci_period_ok = bool(
        klci_coverage.get("exists")
        and str(klci_coverage.get("start", "9999"))[:4] <= "2015"
        and str(klci_coverage.get("end", "0000"))[:4] >= "2025"
    )
    checks.append(_check("klci_coverage_2015_2025", klci_period_ok, klci_coverage))

    news_evidence = check_news_evidence_readiness()
    checks.append(
        _check(
            "malaysian_news_evidence",
            news_evidence.ready,
            news_evidence.to_dict(),
            critical=False,
        )
    )
    fusion_evidence = check_fusion_dataset_readiness(DEFAULT_FUSION_DATASET)
    checks.append(
        _check(
            "historical_fusion_evidence",
            fusion_evidence.ready,
            fusion_evidence.to_dict(),
            critical=False,
        )
    )
    try:
        correction_ledger = dict(correction_audit_summary())
        correction_ledger_ok = bool(correction_ledger.get("available"))
    except Exception as exc:
        correction_ledger = {"available": False, "error": str(exc)}
        correction_ledger_ok = False
    checks.append(
        _check(
            "news_correction_audit_ledger",
            correction_ledger_ok,
            correction_ledger,
        )
    )

    if level == "full":
        python = sys.executable
        tests = _run_command([python, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], PROJECT_ROOT, timeout=1200)
        backend_tests = _run_command([python, "-m", "unittest", "discover", "-s", "backend/tests", "-p", "test_*.py"], PROJECT_ROOT, timeout=1200)
        checks.append(_check("python_tests", tests["returncode"] == 0 and backend_tests["returncode"] == 0, {"pipeline": tests, "backend": backend_tests}))
        if (PROJECT_ROOT / "frontend" / "node_modules").exists():
            bundled_pnpm = Path(
                r"C:\Users\nicho\.cache\codex-runtimes\codex-primary-runtime"
                r"\dependencies\bin\fallback\pnpm.cmd"
            )
            pnpm = os.environ.get("PNPM_HOME", "")
            pnpm_command = (
                str(Path(pnpm) / "pnpm.cmd")
                if pnpm and (Path(pnpm) / "pnpm.cmd").exists()
                else shutil.which("pnpm.cmd") or str(bundled_pnpm)
            )
            frontend_tests = _run_command([pnpm_command, "test", "--", "--run"], PROJECT_ROOT / "frontend", timeout=900)
            checks.append(_check("frontend_tests", frontend_tests["returncode"] == 0, frontend_tests))
            build = _run_command([pnpm_command, "build"], PROJECT_ROOT / "frontend", timeout=900)
            checks.append(_check("frontend_build", build["returncode"] == 0, build))
        else:
            checks.append(_check("frontend_dependencies", False, "frontend/node_modules is missing"))
    if live_providers:
        checks.append(_check("live_provider_checks", False, "Run provider smoke checks through the explicit provider benchmark command.", critical=False))

    manifest = {
        "audit_run_id": run_id,
        "level": level,
        "live_providers": live_providers,
        "registry_signature": registry_signature(),
        "checks": checks,
        "critical_failures": [item["name"] for item in checks if item["critical"] and not item["passed"]],
        "warnings": [item["name"] for item in checks if not item["critical"] and not item["passed"]],
    }
    (output_dir / "audit.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    lines = [f"# StockSense system audit {run_id}", "", f"- Level: `{level}`", f"- Registry: `{manifest['registry_signature']}`", "", "| Check | Status |", "|---|---|"]
    lines.extend(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} |" for item in checks)
    (output_dir / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit the StockSense runtime")
    parser.add_argument("--level", choices=("quick", "full"), default="full")
    parser.add_argument("--live-providers", action="store_true")
    args = parser.parse_args()
    result = run_system_audit(args.level, args.live_providers)
    print(json.dumps(result, indent=2, default=str))
    return 1 if result["critical_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
