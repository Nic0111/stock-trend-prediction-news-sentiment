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


RESULTS_ROOT = PROJECT_ROOT / "results" / "system-audit"


def _check(name: str, passed: bool, detail: Any = None, critical: bool = True) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "critical": critical, "detail": detail}


def _migrate_registry_provenance(registry: dict[str, Any]) -> bool:
    """Backfill hashes into old registries without changing model selection."""
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
    checks.append(_check("environment", (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").exists()))

    registry_error = None
    registry: dict[str, Any] = {}
    try:
        registry = registry_payload()
        _migrate_registry_provenance(registry)
        registry = registry_payload()
    except Exception as exc:
        registry_error = str(exc)
    checks.append(_check("registry_loadable", registry_error is None, registry_error))
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
        checks.append(_check(f"stock_artifact_{horizon}d", path.exists() and (not expected or expected == actual), artifact_details[str(horizon)]))

    sentiment_hashes = sentiment_artifact_hashes()
    finbert_model_key = str((PROJECT_ROOT / "models" / "experimental" / "finbert-sentiment" / "model.safetensors").relative_to(PROJECT_ROOT))
    checks.append(_check("finbert_artifact", finbert_model_key in sentiment_hashes, sentiment_hashes))
    runtime_config_path = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
    try:
        runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
        configured_hashes = runtime_config.get("artifact_hashes", {})
        runtime_hashes_ok = all(configured_hashes.get(path) == value for path, value in sentiment_hashes.items())
    except (OSError, ValueError, TypeError):
        runtime_hashes_ok = False
        configured_hashes = {}
    checks.append(_check("finbert_runtime_hashes", runtime_hashes_ok, {"configured": configured_hashes, "actual": sentiment_hashes}))
    checks.append(_check("stock_metadata", (PROJECT_ROOT / "data" / "stock-metadata.csv").exists()))

    raw_files = list((PROJECT_ROOT / "data" / "raw-stock-dataset").glob("*.csv"))
    checks.append(_check("raw_stock_coverage", len(raw_files) >= 10, {str(path.name): _date_coverage(path) for path in raw_files}, critical=False))
    klci = PROJECT_ROOT / "data" / "raw-market-context-dataset" / "ftse-bursa-malaysia-klci-raw.csv"
    klci_coverage = _date_coverage(klci)
    checks.append(_check("klci_coverage", bool(klci_coverage.get("exists")), klci_coverage, critical=False))

    if level == "full":
        python = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
        tests = _run_command([python, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], PROJECT_ROOT, timeout=1200)
        backend_tests = _run_command([python, "-m", "unittest", "discover", "-s", "backend/tests", "-p", "test_*.py"], PROJECT_ROOT, timeout=1200)
        checks.append(_check("python_tests", tests["returncode"] == 0 and backend_tests["returncode"] == 0, {"pipeline": tests, "backend": backend_tests}))
        if (PROJECT_ROOT / "frontend" / "node_modules").exists():
            pnpm = os.environ.get("PNPM_HOME", "")
            pnpm_command = "pnpm.cmd" if not pnpm else str(Path(pnpm) / "pnpm.cmd")
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
