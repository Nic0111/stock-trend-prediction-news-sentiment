"""Retry only unavailable Bursa recovery rows and merge the label package.

This is an offline-workspace helper. It never changes the canonical Bursa
dataset or model artifacts. The retry workspace is built from the completed
balance-v2 recovery outputs, so already recovered rows are not requested
again. The final merged review pack is written separately to preserve the
hashes and provenance of the original package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(r"C:\stock-investment-analytics")
SOURCE_WORK = ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2"
WORK = ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2-retry-unrecovered-v1"
YEARS = list(range(2015, 2022)) + [2024, 2025]
LOG = WORK / "retry-worker.log"
ERR = WORK / "retry-worker.err.log"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return len(records)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def is_unrecovered(row: dict[str, Any]) -> bool:
    status = str(row.get("recovery_status") or "").strip().casefold()
    source = str(row.get("text_source") or "").strip().casefold()
    return status != "recovered" or source not in {"publisher", "wayback", "common_crawl", "gdelt_webngrams"}


def prepare_workspace() -> dict[str, Any]:
    WORK.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((SOURCE_WORK / "append-manifest.json").read_text(encoding="utf-8"))
    base_dataset = dict(source_manifest["base_dataset"])
    candidate_files: dict[str, dict[str, Any]] = {}
    total = 0
    by_year: dict[str, int] = {}
    for year in YEARS:
        source = SOURCE_WORK / "recovery-root" / str(year) / "recovery" / "raw" / "hybrid-recovered.jsonl"
        retry_rows = [row for row in read_jsonl(source) if is_unrecovered(row)]
        destination = WORK / "candidates" / str(year) / "candidates.jsonl"
        count = write_jsonl(destination, retry_rows)
        by_year[str(year)] = count
        if count:
            candidate_files[str(year)] = {
                "path": str(destination),
                "rows": count,
                "sha256": sha256_file(destination),
            }
            total += count
    manifest = {
        "schema_version": "bursa-news-append-v1",
        "status": "prepared",
        "selection_mode": "unrecovered_retry",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_dataset": base_dataset,
        "candidate_root": str(SOURCE_WORK / "recovery-root"),
        "years": YEARS,
        "recovery_root_dir": str(WORK / "recovery-root"),
        "candidate_files": candidate_files,
        "selected_rows": total,
        "rows_by_year": by_year,
        "production_changes": False,
    }
    write_json(WORK / "append-manifest.json", manifest)
    write_json(WORK / "retry-selection-summary.json", {
        "source_work_dir": str(SOURCE_WORK),
        "years": YEARS,
        "rows_by_year": by_year,
        "selected_rows": total,
        "selection_rule": "recovery_status != recovered or unsupported/empty text_source",
        "production_changes": False,
    })
    return manifest


def completed_years(manifest: dict[str, Any]) -> set[int]:
    done: set[int] = set()
    for year_text in manifest.get("candidate_files", {}):
        year = int(year_text)
        report = WORK / "recovery-root" / str(year) / "recovery" / f"pilot-{year}-ngram-usability-report.json"
        if report.exists():
            try:
                if json.loads(report.read_text(encoding="utf-8")).get("completed") is True:
                    done.add(year)
            except (OSError, json.JSONDecodeError):
                pass
    return done


def run_cli(python: Path, args: list[str], stdout, stderr) -> int:
    command = [str(python), "-m", "pipeline.fusion_cli", *args]
    stdout.write("running: " + " ".join(command) + "\n")
    stdout.flush()
    result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    stdout.write(f"exit={result.returncode}\n")
    stdout.flush()
    return int(result.returncode)


def merge_review_packs(retry_pack: Path) -> Path:
    previous = SOURCE_WORK / "codex-label-package" / "cycle-001" / "review-pack.csv"
    retry_only = retry_pack / "review-pack.csv"
    merged = retry_pack / "review-pack-appended.csv"
    with previous.open("r", encoding="utf-8-sig", newline="") as handle:
        previous_rows = list(csv.DictReader(handle))
        previous_fields = list(previous_rows[0].keys()) if previous_rows else []
    with retry_only.open("r", encoding="utf-8-sig", newline="") as handle:
        retry_rows = list(csv.DictReader(handle))
        retry_fields = list(retry_rows[0].keys()) if retry_rows else []
    fields = previous_fields or retry_fields
    for field in retry_fields:
        if field not in fields:
            fields.append(field)
    merged_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in previous_rows + retry_rows:
        key = str(row.get("article_id") or row.get("source_url") or "").strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged_rows.append({field: row.get(field, "") for field in fields})
    with merged.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged_rows)
    write_json(retry_pack / "merge-summary.json", {
        "previous_review_pack": str(previous),
        "retry_review_pack": str(retry_only),
        "merged_review_pack": str(merged),
        "previous_rows": len(previous_rows),
        "retry_rows": len(retry_rows),
        "merged_rows": len(merged_rows),
        "deduplication_key": "article_id, fallback source_url",
        "production_changes": False,
    })
    return merged


def main() -> int:
    manifest = prepare_workspace()
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    with LOG.open("a", encoding="utf-8") as stdout, ERR.open("a", encoding="utf-8") as stderr:
        stdout.write(f"prepared rows={manifest['selected_rows']} by_year={manifest['rows_by_year']}\n")
        stdout.flush()
        years = set(int(value) for value in manifest["candidate_files"])
        for cycle in range(1, 21):
            done = completed_years(manifest)
            stdout.write(f"recovery_cycle={cycle} completed={sorted(done)} expected={sorted(years)}\n")
            stdout.flush()
            if years.issubset(done):
                break
            rc = run_cli(
                python,
                [
                    "bursa-news-append", "recover", "--work-dir", str(WORK),
                    "--max-runtime-minutes", "45", "--max-download-gib", "5",
                    "--request-interval-seconds", "1",
                ],
                stdout,
                stderr,
            )
            if rc != 0:
                stdout.write(f"recovery cycle returned {rc}; checkpoints remain resumable\n")
                stdout.flush()
            time.sleep(5)
        else:
            stdout.write("maximum retry cycles reached; no export started\n")
            stdout.flush()
            return 2
        triage_rc = run_cli(
            python,
            ["bursa-news-append", "triage", "--work-dir", str(WORK), "--random-control-share", "0.10"],
            stdout,
            stderr,
        )
        if triage_rc != 0:
            return triage_rc
        export_rc = run_cli(
            python,
            ["bursa-news-append", "export-labels", "--work-dir", str(WORK), "--batch-size", "25"],
            stdout,
            stderr,
        )
        if export_rc != 0:
            return export_rc
        retry_pack = WORK / "codex-label-package"
        merged = merge_review_packs(retry_pack)
        stdout.write(f"merged_review_pack={merged}\n")
        stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
