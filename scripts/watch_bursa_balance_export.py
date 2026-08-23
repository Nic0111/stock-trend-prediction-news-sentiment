"""Wait for a bounded Bursa recovery cycle, then prepare its label package.

This watcher is deliberately offline-only: it runs triage and export-labels,
but never imports labels, commits the canonical dataset, or updates a model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"C:\stock-investment-analytics")
WORK = ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2"
LOG = WORK / "post-recovery-export.log"
ERR = WORK / "post-recovery-export.err.log"
MAX_WAIT_SECONDS = 12 * 60 * 60


def write_log(handle, message: str) -> None:
    handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
    handle.flush()


def expected_years() -> list[int]:
    years: list[int] = []
    root = WORK / "candidates"
    for path in sorted(root.glob("*/candidates.jsonl")):
        try:
            year = int(path.parent.name)
        except ValueError:
            continue
        if path.stat().st_size > 0:
            years.append(year)
    return years


def completed_years() -> set[int]:
    done: set[int] = set()
    for year in expected_years():
        report = WORK / "recovery-root" / str(year) / "recovery" / f"pilot-{year}-ngram-usability-report.json"
        if not report.exists():
            continue
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if bool(payload.get("completed")):
            done.add(year)
    return done


def run_cli(python: Path, args: list[str], out, err) -> int:
    command = [str(python), "-m", "pipeline.fusion_cli", *args]
    write_log(out, "running: " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=False)
    write_log(out, f"exit={result.returncode}")
    return int(result.returncode)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    started = time.monotonic()
    with LOG.open("a", encoding="utf-8") as out, ERR.open("a", encoding="utf-8") as err:
        years = expected_years()
        write_log(out, f"watching years={years}")
        while time.monotonic() - started < MAX_WAIT_SECONDS:
            done = completed_years()
            write_log(out, f"completed_years={sorted(done)}")
            if years and set(years).issubset(done):
                break
            time.sleep(60)
        else:
            write_log(out, "watch timeout; no label export was started")
            return 2

        triage_rc = run_cli(
            python,
            [
                "bursa-news-append",
                "triage",
                "--work-dir",
                str(WORK),
                "--random-control-share",
                "0.10",
            ],
            out,
            err,
        )
        if triage_rc != 0:
            write_log(out, "triage failed; export was not started")
            return triage_rc

        return run_cli(
            python,
            [
                "bursa-news-append",
                "export-labels",
                "--work-dir",
                str(WORK),
                "--batch-size",
                "25",
            ],
            out,
            err,
        )


if __name__ == "__main__":
    raise SystemExit(main())
