"""Detached runner for the resumable offline Bursa news recovery cycle."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(r"C:\stock-investment-analytics")
WORK = ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2"
LOG = WORK / "recovery-worker-detached.log"
ERR = WORK / "recovery-worker-detached.err.log"

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
WORK.mkdir(parents=True, exist_ok=True)

def expected_years() -> list[int]:
    years: list[int] = []
    for path in sorted((WORK / "candidates").glob("*/candidates.jsonl")):
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
        path = WORK / "recovery-root" / str(year) / "recovery" / f"pilot-{year}-ngram-usability-report.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("completed") is True:
            done.add(year)
    return done


def main() -> int:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    command = [
        str(python),
        "-u",
        "-m",
        "pipeline.fusion_cli",
        "bursa-news-append",
        "recover",
        "--work-dir",
        str(WORK),
        "--max-runtime-minutes",
        "45",
        "--max-download-gib",
        "5",
        "--request-interval-seconds",
        "1",
        "--skip-archives",
    ]
    years = expected_years()
    with LOG.open("a", encoding="utf-8") as stdout, ERR.open("a", encoding="utf-8") as stderr:
        for cycle in range(1, 21):
            done = completed_years()
            stdout.write(f"cycle={cycle} completed_years={sorted(done)} expected={years}\n")
            stdout.flush()
            if years and set(years).issubset(done):
                return 0
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            stdout.write(f"cycle={cycle} exit={result.returncode}\n")
            stdout.flush()
            time.sleep(5)
        stdout.write("maximum recovery cycles reached; checkpoints remain resumable\n")
        stdout.flush()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
