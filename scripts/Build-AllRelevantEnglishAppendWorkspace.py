"""Stage all reviewed English Bursa articles for a second append cycle.

The balance-aware cycle already appended 93 of the 218 English relevant rows.
This creates a fresh baseline-checked workspace containing all 218 so the
append implementation can deduplicate the existing 93 and add only the 125
new rows.  It does not modify the canonical dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "news-evidence-benchmark" / "bursa-news-llm-assisted-label.jsonl"
DEFAULT_REVIEWED_STAGE = (
    ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "bursa-news-balance-v2-combined-codex-commit-v1"
    / "staging"
    / "cycle-001"
    / "codex-reviewed.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "bursa-news-all-relevant-english-codex-append-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reviewed-stage", type=Path, default=DEFAULT_REVIEWED_STAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    source = args.reviewed_stage.resolve()
    output = args.output_dir.resolve()
    if not dataset.exists() or not source.exists():
        raise SystemExit(f"Missing dataset or reviewed stage: {dataset} / {source}")
    if (output / "append-commit-report.json").exists():
        raise SystemExit(f"Refusing to reuse an already committed workspace: {output}")

    rows = _read_jsonl(source)
    selected = [
        row
        for row in rows
        if str(row.get("entity_relevant") or "").strip().casefold() in {"true", "1", "yes"}
        and str(row.get("language") or "").strip().casefold() == "en"
    ]
    if len(rows) != 381 or len(selected) != 218:
        raise SystemExit(f"Expected 381 reviewed rows and 218 English relevant rows; found {len(rows)} / {len(selected)}")
    ids = [str(row.get("article_id") or "") for row in selected]
    if not all(ids) or len(ids) != len(set(ids)):
        raise SystemExit("The selected reviewed rows contain missing or duplicate article IDs")

    dataset_hash = _sha256(dataset)
    manifest = {
        "schema_version": "bursa-news-append-v1",
        "status": "prepared",
        "selection_mode": "all_relevant_english_reviewed",
        "cycle_index": 1,
        "base_dataset": {
            "dataset_path": str(dataset),
            "dataset_sha256": dataset_hash,
            "dataset_rows": sum(1 for _ in dataset.open("r", encoding="utf-8")),
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        },
        "years": list(range(2015, 2026)),
        "selected_reviewed_rows": len(selected),
        "source_reviewed_stage": str(source),
        "production_changes": False,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    stage = output / "staging" / "codex-reviewed.jsonl"
    _write_jsonl(stage, selected)
    manifest["staging_path"] = str(stage)
    manifest["staging_sha256"] = _sha256(stage)
    _write_json(output / "append-manifest.json", manifest)
    _write_json(
        output / "append-provenance.json",
        {
            "source_reviewed_stage": str(source),
            "source_reviewed_stage_sha256": _sha256(source),
            "source_rows": len(rows),
            "selected_english_relevant_rows": len(selected),
            "excluded_non_english_relevant_rows": len(rows) - len(selected),
            "canonical_dataset": str(dataset),
            "canonical_dataset_sha256": dataset_hash,
            "staging_path": str(stage),
            "staging_sha256": _sha256(stage),
            "production_changed": False,
        },
    )
    print(json.dumps({"status": "prepared", "work_dir": str(output), "staging": str(stage), "rows": len(selected), "production_changed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
