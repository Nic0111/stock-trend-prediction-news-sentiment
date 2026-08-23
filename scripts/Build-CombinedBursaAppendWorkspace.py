"""Build one auditable append workspace from two independently staged label batches.

The first balanced cycle and its later unrecovered retry were both prepared
against the same immutable Bursa-news master hash.  They must therefore be
merged *before* committing, otherwise the second manifest becomes stale after
the first append.  This helper only creates a derived workspace; it never
changes the canonical dataset or any model artifact.
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
DEFAULT_ORIGINAL = ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2"
DEFAULT_RETRY = ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2-retry-unrecovered-v1"
DEFAULT_OUTPUT = ROOT / "data" / "historical-news-fusion" / "gdelt" / "bursa-news-balance-v2-combined-codex-commit-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--original-work-dir", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--retry-work-dir", type=Path, default=DEFAULT_RETRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    original = args.original_work_dir.resolve()
    retry = args.retry_work_dir.resolve()
    output = args.output_dir.resolve()
    manifest_path = original / "append-manifest.json"
    original_stage = original / "staging" / "cycle-001" / "codex-reviewed.jsonl"
    retry_stage = retry / "staging" / "codex-reviewed.jsonl"
    for path in (manifest_path, original_stage, retry_stage):
        if not path.exists():
            raise SystemExit(f"Required source is missing: {path}")

    manifest = _read_json(manifest_path)
    base = dict(manifest.get("base_dataset") or {})
    canonical = Path(str(base.get("dataset_path") or "")).resolve()
    expected_hash = str(base.get("dataset_sha256") or "")
    if not canonical.exists() or not expected_hash:
        raise SystemExit("The original manifest has no usable canonical dataset identity")
    actual_hash = _sha256(canonical)
    if actual_hash != expected_hash:
        raise SystemExit("Canonical dataset hash no longer matches the frozen append baseline")

    original_rows = _read_jsonl(original_stage)
    retry_rows = _read_jsonl(retry_stage)
    if len(original_rows) != 379 or len(retry_rows) != 2:
        raise SystemExit(
            f"Unexpected staged counts: original={len(original_rows)}, retry={len(retry_rows)}; expected 379 + 2"
        )
    rows = original_rows + retry_rows
    ids = [str(row.get("article_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise SystemExit("The staged batches contain missing or duplicate article IDs")

    combined_manifest = dict(manifest)
    combined_manifest["label_package_dir"] = str(output / "codex-label-package" / "cycle-001")
    combined_manifest["combined_append_provenance"] = {
        "purpose": "atomic combination of original balanced labels and unrecovered retry labels",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_workspaces": [str(original), str(retry)],
        "source_staging_files": [str(original_stage), str(retry_stage)],
        "source_staging_sha256": [_sha256(original_stage), _sha256(retry_stage)],
        "source_row_counts": [len(original_rows), len(retry_rows)],
        "combined_row_count": len(rows),
        "production_changed_by_build": False,
    }
    stage = output / "staging" / "cycle-001" / "codex-reviewed.jsonl"
    _write_jsonl(stage, rows)
    _write_json(output / "append-manifest.json", combined_manifest)
    _write_json(
        output / "combined-provenance.json",
        {
            **combined_manifest["combined_append_provenance"],
            "canonical_dataset": str(canonical),
            "canonical_dataset_sha256": actual_hash,
            "combined_staging_file": str(stage),
            "combined_staging_sha256": _sha256(stage),
            "selection_mode": combined_manifest.get("selection_mode"),
        },
    )
    print(
        json.dumps(
            {
                "status": "combined_workspace_ready",
                "output_dir": str(output),
                "staging": str(stage),
                "rows": len(rows),
                "canonical_hash_verified": actual_hash,
                "production_changed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
