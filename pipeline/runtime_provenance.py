"""Shared, lightweight runtime provenance and artifact-integrity helpers."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = PROJECT_ROOT / "models" / "model-registry.json"
SENTIMENT_RUNTIME_CONFIG_FILE = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
FINBERT_MODEL_DIR = PROJECT_ROOT / "models" / "experimental" / "finbert-sentiment"
FINBERT_ARTIFACT_FILENAMES = (
    "config.json",
    "metadata.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
)


@lru_cache(maxsize=128)
def _sha256_for_file_state(
    path_text: str,
    size: int,
    modified_ns: int,
) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file once per stable size/mtime state within this process."""
    resolved = path.resolve()
    stat = resolved.stat()
    return _sha256_for_file_state(str(resolved), stat.st_size, stat.st_mtime_ns)


def registry_payload() -> dict[str, Any]:
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def registry_signature() -> str:
    try:
        return sha256_file(REGISTRY_FILE)
    except OSError:
        return "missing"


def stock_artifact_hashes(registry: dict[str, Any] | None = None) -> dict[str, str]:
    registry = registry or registry_payload()
    hashes: dict[str, str] = {}
    for horizon, entry in registry.get("horizons", {}).items():
        path = PROJECT_ROOT / str(entry.get("artifact_path", ""))
        if path.exists():
            hashes[str(horizon)] = sha256_file(path)
    return hashes


def sentiment_artifact_hashes_for_directory(model_dir: str | Path) -> dict[str, str]:
    """Return deployment-relative hashes for one local FinBERT directory."""
    directory = Path(model_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in (directory / name for name in FINBERT_ARTIFACT_FILENAMES)
        if path.exists()
    }


def sentiment_artifact_hashes() -> dict[str, str]:
    """Hashes for the unchanged PhraseBank production artifact."""
    return sentiment_artifact_hashes_for_directory(FINBERT_MODEL_DIR)


def runtime_manifest() -> dict[str, Any]:
    registry = registry_payload()
    return {
        "registry_signature": registry_signature(),
        "stock_artifact_hashes": stock_artifact_hashes(registry),
        "sentiment_artifact_hashes": sentiment_artifact_hashes(),
    }
