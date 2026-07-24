"""Shared, lightweight runtime provenance and artifact-integrity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = PROJECT_ROOT / "models" / "model-registry.json"
SENTIMENT_RUNTIME_CONFIG_FILE = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
FINBERT_MODEL_DIR = PROJECT_ROOT / "models" / "experimental" / "finbert-sentiment"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def sentiment_artifact_hashes() -> dict[str, str]:
    paths = [
        FINBERT_MODEL_DIR / "config.json",
        FINBERT_MODEL_DIR / "metadata.json",
        FINBERT_MODEL_DIR / "tokenizer.json",
        FINBERT_MODEL_DIR / "tokenizer_config.json",
        FINBERT_MODEL_DIR / "vocab.txt",
        FINBERT_MODEL_DIR / "model.safetensors",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in paths
        if path.exists()
    }


def runtime_manifest() -> dict[str, Any]:
    registry = registry_payload()
    return {
        "registry_signature": registry_signature(),
        "stock_artifact_hashes": stock_artifact_hashes(registry),
        "sentiment_artifact_hashes": sentiment_artifact_hashes(),
    }
