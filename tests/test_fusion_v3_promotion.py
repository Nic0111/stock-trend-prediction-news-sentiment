from __future__ import annotations

import json

import pytest

from pipeline.fusion_v3_promotion import promote_fusion_v3_registry


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v3_promotion_rejects_registry_without_accepted_horizon(tmp_path) -> None:
    registry = tmp_path / "registry.json"
    reproducibility = tmp_path / "reproducibility.json"
    state = tmp_path / "state.json"
    parity = tmp_path / "parity.json"
    _write(
        registry,
        {
            "schema_version": "fusion-registry-v5",
            "accepted_horizons": [],
            "dataset_sha256": "dataset",
        },
    )
    _write(reproducibility, {"passed": True})
    _write(state, {"dataset_sha256": "dataset"})
    _write(parity, {"passed": True})
    with pytest.raises(ValueError, match="No horizon passed"):
        promote_fusion_v3_registry(
            registry,
            reproducibility,
            state,
            parity,
            active_registry_path=tmp_path / "active.json",
        )

def test_v3_promotion_rejects_missing_runtime_parity(tmp_path) -> None:
    registry = tmp_path / "registry.json"
    reproducibility = tmp_path / "reproducibility.json"
    state = tmp_path / "state.json"
    parity = tmp_path / "parity.json"
    _write(
        registry,
        {
            "schema_version": "fusion-registry-v5",
            "accepted_horizons": [1],
            "dataset_sha256": "dataset",
        },
    )
    _write(reproducibility, {"passed": True})
    _write(state, {"dataset_sha256": "dataset"})
    _write(parity, {"passed": False})
    with pytest.raises(ValueError, match="feature parity"):
        promote_fusion_v3_registry(
            registry,
            reproducibility,
            state,
            parity,
            active_registry_path=tmp_path / "active.json",
        )
