from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib

from pipeline.predict_stock_trend import _load_model_cached, _production_model_loader


def test_production_model_loader_reuses_unchanged_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        joblib.dump({"version": 1}, path)
        _load_model_cached.cache_clear()

        first = _production_model_loader(path)
        second = _production_model_loader(path)

        assert first is second


def test_production_model_loader_invalidates_changed_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        joblib.dump({"version": 1}, path)
        _load_model_cached.cache_clear()
        first = _production_model_loader(path)

        joblib.dump({"version": 2, "larger": True}, path)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        second = _production_model_loader(path)

        assert first is not second
        assert second["version"] == 2
