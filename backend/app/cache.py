from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    expires_at: float
    value: T


class MemoryTTLCache(Generic[T]):
    """Small process-local cache for API response reuse and analysis grounding."""

    def __init__(self) -> None:
        self._values: dict[str, _Entry[T]] = {}
        self._lock = RLock()

    def get(self, key: str) -> Optional[T]:
        now = monotonic()
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._values.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: float) -> None:
        with self._lock:
            self._values[key] = _Entry(monotonic() + ttl_seconds, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)

