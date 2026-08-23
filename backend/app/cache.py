from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Event, RLock
from time import monotonic
from typing import Callable, Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    expires_at: float
    value: T


@dataclass(frozen=True)
class CacheStats:
    size: int
    max_entries: int
    hits: int
    misses: int
    evictions: int
    in_flight: int


class MemoryTTLCache(Generic[T]):
    """Bounded process-local LRU/TTL cache with duplicate-work coalescing."""

    def __init__(self, max_entries: int = 128) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._values: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._in_flight: dict[str, Event] = {}
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, entry in self._values.items() if entry.expires_at <= now]
        for key in expired:
            self._values.pop(key, None)

    def get(self, key: str) -> Optional[T]:
        now = monotonic()
        with self._lock:
            self._purge_expired(now)
            entry = self._values.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._values.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            self.delete(key)
            return
        with self._lock:
            self._purge_expired(monotonic())
            self._values[key] = _Entry(monotonic() + ttl_seconds, value)
            self._values.move_to_end(key)
            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)
                self._evictions += 1

    def delete(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def get_or_compute(self, key: str, ttl_seconds: float, compute: Callable[[], T]) -> tuple[T, bool]:
        """Return a cached value or compute it once for all concurrent callers.

        The boolean is ``True`` for a cache hit (including a value produced by
        another caller while this caller waited) and ``False`` for the caller
        that performed the computation.
        """

        cached = self.get(key)
        if cached is not None:
            return cached, True

        with self._lock:
            event = self._in_flight.get(key)
            if event is None:
                event = Event()
                self._in_flight[key] = event
                owner = True
            else:
                owner = False

        if not owner:
            event.wait()
            cached = self.get(key)
            if cached is not None:
                return cached, True
            # The owner failed. Retry as a new owner so errors are not cached.
            return self.get_or_compute(key, ttl_seconds, compute)

        try:
            value = compute()
            self.set(key, value, ttl_seconds)
            return value, False
        finally:
            with self._lock:
                self._in_flight.pop(key, None)
                event.set()

    def stats(self) -> CacheStats:
        with self._lock:
            self._purge_expired(monotonic())
            return CacheStats(
                size=len(self._values),
                max_entries=self._max_entries,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                in_flight=len(self._in_flight),
            )
