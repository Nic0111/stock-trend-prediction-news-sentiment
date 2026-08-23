from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

from backend.app.cache import MemoryTTLCache


def test_cache_evicts_least_recently_used_entry() -> None:
    cache: MemoryTTLCache[int] = MemoryTTLCache(max_entries=2)
    cache.set("a", 1, 60)
    cache.set("b", 2, 60)
    assert cache.get("a") == 1

    cache.set("c", 3, 60)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.stats().evictions == 1


def test_cache_drops_expired_entries() -> None:
    cache: MemoryTTLCache[int] = MemoryTTLCache(max_entries=2)
    cache.set("short", 1, 0.01)
    sleep(0.02)

    assert cache.get("short") is None
    assert cache.stats().size == 0


def test_get_or_compute_coalesces_concurrent_work() -> None:
    cache: MemoryTTLCache[int] = MemoryTTLCache(max_entries=4)
    calls = 0
    lock = Lock()

    def compute() -> int:
        nonlocal calls
        with lock:
            calls += 1
        sleep(0.03)
        return 42

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda _: cache.get_or_compute("answer", 60, compute), range(5)))

    assert calls == 1
    assert [value for value, _ in results] == [42] * 5
    assert sum(1 for _, hit in results if not hit) == 1
