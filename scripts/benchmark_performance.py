from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _request_samples(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    repeats: int,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples: list[float] = []
    cache_states: list[str] = []
    statuses: list[int] = []
    for _ in range(repeats):
        started = time.perf_counter()
        response = client.request(method, path, json=json_body)
        samples.append((time.perf_counter() - started) * 1000)
        statuses.append(response.status_code)
        cache_states.append(response.headers.get("X-Cache", "UNSET"))
        response.raise_for_status()
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return {
        "path": path,
        "method": method,
        "samples_ms": [round(value, 2) for value in samples],
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(ordered[p95_index], 2),
        "statuses": statuses,
        "cache_states": cache_states,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure StockSense cold/warm API performance.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        client.get("/api/v1/health/live").raise_for_status()
        results = {
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "base_url": args.base_url,
            "repeats": args.repeats,
            "measurements": [
                _request_samples(client, "GET", "/api/v1/health/ready", repeats=args.repeats),
                _request_samples(client, "GET", "/api/v1/market-snapshots/1155.KL", repeats=args.repeats),
                _request_samples(client, "GET", "/api/v1/predictions/1155.KL?horizon=5", repeats=args.repeats),
                _request_samples(
                    client,
                    "POST",
                    "/api/v1/analyses",
                    repeats=args.repeats,
                    json_body={"ticker": "1155.KL", "horizon": 5},
                ),
                _request_samples(
                    client,
                    "POST",
                    "/api/v1/comparisons",
                    repeats=args.repeats,
                    json_body={
                        "tickers": ["1155.KL", "1295.KL", "1023.KL", "5347.KL", "5225.KL"],
                        "horizon": 5,
                    },
                ),
            ],
            "cache_diagnostics": client.get("/api/v1/system/performance").json(),
        }

    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
