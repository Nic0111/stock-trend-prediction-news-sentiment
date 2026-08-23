# StockSense performance checks

## API benchmark

Start StockSense, then run:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_performance.py `
  --output results\performance\latest.json
```

The benchmark records repeated readiness, snapshot, prediction, analysis and
five-company comparison timings together with bounded-cache statistics from
`/api/v1/system/performance`.

API responses expose:

- `Server-Timing` for total server duration.
- `X-Cache` on snapshot, prediction and analysis requests.

In a development browser, `window.__stocksensePerformance` contains LCP,
long-task and overview seek diagnostics.

## Overview media

The overview now assigns the MP4 URL directly to each video element. This is
intentional: converting a clip to a Blob first prevents useful byte-range
streaming and forces the complete 1080p60 file into browser memory.

Only the active scene and its next/transition scene are attached at once.

To stage seek-optimized 1080p60 copies after installing FFmpeg:

```powershell
.\scripts\Optimize-OverviewVideos.ps1
```

The script uses high-quality H.264, 60 fps, a 30-frame GOP and `faststart`.
It has a per-file watchdog, skips completed outputs and writes a manifest.
It never replaces the current media automatically; visually inspect the
staged directory before changing the media paths.
