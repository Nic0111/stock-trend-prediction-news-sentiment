"""Merge the corrected automatic 1295.KL extension without changing base rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/fusion/fusion-window-candidates-v4-expanded-v13-auto-rescored.parquet"
TARGET = ROOT / "data/fusion/fusion-window-candidates-v4-expanded-v14-auto-corrected-1295-only.parquet"
OUT = ROOT / "data/fusion/fusion-window-candidates-v4-expanded-v14-auto-corrected.parquet"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = pd.read_parquet(BASE)
    target = pd.read_parquet(TARGET)
    ticker = target["Ticker"].astype(str).str.strip()
    target = target.loc[ticker.eq("1295.KL")].copy()
    base_other = base.loc[~base["Ticker"].astype(str).str.strip().eq("1295.KL")].copy()
    merged = pd.concat([base_other, target], ignore_index=True)
    # Keep the original v13 rows for the nine unaffected companies verbatim.
    # The targeted extension is already deduplicated against its base corpus.
    merged = merged.sort_values(["Ticker", "horizon", "observation_date", "news_window_days", "news_half_life_days"], kind="stable").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    shards = []
    for horizon, suffix in ((1, "1d"), (5, "5d"), (10, "10d")):
        path = OUT.with_name(f"{OUT.stem}-{suffix}.parquet")
        part = merged.loc[merged["horizon"].astype(int).eq(horizon)].copy()
        part.to_parquet(path, index=False)
        shards.append({"path": str(path.relative_to(ROOT)), "rows": int(len(part)), "sha256": sha(path)})
    manifest = {
        "schema_version": "fusion-v4-targeted-extension-v1",
        "base_dataset": str(BASE.relative_to(ROOT)),
        "target_extension": str(TARGET.relative_to(ROOT)),
        "output_path": str(OUT.relative_to(ROOT)),
        "rows": int(len(merged)),
        "horizons": [1, 5, 10],
        "target_ticker": "1295.KL",
        "target_rows": int(len(target)),
        "base_rows_preserved_for_other_tickers": True,
        "manual_labels_used": False,
        "article_corpus_root": "data/historical-news-fusion/normalized-production-v3-official-prepared-expanded-v14-auto-corrected",
        "shards": shards,
        "sha256": sha(OUT),
    }
    manifest_path = OUT.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
