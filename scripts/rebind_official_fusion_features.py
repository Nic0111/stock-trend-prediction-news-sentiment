"""Rebind an existing point-in-time news feature table to a verified stock OOF run.

The news feature build is deterministic and expensive. This utility preserves all
existing point-in-time news columns and replaces only the stock probability fields
from an independently verified nested OOF export. It writes a new Parquet file and
never changes the active fusion dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


KEYS = ["Date", "Ticker", "horizon"]


def rebind(source: Path, probabilities: Path, output: Path) -> dict[str, object]:
    base = pd.read_parquet(source)
    oof = pd.read_csv(probabilities)
    for frame, name in ((base, "source"), (oof, "OOF")):
        missing = [column for column in KEYS if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing key columns: {missing}")
    base["Date"] = pd.to_datetime(base["Date"], errors="raise").dt.normalize()
    oof["Date"] = pd.to_datetime(oof["Date"], errors="raise").dt.normalize()
    base["Ticker"] = base["Ticker"].astype(str).str.upper().str.strip()
    oof["Ticker"] = oof["Ticker"].astype(str).str.upper().str.strip()
    base["horizon"] = pd.to_numeric(base["horizon"], errors="raise").astype(int)
    oof["horizon"] = pd.to_numeric(oof["horizon"], errors="raise").astype(int)
    if base.duplicated(KEYS).any() or oof.duplicated(KEYS).any():
        raise ValueError("Duplicate key rows found")
    stock_columns = {
        "probability_up": "stock_probability_up",
        "decision_threshold": "stock_decision_threshold",
        "threshold_margin": "stock_threshold_margin",
        "evidence_strength": "stock_evidence_strength",
    }
    merged = base.drop(
        columns=list(stock_columns.values()), errors="ignore"
    ).merge(
        oof[KEYS + list(stock_columns)], on=KEYS, how="left", validate="one_to_one"
    )
    for source_column, destination_column in stock_columns.items():
        merged = merged.rename(columns={source_column: destination_column})
    if merged["stock_probability_up"].isna().any():
        raise ValueError("OOF probabilities do not cover every fusion row")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    payload = {
        "schema_version": "official-v3-fusion-rebind-v1",
        "source": str(source),
        "stock_oof": str(probabilities),
        "output": str(output),
        "rows": int(len(merged)),
        "tickers": int(merged["Ticker"].nunique()),
        "horizons": sorted(merged["horizon"].unique().tolist()),
        "stock_registry_run_id": sorted(oof["stock_registry_run_id"].dropna().astype(str).unique().tolist()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--stock-oof", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rebind(args.source, args.stock_oof, args.output)


if __name__ == "__main__":
    main()
