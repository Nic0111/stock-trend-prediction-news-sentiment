"""Fail-closed readiness checks for Malaysian news evidence evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = PROJECT_ROOT / "config" / "news_evidence_policy.json"


@dataclass(frozen=True)
class NewsEvidenceReadiness:
    ready: bool
    policy_version: str
    dataset_path: str
    row_count: int
    ticker_counts: dict[str, int]
    split_counts: dict[str, int]
    errors: tuple[str, ...]

    @property
    def reason(self) -> str:
        # Keep readiness as a boolean API field.  A plain ``ready`` label is
        # too easy to misread as production acceptance, especially while the
        # dataset is still only an LLM-assisted development benchmark.
        return "Dataset checks passed" if self.ready else "; ".join(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "reason": self.reason}


def check_news_evidence_readiness(
    policy_path: Path = DEFAULT_POLICY,
) -> NewsEvidenceReadiness:
    policy_file = Path(policy_path)
    try:
        policy = json.loads(policy_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return NewsEvidenceReadiness(
            False, "unavailable", "", 0, {}, {}, (f"news evidence policy unreadable: {exc}",)
        )
    dataset = PROJECT_ROOT / str(policy.get("benchmark_path", ""))
    if not dataset.is_file():
        return NewsEvidenceReadiness(
            False,
            str(policy.get("policy_version", "unknown")),
            str(dataset),
            0,
            {},
            {},
            (f"Malaysian labelled news benchmark not found: {dataset}",),
        )
    try:
        # Use the same table reader as the canonical Bursa validator so the
        # readiness wrapper accepts the materialized JSONL benchmark as well as
        # CSV exports supplied by annotators.
        from pipeline.bursa_news_dataset import read_table

        frame = read_table(dataset)
    except Exception as exc:
        return NewsEvidenceReadiness(
            False,
            str(policy.get("policy_version", "unknown")),
            str(dataset),
            0,
            {},
            {},
            (f"Malaysian labelled news benchmark could not be read: {exc}",),
        )
    errors: list[str] = []
    required_columns = [str(value) for value in policy.get("required_columns", [])]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
    readiness_gates = policy.get("readiness_gates", {})
    enforce_dataset_size = bool(
        readiness_gates.get("enforce_dataset_size", True)
    )
    if enforce_dataset_size and len(frame) < int(policy.get("minimum_rows", 0)):
        errors.append(
            f"benchmark has {len(frame)} rows; requires {int(policy.get('minimum_rows', 0))}"
        )
    ticker_counts = (
        frame.get("ticker", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    )
    minimum_per_ticker = int(policy.get("minimum_rows_per_ticker", 0))
    for ticker in policy.get("required_tickers", []):
        count = int(ticker_counts.get(str(ticker), 0))
        if enforce_dataset_size and count < minimum_per_ticker:
            errors.append(
                f"{ticker} has {count} labelled rows; "
                f"requires {minimum_per_ticker}"
            )
        elif (
            not enforce_dataset_size
            and bool(readiness_gates.get("require_all_tickers", True))
            and count == 0
        ):
            errors.append(f"required ticker has no labelled rows: {ticker}")
    split_counts = (
        frame.get("split", pd.Series(dtype=str)).astype(str).str.casefold().value_counts().to_dict()
    )
    for split in policy.get("required_splits", []):
        if int(split_counts.get(str(split).casefold(), 0)) == 0:
            errors.append(f"required split is absent: {split}")
    quality_policy_value = str(policy.get("dataset_quality_policy_path", "")).strip()
    if quality_policy_value:
        try:
            from pipeline.bursa_news_dataset import (
                load_policy as load_bursa_policy,
                validate_bursa_news_dataset,
            )

            quality_policy_path = PROJECT_ROOT / quality_policy_value
            quality_report = validate_bursa_news_dataset(
                frame,
                policy=load_bursa_policy(quality_policy_path),
                dataset_path=str(dataset),
            )
            errors.extend(
                error for error in quality_report.errors if error not in errors
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"Bursa news quality policy could not be evaluated: {exc}")
    return NewsEvidenceReadiness(
        not errors,
        str(policy.get("policy_version", "unknown")),
        str(dataset),
        len(frame),
        {str(key): int(value) for key, value in ticker_counts.items()},
        {str(key): int(value) for key, value in split_counts.items()},
        tuple(errors),
    )
