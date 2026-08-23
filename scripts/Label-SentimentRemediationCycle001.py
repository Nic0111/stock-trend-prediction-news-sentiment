"""Create auditable Codex labels for sentiment-remediation cycle 001.

This one-off script records the content review of the frozen 393-row package.
It deliberately does not call FinBERT, consume existing model predictions, or
use a remote labelling API. Rows absent from ``RELEVANT`` were reviewed as
incidental company mentions, broad-market stories, different entities, archive
pages, or otherwise unsuitable issuer-level sentiment evidence.

The output retains truthful ``codex`` / ``codex_single_review`` provenance. It
must not be represented as independent human ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "sentiment-remediation"
    / "low-coverage-2015-2026-v1"
    / "historical"
    / "codex-label-package"
    / "cycle-001"
    / "review-pack.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "codex-reviewed-20260809"


# Frozen review-pack row number -> (sentiment, event category, confidence).
# Only material, issuer-specific evidence is listed. Everything else receives
# an explicit entity-negative decision.
RELEVANT: dict[int, tuple[str, str, float]] = {
    # Maybank
    7: ("positive", "operations", 0.92),
    20: ("positive", "financing", 0.92),
    24: ("positive", "product_service", 0.90),
    34: ("positive", "product_service", 0.91),
    37: ("positive", "esg", 0.82),
    41: ("positive", "product_service", 0.90),
    171: ("positive", "financing", 0.93),
    185: ("positive", "product_service", 0.94),
    189: ("positive", "product_service", 0.95),
    193: ("positive", "product_service", 0.94),
    298: ("positive", "product_service", 0.95),
    # Public Bank
    25: ("positive", "product_service", 0.87),
    26: ("positive", "product_service", 0.87),
    30: ("positive", "corporate_action", 0.94),
    31: ("positive", "corporate_action", 0.94),
    45: ("neutral", "financing", 0.82),
    46: ("positive", "corporate_action", 0.86),
    55: ("positive", "financing", 0.95),
    56: ("positive", "financing", 0.95),
    202: ("positive", "corporate_action", 0.95),
    213: ("negative", "earnings", 0.93),
    342: ("neutral", "corporate_action", 0.90),
    # PETRONAS Chemicals
    51: ("neutral", "operations", 0.84),
    53: ("neutral", "operations", 0.84),
    54: ("neutral", "operations", 0.84),
    60: ("negative", "earnings", 0.97),
    69: ("negative", "operations", 0.94),
    70: ("negative", "operations", 0.94),
    200: ("positive", "earnings", 0.88),
    205: ("negative", "earnings", 0.97),
    207: ("negative", "earnings", 0.96),
    211: ("negative", "earnings", 0.92),
    217: ("neutral", "financing", 0.78),
    228: ("negative", "regulatory_legal", 0.90),
    245: ("neutral", "operations", 0.82),
    265: ("positive", "earnings", 0.95),
    267: ("negative", "operations", 0.88),
    287: ("positive", "earnings", 0.94),
    288: ("positive", "earnings", 0.96),
    290: ("negative", "financing", 0.84),
    367: ("negative", "other", 0.91),
    371: ("negative", "operations", 0.87),
    378: ("negative", "earnings", 0.97),
    379: ("negative", "earnings", 0.97),
    380: ("negative", "earnings", 0.97),
    # MISC
    84: ("neutral", "earnings", 0.86),
    86: ("negative", "regulatory_legal", 0.96),
    87: ("positive", "operations", 0.93),
    89: ("negative", "earnings", 0.95),
    90: ("positive", "contract", 0.96),
    91: ("positive", "corporate_action", 0.92),
    92: ("positive", "corporate_action", 0.92),
    93: ("positive", "contract", 0.94),
    94: ("positive", "contract", 0.88),
    97: ("positive", "contract", 0.95),
    98: ("positive", "contract", 0.95),
    99: ("positive", "contract", 0.96),
    100: ("positive", "contract", 0.95),
    101: ("positive", "contract", 0.95),
    105: ("positive", "operations", 0.86),
    108: ("positive", "contract", 0.94),
    111: ("positive", "contract", 0.94),
    113: ("positive", "operations", 0.86),
    115: ("neutral", "earnings", 0.88),
    124: ("positive", "esg", 0.94),
    131: ("neutral", "corporate_action", 0.86),
    132: ("positive", "contract", 0.94),
    210: ("positive", "earnings", 0.97),
    226: ("neutral", "corporate_action", 0.84),
    236: ("positive", "contract", 0.96),
    243: ("positive", "esg", 0.93),
    259: ("positive", "corporate_action", 0.94),
    261: ("positive", "corporate_action", 0.94),
    270: ("positive", "earnings", 0.97),
    393: ("positive", "contract", 0.96),
    # IHH Healthcare
    95: ("neutral", "management", 0.88),
    103: ("negative", "corporate_action", 0.89),
    104: ("negative", "corporate_action", 0.89),
    106: ("negative", "corporate_action", 0.88),
    107: ("neutral", "corporate_action", 0.90),
    116: ("negative", "regulatory_legal", 0.96),
    117: ("negative", "regulatory_legal", 0.96),
    118: ("neutral", "corporate_action", 0.88),
    119: ("neutral", "regulatory_legal", 0.82),
    120: ("positive", "corporate_action", 0.94),
    121: ("negative", "regulatory_legal", 0.93),
    122: ("neutral", "regulatory_legal", 0.88),
    125: ("positive", "earnings", 0.97),
    126: ("positive", "earnings", 0.97),
    128: ("neutral", "corporate_action", 0.80),
    129: ("positive", "corporate_action", 0.94),
    130: ("neutral", "corporate_action", 0.82),
    214: ("positive", "earnings", 0.97),
    240: ("positive", "operations", 0.93),
    242: ("positive", "operations", 0.93),
    244: ("positive", "operations", 0.93),
    262: ("positive", "product_service", 0.95),
    269: ("positive", "earnings", 0.95),
    272: ("positive", "corporate_action", 0.88),
    273: ("positive", "other", 0.80),
    365: ("positive", "corporate_action", 0.86),
    370: ("negative", "operations", 0.91),
    373: ("positive", "corporate_action", 0.88),
    374: ("positive", "corporate_action", 0.86),
    # Press Metal
    88: ("positive", "regulatory_legal", 0.88),
    114: ("negative", "regulatory_legal", 0.95),
    134: ("negative", "earnings", 0.97),
    135: ("negative", "earnings", 0.96),
    136: ("negative", "earnings", 0.96),
    139: ("positive", "regulatory_legal", 0.93),
    140: ("positive", "regulatory_legal", 0.93),
    142: ("positive", "earnings", 0.96),
    144: ("negative", "earnings", 0.96),
    145: ("positive", "earnings", 0.98),
    150: ("negative", "regulatory_legal", 0.85),
    153: ("positive", "macro_sector_exposure", 0.82),
    154: ("positive", "earnings", 0.98),
}


def _rationale(sentiment: str | None, entity_relevant: bool) -> str:
    if not entity_relevant:
        return (
            "Codex review: the recovered item is not material issuer-level evidence; "
            "the match is incidental, broad-market, a different entity, an analyst-only "
            "mention, or a non-substantive/archive page."
        )
    implications = {
        "positive": "a likely favourable issuer-level implication",
        "negative": "a likely adverse issuer-level implication",
        "neutral": "no clearly favourable or adverse issuer-level implication",
    }
    return "Codex review: the article directly describes company-specific evidence with " + implications[sentiment or "neutral"] + "."


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "article_id",
        "content_sha256",
        "label_text_sha256",
        "batch_sha256",
        "entity_relevant",
        "sentiment",
        "event_category",
        "language",
        "confidence",
        "evidence",
        "rationale",
        "label_status",
        "labelled_by",
        "labelled_at",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _label(row: dict[str, str], number: int, labelled_at: str) -> dict[str, Any]:
    title = " ".join(str(row.get("title") or "").split())
    selected = RELEVANT.get(number)
    entity_relevant = selected is not None
    sentiment = selected[0] if selected else ""
    event_category = selected[1] if selected else "other"
    confidence = selected[2] if selected else 0.92
    return {
        "article_id": row["article_id"],
        "content_sha256": row["content_sha256"],
        "label_text_sha256": row["label_text_sha256"],
        "batch_sha256": row["batch_sha256"],
        "entity_relevant": str(entity_relevant).lower(),
        "sentiment": sentiment,
        "event_category": event_category,
        "language": "en",
        "confidence": f"{confidence:.2f}",
        "evidence": json.dumps([title[:240]], ensure_ascii=False),
        "rationale": _rationale(sentiment or None, entity_relevant),
        "label_status": "labelled",
        "labelled_by": "codex",
        "labelled_at": labelled_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Label the frozen 393-row sentiment-remediation package.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = _read_csv(args.input)
    if len(rows) != 393:
        raise SystemExit(f"Expected the frozen 393-row package, found {len(rows)} rows")
    if not set(RELEVANT).issubset(range(1, len(rows) + 1)):
        raise SystemExit("A reviewed row number is outside the frozen package")

    labelled_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    labels = [_label(row, number, labelled_at) for number, row in enumerate(rows, start=1)]
    output_path = args.output_dir / "review-pack-codex-reviewed.csv"
    _write_csv(output_path, labels)

    relevant_by_ticker = Counter(
        row["ticker"] for row, label in zip(rows, labels) if label["entity_relevant"] == "true"
    )
    summary = {
        "input": str(args.input),
        "output": str(output_path),
        "rows": len(labels),
        "entity_relevant": sum(label["entity_relevant"] == "true" for label in labels),
        "entity_negative": sum(label["entity_relevant"] == "false" for label in labels),
        "sentiment_counts": dict(
            sorted(Counter(label["sentiment"] or "entity_negative" for label in labels).items())
        ),
        "ticker_relevant_counts": dict(sorted(relevant_by_ticker.items())),
        "provenance": {
            "labelled_by": "codex",
            "expected_import_source": "codex_single_review",
            "independent_human_ground_truth": False,
            "finbert_predictions_used": False,
            "remote_label_api_used": False,
            "review_policy": "material selected-company evidence only",
        },
    }
    (args.output_dir / "label-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
