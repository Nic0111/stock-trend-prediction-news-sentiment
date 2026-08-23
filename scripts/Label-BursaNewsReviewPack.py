"""Create the auditable Codex review labels for the fixed retry review pack.

This is deliberately a *one-off, content-reviewed* labelling script.  It does
not call FinBERT, use existing model predictions, or access the network.  The
explicit row map below records the substantive company-specific decisions made
against the frozen 381-row review pack.  Rows not listed in the map were
reviewed as incidental, broad-market, duplicate-index, archive, or unrelated
mentions and receive an explicit entity-negative label.

The project importer retains ``codex_single_review`` as the original source
and records accepted rows as ``llm_assisted_label``. The source provenance is
therefore explicit and is not represented as an independent human double-label
consensus.
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
    / "bursa-news-balance-v2-retry-unrecovered-v1"
    / "codex-label-package"
    / "review-pack-appended.csv"
)
DEFAULT_ORIGINAL_IDS = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "bursa-news-balance-v2"
    / "codex-label-package"
    / "cycle-001"
    / "review-pack.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "codex-reviewed-20260803"


# row number -> (sentiment, event category, confidence[, language])
#
# These entries are direct company or controlled-subsidiary evidence reviewed
# from the title and recovered article text.  Keep the identifiers readable so
# a future examiner can trace every label back to the frozen review pack.
RELEVANT: dict[int, tuple[str, str, float] | tuple[str, str, float, str]] = {
    # CIMB Group
    136: ("positive", "product_service", 0.92),
    156: ("positive", "product_service", 0.91),
    # Maybank
    11: ("neutral", "management", 0.88),
    288: ("positive", "product_service", 0.82),
    289: ("neutral", "other", 0.82),
    # Public Bank
    22: ("negative", "regulatory_legal", 0.92),
    27: ("neutral", "operations", 0.86),
    28: ("neutral", "operations", 0.82),
    36: ("positive", "earnings", 0.96),
    171: ("positive", "corporate_action", 0.96),
    303: ("negative", "macro_sector_exposure", 0.78),
    # MISC
    45: ("positive", "regulatory_legal", 0.90),
    46: ("positive", "operations", 0.90),
    48: ("positive", "earnings", 0.85),
    49: ("neutral", "corporate_action", 0.80),
    51: ("positive", "operations", 0.86),
    53: ("neutral", "other", 0.78),
    57: ("negative", "earnings", 0.95),
    62: ("neutral", "operations", 0.86),
    64: ("negative", "earnings", 0.94),
    66: ("negative", "earnings", 0.94),
    67: ("positive", "operations", 0.88),
    68: ("negative", "earnings", 0.94),
    74: ("neutral", "regulatory_legal", 0.84),
    77: ("positive", "contract", 0.91),
    78: ("neutral", "operations", 0.84),
    83: ("neutral", "corporate_action", 0.90),
    84: ("neutral", "corporate_action", 0.90),
    87: ("positive", "other", 0.92),
    88: ("positive", "other", 0.92),
    169: ("negative", "other", 0.75),
    207: ("neutral", "earnings", 0.78),
    214: ("positive", "contract", 0.90),
    219: ("positive", "esg", 0.94),
    242: ("positive", "earnings", 0.94),
    243: ("negative", "earnings", 0.88),
    249: ("neutral", "other", 0.75),
    250: ("neutral", "management", 0.94),
    252: ("neutral", "earnings", 0.75),
    259: ("positive", "corporate_action", 0.92),
    260: ("positive", "corporate_action", 0.92),
    269: ("positive", "contract", 0.94),
    270: ("positive", "contract", 0.92),
    276: ("positive", "esg", 0.93),
    279: ("negative", "earnings", 0.75),
    280: ("positive", "corporate_action", 0.86),
    316: ("positive", "contract", 0.91),
    357: ("positive", "operations", 0.88),
    372: ("positive", "operations", 0.93),
    376: ("positive", "contract", 0.92),
    377: ("positive", "contract", 0.96),
    378: ("positive", "contract", 0.92),
    # YTL
    72: ("positive", "financing", 0.94),
    93: ("neutral", "other", 0.75),
    97: ("negative", "earnings", 0.95),
    100: ("neutral", "other", 0.78),
    112: ("positive", "corporate_action", 0.90),
    114: ("positive", "operations", 0.85),
    127: ("positive", "other", 0.78),
    181: ("positive", "earnings", 0.95),
    210: ("positive", "operations", 0.95),
    228: ("neutral", "other", 0.78),
    235: ("positive", "earnings", 0.96),
    244: ("neutral", "corporate_action", 0.80),
    246: ("neutral", "corporate_action", 0.90),
    247: ("positive", "corporate_action", 0.90),
    248: ("neutral", "other", 0.78),
    251: ("positive", "operations", 0.90),
    253: ("positive", "earnings", 0.92),
    254: ("positive", "earnings", 0.90),
    255: ("negative", "regulatory_legal", 0.95),
    256: ("neutral", "regulatory_legal", 0.90),
    257: ("negative", "regulatory_legal", 0.94),
    258: ("negative", "regulatory_legal", 0.95),
    261: ("negative", "regulatory_legal", 0.92),
    272: ("neutral", "other", 0.78),
    274: ("positive", "corporate_action", 0.92),
    275: ("positive", "contract", 0.92),
    277: ("positive", "operations", 0.90),
    358: ("positive", "contract", 0.95),
    359: ("positive", "product_service", 0.94),
    360: ("positive", "product_service", 0.94),
    361: ("positive", "esg", 0.85),
    362: ("positive", "corporate_action", 0.90),
    364: ("positive", "esg", 0.85),
    365: ("positive", "corporate_action", 0.86),
    367: ("positive", "product_service", 0.94),
    368: ("positive", "esg", 0.93),
    369: ("positive", "product_service", 0.94),
    370: ("positive", "financing", 0.92),
    371: ("positive", "operations", 0.92),
    # PETRONAS Chemicals
    1: ("positive", "earnings", 0.96),
    6: ("positive", "operations", 0.94),
    7: ("positive", "operations", 0.94),
    8: ("positive", "operations", 0.94),
    37: ("positive", "earnings", 0.90),
    39: ("positive", "corporate_action", 0.86),
    174: ("positive", "operations", 0.94),
    184: ("negative", "earnings", 0.96),
    196: ("neutral", "corporate_action", 0.85),
    212: ("neutral", "regulatory_legal", 0.94),
    307: ("positive", "earnings", 0.97),
    336: ("negative", "regulatory_legal", 0.90),
    337: ("positive", "operations", 0.94),
    # IHH Healthcare
    189: ("positive", "earnings", 0.97),
    193: ("positive", "corporate_action", 0.93),
    204: ("neutral", "management", 0.84),
    208: ("positive", "other", 0.80),
    221: ("positive", "other", 0.80),
    222: ("neutral", "corporate_action", 0.90),
    232: ("neutral", "corporate_action", 0.90),
    237: ("positive", "earnings", 0.97),
    238: ("negative", "earnings", 0.96),
    239: ("negative", "earnings", 0.96),
    241: ("neutral", "other", 0.78),
    340: ("neutral", "corporate_action", 0.80),
    342: ("neutral", "regulatory_legal", 0.90),
    343: ("neutral", "regulatory_legal", 0.90),
    344: ("neutral", "regulatory_legal", 0.90),
    345: ("neutral", "regulatory_legal", 0.90),
    346: ("neutral", "regulatory_legal", 0.90),
    347: ("neutral", "regulatory_legal", 0.90),
    352: ("positive", "operations", 0.88),
    353: ("negative", "regulatory_legal", 0.80),
    355: ("positive", "operations", 0.90),
    356: ("positive", "operations", 0.90),
    # Tenaga Nasional
    65: ("neutral", "corporate_action", 0.85, "other"),
    90: ("neutral", "corporate_action", 0.90),
    102: ("negative", "operations", 0.92),
    157: ("positive", "contract", 0.91),
    168: ("negative", "regulatory_legal", 0.96),
    182: ("positive", "esg", 0.92),
    183: ("positive", "esg", 0.92),
    198: ("positive", "product_service", 0.92),
    203: ("positive", "corporate_action", 0.88),
    209: ("neutral", "regulatory_legal", 0.78),
    218: ("negative", "other", 0.88),
    231: ("positive", "esg", 0.90),
    233: ("positive", "product_service", 0.95),
    240: ("negative", "other", 0.89),
    298: ("positive", "product_service", 0.92),
    306: ("positive", "product_service", 0.95),
    309: ("positive", "esg", 0.85),
    311: ("positive", "product_service", 0.95),
    312: ("positive", "contract", 0.94),
    313: ("neutral", "operations", 0.82),
    321: ("negative", "regulatory_legal", 0.88),
    333: ("positive", "operations", 0.93),
    338: ("positive", "product_service", 0.95),
    350: ("positive", "product_service", 0.95),
    381: ("neutral", "other", 0.78),
    # CelcomDigi
    117: ("neutral", "corporate_action", 0.93),
    118: ("neutral", "management", 0.85),
    119: ("neutral", "corporate_action", 0.88),
    120: ("negative", "regulatory_legal", 0.86),
    121: ("neutral", "regulatory_legal", 0.90),
    122: ("neutral", "product_service", 0.85),
    123: ("neutral", "corporate_action", 0.90),
    126: ("neutral", "other", 0.85),
    143: ("positive", "product_service", 0.94),
    155: ("positive", "product_service", 0.95),
    158: ("positive", "product_service", 0.96),
    166: ("positive", "esg", 0.90),
    172: ("positive", "earnings", 0.97),
    173: ("positive", "operations", 0.96),
    176: ("negative", "earnings", 0.93),
    185: ("positive", "operations", 0.90),
    186: ("positive", "product_service", 0.90),
    187: ("positive", "product_service", 0.90),
    188: ("positive", "product_service", 0.90),
    191: ("positive", "operations", 0.94),
    192: ("positive", "operations", 0.94),
    194: ("positive", "product_service", 0.90),
    197: ("positive", "product_service", 0.95),
    199: ("neutral", "corporate_action", 0.80),
    215: ("negative", "other", 0.84),
    217: ("positive", "product_service", 0.94),
    220: ("positive", "operations", 0.90),
    223: ("positive", "operations", 0.92),
    225: ("positive", "operations", 0.92),
    226: ("neutral", "product_service", 0.77),
    230: ("positive", "operations", 0.94),
    234: ("negative", "operations", 0.96),
    310: ("positive", "product_service", 0.92),
    323: ("positive", "operations", 0.83),
    329: ("positive", "product_service", 0.94),
    332: ("positive", "product_service", 0.95),
    335: ("positive", "operations", 0.94),
    348: ("neutral", "earnings", 0.82),
    349: ("positive", "earnings", 0.96),
    354: ("positive", "product_service", 0.94),
    # Press Metal
    60: ("neutral", "other", 0.78),
    61: ("positive", "corporate_action", 0.94),
    85: ("neutral", "financing", 0.90),
    89: ("positive", "other", 0.87),
    91: ("neutral", "other", 0.80),
    94: ("positive", "other", 0.88),
    96: ("positive", "operations", 0.94),
    99: ("negative", "macro_sector_exposure", 0.90),
    101: ("positive", "other", 0.92),
    110: ("positive", "corporate_action", 0.94),
    111: ("negative", "earnings", 0.90),
    115: ("positive", "corporate_action", 0.94),
    245: ("neutral", "other", 0.80),
    263: ("negative", "operations", 0.96),
    264: ("positive", "corporate_action", 0.94),
    266: ("positive", "corporate_action", 0.94),
    267: ("positive", "corporate_action", 0.94),
    268: ("positive", "corporate_action", 0.94),
    271: ("negative", "macro_sector_exposure", 0.90),
    273: ("positive", "esg", 0.92),
    278: ("neutral", "other", 0.85),
    281: ("positive", "earnings", 0.96),
    363: ("positive", "earnings", 0.90),
    366: ("positive", "contract", 0.95),
    373: ("neutral", "other", 0.82),
    374: ("neutral", "earnings", 0.80),
    375: ("positive", "operations", 0.95),
    380: ("positive", "other", 0.95),
}


def _rationale(sentiment: str | None, entity_relevant: bool) -> str:
    if not entity_relevant:
        return (
            "Codex review: the recovered article is not materially about the selected "
            "Bursa issuer; the mention is incidental, broad market context, a different entity, "
            "or a non-news/archive page."
        )
    descriptions = {
        "positive": "a likely favourable financial, operating, contract, or strategic implication.",
        "negative": "a likely adverse financial, operating, regulatory, or strategic implication.",
        "neutral": "no clearly favourable or adverse company implication.",
    }
    return "Codex review: the article directly describes company-specific evidence with " + descriptions[sentiment or "neutral"]


def _label_row(row: dict[str, str], row_number: int, labelled_at: str) -> dict[str, Any]:
    title = " ".join(str(row.get("title") or "").split())
    base: dict[str, Any] = {
        "article_id": row["article_id"],
        "content_sha256": row["content_sha256"],
        "label_text_sha256": row["label_text_sha256"],
        "batch_sha256": row["batch_sha256"],
        "entity_relevant": "false",
        "sentiment": "",
        "event_category": "other",
        "language": "en",
        "confidence": "0.90",
        "evidence": json.dumps([title[:240]], ensure_ascii=False),
        "rationale": _rationale(None, False),
        "label_status": "labelled",
        "labelled_by": "codex",
        "labelled_at": labelled_at,
    }
    selected = RELEVANT.get(row_number)
    if selected is None:
        return base
    sentiment, event_category, confidence, *optional_language = selected
    language = optional_language[0] if optional_language else "en"
    base.update(
        {
            "entity_relevant": "true",
            "sentiment": sentiment,
            "event_category": event_category,
            "language": language,
            "confidence": f"{confidence:.2f}",
            "rationale": _rationale(sentiment, True),
        }
    )
    return base


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Codex-reviewed labels for the frozen 381-row Bursa pack.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--original-review-pack", type=Path, default=DEFAULT_ORIGINAL_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = _read_csv(args.input)
    if len(rows) != 381:
        raise SystemExit(f"Expected the frozen 381-row appended review pack, found {len(rows)} rows")
    original_ids = {row["article_id"] for row in _read_csv(args.original_review_pack)}
    if len(original_ids) != 379:
        raise SystemExit(f"Expected 379 original review IDs, found {len(original_ids)}")
    if not set(RELEVANT).issubset(set(range(1, len(rows) + 1))):
        raise SystemExit("A reviewed row number is outside the frozen review pack")

    labelled_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    labels = [_label_row(row, number, labelled_at) for number, row in enumerate(rows, start=1)]
    original = [label for label in labels if label["article_id"] in original_ids]
    retry = [label for label in labels if label["article_id"] not in original_ids]
    if len(original) != 379 or len(retry) != 2:
        raise SystemExit(f"Unexpected package split: original={len(original)}, retry={len(retry)}")

    output_dir = args.output_dir
    _write_csv(output_dir / "original-codex-reviewed.csv", original)
    _write_csv(output_dir / "retry-codex-reviewed.csv", retry)
    _write_csv(output_dir / "review-pack-appended-codex-reviewed.csv", labels)
    summary = {
        "input": str(args.input),
        "rows": len(labels),
        "original_rows": len(original),
        "retry_rows": len(retry),
        "entity_relevant": sum(item["entity_relevant"] == "true" for item in labels),
        "entity_negative": sum(item["entity_relevant"] == "false" for item in labels),
        "sentiment_counts": dict(
            sorted(Counter(item["sentiment"] or "unlabelled_entity_negative" for item in labels).items())
        ),
        "ticker_relevant_counts": dict(
            sorted(
                Counter(
                    row["ticker"]
                    for row, label in zip(rows, labels)
                    if label["entity_relevant"] == "true"
                ).items()
            )
        ),
        "provenance": {
            "labelled_by": "codex",
            "label_source_original_expected_after_import": "codex_single_review",
            "finbert_predictions_used": False,
            "network_or_llm_api_used": False,
            "review_policy": "material selected-company evidence only; incidental candidates are entity-negative",
        },
        "outputs": {
            "all": str(output_dir / "review-pack-appended-codex-reviewed.csv"),
            "original": str(output_dir / "original-codex-reviewed.csv"),
            "retry": str(output_dir / "retry-codex-reviewed.csv"),
        },
    }
    (output_dir / "label-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
