from __future__ import annotations

import pandas as pd
import pytest

from pipeline.official_disclosure_news import (
    classify_material_event,
    prepare_official_disclosures,
)


def test_material_taxonomy_and_routine_exclusion() -> None:
    assert classify_material_event("Quarterly financial results") == "financial_results_guidance"
    assert classify_material_event("Interim dividend entitlement date") == "dividends_distributions"
    assert classify_material_event("Notice of meeting") == "routine_administrative"

    result = prepare_official_disclosures(
        pd.DataFrame(
            [
                {"title": "Contract awarded", "description": "New order win"},
                {"title": "Notice of meeting", "description": "Routine notice"},
            ]
        ),
        source_kind="bursa_announcement",
    )
    assert len(result) == 1
    assert result.iloc[0]["event_category"] == "contracts_order_wins"
    assert result.iloc[0]["disclosure_type"] == "bursa_announcement"
    assert result.iloc[0]["entity_confidence"] == 1.0
    assert result.iloc[0]["source_kind"] == "bursa_announcement"
    assert result.iloc[0]["text_source"] == "official_export"


def test_unknown_source_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        prepare_official_disclosures(
            pd.DataFrame([{"title": "Announcement"}]),
            source_kind="search_snippet",
        )
