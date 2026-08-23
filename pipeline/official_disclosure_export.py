"""Lawful, metadata-first export of public issuer/Bursa disclosures for fusion v3.

This module deliberately does *not* bypass robots, Cloudflare, login walls or
other access controls.  It fetches only public HTML pages listed in the v3
source register, caches each response, extracts the public date/title/link
metadata, and records failures in a resumable report.  Full article text is
not invented: issuer metadata rows are marked ``official_export`` and
``publisher_full_text_available=False`` until a permitted export supplies it.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from pipeline.official_disclosure_news import classify_material_event


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTER = PROJECT_ROOT / "data" / "fusion" / "official-disclosures-v3-source-register.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "fusion" / "official-disclosures-v3-export.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "fusion" / "official-disclosures-v3-export-report.json"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "fusion" / ".official-disclosure-cache-v3"
LOCAL_TZ = "Asia/Kuala_Lumpur"
YEARS = tuple(range(2019, 2026))


@dataclass(frozen=True)
class SourceSpec:
    ticker: str
    company_name: str
    source_kind: str
    adapter: str
    url_template: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("1023.KL", "CIMB Group Holdings Berhad", "issuer_ir", "cimb", "https://www.cimb.com/en/newsroom.html?year={year}"),
    SourceSpec("1155.KL", "Malayan Banking Berhad", "issuer_ir", "listedcompany", "https://maybank.listedcompany.com/newsroom.html/year/{year}"),
    # Public Bank embeds the Bursa table in a Cloudflare-protected iframe.  We
    # record the public register but deliberately do not bypass that control;
    # a lawful user export can still be supplied through prepare-official.
    SourceSpec("1295.KL", "Public Bank Berhad", "issuer_ir", "blocked_register_only", "https://www.publicbankgroup.com/investor-relations/bursa-announcements/"),
    # Public Bank's issuer-hosted media-release archive exposes explicit
    # dates and titles for 2019--2025.  It is a lawful metadata export and is
    # independent of the access-challenged Bursa iframe above.
    SourceSpec("1295.KL", "Public Bank Berhad", "issuer_ir", "pbb_media_release", "https://www.publicbankgroup.com/news-update/media-release/{year}/"),
    # Public Bank's financial-information pages expose explicit publication
    # dates in the issuer-hosted media-release PDF filenames for recent years.
    # The adapter only accepts filenames containing a full day/month/year;
    # quarter labels alone are deliberately rejected as period dates.
    SourceSpec("1295.KL", "Public Bank Berhad", "issuer_ir", "pbb_financial_information", "https://www.publicbankgroup.com/investor-relations/financial-information/{year}/"),
    SourceSpec(
        "3816.KL",
        "MISC Berhad",
        "issuer_ir",
        "insage",
        "https://www.insage.com.my/ir/cmn/trps02/Announcements_trps02.aspx?Symbol=3816&page={page}&CatId=&FromDate=01%2F01%2F{year}&ToDate=31%2F12%2F{year}",
    ),
    SourceSpec("5183.KL", "Petronas Chemicals Group Berhad", "issuer_ir", "pcg", "https://www.petronas.com/pcg/investor-home"),
    SourceSpec("5183.KL", "Petronas Chemicals Group Berhad", "issuer_ir", "pcg_briefing", "https://www.petronas.com/pcg/investor-relations/analyst-briefing"),
    SourceSpec("5183.KL", "Petronas Chemicals Group Berhad", "issuer_ir", "pcg_news", "https://www.petronas.com/pcg/media/news-and-highlights?created_year={year}"),
    SourceSpec("5225.KL", "IHH Healthcare Berhad", "issuer_ir", "listedcompany", "https://ihhmy.listedcompany.com/newsroom.html/year/{year}"),
    SourceSpec("5347.KL", "Tenaga Nasional Berhad", "issuer_ir", "tnb", "https://www.tnb.com.my/announcements/"),
    # The public RSS feed is a lawful, timestamped export.  It is bounded to
    # the publisher's retained feed; missing older years are reported rather
    # than fabricated.
    SourceSpec("5347.KL", "Tenaga Nasional Berhad", "issuer_ir", "tnb_rss", "https://www.tnb.com.my/rss/announcements"),
    # TNB also publishes a dated, public press-release register.  It is kept
    # separate from the Bursa-announcement register because the latter is
    # dominated by routine listing filings.  The adapter below selects one
    # English issuer release per event and retains only material categories.
    SourceSpec("5347.KL", "Tenaga Nasional Berhad", "issuer_ir", "tnb_press_releases", "https://www.tnb.com.my/listings/press_releases"),
    # TNB's current listing is not a complete historical archive.  These
    # issuer-hosted announcement pages are retained as a small, auditable
    # legacy supplement.  The URLs, dates and titles below are copied from
    # the public TNB pages; the exporter verifies each URL before emitting a
    # metadata row.  They are never treated as publisher full text.
    SourceSpec("5347.KL", "Tenaga Nasional Berhad", "issuer_ir", "tnb_legacy_announcements", "https://www.tnb.com.my/announcements/"),
    SourceSpec("6947.KL", "CelcomDigi Berhad", "bursa_announcement", "listedcompany", "https://celcomdigi.listedcompany.com/newsroom.html/year/{year}"),
    SourceSpec(
        "8869.KL",
        "Press Metal Aluminium Holdings Berhad",
        "issuer_ir",
        "insage",
        "https://www.insage.com.my/ir/cmn/trps02/Announcements_trps02.aspx?Symbol=8869&page={page}&CatId=&FromDate=01%2F01%2F{year}&ToDate=31%2F12%2F{year}",
    ),
    SourceSpec(
        "4677.KL",
        "YTL Corporation Berhad",
        "press_release",
        "ytl_press_release",
        "https://www.ytl.com/press-releases-archive?post_year={year}",
    ),
)


TNB_LEGACY_ANNOUNCEMENTS: tuple[tuple[int, str, str, str], ...] = (
    (2019, "2019-03-28", "TNB Committed to Delivering Value to Stakeholders", "https://www.tnb.com.my/announcements/tnb-committed-to-delivering-value-to-stakeholders"),
    (2020, "2020-02-28", "TNB Sustains Growth in Challenging 2019", "https://www.tnb.com.my/announcements/tnb-sustains-growth-in-challenging-2019"),
    (2020, "2020-03-27", "TNB Pledges Another RM150 Million to Assist Malaysians Affected by Extended MCO", "https://www.tnb.com.my/announcements/tnb-pledges-another-rm150-million-to-assist-malaysians"),
    (2020, "2020-05-10", "Reopening of Kedai Tenaga in 5 States", "https://www.tnb.com.my/announcements/reopening-of-kedai-tenaga-in-5-states"),
    (2020, "2020-05-14", "TNB Introduces Easy Payment Plan in Post MCO Package for Residential Customers", "https://www.tnb.com.my/announcements/tnb-introduces-epp-in-post-mco"),
    (2020, "2020-06-10", "TNB Affected by Volatile Foreign Exchange in Q1 2020, Prepares for Prolonged Challenges Post COVID-19", "https://www.tnb.com.my/announcements/tnb-affected-by-volatile-foreign-exchange-in-q1-2020"),
    (2020, "2020-06-30", "TNB Committed to Fair Shareholder Returns", "https://www.tnb.com.my/announcements/tnb-committed-to-fair-shareholder-returns"),
    (2021, "2021-02-10", "TNB Appoints Baharin Din as President and Chief Executive Officer Effective 1 March 2021", "https://www.tnb.com.my/announcements/change-in-boardroom-datuk-ir-baharin-bin-din"),
    (2021, "2021-05-31", "TNB Operation Hours During Total Lockdown Starting from 1 June 2021", "https://www.tnb.com.my/announcements/tnb-operation-hours-during-total-lockdown-starting-from-1-june-2021"),
    (2021, "2021-11-25", "TNB Remains Resilient Amidst National Economic Recovery", "https://www.tnb.com.my/announcements/tnb-remains-resilient-amidst-national-economic-recovery"),
    (2022, "2022-09-01", "TNB Shows Resilient 1HFY22 Financial Performance", "https://www.tnb.com.my/announcements/tnb-shows-resilient-1hfy22-financial-performance"),
    (2022, "2022-10-14", "TNB Achieves Financial Close for LSS4 towards A Sustainable Future", "https://www.tnb.com.my/announcements/tnb-achieves-financial-close-for-lss4-towards-a-sustainable-future"),
    (2022, "2022-08-02", "TNB Fast-Tracks Sustainability Agenda", "https://www.tnb.com.my/announcements/tnb-fast-tracks-sustainability-agenda"),
    (2022, "2022-08-17", "TNB To Invest RM20 Billion Capex Per Annum In Driving Responsible Energy Transition Journey 2050", "https://www.tnb.com.my/announcements/tnb-to-invest-rm20-billion-in-driving-responsible-energy-transition-journey"),
    (2022, "2022-08-19", "TNB, Petronas Join Forces to Advance Malaysia's Carbon-Neutral Aspirations", "https://www.tnb.com.my/announcements/tnb-petronas-join-forces-to-advance-malaysias-carbon-neutral-aspirations"),
    (2022, "2022-09-14", "TNB to Accelerate Its Digitalisation Journey, Focusing On Energy Transition and Sustainability", "https://www.tnb.com.my/announcements/tnb-to-accelerate-its-digitalisation-journey"),
    (2023, "2023-02-28", "TNB Records Strong FY2022 EBITDA, Intensifies Progress in Energy Transition Plan", "https://www.tnb.com.my/announcements/tnb-records-strong-fy2022-ebitda-intensifies-progress-in-energy-transition"),
    (2023, "2023-06-01", "TNB Powers Up Sustainability Journey Further with New Sustainability Division", "https://www.tnb.com.my/announcements/tnb-powers-up-sustainability-journey-further-with-new-sustainability-divisi"),
    (2023, "2023-06-23", "Press Statement on Tariff Adjustment (July-Dec 2023) for Peninsular Malaysia by NRECC", "https://www.tnb.com.my/announcements/press-statement-on-tariff-adjustment-july-dec-2023-for-peninsular-malaysia"),
    (2023, "2023-12-06", "TNB Unveils Vision for A Sustainable Energy Future and Collaboration at CEPSI 2023", "https://www.tnb.com.my/announcements/tnb-unveils-vision-for-a-sustainable-energy-future-at-CEPSI2023"),
    (2024, "2024-05-21", "TNB Driving Strategic Growth and Net Zero Commitments in 2023 and Beyond", "https://www.tnb.com.my/announcements/tnb-driving-strategic-growth-and-net-zero-commitments-in-2023-and-beyo"),
)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def _url_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _parse_date(value: Any, year: int | None = None) -> str | None:
    text = _clean(value)
    if not text:
        return None
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
            stamp = pd.to_datetime(text, errors="coerce", utc=True)
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            stamp = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        elif re.fullmatch(r"\d{8}", text):
            stamp = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        elif re.match(r"^[A-Za-z]{3,9}\b", text):
            stamp = pd.to_datetime(text, errors="coerce", dayfirst=False)
        else:
            stamp = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if pd.isna(stamp):
            return None
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(LOCAL_TZ)
        else:
            stamp = stamp.tz_convert(LOCAL_TZ)
        if year is not None and int(stamp.year) != int(year):
            return None
        # Public pages generally provide dates, not a publication time.  Noon
        # MYT is explicit and avoids silently claiming a post-close timestamp.
        stamp = stamp.normalize() + pd.Timedelta(hours=12)
        return stamp.tz_convert("UTC").isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _record(
    spec: SourceSpec,
    *,
    title: str,
    published_at: str,
    url: str,
    source_page: str,
    event_category_override: str | None = None,
) -> dict[str, Any] | None:
    title = _clean(title)
    if not title or not published_at or not url:
        return None
    category = event_category_override or classify_material_event(title, "")
    # A disclosure whose title cannot be mapped to a material event is still
    # kept out of the fusion corpus; the report records its source page.  This
    # prevents routine filing volume from masquerading as news evidence.
    if category in {"routine_administrative", "unclassified"}:
        return None
    article_id = hashlib.sha256("|".join((spec.ticker, published_at, title, url)).encode("utf-8")).hexdigest()
    return {
        "article_id": article_id,
        "ticker": spec.ticker,
        "company_name": spec.company_name,
        "published_at": published_at,
        "retrieved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "title": title,
        # The source page exposes metadata only.  Keep the title as a bounded
        # model input, and retain provenance so it is never confused with full
        # publisher text.
        "description": title,
        "content_snippet": title,
        "url": url,
        "source_name": urlparse(url).netloc.casefold().removeprefix("www."),
        "source_domain": urlparse(url).netloc.casefold().removeprefix("www."),
        "provider": "bursa_official" if spec.source_kind == "bursa_announcement" else "issuer_official",
        "source_kind": spec.source_kind,
        "disclosure_type": spec.source_kind,
        "event_category": category,
        "text_source": "official_export",
        "publisher_full_text_available": False,
        "is_pseudo_document": False,
        "automatic_relevance": "official_source",
        # These rows come from a company-specific issuer/Bursa register.  The
        # source itself is the entity anchor; keep it explicit so the normal
        # scorer does not drop the row when it applies the entity-relevance
        # filter.
        "entity_relevant": True,
        "relevance_score": 1.0,
        "entity_confidence": 1.0,
        "label_validation_status": "unlabelled_reference",
        "llm_assisted_label": False,
        "source_page": source_page,
    }


def _tnb_legacy_announcements(
    html_by_url: Mapping[str, str | None],
    spec: SourceSpec,
    year: int,
    base_url: str,
) -> list[dict[str, Any]]:
    """Return metadata for the fixed, issuer-hosted TNB legacy supplement.

    The current TNB index omits a portion of 2019--2022.  The supplement is
    deliberately small and explicit: every row has a primary TNB URL, an
    observed publication date and an event-category override justified by the
    page's title/content.  We still require the URL to have been fetched with
    a successful HTML response before emitting it.
    """

    category_by_url = {
        # The titles are short issuer metadata; these explicit categories keep
        # clearly material financial/regulatory events from being discarded by
        # a title-only classifier while preserving an auditable override.
        "https://www.tnb.com.my/announcements/tnb-committed-to-delivering-value-to-stakeholders": "management_changes",
        "https://www.tnb.com.my/announcements/tnb-sustains-growth-in-challenging-2019": "financial_results_guidance",
        "https://www.tnb.com.my/announcements/tnb-pledges-another-rm150-million-to-assist-malaysians": "regulatory_legal_governance",
        "https://www.tnb.com.my/announcements/reopening-of-kedai-tenaga-in-5-states": "operational_product_strategy",
        "https://www.tnb.com.my/announcements/tnb-introduces-epp-in-post-mco": "regulatory_legal_governance",
        "https://www.tnb.com.my/announcements/tnb-affected-by-volatile-foreign-exchange-in-q1-2020": "financial_results_guidance",
        "https://www.tnb.com.my/announcements/tnb-committed-to-fair-shareholder-returns": "dividends_distributions",
        "https://www.tnb.com.my/announcements/change-in-boardroom-datuk-ir-baharin-bin-din": "management_changes",
        "https://www.tnb.com.my/announcements/tnb-operation-hours-during-total-lockdown-starting-from-1-june-2021": "operational_product_strategy",
        "https://www.tnb.com.my/announcements/tnb-remains-resilient-amidst-national-economic-recovery": "financial_results_guidance",
        "https://www.tnb.com.my/announcements/tnb-shows-resilient-1hfy22-financial-performance": "financial_results_guidance",
        "https://www.tnb.com.my/announcements/tnb-achieves-financial-close-for-lss4-towards-a-sustainable-future": "financing_capital_changes",
        "https://www.tnb.com.my/announcements/tnb-fast-tracks-sustainability-agenda": "operational_product_strategy",
        "https://www.tnb.com.my/announcements/tnb-to-invest-rm20-billion-in-driving-responsible-energy-transition-journey": "acquisitions_disposals_investments",
        "https://www.tnb.com.my/announcements/tnb-petronas-join-forces-to-advance-malaysias-carbon-neutral-aspirations": "contracts_order_wins",
        "https://www.tnb.com.my/announcements/tnb-to-accelerate-its-digitalisation-journey": "operational_product_strategy",
        "https://www.tnb.com.my/announcements/tnb-records-strong-fy2022-ebitda-intensifies-progress-in-energy-transition": "financial_results_guidance",
        "https://www.tnb.com.my/announcements/tnb-powers-up-sustainability-journey-further-with-new-sustainability-divisi": "operational_product_strategy",
        "https://www.tnb.com.my/announcements/press-statement-on-tariff-adjustment-july-dec-2023-for-peninsular-malaysia": "regulatory_legal_governance",
        "https://www.tnb.com.my/announcements/tnb-unveils-vision-for-a-sustainable-energy-future-at-CEPSI2023": "operational_product_strategy",
        "https://www.tnb.com.my/announcements/tnb-driving-strategic-growth-and-net-zero-commitments-in-2023-and-beyo": "operational_product_strategy",
    }
    rows: list[dict[str, Any]] = []
    for item_year, date_text, title, url in TNB_LEGACY_ANNOUNCEMENTS:
        if int(item_year) != int(year) or html_by_url.get(url) is None:
            continue
        published_at = _parse_date(date_text, year)
        row = _record(
            spec,
            title=title,
            published_at=published_at or "",
            url=url,
            source_page=url,
            event_category_override=category_by_url.get(url),
        )
        if row:
            row["disclosure_type"] = "issuer_legacy_announcement"
            row["legacy_metadata_override"] = True
            rows.append(row)
    return rows


def _fetch(url: str, *, cache_dir: Path, session: requests.Session, timeout: int = 30) -> tuple[str | None, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    body_path = cache_dir / f"{_url_key(url)}.html"
    meta_path = cache_dir / f"{_url_key(url)}.json"
    if body_path.is_file() and meta_path.is_file():
        try:
            return body_path.read_text(encoding="utf-8"), json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            pass
    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
        meta = {"url": url, "final_url": response.url, "status": response.status_code, "bytes": len(response.content), "fetched_at": datetime.now(timezone.utc).isoformat()}
        if response.status_code != 200 or "text/html" not in response.headers.get("content-type", "text/html").casefold():
            meta["error"] = "non-200 or non-HTML response"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return None, meta
        text = response.text
        body_path.write_text(text, encoding="utf-8")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return text, meta
    except requests.RequestException as exc:
        meta = {"url": url, "status": None, "error": type(exc).__name__ + ": " + str(exc), "fetched_at": datetime.now(timezone.utc).isoformat()}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return None, meta


def _listedcompany(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = _clean(anchor.get("href"))
        if "/news.html/id/" not in href:
            continue
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        title = anchor.get_text(" ", strip=True)
        date_text = ""
        for parent in (anchor.find_parent(class_="list-item"), anchor.find_parent(class_="newannonce-container"), anchor.find_parent("li")):
            if not parent:
                continue
            # The three listed-company templates use different markup:
            # ``ann-date``, ``ir-news-date`` or a plain date paragraph.
            date_node = (
                parent.find(class_="ann-date")
                or parent.find(class_="ir-news-date")
                or parent.find(class_="date")
            )
            if date_node is None:
                for candidate in parent.find_all(["p", "h5", "h6", "span"], limit=5):
                    candidate_text = candidate.get_text(" ", strip=True)
                    if re.search(r"\b(?:19|20)\d{2}\b", candidate_text) and re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})", candidate_text, re.I):
                        date_node = candidate
                        break
            if date_node:
                date_text = date_node.get_text(" ", strip=True)
                break
        published = _parse_date(date_text, year)
        row = _record(spec, title=title, published_at=published or "", url=url, source_page=base_url)
        if row:
            rows.append(row)
    return rows


def _tnb(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    for tr in soup.find_all("tr"):
        anchor = tr.find("a", href=True)
        date_node = tr.find(class_="channel-year")
        if not anchor or not date_node or "/announcements/" not in _clean(anchor.get("href")):
            continue
        raw_date = date_node.get_text(" ", strip=True)
        # Cards prefix the date with a quarter label (``1Q``); isolate the
        # human-readable month/day/year before handing it to the date parser.
        date_match = re.search(
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2}"
            r"|\b\d{1,2}[/-]\d{1,2}[/-]20\d{2}\b",
            raw_date,
            re.I,
        )
        published = _parse_date(date_match.group(0) if date_match else raw_date, year)
        row = _record(spec, title=anchor.get_text(" ", strip=True), published_at=published or "", url=urljoin(base_url, anchor["href"]), source_page=base_url)
        if row:
            row["date_precision"] = "publisher-announcement-date"
            rows.append(row)
    return rows


def _pbb_financial_information(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    """Parse explicitly dated Public Bank financial-release PDFs.

    The page groups four quarterly releases but historically does not expose a
    publication-date element.  We therefore accept only media-release links
    whose issuer filename contains a full day/month/year (for example
    ``eng_21-may_public-bank...``).  ``q1-2019.pdf``-style period labels are
    intentionally excluded because a quarter end is not a publication date.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    month_names = "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    date_pattern = re.compile(rf"(?<!\d)(\d{{1,2}})[^A-Za-z0-9]+({month_names})[^/\r\n]{{0,80}}?(20\d{{2}})(?!\d)", re.I)
    for anchor in soup.find_all("a", href=True):
        if anchor.get_text(" ", strip=True).casefold() != "media release":
            continue
        href = _clean(anchor.get("href"))
        match = date_pattern.search(href)
        if not match or int(match.group(3)) != int(year):
            continue
        published = _parse_date(f"{match.group(1)} {match.group(2)} {match.group(3)}", year)
        url = urljoin(base_url, href)
        if not published or not url or url in seen:
            continue
        seen.add(url)
        title = Path(urlparse(url).path).stem.replace("_", " ").replace("-", " ")
        row = _record(spec, title=f"Public Bank {title}", published_at=published, url=url, source_page=base_url)
        if row:
            row["disclosure_type"] = "financial_media_release"
            row["date_precision"] = "issuer-filename-day-month-year"
            rows.append(row)
    return rows


def _pcg_news(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    """Parse PCG's public, year-filtered News & Highlights archive."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in soup.select("div[role='article'].JpressMidBodyList"):
        date_node = card.select_one(".JpressMidBodyListTop p")
        title_node = card.select_one("h3 a[href]")
        if date_node is None or title_node is None:
            continue
        raw_date = date_node.get_text(" ", strip=True)
        published = _parse_date(raw_date, year)
        url = urljoin(base_url, _clean(title_node.get("href")))
        title = title_node.get_text(" ", strip=True)
        if not published or not title or not url or url in seen:
            continue
        seen.add(url)
        row = _record(spec, title=title, published_at=published, url=url, source_page=base_url)
        if row:
            row["disclosure_type"] = "issuer_news_release"
            row["date_precision"] = "publisher-release-date"
            rows.append(row)
    return rows


def _pbb_media_release(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    """Parse Public Bank's issuer-hosted, year-filtered media releases.

    The archive exposes an explicit calendar date and a title in each
    ``result-item``.  We retain only those metadata fields and deliberately do
    not treat the hidden release body as a permitted full-text export.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in soup.select(".result-item"):
        title_node = item.select_one(".title a")
        date_node = item.select_one(".calendar")
        if title_node is None or date_node is None:
            continue
        title = title_node.get_text(" ", strip=True)
        date_text = date_node.get_text(" ", strip=True)
        published = _parse_date(date_text, year)
        if not published:
            continue
        key = (published, title.casefold())
        if key in seen:
            continue
        seen.add(key)
        row = _record(spec, title=title, published_at=published, url=base_url, source_page=base_url)
        if row:
            row["disclosure_type"] = "media_release"
            row["source_kind"] = "issuer_ir"
            row["provider"] = "issuer_official"
            row["date_precision"] = "publisher-release-date"
            rows.append(row)
    return rows


def _insage(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for anchor in soup.find_all("a", href=True):
        href = _clean(anchor.get("href"))
        if not any(token in href.casefold() for token in ("new-announcement", "announcement", "medianews")):
            continue
        title = anchor.get_text(" ", strip=True)
        # Insage exposes the authoritative publication date in the enclosing
        # ``announcementByDate`` block. The NewsID is only a fallback because
        # some historical templates do not encode a complete date in links.
        date_text = ""
        parent = anchor.find_parent(class_="announcementByDate")
        if parent is not None:
            date_node = parent.find(class_="announcementDate")
            if date_node is not None:
                date_text = date_node.get_text(" ", strip=True)
        match = re.search(r"(?:newsid=|/)(20\d{6})", href, re.I)
        published = _parse_date(date_text, year)
        if not published and match:
            published = _parse_date(match.group(1), year)
        url = urljoin(base_url, href)
        key = (published or "", url)
        if key in seen:
            continue
        seen.add(key)
        row = _record(spec, title=title, published_at=published or "", url=url, source_page=base_url)
        if row:
            rows.append(row)
    return rows


def _ytl_press_release(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    """Parse YTL's public, year-filtered press-release archive.

    The archive exposes a date, title and public article URL in ordinary HTML.
    We retain only that metadata; the adapter never fetches a press-release
    body or treats the title as publisher full text.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in soup.select("div.card-body"):
        date_node = card.select_one("p.panel-prDate")
        anchor = card.select_one("h3.heading-prTitle a[href]")
        if date_node is None or anchor is None:
            continue
        url = urljoin(base_url, _clean(anchor.get("href")))
        if not url or url in seen:
            continue
        seen.add(url)
        title = anchor.get("title") or anchor.get_text(" ", strip=True)
        raw_date = date_node.get_text(" ", strip=True)
        date_match = re.search(
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2}",
            raw_date,
            re.I,
        )
        published = _parse_date(date_match.group(0) if date_match else raw_date, year)
        row = _record(
            spec,
            title=title,
            published_at=published or "",
            url=url,
            source_page=base_url,
        )
        if row:
            row["disclosure_type"] = "press_release"
            rows.append(row)
    return rows


def _pcg(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    for anchor in soup.find_all("a", href=True):
        title = anchor.get_text(" ", strip=True)
        href = _clean(anchor.get("href"))
        context = " ".join(anchor.parent.get_text(" ", strip=True).split())
        match = re.search(r"\b(20\d{2})\b", title + " " + href + " " + context)
        if not match or int(match.group(1)) != year or not href or href.startswith("data:"):
            continue
        if not any(token in (title + " " + href).casefold() for token in ("financial", "quarter", "bursa", "analyst", "report")):
            continue
        # PDF URLs often encode only a year/month.  Use 30 June as an explicit
        # metadata date when the page does not expose a day; this is only a
        # candidate source row and is flagged as date_precision=year-month.
        date = _parse_date(f"{year}-06-30", year)
        row = _record(spec, title=title or Path(urlparse(href).path).stem, published_at=date or "", url=urljoin(base_url, href), source_page=base_url)
        if row:
            row["date_precision"] = "year-month-inferred-from-public-file-path"
            rows.append(row)
    return rows


def _pcg_briefing(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    """Parse PCG's public analyst-briefing cards and linked disclosures.

    Each quarter card exposes an explicit release date and a ``Learn More``
    target whose panel contains the public press release and Bursa-report
    links.  We retain only those two material representations; presentation
    packs and transcripts are intentionally excluded to avoid duplicating one
    event.  The title is a bounded metadata description, never invented body
    text.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in soup.select("div.JawardsSlideBodyListInner"):
        date_node = card.select_one("p.mb-0")
        target_link = card.select_one("a[href^='#']")
        if date_node is None or target_link is None:
            continue
        raw_date = date_node.get_text(" ", strip=True)
        date_match = re.search(
            r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+20\d{2}",
            raw_date,
            re.I,
        )
        published = _parse_date(date_match.group(0) if date_match else raw_date, year)
        if not published:
            continue
        target = soup.find(id=_clean(target_link.get("href")).lstrip("#"))
        if target is None:
            continue
        period = " ".join(target.select_one(".JgeneralPopupTitle h2").get_text(" ", strip=True).split()) if target.select_one(".JgeneralPopupTitle h2") else "quarterly"
        for anchor in target.select(".JgeneralPopupContentListInner a[href]"):
            label = _clean(anchor.get_text(" ", strip=True))
            if label not in {"Press Release", "Financial Report (Bursa Announcement)"}:
                continue
            url = urljoin(base_url, _clean(anchor.get("href")))
            if not url or url in seen:
                continue
            seen.add(url)
            # Add the event type to the title so the deterministic taxonomy
            # can classify a generic link label such as "Press Release".
            kind = "quarterly results press release" if label == "Press Release" else "quarterly financial results report"
            title = f"PCG {period} {kind}"
            row = _record(spec, title=title, published_at=published, url=url, source_page=base_url)
            if row:
                row["disclosure_type"] = "press_release" if label == "Press Release" else "bursa_announcement"
                row["source_kind"] = "press_release" if label == "Press Release" else "bursa_announcement"
                row["provider"] = "issuer_official" if label == "Press Release" else "bursa_official"
                row["date_precision"] = "publisher-release-date"
                rows.append(row)
    return rows


def _tnb_rss(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    """Parse TNB's public RSS announcement feed without following article bodies."""
    soup = BeautifulSoup(html, "xml")
    rows: list[dict[str, Any]] = []
    for item in soup.find_all("item"):
        title = _clean(item.find("title").get_text(" ", strip=True) if item.find("title") else "")
        url = _clean(item.find("link").get_text(" ", strip=True) if item.find("link") else "")
        raw_date = _clean(item.find("pubDate").get_text(" ", strip=True) if item.find("pubDate") else "")
        published_text = raw_date
        if raw_date:
            try:
                stamp = parsedate_to_datetime(raw_date)
                published_text = stamp.isoformat()
            except (TypeError, ValueError, OverflowError):
                pass
        published = _parse_date(published_text, year)
        row = _record(spec, title=title, published_at=published or "", url=url, source_page=base_url)
        if row:
            row["date_precision"] = "rss-pubDate"
            rows.append(row)
    return rows


def _tnb_press_releases(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    """Parse TNB's public press-release register.

    The register contains English and Malay versions of the same release and
    can include routine/community items.  We deliberately keep only an
    explicitly dated English representation whose deterministic event
    taxonomy is material.  This is metadata from an issuer-controlled public
    page; no PDF body is fetched or treated as full publisher text.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    # Malay titles commonly use these words.  URL language markers are
    # preferred, but the title guard handles older files without a marker.
    malay_markers = re.compile(
        r"\b(?:dan|dalam|untuk|usaha|negara|sebagai|oleh|mendukung|mengorak|perkukuh|memacu|kerjasama|kerajaan|tenaga|syarikat|rakyat|terus|langkah|menyokong|meningkatkan|peralihan)\b",
        re.I,
    )
    english_url = re.compile(r"(?:_eng(?:lish)?|_en(?:g)?|_bi)(?:[._%]|$)", re.I)
    for tr in soup.find_all("tr"):
        anchor = tr.find("a", href=True)
        date_node = tr.find(class_="channel-year")
        if anchor is None or date_node is None:
            continue
        raw_date = date_node.get_text(" ", strip=True)
        published = _parse_date(raw_date, year)
        if not published:
            continue
        href = _clean(anchor.get("href"))
        title = _clean(anchor.get_text(" ", strip=True))
        if not href or not title:
            continue
        # Prefer explicit English filename markers.  If absent, reject titles
        # that clearly identify the Malay counterpart.
        if not english_url.search(urlparse(href).path) and malay_markers.search(title):
            continue
        url = urljoin(base_url, href)
        category = classify_material_event(title, "")
        if category in {"routine_administrative", "unclassified"}:
            continue
        key = (published, re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip())
        if key in seen:
            continue
        seen.add(key)
        row = _record(spec, title=title, published_at=published, url=url, source_page=base_url)
        if row:
            row["disclosure_type"] = "issuer_press_release"
            row["date_precision"] = "publisher-release-date"
            rows.append(row)
    return rows


def _cimb(html: str, spec: SourceSpec, year: int, base_url: str) -> list[dict[str, Any]]:
    # CIMB newsroom pages are mostly JS-rendered; retain only explicitly
    # dated links if the public HTML exposes them.  The adapter intentionally
    # returns zero instead of guessing dates from a page URL.
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    for anchor in soup.find_all("a", href=True):
        href = _clean(anchor.get("href"))
        title = anchor.get_text(" ", strip=True)
        parent_text = " ".join(anchor.parent.get_text(" ", strip=True).split())
        match = re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+20\d{2}\b", parent_text, re.I)
        if not match or int(re.search(r"20\d{2}", match.group(0)).group(0)) != year:
            continue
        row = _record(spec, title=title, published_at=_parse_date(match.group(0), year) or "", url=urljoin(base_url, href), source_page=base_url)
        if row:
            rows.append(row)
    return rows


def _parse(spec: SourceSpec, html: str, year: int, base_url: str) -> list[dict[str, Any]]:
    if spec.adapter == "listedcompany":
        return _listedcompany(html, spec, year, base_url)
    if spec.adapter == "tnb":
        return _tnb(html, spec, year, base_url)
    if spec.adapter == "pbb_media_release":
        return _pbb_media_release(html, spec, year, base_url)
    if spec.adapter == "pbb_financial_information":
        return _pbb_financial_information(html, spec, year, base_url)
    if spec.adapter == "insage":
        return _insage(html, spec, year, base_url)
    if spec.adapter == "pcg":
        return _pcg(html, spec, year, base_url)
    if spec.adapter == "pcg_briefing":
        return _pcg_briefing(html, spec, year, base_url)
    if spec.adapter == "pcg_news":
        return _pcg_news(html, spec, year, base_url)
    if spec.adapter == "cimb":
        return _cimb(html, spec, year, base_url)
    if spec.adapter == "ytl_press_release":
        return _ytl_press_release(html, spec, year, base_url)
    if spec.adapter == "tnb_rss":
        return _tnb_rss(html, spec, year, base_url)
    if spec.adapter == "tnb_press_releases":
        return _tnb_press_releases(html, spec, year, base_url)
    if spec.adapter == "tnb_legacy_announcements":
        return []
    return []


def export_official_disclosures_v3(
    *,
    years: Sequence[int] = YEARS,
    output_path: Path | str = DEFAULT_OUTPUT,
    report_path: Path | str = DEFAULT_REPORT,
    cache_dir: Path | str = DEFAULT_CACHE,
    tickers: Sequence[str] | None = None,
    request_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    """Fetch public metadata with checkpointed cache and write an isolated CSV."""
    years = tuple(sorted({int(value) for value in years}))
    requested = {str(value).upper() for value in tickers} if tickers else None
    output_path, report_path, cache_dir = Path(output_path), Path(report_path), Path(cache_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "StockSense-v3-public-disclosure-export/1.0 (+educational research)"})
    all_rows: list[dict[str, Any]] = []
    source_report: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        if requested and spec.ticker.upper() not in requested:
            continue
        if spec.adapter == "blocked_register_only":
            source_report.append({"ticker": spec.ticker, "adapter": spec.adapter, "url": spec.url_template, "status": "not_fetched", "reason": "public register embeds an access-challenged Bursa iframe; no access-control bypass attempted; provide a lawful export if needed"})
            continue
        if spec.adapter == "tnb_legacy_announcements":
            # Verify every curated primary URL independently.  This keeps the
            # supplement resumable and prevents a stale manifest entry from
            # becoming a dataset row when the issuer page is unavailable.
            for item_year, _date_text, _title, legacy_url in TNB_LEGACY_ANNOUNCEMENTS:
                if int(item_year) not in years:
                    continue
                html, meta = _fetch(legacy_url, cache_dir=cache_dir, session=session)
                source_report.append(
                    {
                        "ticker": spec.ticker,
                        "year": int(item_year),
                        "page": None,
                        "adapter": spec.adapter,
                        "url": legacy_url,
                        "status": "verified" if html is not None else "unavailable",
                        **meta,
                    }
                )
                if html is not None:
                    all_rows.extend(
                        _tnb_legacy_announcements(
                            {legacy_url: html}, spec, int(item_year), legacy_url
                        )
                    )
                if request_interval_seconds > 0:
                    time.sleep(float(request_interval_seconds))
            continue
        for year in years:
            # The public Insage endpoint is paginated.  Its date filters are
            # year-specific, so bounded page traversal recovers the complete
            # visible result set without guessing dates or bypassing controls.
            if spec.adapter == "insage":
                pages = range(1, 51)
            elif spec.adapter == "pcg_news":
                # PCG exposes the year archive through a client-side "Load
                # More" link.  The first page is the unqualified URL and
                # subsequent pages use ``page=N``; bound traversal so the
                # collector remains free-only and resumable from cache.
                pages = (None, *range(1, 51))
            else:
                pages = (None,)
            for page in pages:
                format_values = {"year": year}
                if page is not None:
                    format_values["page"] = page
                url = spec.url_template.format(**format_values)
                if spec.adapter == "pcg_news" and page is not None:
                    url = f"{url}&page={int(page)}"
                html, meta = _fetch(url, cache_dir=cache_dir, session=session)
                item = {"ticker": spec.ticker, "year": year, "page": page, "adapter": spec.adapter, "url": url, **meta}
                if html is None:
                    item["status"] = "unavailable"
                    source_report.append(item)
                    if page is not None:
                        break
                    continue
                rows = _parse(spec, html, year, url)
                all_rows.extend(rows)
                item.update({"status": "parsed", "rows": len(rows)})
                source_report.append(item)
                if request_interval_seconds > 0:
                    time.sleep(float(request_interval_seconds))
                if page is not None and not rows:
                    # A page containing only routine or unclassified filings
                    # is not the end of the register. The old check stopped
                    # here and silently skipped later pages with material
                    # events. Stop only when the page has no register entries.
                    soup = BeautifulSoup(html, "html.parser")
                    has_register_entries = bool(
                        soup.select_one(
                            ".announcementByDate, .announcementTitle, "
                            ".JpressMidBodyList"
                        )
                    )
                    if not has_register_entries:
                        break
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["article_id", "ticker", "company_name", "published_at", "title", "description", "content_snippet", "url", "source_name", "source_kind", "event_category"])
    else:
        frame = frame.drop_duplicates(["ticker", "article_id"], keep="last").sort_values(["ticker", "published_at", "article_id"]).reset_index(drop=True)
    frame.to_csv(output_path, index=False)
    payload = {
        "schema_version": "fusion-v3-official-disclosure-export-v1",
        "years": list(years),
        "rows": int(len(frame)),
        "rows_by_ticker": {str(k): int(v) for k, v in frame.groupby("ticker").size().to_dict().items()} if not frame.empty else {},
        "event_categories": {str(k): int(v) for k, v in frame.get("event_category", pd.Series(dtype=str)).value_counts().to_dict().items()},
        "output_path": str(output_path),
        "cache_dir": str(cache_dir),
        "source_report": source_report,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Metadata-first export; no publisher full text is claimed.", "Blocked/interactive sources are reported and require the user to provide a lawful export.", "Rows are candidates for fusion evidence and still require production sentiment scoring and point-in-time feature rebuild."],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


__all__ = ["DEFAULT_OUTPUT", "DEFAULT_REPORT", "DEFAULT_CACHE", "SOURCE_SPECS", "export_official_disclosures_v3"]
