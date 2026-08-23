from __future__ import annotations

from pipeline.official_disclosure_export import (
    SOURCE_SPECS,
    _insage,
    _pbb_financial_information,
    _pbb_media_release,
    _pcg_briefing,
    _pcg_news,
    _tnb,
    _tnb_legacy_announcements,
    _tnb_press_releases,
    _tnb_rss,
    _ytl_press_release,
)


def test_public_source_register_has_paginated_insage_specs() -> None:
    specs = {spec.ticker: spec for spec in SOURCE_SPECS}
    assert "page={page}" in specs["3816.KL"].url_template
    assert "FromDate" in specs["3816.KL"].url_template
    assert "page={page}" in specs["8869.KL"].url_template
    assert specs["4677.KL"].adapter == "ytl_press_release"
    assert specs["4677.KL"].source_kind == "press_release"


def test_insage_parser_extracts_exact_new_announcement_date() -> None:
    html = """
    <html><body>
      <a href="https://example.ir/new-announcement.htm?NewsID=202401315100001&amp;Symbol=3816">Fourth quarter profit rises</a>
      <a href="https://example.ir/new-announcement.htm?NewsID=202312315100001&amp;Symbol=3816">Old year dividend</a>
    </body></html>
    """
    rows = _insage(html, SOURCE_SPECS[1].__class__("3816.KL", "MISC Berhad", "issuer_ir", "insage", ""), 2024, "https://example.com")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "3816.KL"
    assert rows[0]["published_at"].startswith("2024-01-31")


def test_insage_parser_prefers_enclosing_announcement_date() -> None:
    html = """
    <div class="announcementByDate">
      <div class="announcementDate">Monday - 16 Dec 2024</div>
      <div class="announcementTitle">
        <a href="https://example.ir/new-announcement.htm?NewsID=202412165000002&amp;Symbol=3816">
          Provision of Long-Term LNG Shipping Services
        </a>
      </div>
    </div>
    """
    spec = SOURCE_SPECS[1].__class__("3816.KL", "MISC Berhad", "issuer_ir", "insage", "")
    rows = _insage(html, spec, 2024, "https://example.com")
    assert len(rows) == 1
    assert rows[0]["published_at"].startswith("2024-12-16")


def test_ytl_parser_extracts_year_filtered_metadata_only() -> None:
    html = """
    <div class="card-body">
      <p class="panel-prDate">Thursday 20 November 2025</p>
      <h3 class="heading-prTitle"><a href="/press-releases/results">YTL Corp quarterly revenue rises</a></h3>
    </div>
    <div class="card-body">
      <p class="panel-prDate">Thursday 20 November 2024</p>
      <h3 class="heading-prTitle"><a href="/press-releases/old">YTL Corp old result</a></h3>
    </div>
    """
    spec = SOURCE_SPECS[-1]
    rows = _ytl_press_release(html, spec, 2025, "https://www.ytl.com/press-releases-archive?post_year=2025")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "4677.KL"
    assert rows[0]["source_kind"] == "press_release"
    assert rows[0]["publisher_full_text_available"] is False
    assert rows[0]["url"].endswith("/press-releases/results")


def test_pcg_briefing_parser_uses_explicit_card_date_and_material_links() -> None:
    html = """
    <div class="JawardsSlideBodyListInner">
      <p class="mb-0">1Q<br/>May 20, 2025</p>
      <a href="#q1">Learn More</a>
    </div>
    <div class="JgeneralPopupContainer" id="q1">
      <div class="JgeneralPopupTitle"><h2>1Q<br/>May 20, 2025</h2></div>
      <div class="JgeneralPopupContentListInner">
        <p><a href="/media/pcg-results">Press Release</a></p>
        <p><a href="/files/PCG%2031%20March%202025%20Quarterly%20Bursa%20Announcement.pdf">Financial Report (Bursa Announcement)</a></p>
        <p><a href="/files/presentation.pdf">Presentation Pack</a></p>
      </div>
    </div>
    """
    spec = next(spec for spec in SOURCE_SPECS if spec.adapter == "pcg_briefing")
    rows = _pcg_briefing(html, spec, 2025, "https://www.petronas.com/pcg/investor-relations/analyst-briefing")
    assert len(rows) == 2
    assert {row["source_kind"] for row in rows} == {"press_release", "bursa_announcement"}
    assert all(row["published_at"].startswith("2025-05-20") for row in rows)


def test_tnb_rss_parser_uses_pubdate_and_filters_year() -> None:
    html = """
    <rss><channel>
      <item><title>TNB quarterly financial results</title><link>https://www.tnb.com.my/announcements/results</link><pubDate>Thu, 20 Feb 2025 10:00:00 +0800</pubDate></item>
      <item><title>Old quarterly financial results</title><link>https://www.tnb.com.my/announcements/old</link><pubDate>Thu, 20 Feb 2024 10:00:00 +0800</pubDate></item>
    </channel></rss>
    """
    spec = next(spec for spec in SOURCE_SPECS if spec.adapter == "tnb_rss")
    rows = _tnb_rss(html, spec, 2025, "https://www.tnb.com.my/rss/announcements")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "5347.KL"
    assert rows[0]["date_precision"] == "rss-pubDate"


def test_pbb_media_release_parser_uses_calendar_date_and_material_titles() -> None:
    html = """
    <div class='result-item'>
      <div class='calendar'><div class='day'>29</div><div class='month-year'>Nov 2024</div></div>
      <div class='title'><a class='show-content-modal' href=''>Public Bank Group Achieved Pre-Tax Profit of RM2.45 Billion</a></div>
    </div>
    <div class='result-item'>
      <div class='calendar'><div class='day'>8</div><div class='month-year'>Nov 2024</div></div>
      <div class='title'><a class='show-content-modal' href=''>Comments On The Budget 2025</a></div>
    </div>
    """
    spec = next(spec for spec in SOURCE_SPECS if spec.adapter == "pbb_media_release")
    rows = _pbb_media_release(html, spec, 2024, "https://www.publicbankgroup.com/news-update/media-release/2024/")
    assert len(rows) == 2
    assert all(row["ticker"] == "1295.KL" for row in rows)
    assert all(row["date_precision"] == "publisher-release-date" for row in rows)
    by_title = {row["title"]: row for row in rows}
    assert by_title["Public Bank Group Achieved Pre-Tax Profit of RM2.45 Billion"]["event_category"] == "financial_results_guidance"
    assert by_title["Comments On The Budget 2025"]["event_category"] == "regulatory_legal_governance"


def test_pbb_financial_parser_requires_full_filename_date() -> None:
    html = """
    <h4>First Quarter</h4>
    <a class='pdf' href='/media/x/eng_21-may_public-bank-group-profit-2025.pdf'>Media Release</a>
    <h4>Fourth Quarter</h4>
    <a class='pdf' href='/media/y/q4-2019.pdf'>Media Release</a>
    """
    spec = next(spec for spec in SOURCE_SPECS if spec.adapter == "pbb_financial_information")
    rows = _pbb_financial_information(html, spec, 2025, "https://www.publicbankgroup.com/investor-relations/financial-information/2025/")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "1295.KL"
    assert rows[0]["date_precision"] == "issuer-filename-day-month-year"


def test_pcg_news_parser_uses_public_release_date() -> None:
    html = """
    <div role='article' class='node JpressMidBodyList'>
      <div class='JpressMidBodyListTop'><p>21 Nov, 2024</p></div>
      <h3><a href='/pcg/media/media-releases/pcg-posts-pat'>PCG Posts pre-tax profit of RM100 Million in 3Q 2024</a></h3>
    </div>
    <div role='article' class='node JpressMidBodyList'>
      <div class='JpressMidBodyListTop'><p>20 Nov, 2023</p></div>
      <h3><a href='/pcg/media/media-releases/old'>PCG Posts PAT of RM90 Million in 3Q 2023</a></h3>
    </div>
    """
    spec = next(spec for spec in SOURCE_SPECS if spec.adapter == "pcg_news")
    rows = _pcg_news(html, spec, 2024, "https://www.petronas.com/pcg/media/news-and-highlights?created_year=2024")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "5183.KL"
    assert rows[0]["date_precision"] == "publisher-release-date"


def test_tnb_parser_accepts_numeric_announcement_dates() -> None:
    html = """
    <table><tr>
      <td><a href='/announcements/tnb-contract'>TNB Secures New Contract</a></td>
      <td class='channel-year' data-value='2025'>15/04/2025</td>
    </tr><tr>
      <td><a href='/announcements/old'>TNB Old Contract</a></td>
      <td class='channel-year' data-value='2024'>15/04/2024</td>
    </tr></table>
    """
    spec = next(spec for spec in SOURCE_SPECS if spec.adapter == "tnb")
    rows = _tnb(html, spec, 2025, "https://www.tnb.com.my/announcements/")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "5347.KL"
    assert rows[0]["date_precision"] == "publisher-announcement-date"


def test_tnb_press_release_parser_keeps_one_dated_english_material_event() -> None:
    html = """
    <table><tr>
      <td><a href='/assets/press_releases/2025041508_ENG.pdf'>TNB Expands Global Footprint With A New Maintenance Contract</a></td>
      <td class='channel-year'>15/04/2025</td>
    </tr><tr>
      <td><a href='/assets/press_releases/2025041508_BM.pdf'>TNB Memperluas Kehadiran Global Dengan Kontrak Baharu</a></td>
      <td class='channel-year'>15/04/2025</td>
    </tr><tr>
      <td><a href='/assets/press_releases/2025030101_ENG.pdf'>TNB Community Sports Day</a></td>
      <td class='channel-year'>01/03/2025</td>
    </tr></table>
    """
    spec = next(spec for spec in SOURCE_SPECS if spec.adapter == "tnb_press_releases")
    rows = _tnb_press_releases(html, spec, 2025, "https://www.tnb.com.my/listings/press_releases")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "5347.KL"
    assert rows[0]["disclosure_type"] == "issuer_press_release"
    assert rows[0]["date_precision"] == "publisher-release-date"


def test_tnb_legacy_supplement_requires_verified_primary_url() -> None:
    spec = next(spec for spec in SOURCE_SPECS if spec.adapter == "tnb_legacy_announcements")
    url = "https://www.tnb.com.my/announcements/tnb-shows-resilient-1hfy22-financial-performance"
    rows = _tnb_legacy_announcements({url: "<html>issuer page</html>"}, spec, 2022, url)
    assert len(rows) == 1
    assert rows[0]["legacy_metadata_override"] is True
    assert rows[0]["source_page"] == url
    assert _tnb_legacy_announcements({}, spec, 2022, url) == []
