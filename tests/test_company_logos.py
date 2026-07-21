from __future__ import annotations

import unittest

from pipeline.company_logos import COMPANY_DOMAINS, COMPANY_LOGO_URLS, get_company_logo


class CompanyLogoTests(unittest.TestCase):
    def test_all_supported_tickers_have_hd_logo_sources_and_fallback_initials(self) -> None:
        for ticker, domain in COMPANY_DOMAINS.items():
            logo = get_company_logo(ticker, f"{ticker} Example Berhad")
            self.assertEqual(logo.domain, domain)
            self.assertEqual(logo.primary_url, COMPANY_LOGO_URLS[ticker])
            self.assertTrue(logo.fallback_url.startswith("https://icons.duckduckgo.com/"))
            self.assertTrue(logo.initials)

    def test_unknown_ticker_has_no_silent_logo_mapping(self) -> None:
        with self.assertRaises(ValueError):
            get_company_logo("UNKNOWN.KL", "Unknown Company Berhad")


if __name__ == "__main__":
    unittest.main()
