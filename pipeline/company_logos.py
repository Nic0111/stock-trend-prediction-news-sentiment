from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from urllib.parse import quote


COMPANY_DOMAINS: Dict[str, str] = {
    "1155.KL": "maybank.com",
    "1295.KL": "publicbank.com.my",
    "1023.KL": "cimb.com",
    "5347.KL": "tnb.com.my",
    "5225.KL": "ihhhealthcare.com",
    "6947.KL": "celcomdigi.com",
    "5183.KL": "petronas.com",
    "8869.KL": "pressmetal.com",
    "3816.KL": "miscgroup.com",
    "4677.KL": "ytl.com",
}


# Prefer official brand assets or Wikimedia-hosted SVG/PNG originals over generic favicons.
# The favicon URL remains a secondary network fallback; initials are rendered underneath both.
COMPANY_LOGO_URLS: Dict[str, str] = {
    "1155.KL": "https://www.maybank.com/iwov-resources/images/icons/maybank_logo.svg",
    "1295.KL": "https://upload.wikimedia.org/wikipedia/en/2/21/Public_Bank_Berhad_logo.svg",
    "1023.KL": "https://upload.wikimedia.org/wikipedia/commons/8/8f/CIMB_Group_Logo.svg",
    "5347.KL": "https://upload.wikimedia.org/wikipedia/en/2/27/Tenaga_Nasional_Logo.svg",
    "5225.KL": "https://www.ihhhealthcare.com/ihh/assets/images/reskin/logo-ihh.svg",
    "6947.KL": "https://upload.wikimedia.org/wikipedia/en/d/d6/CelcomDigi_Logo.svg",
    "5183.KL": "https://www.petronas.com/themes/custom/petronas/images/petronas-logo-dark.svg",
    "8869.KL": "https://www.pressmetal.com/images/logo.png",
    "3816.KL": "https://www.miscgroup.com/awcontent/web/assets-desktop/images/logo.svg",
    "4677.KL": "https://upload.wikimedia.org/wikipedia/commons/b/ba/YTL_Corporation_Berhad.png",
}


@dataclass(frozen=True)
class CompanyLogo:
    ticker: str
    domain: str
    primary_url: str
    fallback_url: str
    initials: str


def _build_initials(company_name: str) -> str:
    words = [word for word in company_name.replace("&", " ").split() if word]
    legal_words = {"berhad", "holdings", "group", "corporation", "limited", "bhd"}
    meaningful = [word for word in words if word.casefold() not in legal_words]
    selected = meaningful or words
    if not selected:
        return "?"
    if len(selected) == 1:
        return selected[0][:2].upper()
    return (selected[0][0] + selected[1][0]).upper()


def get_company_logo(ticker: str, company_name: str, *, size: int = 256) -> CompanyLogo:
    normalized_ticker = ticker.strip().upper()
    domain = COMPANY_DOMAINS.get(normalized_ticker, "")
    if not domain:
        raise ValueError(f"No company logo domain is configured for {normalized_ticker}")
    encoded_domain = quote(domain, safe=".")
    return CompanyLogo(
        ticker=normalized_ticker,
        domain=domain,
        primary_url=COMPANY_LOGO_URLS.get(
            normalized_ticker,
            f"https://www.google.com/s2/favicons?domain={encoded_domain}&sz={size}",
        ),
        fallback_url=f"https://icons.duckduckgo.com/ip3/{encoded_domain}.ico",
        initials=_build_initials(company_name),
    )
