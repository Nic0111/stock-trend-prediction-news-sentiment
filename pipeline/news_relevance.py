"""Provider-neutral rules for screening non-news sources.

The provider adapters keep the raw payloads for auditability, but only sources
that can plausibly be financial news should reach sentiment scoring and the
dashboard.  This module intentionally uses a small, explicit domain list so
that normal company technology reporting is not removed by broad keyword
rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


NON_NEWS_PACKAGE_REGISTRY_DOMAINS = frozenset(
    {
        "pypi.org",
        "npmjs.com",
        "npmjs.org",
        "packagist.org",
        "rubygems.org",
        "crates.io",
        "nuget.org",
        "libraries.io",
        "pkg.go.dev",
        "anaconda.org",
        "conda-forge.org",
        "mvnrepository.com",
        "search.maven.org",
        "central.sonatype.com",
    }
)


# These profiles intentionally remain small and explicit.  The supported
# universe is fixed, so a transparent profile is safer than asking a general
# language model to decide whether an acronym refers to a Bursa issuer.
COMPANY_ENTITY_PROFILES: Mapping[str, Mapping[str, Sequence[str]]] = {
    "1155.KL": {
        "distinctive": ("Maybank", "Malayan Banking", "Maybank Group"),
        "ambiguous": (),
        "anchors": ("Malaysia", "Bursa", "Malaysian banking", "banking group"),
        "collisions": ("Maybank Highway",),
    },
    "1295.KL": {
        "distinctive": ("Public Bank Berhad", "PBBank", "Public Bank Group"),
        "ambiguous": ("Public Bank",),
        "anchors": ("Malaysia", "Bursa", "Malaysian banking", "public bank berhad"),
        "collisions": ("public bank ballot", "public bank system", "municipal bank"),
    },
    "1023.KL": {
        "distinctive": ("CIMB", "CIMB Group", "CIMB Bank"),
        "ambiguous": (),
        "anchors": ("Malaysia", "Bursa", "Malaysian banking", "CIMB Niaga"),
        "collisions": (),
    },
    "5347.KL": {
        "distinctive": ("Tenaga Nasional", "Tenaga Nasional Berhad", "TNB Electron", "TNB Energy"),
        "ambiguous": ("TNB",),
        "anchors": ("Malaysia", "Bursa", "electricity", "power", "utility", "grid", "substation", "EV", "charger", "charging"),
        "collisions": ("Thomasville National Bank", "TNB Financial", "TNB Bancshares"),
    },
    "5225.KL": {
        "distinctive": ("IHH Healthcare", "IHH Healthcare Berhad"),
        "ambiguous": ("IHH",),
        "anchors": ("Malaysia", "Bursa", "healthcare", "hospital", "medical", "Gleneagles", "Parkway"),
        "collisions": (),
    },
    "6947.KL": {
        "distinctive": ("CelcomDigi", "Celcom Digi", "CelcomDigi Berhad"),
        "ambiguous": (),
        "anchors": ("Malaysia", "Bursa", "telecommunications", "telco", "mobile network"),
        "collisions": (),
    },
    "5183.KL": {
        "distinctive": ("PETRONAS Chemicals", "PETRONAS Chemicals Group", "PetChem"),
        "ambiguous": ("PCG",),
        "anchors": ("Malaysia", "Bursa", "PETRONAS", "petrochemical", "chemicals", "polymer"),
        "collisions": ("Pregnancy Care Group", "Philippine Coast Guard", "PG&E", "Precision Components Group"),
    },
    "8869.KL": {
        "distinctive": ("Press Metal Aluminium", "Press Metal Aluminium Holdings", "Press Metal"),
        "ambiguous": (),
        "anchors": ("Malaysia", "Bursa", "aluminium", "aluminum", "smelter", "metal"),
        "collisions": (),
    },
    "3816.KL": {
        "distinctive": ("MISC Berhad", "MISC Berhad", "Malaysia Marine and Heavy Engineering"),
        "ambiguous": ("MISC",),
        "anchors": ("Malaysia", "Bursa", "maritime", "shipping", "LNG", "tanker", "offshore", "PETRONAS"),
        "collisions": ("miscellaneous",),
    },
    "4677.KL": {
        "distinctive": ("YTL Corporation", "YTL Corporation Berhad", "YTL Power", "YTL Cement"),
        "ambiguous": ("YTL",),
        "anchors": ("Malaysia", "Bursa", "power", "cement", "property", "utilities", "telecommunications"),
        "collisions": (),
    },
}


@dataclass(frozen=True)
class EntityRelevance:
    """Auditable entity decision made before sentiment scoring."""

    status: str
    score: float
    reasons: tuple[str, ...]
    matched_terms: tuple[str, ...] = ()
    collision_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "automatic_relevance": self.status,
            "relevance_score": self.score,
            "relevance_reasons": list(self.reasons),
            "matched_entity_terms": list(self.matched_terms),
            "collision_signals": list(self.collision_signals),
        }


def normalize_source_domain(value: Any) -> str:
    """Return a lowercase registrable-looking host without ``www.``."""

    text = str(value or "").strip().casefold()
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"//{text}")
    host = (parsed.hostname or "").strip().casefold().removeprefix("www.")
    return host


def classify_non_news_source(url: Any = "", source_domain: Any = "") -> str | None:
    """Return an exclusion reason for known package registries, if applicable."""

    domains = {
        normalize_source_domain(url),
        normalize_source_domain(source_domain),
    }
    for domain in domains:
        if not domain:
            continue
        if domain in NON_NEWS_PACKAGE_REGISTRY_DOMAINS or any(
            domain.endswith(f".{blocked}")
            for blocked in NON_NEWS_PACKAGE_REGISTRY_DOMAINS
        ):
            return "package_registry"
    return None


def _contains_phrase(text: str, phrase: str) -> bool:
    # Keep matching independent from URL handling; punctuation and whitespace
    # variations should not change an entity decision.
    import re

    haystack = re.sub(r"[^\w]+", " ", str(text or "").casefold()).strip()
    needle = re.sub(r"[^\w]+", " ", str(phrase or "").casefold()).strip()
    if not haystack or not needle:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in needle.split()) + r"(?!\w)"
    return bool(re.search(pattern, haystack))


def classify_entity_relevance(
    ticker: str,
    title: str,
    description: str,
    *,
    matched_alias: str = "",
    provider_entity_match: bool = False,
) -> EntityRelevance:
    """Classify a provider candidate with a precision-first entity policy."""

    profile = COMPANY_ENTITY_PROFILES.get(str(ticker or "").upper())
    if not profile:
        return EntityRelevance("review", 0.30, ("No entity profile is configured",))
    title_text = str(title or "")
    body_text = str(description or "")
    combined = f"{title_text} {body_text}"
    score = 0.0
    reasons: list[str] = []
    matched: list[str] = []
    collisions: list[str] = []

    if provider_entity_match:
        score += 0.80
        reasons.append("Exact provider entity match")
    for term in profile.get("distinctive", ()):
        if _contains_phrase(title_text, term):
            score = max(score, 0.70 if len(term.split()) == 1 else 0.80)
            reasons.append(f"Distinctive company term in title: {term}")
            matched.append(term)
        elif _contains_phrase(body_text, term):
            score = max(score, 0.60 if len(term.split()) == 1 else 0.70)
            reasons.append(f"Distinctive company term in article: {term}")
            matched.append(term)
    for term in profile.get("ambiguous", ()):
        if _contains_phrase(combined, term):
            score = max(score, score + 0.25)
            reasons.append(f"Ambiguous alias requires context: {term}")
            matched.append(term)
    anchor_hits = [anchor for anchor in profile.get("anchors", ()) if _contains_phrase(combined, anchor)]
    if anchor_hits:
        score = min(1.0, score + min(0.30, 0.15 * len(anchor_hits)))
        reasons.append("Company/market context anchors found")
        matched.extend(anchor_hits)
    for collision in profile.get("collisions", ()):
        if _contains_phrase(combined, collision):
            collisions.append(collision)
    if collisions:
        score = max(0.0, score - 0.80)
        reasons.append("Known entity collision detected")
    if matched_alias and matched_alias not in matched:
        matched.append(matched_alias)

    if collisions or score < 0.30:
        status = "unrelated"
    elif score < 0.70:
        status = "review"
    else:
        status = "included"
    if not reasons:
        reasons.append("No sufficiently specific company evidence")
    return EntityRelevance(status, round(min(1.0, max(0.0, score)), 4), tuple(dict.fromkeys(reasons)), tuple(dict.fromkeys(matched)), tuple(collisions))


def stable_article_id(article: Mapping[str, Any]) -> str:
    """Return a provider-independent identity for correction requests."""

    import re
    from urllib.parse import urlparse, urlunparse

    url = str(article.get("url") or "").strip()
    if url:
        parsed = urlparse(url)
        query = "&".join(
            item for item in parsed.query.split("&")
            if item and not item.casefold().startswith(("utm_", "fbclid=", "gclid="))
        )
        identity = urlunparse((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path.rstrip("/"), "", query, ""))
    else:
        title = re.sub(r"\s+", " ", str(article.get("title") or "").casefold()).strip()
        domain = re.sub(r"\s+", " ", str(article.get("source_domain") or "").casefold()).strip()
        published = str(article.get("published_at") or "")[:10]
        identity = f"{title}|{domain}|{published}"
    return "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def stable_event_id(article: Mapping[str, Any]) -> str:
    """Return a conservative event identity for diversity reporting.

    This is deliberately not used to remove articles.  It groups exact or
    near-syndicated headlines into an auditable event count so coverage cannot
    be overstated merely because the same story was republished.
    """

    import re
    from datetime import datetime, timezone

    title = re.sub(r"[^\w]+", " ", str(article.get("title") or "").casefold()).strip()
    # Provider boilerplate and very common words add little identity value.
    stop = {
        "a", "an", "and", "at", "by", "for", "from", "in", "of", "on",
        "the", "to", "with", "says", "said", "update",
    }
    tokens = [token for token in title.split() if token not in stop][:14]
    published = str(article.get("published_at") or "")
    try:
        parsed = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        # A three-day bucket keeps a syndicated event together without merging
        # recurring company announcements across a long context window.
        date_bucket = parsed.date().toordinal() // 3
    except ValueError:
        date_bucket = published[:10]
    identity = f"{' '.join(tokens)}|{date_bucket}"
    if not tokens:
        return stable_article_id(article)
    return "event-sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
