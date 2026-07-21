from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pipeline.dashboard_service import DashboardAnalysisResult, run_dashboard_analysis
from pipeline.company_logos import CompanyLogo, get_company_logo
from pipeline.download_financial_news_data import (
    NEWS_CONTEXT_CONFIG_VERSION,
    NEWS_HISTORICAL_CONTEXT_END_DAYS,
    NEWS_HISTORICAL_CONTEXT_START_DAYS,
    NEWS_RETRIEVAL_LOOKBACK_DAYS,
    NewsContextResult,
    article_age_days,
    load_stock_metadata,
)
from pipeline.marketaux_news_provider import retrieve_news_with_marketaux_fallback
from pipeline.market_snapshot import MarketSnapshot, fetch_market_snapshot
from pipeline.openrouter_client import (
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_MODEL,
    OpenRouterConfigurationError,
    OpenRouterError,
    complete_chat,
    get_openrouter_api_key,
)
from pipeline.predict_stock_trend import PredictionResult, predict_stock_trend


LOCKED_SELECTOR_COMPONENT = components.declare_component(
    "stocksense_locked_company_selector_v2",
    path=str(Path(__file__).resolve().parent / "assets" / "locked_company_selector"),
)


st.set_page_config(
    page_title="StockSense",
    page_icon="📈",
    layout="wide",
)


APPLE_THEME_CSS = """
<style>
:root {
    --ink: #1d1d1f;
    --secondary-ink: #6e6e73;
    --surface: rgba(255, 255, 255, 0.92);
    --surface-muted: #f5f5f7;
    --line: rgba(0, 0, 0, 0.08);
    --blue: #0071e3;
    --blue-dark: #0062c4;
    --green: #248a3d;
    --green-soft: #e9f7ed;
    --red: #d70015;
    --red-soft: #fff0f1;
    --amber: #9a6700;
    --amber-soft: #fff7df;
    --shadow: 0 18px 50px rgba(0, 0, 0, 0.07);
}

html, body, .stApp, .stApp button, .stApp input, .stApp textarea, .stApp select {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif;
}

.stApp {
    background-color: #f5f5f7;
    color: var(--ink);
}

/* Desktop-first type scale: readable at a glance on a 16:9 workspace. */
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stSelectbox"] [role="button"],
.stApp [data-testid="stRadio"] label,
.stApp button {
    font-size: 1rem;
}

[class*="st-key-remove_ticker_action_"] button {
    border-color: rgba(215, 0, 21, 0.22) !important;
    background: #fff0f1 !important;
    color: #b4232f !important;
}

[class*="st-key-remove_ticker_action_"] button:hover {
    border-color: rgba(215, 0, 21, 0.36) !important;
    background: #ffe1e4 !important;
    color: #a31322 !important;
}

.stApp [data-testid="stCaptionContainer"] p {
    font-size: 0.9rem;
    line-height: 1.45;
}

.stApp [data-testid="stMetricLabel"] {
    font-size: 0.92rem;
}

.stApp [data-testid="stMetricValue"] {
    font-size: 1.55rem;
}

.stApp [data-testid="stMetricDelta"] {
    font-size: 0.92rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stPageLink"] {
    font-size: 1.05rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 5rem;
}

[data-testid="stSidebar"] {
    background: rgba(250, 250, 252, 0.9);
    border-right: 1px solid var(--line);
    backdrop-filter: blur(24px);
}

[data-testid="stSidebarContent"] {
    display: flex;
    min-height: 100vh;
    flex-direction: column;
}

[data-testid="stSidebarNav"] {
    display: none;
}

[data-testid="stSidebarUserContent"] {
    display: flex;
    min-height: calc(100vh - 3.5rem);
    flex: 1 1 auto;
    flex-direction: column;
    padding-bottom: 1.25rem !important;
}

[data-testid="stSidebarUserContent"] > div {
    display: flex;
    min-height: 100%;
    flex: 1 1 auto;
    flex-direction: column;
}

[class*="st-key-sidebar_main_navigation"] {
    display: flex;
    flex-direction: column;
    gap: 0.12rem;
}

.sidebar-section-label {
    display: block;
    margin: 1.7rem 0.75rem 0.7rem;
    padding-bottom: 0.45rem;
    color: var(--secondary-ink);
    font-size: 0.8rem;
    font-weight: 760;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.sidebar-section-label:first-child {
    margin-top: 0;
}

[class*="st-key-sidebar_utility_navigation"] {
    padding-top: 1rem;
    border-top: 1px solid var(--line);
}

[data-testid="stSidebarUserContent"] [data-testid="stLayoutWrapper"]:has([class*="st-key-sidebar_utility_navigation"]) {
    margin-top: auto !important;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] {
    margin: 0.1rem 0;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    min-height: 3rem;
    align-items: center;
    border-radius: 13px;
    padding: 0.7rem 0.8rem;
    text-decoration: none;
    transition: background 140ms ease, color 140ms ease;
}

[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
    color: var(--ink) !important;
    display: flex;
    gap: 0.7rem;
    width: 100%;
    font-size: 1.05rem !important;
}

[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] {
    color: inherit !important;
    font-size: 1.45rem !important;
}

[data-testid="stSidebar"] .sidebar-brand {
    gap: 0.95rem;
    margin-bottom: 1.35rem !important;
    padding: 0 0.1rem 1.15rem;
    border-bottom: 1px solid var(--line);
}

[data-testid="stSidebar"] .sidebar-brand .brand-logo {
    width: 3.55rem;
    height: 3.55rem;
    border-radius: 16px;
}

[data-testid="stSidebar"] .sidebar-brand .brand-wordmark {
    font-size: 1.35rem;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
    background: rgba(0, 113, 227, 0.07);
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
    background: rgba(0, 113, 227, 0.13);
    color: var(--blue);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

div[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 22px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.04);
    padding: 1.1rem;
}

[data-testid="stFormSubmitButton"] button {
    min-height: 2.85rem;
    border: 0;
    border-radius: 999px;
    background: var(--blue);
    color: white;
    font-weight: 650;
    letter-spacing: -0.01em;
    box-shadow: 0 7px 18px rgba(0, 113, 227, 0.22);
    transition: transform 140ms ease, background 140ms ease, box-shadow 140ms ease;
}

[data-testid="stFormSubmitButton"] button:hover {
    background: var(--blue-dark);
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 9px 24px rgba(0, 113, 227, 0.28);
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stRadio"] {
    border-radius: 14px;
}

[data-testid="stAlert"] {
    border-radius: 18px;
    border-width: 1px;
}

[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid var(--line);
    border-radius: 18px;
    overflow: hidden;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.88);
    border-color: var(--line);
    border-radius: 20px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.04);
}

.apple-hero {
    position: relative;
    overflow: hidden;
    margin-bottom: 1.8rem;
    padding: clamp(1.7rem, 4vw, 3.4rem);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-radius: 32px;
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.98), rgba(242, 247, 255, 0.92));
    box-shadow: var(--shadow);
}

.apple-hero::after {
    content: "";
    position: absolute;
    width: 18rem;
    height: 18rem;
    right: -6rem;
    top: -9rem;
    border-radius: 50%;
    background: rgba(0, 113, 227, 0.10);
    filter: blur(2px);
}

.brand-row, .signal-title-row, .article-meta, .source-row {
    display: flex;
    align-items: center;
}

.brand-row {
    gap: 0.8rem;
    margin-bottom: 1rem;
}

.brand-logo {
    display: block;
    width: 2.8rem;
    height: 2.8rem;
    flex: 0 0 auto;
    border-radius: 13px;
    object-fit: contain;
}

.brand-copy {
    display: flex;
    flex-direction: column;
    gap: 0.08rem;
}

.brand-wordmark {
    color: var(--ink);
    font-size: 1.14rem;
    font-weight: 760;
    letter-spacing: -0.035em;
    line-height: 1;
}

.brand-mark {
    display: grid;
    width: 2.8rem;
    height: 2.8rem;
    place-items: center;
    border-radius: 13px;
    background: var(--blue);
    color: white;
    box-shadow: 0 9px 22px rgba(0, 113, 227, 0.25);
}

.brand-label, .eyebrow {
    color: var(--secondary-ink);
    font-size: 0.82rem;
    font-weight: 720;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.apple-hero h1 {
    position: relative;
    z-index: 1;
    max-width: 800px;
    margin: 0;
    color: var(--ink);
    font-size: clamp(2.25rem, 5vw, 4.4rem);
    font-weight: 720;
    letter-spacing: -0.055em;
    line-height: 0.98;
}

.apple-hero p {
    position: relative;
    z-index: 1;
    max-width: 700px;
    margin: 1rem 0 0;
    color: var(--secondary-ink);
    font-size: 1.14rem;
    line-height: 1.55;
}

.page-header {
    margin-bottom: 1.6rem;
    padding-bottom: 1.15rem;
    border-bottom: 1px solid var(--line);
}

.page-header h1 {
    margin: 0.28rem 0 0;
    color: var(--ink);
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 720;
    letter-spacing: -0.055em;
    line-height: 1;
}

.page-header p {
    max-width: 720px;
    margin: 0.6rem 0 0;
    color: var(--secondary-ink);
    font-size: 1.05rem;
    line-height: 1.5;
}

.workspace-bar {
    margin-bottom: 1.6rem;
    padding: 1rem 1.1rem;
    border: 1px solid var(--line);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.88);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.04);
}

.workspace-caption {
    margin-top: 0.35rem;
    color: var(--secondary-ink);
    font-size: 0.9rem;
}

.snapshot-price {
    margin-top: 0.4rem;
    color: var(--ink);
    font-size: clamp(2.3rem, 6vw, 4.5rem);
    font-weight: 730;
    letter-spacing: -0.06em;
    line-height: 0.95;
}

.snapshot-currency {
    margin-left: 0.35rem;
    color: var(--secondary-ink);
    font-size: 1rem;
    font-weight: 650;
    letter-spacing: 0;
}

.compare-product {
    min-height: 27rem;
    padding: 1.35rem 1.1rem 1.1rem;
    border-top: 1px solid rgba(0, 0, 0, 0.14);
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    background: rgba(255, 255, 255, 0.4);
}

.compare-product-title {
    min-height: 2.8rem;
    color: var(--ink);
    font-size: 1.12rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.25;
    text-align: center;
}

.compare-ticker {
    margin-top: 0.25rem;
    color: var(--secondary-ink);
    font-size: 0.88rem;
    text-align: center;
}

.compare-product-header {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 3.8rem;
}

.compare-product-heading {
    flex: 1 1 auto;
    min-width: 0;
}

.company-logo.compare-logo {
    width: 5.2rem;
    height: 5.2rem;
    margin: 0.85rem auto 0.65rem;
    border-radius: 20px;
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.09);
}

.company-logo.compare-logo img { padding: 0.8rem; }

.compare-icon {
    display: grid;
    width: 5.5rem;
    height: 5.5rem;
    margin: 1.35rem auto 0.8rem;
    place-items: center;
    border-radius: 26px;
    background: rgba(0, 113, 227, 0.07);
    color: var(--blue);
}

.compare-icon.positive { color: var(--green); background: var(--green-soft); }
.compare-icon.negative { color: var(--red); background: var(--red-soft); }
.compare-icon.neutral { color: #6e6e73; background: rgba(118, 118, 128, 0.10); }

.compare-price {
    color: var(--ink);
    font-size: 2.2rem;
    font-weight: 730;
    letter-spacing: -0.05em;
    text-align: center;
}

.compare-delta {
    margin-top: 0.32rem;
    font-size: 1rem;
    font-weight: 700;
    text-align: center;
}

.compare-delta.positive { color: var(--green); }
.compare-delta.negative { color: var(--red); }
.compare-delta.neutral { color: var(--secondary-ink); }

.compare-caption {
    margin: 0.38rem 0 0.7rem;
    color: var(--secondary-ink);
    font-size: 0.88rem;
    text-align: center;
}

.compare-stat-list {
    margin-top: 0.8rem;
    border-top: 1px solid rgba(0, 0, 0, 0.08);
}

.compare-stat-row {
    display: flex;
    justify-content: space-between;
    gap: 0.6rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    color: var(--secondary-ink);
    font-size: 0.9rem;
}

.compare-stat-row strong { color: var(--ink); font-weight: 680; }
.compare-stat-list.positive .semantic-color { color: var(--green); }
.compare-stat-list.negative .semantic-color { color: var(--red); }
.compare-stat-list.neutral .semantic-color { color: #6e6e73; }

.comparison-note {
    margin-top: 1.2rem;
    padding: 1rem 1.1rem;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.65);
    color: var(--secondary-ink);
    font-size: 1rem;
    line-height: 1.5;
}

.learn-panel {
    padding: 1.25rem 1.35rem;
    border-left: 3px solid var(--blue);
    background: rgba(0, 113, 227, 0.055);
    color: #3a3a3c;
    font-size: 1rem;
    line-height: 1.55;
}

.chat-intro {
    margin-bottom: 1rem;
    padding: 1rem 1.15rem;
    border: 1px solid rgba(0, 113, 227, 0.14);
    border-radius: 18px;
    background: rgba(0, 113, 227, 0.055);
    color: #3a3a3c;
    font-size: 1rem;
    line-height: 1.5;
}

.suggestion-button button {
    min-height: 2.5rem;
    border-radius: 999px;
}

.comparison-add-action-spacer {
    height: 1.72rem;
}

/* Keep the selector row visually separated from the company marks above it. */
.comparison-selector-gap {
    height: 1rem;
}

[class*="st-key-remove_ticker_action_"] {
    margin-top: -0.55rem !important;
}

[class*="st-key-comparison_add_ticker"] button {
    min-height: 2.85rem;
    height: 2.85rem;
    padding: 0 0.55rem;
    border-radius: 14px;
    border-color: rgba(36, 138, 61, 0.24);
    background: var(--green-soft);
    color: var(--green);
    font-size: 1.35rem;
    font-weight: 560;
    line-height: 1;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
}

[class*="st-key-comparison_add_ticker"] button:hover {
    border-color: rgba(36, 138, 61, 0.42);
    background: #d9f3e0;
    color: #1f7a35;
}

[class*="st-key-comparison_add_ticker"] button:disabled {
    color: var(--secondary-ink);
    background: rgba(0, 0, 0, 0.03);
}

.nav-footer {
    margin-top: 1.3rem;
    color: var(--secondary-ink);
    font-size: 0.82rem;
    line-height: 1.45;
}

.company-identity {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.7rem 0 1rem;
}

.company-identity.compact { margin-top: 0.35rem; }

.company-logo {
    position: relative;
    display: grid;
    flex: 0 0 auto;
    width: 3.2rem;
    height: 3.2rem;
    place-items: center;
    overflow: hidden;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 16px;
    background: white;
    color: var(--blue);
    font-size: 0.88rem;
    font-weight: 760;
    letter-spacing: -0.02em;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
}

.company-logo img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    padding: 0.35rem;
    background: white;
    object-fit: contain;
}

.selector-logo-slot {
    display: flex;
    min-height: 6rem;
    align-items: flex-end;
    justify-content: center;
    padding-bottom: 0.25rem;
}

.company-logo.selector-logo {
    width: 5.5rem;
    height: 5.5rem;
    border-radius: 18px;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}

.company-logo.selector-logo img { padding: 0.45rem; }

.company-name {
    color: var(--ink);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.company-ticker {
    margin-top: 0.22rem;
    color: var(--secondary-ink);
    font-size: 0.88rem;
}

.compare-trend-badge {
    display: grid;
    width: 2.45rem;
    height: 2.45rem;
    margin: 0 auto 0.5rem;
    place-items: center;
    border-radius: 13px;
}

.compare-trend-badge.positive { color: var(--green); background: var(--green-soft); }
.compare-trend-badge.negative { color: var(--red); background: var(--red-soft); }
.compare-trend-badge.neutral { color: #6e6e73; background: rgba(118, 118, 128, 0.10); }
.compare-trend-badge svg { width: 25px; height: 25px; }

.signal-card {
    height: 100%;
    min-height: 245px;
    padding: 1.5rem;
    border: 1px solid var(--line);
    border-radius: 26px;
    background: var(--surface);
    box-shadow: var(--shadow);
}

.signal-card.positive {
    background: linear-gradient(145deg, white, var(--green-soft));
}

.signal-card.negative {
    background: linear-gradient(145deg, white, var(--red-soft));
}

.signal-card.neutral {
    background: linear-gradient(145deg, white, #f3f4f6);
}

.signal-title-row {
    justify-content: space-between;
    gap: 0.8rem;
}

.source-pill, .tone-pill, .article-tone {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    border-radius: 999px;
    font-size: 0.84rem;
    font-weight: 680;
}

.source-pill {
    padding: 0.35rem 0.65rem;
    color: #3a3a3c;
    background: rgba(118, 118, 128, 0.12);
}

.source-pill.live::before {
    content: "";
    width: 0.42rem;
    height: 0.42rem;
    margin-right: 0.38rem;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 4px rgba(36, 138, 61, 0.12);
}

.source-pill.fallback { color: var(--amber); background: var(--amber-soft); }
.source-pill.unavailable { color: var(--red); background: var(--red-soft); }

.signal-primary {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.45rem;
}

.trend-icon {
    display: grid;
    flex: 0 0 auto;
    width: 4.6rem;
    height: 4.6rem;
    place-items: center;
    border-radius: 22px;
}

.positive .trend-icon { color: var(--green); background: rgba(36, 138, 61, 0.11); }
.negative .trend-icon { color: var(--red); background: rgba(215, 0, 21, 0.09); }
.neutral .trend-icon { color: #6e6e73; background: rgba(118, 118, 128, 0.10); }

.signal-value {
    color: var(--ink);
    font-size: clamp(2.25rem, 4vw, 3.5rem);
    font-weight: 730;
    letter-spacing: -0.05em;
    line-height: 1;
}

.positive .semantic-color { color: var(--green); }
.negative .semantic-color { color: var(--red); }
.neutral .semantic-color { color: #6e6e73; }

.signal-subtitle {
    margin-top: 0.4rem;
    color: var(--secondary-ink);
    font-size: 1rem;
}

.mini-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-top: 1.25rem;
}

.mini-stat {
    padding: 0.7rem;
    border: 1px solid rgba(0, 0, 0, 0.055);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.62);
}

.mini-stat span { display: block; color: var(--secondary-ink); font-size: 0.82rem; }
.mini-stat strong { color: var(--ink); font-size: 1.02rem; font-weight: 680; }

.sentiment-scale {
    position: relative;
    height: 0.52rem;
    margin-top: 1.2rem;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--red), #d1d1d6 50%, var(--green));
}

.sentiment-scale::after {
    content: "";
    position: absolute;
    left: var(--score-position);
    top: 50%;
    width: 1rem;
    height: 1rem;
    border: 3px solid white;
    border-radius: 50%;
    background: var(--marker-color);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.28);
    transform: translate(-50%, -50%);
}

.scale-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.45rem;
    color: var(--secondary-ink);
    font-size: 0.8rem;
}

.section-heading {
    margin: 2.2rem 0 1rem;
    color: var(--ink);
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
}

.tone-pill, .article-tone {
    padding: 0.3rem 0.6rem;
}

.tone-pill.positive, .article-tone.positive { color: var(--green); background: var(--green-soft); }
.tone-pill.negative, .article-tone.negative { color: var(--red); background: var(--red-soft); }
.tone-pill.neutral, .article-tone.neutral { color: #6e6e73; background: rgba(118, 118, 128, 0.11); }

.article-card-meta {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 1px minmax(9rem, auto);
    align-items: center;
    gap: 1rem;
    min-height: 8.6rem;
    padding: 1.05rem 1.15rem 0.95rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.07);
}

.article-source-block {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 0.3rem;
}

.article-source-label {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: #26384f;
    font-size: 0.9rem;
    font-weight: 470;
    letter-spacing: -0.02em;
}

.article-source-prefix { font-weight: 760; }

.article-source-line,
.article-website {
    display: inline-flex;
    align-items: center;
}

.article-source-line { gap: 0.4rem; }

.article-source-icon,
.article-website-icon {
    display: inline-grid;
    width: 1rem;
    height: 1rem;
    place-items: center;
    flex: 0 0 auto;
    color: #667383;
}

.article-source-icon svg,
.article-website-icon svg { width: 100%; height: 100%; }

.article-source-detail {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.45rem;
    margin-top: 0.65rem;
}

.article-website {
    gap: 0.42rem;
    color: var(--ink);
    font-size: 1.05rem;
    font-weight: 520;
    letter-spacing: -0.025em;
}

.article-date {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--secondary-ink);
    font-size: 0.86rem;
    letter-spacing: -0.015em;
}

.article-calendar-icon {
    display: inline-grid;
    width: 1.05rem;
    height: 1.05rem;
    place-items: center;
    color: #667383;
}

.article-calendar-icon svg { width: 100%; height: 100%; }

.article-meta-divider {
    width: 1px;
    height: 6.2rem;
    background: #d6dce5;
}

.article-status-stack {
    display: flex;
    flex: 0 0 auto;
    min-width: 8.8rem;
    flex-direction: column;
    align-items: stretch;
    gap: 0.45rem;
}

.article-status-stack .article-tone,
.article-confidence {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 2.2rem;
    padding: 0.35rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 520;
    white-space: nowrap;
}

.article-status-stack .article-tone {
    gap: 0.4rem;
    border: 1px solid rgba(215, 0, 21, 0.16);
}

.article-status-stack .article-tone.positive {
    border-color: rgba(36, 138, 61, 0.18);
}

.article-status-stack .article-tone.neutral {
    border-color: rgba(118, 118, 128, 0.18);
}

.article-status-icon,
.article-confidence-icon {
    display: inline-grid;
    width: 1.3rem;
    height: 1.3rem;
    place-items: center;
    flex: 0 0 auto;
    border-radius: 50%;
}

.article-status-icon {
    color: white;
    background: var(--red);
}

.article-tone.positive .article-status-icon { background: var(--green); }
.article-tone.neutral .article-status-icon { background: #7d8490; }

.article-status-icon svg { width: 0.95rem; height: 0.95rem; }

.article-confidence {
    justify-content: flex-start;
    gap: 0.4rem;
    border: 1px solid rgba(0, 113, 227, 0.18);
    color: #1768bd;
    background: rgba(0, 113, 227, 0.06);
}

.article-confidence-icon {
    color: #1768bd;
    border-radius: 0;
}

.article-confidence-icon svg { width: 1.15rem; height: 1.15rem; }

.article-body {
    padding: 0.95rem 1.15rem 1.05rem;
}

.article-body h3 {
    margin: 0;
    color: var(--ink);
    font-size: clamp(1.15rem, 1.7vw, 1.45rem);
    font-weight: 690;
    letter-spacing: -0.035em;
    line-height: 1.22;
}

.article-summary {
    max-width: 42rem;
    margin-top: 0.65rem;
    color: var(--secondary-ink);
    font-size: 0.9rem;
    line-height: 1.45;
}

[data-testid="stLinkButton"] {
    margin: 0 1.15rem 1.05rem;
}

[data-testid="stLinkButton"] {
    width: fit-content;
}

[data-testid="stLinkButton"] a {
    display: inline-flex;
    width: fit-content;
    align-items: center;
    justify-content: center;
    text-decoration: none;
}

[data-testid="stLinkButton"] a,
[data-testid="stLinkButton"] button {
    min-height: 2.2rem;
    padding: 0.42rem 0.72rem;
    border: 1px solid rgba(29, 29, 31, 0.16);
    border-radius: 0.7rem;
    color: var(--ink);
    background: rgba(255, 255, 255, 0.9);
    font-size: 0.82rem;
    line-height: 1.1;
    box-shadow: none;
}

[data-testid="stLinkButton"] a:hover,
[data-testid="stLinkButton"] button:hover {
    border-color: rgba(0, 113, 227, 0.45);
    color: var(--blue-dark);
    background: #fff;
}

@media (max-width: 560px) {
    .article-card-meta {
        grid-template-columns: 1fr;
        align-items: stretch;
        min-height: auto;
        padding: 1.2rem 1.2rem 1.1rem;
    }
    .article-meta-divider {
        width: 100%;
        height: 1px;
    }
    .article-status-stack { width: 100%; min-width: 0; flex-direction: row; }
    .article-status-stack > * { flex: 1 1 0; }
    .article-body { padding: 1.2rem; }
    [data-testid="stLinkButton"] { margin: 0 1.2rem 1.2rem; }
}

@media (max-width: 760px) {
    .apple-hero { border-radius: 24px; }
    .signal-card { min-height: auto; border-radius: 22px; }
    .mini-grid { grid-template-columns: 1fr; }
}
</style>
"""


def _inject_theme() -> None:
    # Keep the original light workspace appearance after restoring the
    # previous sidebar navigation.
    st.session_state.pop("dark_mode", None)
    dark_mode = False
    dark_overrides = ""
    background_uri = _market_background_data_uri()
    background_overrides = f"""
<style>
.stApp {{
    background-color: #e9edf3;
    background-image:
        linear-gradient(rgba(235, 239, 245, 0.78), rgba(235, 239, 245, 0.78)),
        radial-gradient(circle at 80% 0%, rgba(0, 113, 227, 0.055), transparent 30rem),
        url("{background_uri}");
    background-repeat: no-repeat, no-repeat, no-repeat;
    background-position: 0 0, 0 0, center center;
    background-size: 100% 100%, auto, cover;
    background-attachment: scroll, scroll, fixed;
}}

@media (max-width: 760px) {{
    .stApp {{
        background-position: 0 0, 0 0, center center;
        background-size: 100% 100%, auto, cover;
    }}
}}
</style>
"""
    if dark_mode:
        dark_overrides = """
<style>
:root {
    --ink: #f5f5f7;
    --secondary-ink: #aeb7c6;
    --surface: rgba(24, 32, 48, 0.94);
    --surface-muted: #111827;
    --line: rgba(255, 255, 255, 0.14);
    --blue: #5aa2ff;
    --blue-dark: #86baff;
    --green: #55c878;
    --green-soft: rgba(37, 138, 61, 0.2);
    --red: #ff6b7a;
    --red-soft: rgba(215, 0, 21, 0.2);
    --amber: #f0c36a;
    --amber-soft: rgba(154, 103, 0, 0.22);
    --shadow: 0 18px 50px rgba(0, 0, 0, 0.26);
}

.stApp {
    background:
        radial-gradient(circle at 80% 0%, rgba(67, 132, 224, 0.2), transparent 30rem),
        #0b1220;
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.97);
    border-right-color: rgba(255, 255, 255, 0.12);
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a {
    color: #b8c7dc;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
    background: rgba(52, 132, 255, 0.2);
    color: #8fc2ff;
}

[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
    background: rgba(52, 132, 255, 0.12);
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stWidgetLabel"] p,
.stApp label {
    color: var(--ink);
}

.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stCaptionContainer"] {
    color: var(--secondary-ink);
}

[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stForm"],
[data-testid="stExpander"] {
    background: rgba(24, 32, 48, 0.94);
    border-color: var(--line);
}

.comparison-note,
.chat-intro,
.learn-panel,
.apple-hero,
.signal-card {
    background: rgba(24, 32, 48, 0.9);
    border-color: var(--line);
    color: var(--ink);
}

.company-logo {
    background: #f5f5f7;
    border-color: rgba(255, 255, 255, 0.2);
}

[class*="st-key-comparison_add_ticker"] button {
    border-color: rgba(255, 255, 255, 0.16);
    background: rgba(24, 32, 48, 0.96);
    color: var(--blue);
}

[class*="st-key-comparison_add_ticker"] button:hover {
    background: rgba(52, 132, 255, 0.18);
    border-color: rgba(143, 194, 255, 0.5);
}
</style>
"""
    st.markdown(APPLE_THEME_CSS + background_overrides + dark_overrides, unsafe_allow_html=True)


def _graph_icon(direction: str) -> str:
    normalized = direction.casefold()
    if normalized in {"up", "positive"}:
        chart_path = "M4 18L9 13L13 16L21 7"
        arrow_path = "M15 7H21V13"
    elif normalized in {"down", "negative"}:
        chart_path = "M4 6L9 11L13 8L21 17"
        arrow_path = "M15 17H21V11"
    else:
        chart_path = "M4 12L9 10L13 14L20 11"
        arrow_path = "M17 9L20 11L17 14"
    return (
        '<svg width="42" height="42" viewBox="0 0 24 24" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<path d="{chart_path}" stroke="currentColor" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="{arrow_path}" stroke="currentColor" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
    )


@st.cache_data(show_spinner=False)
def _brand_logo_data_uri() -> str:
    logo_path = Path(__file__).resolve().parent / "assets" / "stocksense-logo.png"
    try:
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner=False)
def _market_background_data_uri() -> str:
    background_path = Path(__file__).resolve().parent / "assets" / "stocksense-background-d.png"
    try:
        encoded = base64.b64encode(background_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded}"


def _brand_icon() -> str:
    return (
        '<svg width="27" height="27" viewBox="0 0 24 24" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<path d="M3 17L8 12L12 15L20 7" stroke="currentColor" stroke-width="2.1" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M15 7H20V12" stroke="currentColor" stroke-width="2.1" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
    )


def _brand_logo_markup() -> str:
    logo_uri = _brand_logo_data_uri()
    if not logo_uri:
        return f'<div class="brand-mark">{_brand_icon()}</div>'
    return f'<img class="brand-logo" src="{logo_uri}" alt="StockSense logo" />'


def _tone_from_score(score: Any) -> tuple[str, str, str]:
    if score is None:
        return "neutral", "Unavailable", "#6e6e73"
    numeric_score = float(score)
    if numeric_score > 0.05:
        return "positive", "Positive tilt", "#248a3d"
    if numeric_score < -0.05:
        return "negative", "Negative tilt", "#d70015"
    return "neutral", "Balanced tone", "#6e6e73"


def _source_pill(source: str) -> str:
    return f'<span class="source-pill {html.escape(source)}">{html.escape(_source_label(source))}</span>'


def _stock_signal_presentation(result: PredictionResult) -> tuple[str, str, str, str]:
    """Return an Up/Down presentation while keeping validation status separate."""
    direction = result.prediction_label.casefold()
    direction = "down" if direction == "down" else "up"
    tone = "negative" if direction == "down" else "positive"
    if direction == "down":
        subtitle = (
            f"Up probability {result.probability_up:.1%} · below the {result.decision_threshold:.1%} Up cutoff"
            if result.probability_up >= 0.5
            else f"{1.0 - result.probability_up:.1%} probability of downward movement"
        )
    else:
        subtitle = f"{result.probability_up:.1%} probability of upward movement"
    return tone, "Down" if direction == "down" else "Up", direction, subtitle


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_company_logo(ticker: str, company_name: str) -> CompanyLogo | None:
    try:
        return get_company_logo(ticker, company_name)
    except ValueError:
        return None


def _company_logo_markup(ticker: str, company_name: str, variant: str = "") -> str:
    logo = _cached_company_logo(ticker, company_name)
    if logo is None:
        initials = html.escape(company_name[:2].upper())
        return f'<div class="company-logo {html.escape(variant)}"><span>{initials}</span></div>'
    primary_url = html.escape(logo.primary_url, quote=True)
    fallback_url = html.escape(logo.fallback_url, quote=True)
    initials = html.escape(logo.initials)
    alt_text = html.escape(f"{company_name} logo", quote=True)
    return (
        f'<div class="company-logo {html.escape(variant)}">'
        f'<span>{initials}</span>'
        f'<img src="{primary_url}" alt="{alt_text}" '
        f'onerror="this.onerror=null;this.src=\'{fallback_url}\';" />'
        "</div>"
    )


def _render_company_identity(ticker: str, company_name: str, variant: str = "") -> None:
    st.markdown(
        f"""
        <div class="company-identity {html.escape(variant)}">
            {_company_logo_markup(ticker, company_name)}
            <div>
                <div class="company-name">{html.escape(company_name)}</div>
                <div class="company-ticker">{html.escape(ticker)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_selector_logo(ticker: str, metadata: dict[str, str]) -> None:
    """Render a company logo outside the bordered selector surface."""
    st.markdown(
        f"<div class='selector-logo-slot'>{_company_logo_markup(ticker, metadata[ticker], 'selector-logo')}</div>",
        unsafe_allow_html=True,
    )


def _render_company_selector(
    label: str,
    tickers: list[str],
    metadata: dict[str, str],
    current_ticker: str,
    key: str,
) -> str:
    """Render the stable native company selector without dynamic logo content."""
    if current_ticker not in tickers:
        current_ticker = tickers[0]
    return st.selectbox(
        label,
        tickers,
        index=tickers.index(current_ticker),
        format_func=lambda value: f"{metadata[value]} ({value})",
        key=key,
    )


COMPARISON_MIN_TICKERS = 2
COMPARISON_MAX_TICKERS = 5
COMPARISON_STATE_VERSION = 1


def _comparison_values_from_state(
    tickers: list[str],
    default_slots: list[str],
) -> list[str]:
    stored = st.session_state.get("comparison_tickers")
    if isinstance(stored, list) and stored:
        candidate_values = list(stored[:COMPARISON_MAX_TICKERS])
    else:
        candidate_values = [
            st.session_state.get(f"comparison_slot_{index}", value)
            for index, value in enumerate(default_slots)
        ]

    values: list[str] = []
    for value in candidate_values:
        if value in tickers and value not in values:
            values.append(value)
    for value in default_slots:
        if len(values) >= COMPARISON_MIN_TICKERS:
            break
        if value in tickers and value not in values:
            values.append(value)
    return values[:COMPARISON_MAX_TICKERS]


def _set_comparison_values(values: list[str]) -> None:
    normalized = list(dict.fromkeys(values))[:COMPARISON_MAX_TICKERS]
    st.session_state["comparison_tickers"] = normalized
    for index, value in enumerate(normalized):
        st.session_state[f"comparison_slot_{index}"] = value
    for index in range(len(normalized), COMPARISON_MAX_TICKERS):
        st.session_state.pop(f"comparison_slot_{index}", None)


def _add_comparison_ticker(tickers: list[str]) -> None:
    values = list(st.session_state.get("comparison_tickers", []))
    if len(values) >= COMPARISON_MAX_TICKERS:
        return
    available = [ticker for ticker in tickers if ticker not in values]
    if available:
        _set_comparison_values(values + [available[0]])


def _remove_comparison_ticker(index: int) -> None:
    values = list(st.session_state.get("comparison_tickers", []))
    if index < COMPARISON_MIN_TICKERS or index >= len(values):
        return
    values.pop(index)
    _set_comparison_values(values)


def _comparison_option_records(
    tickers: list[str],
    metadata: dict[str, str],
    locked_values: set[str],
) -> list[dict[str, str | bool]]:
    """Build every option and mark selected values as truly disabled."""
    locked = set(locked_values)
    return [
        {
            "value": ticker,
            "label": f"{metadata[ticker]} ({ticker})",
            "locked": ticker in locked,
        }
        for ticker in tickers
    ]


def _comparison_format_func(
    metadata: dict[str, str],
    locked_values: set[str],
):
    locked = set(locked_values)

    def format_value(value: str) -> str:
        label = f"{metadata[value]} ({value})"
        return f"🔒 {label} · Selected" if value in locked else label

    return format_value


def _guard_comparison_selection(index: int) -> None:
    values = list(st.session_state.get("comparison_tickers", []))
    key = f"comparison_slot_{index}"
    candidate = st.session_state.get(key)
    if index >= len(values) or not isinstance(candidate, str):
        return
    if candidate in values[:index] or candidate in values[index + 1 :]:
        st.session_state[key] = values[index]
        return
    values[index] = candidate
    st.session_state["comparison_tickers"] = values


def _render_locked_company_selector(
    label: str,
    tickers: list[str],
    metadata: dict[str, str],
    current_ticker: str,
    locked_values: set[str],
    key: str,
) -> str:
    if current_ticker not in tickers:
        current_ticker = tickers[0]
    result = LOCKED_SELECTOR_COMPONENT(
        label=label,
        options=_comparison_option_records(tickers, metadata, locked_values),
        selected=current_ticker,
        default=current_ticker,
        height=76,
        key=key,
    )
    return result if isinstance(result, str) and result in tickers else current_ticker


@st.cache_data
def _load_company_metadata() -> dict[str, str]:
    return load_stock_metadata()


def _source_label(source: str) -> str:
    return {
        "live": "Live",
        "fallback": "Saved fallback",
        "unavailable": "Unavailable",
    }.get(source, source.title())


def _render_source_status(source: str, warning: str | None) -> None:
    if source == "fallback":
        st.warning(warning or "Showing saved fallback data.", icon="⚠️")
    elif source == "unavailable":
        st.error(warning or "This signal is unavailable.", icon="❌")


def _render_stock_signal(analysis: DashboardAnalysisResult) -> None:
    with st.container(border=True):
        _render_stock_signal_content(analysis)


def _render_stock_signal_content(analysis: DashboardAnalysisResult) -> None:
    st.markdown('<div class="section-heading">Stock Trend</div>', unsafe_allow_html=True)
    if analysis.stock is None:
        _render_source_status(analysis.stock_source, analysis.stock_warning)
        st.error("A stock-direction result is not available for this selection.")
        return

    result = analysis.stock
    tone, display_label, icon_direction, subtitle = _stock_signal_presentation(result)
    evidence_status = "Accepted" if result.directional_signal_available else "Experimental · research only"
    st.markdown(
        f"""
        <div class="signal-card {tone}">
            <div class="signal-title-row">
                <span class="eyebrow">Model outlook</span>
                {_source_pill(analysis.stock_source)}
            </div>
            <div class="signal-primary">
                <div class="trend-icon">{_graph_icon(icon_direction)}</div>
                <div>
                    <div class="signal-value semantic-color">{html.escape(display_label)}</div>
                    <div class="signal-subtitle">{html.escape(subtitle)}</div>
                </div>
            </div>
            <div class="mini-grid">
                <div class="mini-stat"><span>Market observation</span><strong>{html.escape(result.as_of_date)}</strong></div>
                <div class="mini-stat"><span>Horizon</span><strong>{result.horizon_days} trading day(s)</strong></div>
                <div class="mini-stat"><span>Evidence</span><strong>{evidence_status}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_source_status(analysis.stock_source, analysis.stock_warning)
    if result.stale_data_warning:
        st.warning(
            result.freshness_reason
            or "The market observation is not the latest valid trading session."
        )
    if not result.directional_signal_available:
        st.warning(
            f"This {result.horizon_days}-day model direction is experimental and is shown for research transparency. "
            "It did not pass the frozen acceptance benchmark.",
            icon="⚠️",
        )
    else:
        st.caption("This horizon passed the frozen acceptance benchmark.")

    if result.supporting_factors or result.opposing_factors:
        st.markdown('<div class="section-heading">Why this direction</div>', unsafe_allow_html=True)
        st.caption(
            "Local model sensitivity versus typical development-period values. "
            "These factors explain the model output; they do not establish cause."
        )
        support_column, oppose_column = st.columns(2)
        with support_column:
            st.markdown(f"**Supports {result.prediction_label}**")
            if result.supporting_factors:
                for factor in result.supporting_factors:
                    st.markdown(
                        f"- {factor['label']} · {factor['impact_probability_points']:.2f} pp"
                    )
            else:
                st.caption("No dominant supporting factor was identified.")
        with oppose_column:
            st.markdown(f"**Works against {result.prediction_label}**")
            if result.opposing_factors:
                for factor in result.opposing_factors:
                    st.markdown(
                        f"- {factor['label']} · {factor['impact_probability_points']:.2f} pp"
                    )
            else:
                st.caption("No dominant opposing factor was identified.")

    with st.expander("Stock model details"):
        details = {
            "Model": result.model_name,
            "Model version": result.model_version,
            "Decision threshold": f"{result.decision_threshold:.2f}",
            "Refinement round": str(result.refinement_round),
            "Registry status": result.status,
            "Threshold margin strength": result.evidence_strength,
        }
        for label, value in details.items():
            st.write(f"**{label}:** {value}")


def _format_score(score: Any) -> str:
    if score is None:
        return "Unavailable"
    return f"{float(score):+.2f}"


def _render_news_signal(analysis: DashboardAnalysisResult) -> None:
    with st.container(border=True):
        _render_news_signal_content(analysis)


def _render_news_signal_content(analysis: DashboardAnalysisResult) -> None:
    st.markdown('<div class="section-heading">News Sentiment</div>', unsafe_allow_html=True)
    if analysis.news is None:
        _render_source_status(analysis.news_source, analysis.news_warning)
        st.error("A news-sentiment result is not available for this selection.")
        return

    context = analysis.news.contexts.get(str(analysis.horizon_days))
    if context is None:
        st.error("No sentiment context was produced for this horizon.")
        return

    score = context["weighted_sentiment_score"]
    tone, tone_label, marker_color = _tone_from_score(score)
    score_position = 50.0 if score is None else min(100.0, max(0.0, (float(score) + 1.0) * 50.0))
    icon_direction = "neutral" if tone == "neutral" else tone
    st.markdown(
        f"""
        <div class="signal-card {tone}">
            <div class="signal-title-row">
                <span class="eyebrow">Weighted news tone</span>
                {_source_pill(analysis.news_source)}
            </div>
            <div class="signal-primary">
                <div class="trend-icon">{_graph_icon(icon_direction)}</div>
                <div>
                    <div class="signal-value semantic-color">{_format_score(score)}</div>
                    <div class="signal-subtitle"><span class="tone-pill {tone}">{tone_label}</span></div>
                </div>
            </div>
            <div class="sentiment-scale" style="--score-position: {score_position:.1f}%; --marker-color: {marker_color};"></div>
            <div class="scale-labels"><span>More negative</span><span>Balanced</span><span>More positive</span></div>
            <div class="mini-grid">
                <div class="mini-stat"><span>Coverage</span><strong>{html.escape(str(context['coverage_confidence']))}</strong></div>
                <div class="mini-stat"><span>Usable articles</span><strong>{context['usable_article_count']} of {context['candidate_article_count']}</strong></div>
                <div class="mini-stat"><span>Sources</span><strong>{context['unique_source_count']}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_source_status(analysis.news_source, analysis.news_warning)
    provider_metadata = analysis.news.provider_metadata or {}
    if bool(provider_metadata.get("marketaux_used")):
        trigger = str(provider_metadata.get("marketaux_trigger") or "limited NewsAPI coverage").replace("_", " ")
        cache_state = str(provider_metadata.get("marketaux_cache_state") or "live")
        deduplicated = int(provider_metadata.get("deduplicated_article_count", 0) or 0)
        st.info(
            "Additional Marketaux coverage was used because NewsAPI had "
            f"{trigger}. Marketaux data was {cache_state}; {deduplicated} duplicate article(s) were removed.",
            icon="ℹ️",
        )
    retrieval_days = int(
        provider_metadata.get("retrieval_lookback_days", NEWS_RETRIEVAL_LOOKBACK_DAYS)
    )
    st.caption(
        f"{retrieval_days}-day retrieval · {context['window_days']}-day context · "
        f"Positive: {context['positive_count']} · Neutral: {context['neutral_count']} · "
        f"Negative: {context['negative_count']}"
    )
    if score is None:
        if int(context.get("candidate_article_count", 0)) == 0:
            historical_count = int(provider_metadata.get("historical_article_count", 0) or 0)
            if historical_count:
                st.info(
                    f"No articles fell inside the selected {context['window_days']}-day sentiment window. "
                    f"{historical_count} older article(s) are shown below as background only and are not used in this tone."
                )
            else:
                st.info(
                    f"No articles fell inside the selected {context['window_days']}-day sentiment window. "
                    f"Related articles below may come from the wider {retrieval_days}-day retrieval window "
                    "and are not used in this tone."
                )
        else:
            st.info(
                f"Articles were found in the selected {context['window_days']}-day window, but none had "
                "confident enough sentiment to use in the tone."
            )


def _article_context_note(article: dict[str, Any], analysis: DashboardAnalysisResult) -> str | None:
    """Explain why a retrieved article may not contribute to the selected tone."""
    if analysis.news is None:
        return None
    context = analysis.news.contexts.get(str(analysis.horizon_days), {})
    window_days = int(context.get("window_days", 0) or 0)
    published_at = article.get("published_at")
    as_of_date = analysis.news.as_of_date
    if not published_at or not as_of_date or not window_days:
        return None
    try:
        age_days = article_age_days(str(published_at), pd.Timestamp(as_of_date).date())
    except (TypeError, ValueError):
        return None
    if not pd.notna(age_days) or age_days == float("inf"):
        return None
    if age_days > window_days:
        age_text = f"{age_days:.1f}".rstrip("0").rstrip(".")
        return (
            f"Retrieved {age_text} days ago; outside the selected {window_days}-day "
            "sentiment window, so it is not included in the tone."
        )
    if bool(article.get("sentiment_uncertain", True)) or article.get("sentiment_score") not in {-1, 0, 1}:
        return "Related article, but its sentiment was not confident enough to use in the tone."
    return "Included in the selected sentiment context."


def _render_article_card(
    article: dict[str, Any],
    context_note: str | None = None,
    *,
    show_tone: bool = True,
) -> None:
    title = html.escape(str(article.get("title") or "Untitled article"))
    provider = str(article.get("provider") or "newsapi").casefold()
    provider_label = {"newsapi": "NewsAPI", "marketaux": "Marketaux"}.get(provider, provider.title())
    source = html.escape(f"{article.get('source_name') or 'Unknown source'} · {provider_label}")
    published_at = html.escape(str(article.get("published_at") or "Unknown date"))
    sentiment = str(article.get("sentiment_label") or "neutral").casefold()
    article_tone = sentiment if sentiment in {"positive", "negative"} else "neutral"
    tone_markup = (
        f'<span class="article-tone {article_tone}">{html.escape(sentiment.title())}</span>'
        if show_tone
        else ""
    )
    description = html.escape(
        str(article.get("description") or article.get("content_snippet") or "")
    )
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="article-meta" style="justify-content:space-between; gap:0.75rem;">
                <span class="eyebrow">{source} · {published_at}</span>
                {tone_markup}
            </div>
            <div style="margin-top:0.7rem; color:#1d1d1f; font-size:1.1rem; font-weight:680; line-height:1.4;">
                {title}
            </div>
            {f'<div style="margin-top:0.55rem; color:#6e6e73; line-height:1.45;">{description}</div>' if description else ''}
            """,
            unsafe_allow_html=True,
        )
        if context_note:
            st.caption(context_note)
        url = str(article.get("url") or "")
        if url:
            st.link_button("Read article ↗", url)


def _format_article_timestamp(value: Any) -> str:
    """Format provider timestamps as a readable 12-hour date/time label."""
    # Preserve the timestamp's supplied timezone rather than silently
    # converting provider publication times to a different display zone.
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "Date unavailable"
    return timestamp.strftime("%d %b %Y · %I:%M %p")


def _article_provider_labels(article: dict[str, Any]) -> str:
    """Return provider provenance for the article metadata header."""
    values = article.get("provider_sources") or [article.get("provider") or "newsapi"]
    labels: list[str] = []
    provider_labels = {"newsapi": "NewsAPI", "marketaux": "Marketaux"}
    for value in values:
        key = str(value or "").casefold().strip()
        label = provider_labels.get(key, str(value or "").strip().title())
        if label and label not in labels:
            labels.append(label)
    return " / ".join(labels) or "Unknown"


def _shield_icon_svg() -> str:
    return (
        '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<path d="M12 3L19 6V11.2C19 15.7 16.2 19.2 12 21C7.8 19.2 5 15.7 5 11.2V6L12 3Z" '
        'stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>'
        '<path d="M8.5 12.1L10.8 14.3L15.7 9.5" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
    )


def _render_article_card_design_b(
    article: dict[str, Any],
    *,
    show_tone: bool = True,
) -> None:
    """Render the structured metadata article card selected by the user."""
    title = html.escape(str(article.get("title") or "Untitled article"))
    website = str(article.get("source_domain") or "").strip()
    if not website:
        website = urlparse(str(article.get("url") or "")).netloc
    website = website.removeprefix("www.") or str(article.get("source_name") or "Unknown website")
    provider_labels = html.escape(_article_provider_labels(article))
    website_markup = html.escape(website)
    published_at = html.escape(_format_article_timestamp(article.get("published_at")))
    sentiment = str(article.get("sentiment_label") or "neutral").casefold()
    article_tone = sentiment if sentiment in {"positive", "negative"} else "neutral"
    confidence = article.get("sentiment_probability")
    try:
        confidence_text = f"Confidence {float(confidence) * 100:.0f}%"
    except (TypeError, ValueError):
        confidence_text = "Confidence unavailable"
    description = html.escape(str(article.get("description") or article.get("content_snippet") or ""))
    status_markup = ""
    if show_tone:
        status_markup = f"""
            <div class="article-status-stack">
                <span class="article-tone {article_tone}">
                    <span class="article-status-icon">{_graph_icon("negative" if article_tone == "negative" else article_tone)}</span>
                    <span>{html.escape(sentiment.title())}</span>
                </span>
                <span class="article-confidence">
                    <span class="article-confidence-icon">{_shield_icon_svg()}</span>
                    <span>{html.escape(confidence_text)}</span>
                </span>
            </div>
        """
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="article-card-meta">
                <div class="article-source-block">
                    <div class="article-source-label">
                        <span class="article-source-icon">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                                <rect x="4" y="3.5" width="16" height="17" rx="2" stroke="currentColor" stroke-width="1.7"/>
                                <path d="M7.5 7.5H16.5M7.5 11.5H16.5M7.5 15.5H13" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
                            </svg>
                        </span>
                        <span class="article-source-line"><span class="article-source-prefix">Sources:</span> {provider_labels}</span>
                    </div>
                    <div class="article-source-detail">
                        <span class="article-website">
                            <span class="article-website-icon">
                                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                                    <circle cx="12" cy="12" r="8.5" stroke="currentColor" stroke-width="1.7"/>
                                    <path d="M3.8 12H20.2M12 3.5C14.1 5.8 15.2 8.65 15.2 12C15.2 15.35 14.1 18.2 12 20.5C9.9 18.2 8.8 15.35 8.8 12C8.8 8.65 9.9 5.8 12 3.5Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                </svg>
                            </span>
                            {website_markup}
                        </span>
                        <span class="article-date">
                            <span class="article-calendar-icon">
                                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                                    <rect x="3.5" y="5.5" width="17" height="15" rx="2" stroke="currentColor" stroke-width="1.8"/>
                                    <path d="M7 3.5V7.5M17 3.5V7.5M3.5 9.5H20.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                                </svg>
                            </span>
                            {published_at}
                        </span>
                    </div>
                </div>
                <div class="article-meta-divider"></div>
                {status_markup}
            </div>
            <div class="article-body">
                <h3>{title}</h3>
                {f'<div class="article-summary">{description}</div>' if description else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )
        url = str(article.get("url") or "")
        if url:
            st.link_button("Read article \u2197", url)


def _render_news_articles(analysis: DashboardAnalysisResult) -> None:
    if analysis.news is None:
        return
    _render_news_articles_content(analysis)


def _render_news_articles_content(analysis: DashboardAnalysisResult) -> None:
    st.markdown('<div class="section-heading">Relevant News Articles</div>', unsafe_allow_html=True)
    articles = list(analysis.news.articles or [])
    if not articles:
        st.info("No relevant articles were found in the retrieval window.")
    else:
        for start in range(0, len(articles), 2):
            columns = st.columns(2, gap="medium")
            for column, article in zip(columns, articles[start : start + 2]):
                with column:
                    _render_article_card_design_b(article)

    historical_articles = list(analysis.news.historical_articles or [])
    if historical_articles:
        st.markdown(
            '<div class="section-heading" style="margin-top:2rem;">Older Company Context</div>',
            unsafe_allow_html=True,
        )
        st.warning(
            f"No relevant article was found within the current {NEWS_RETRIEVAL_LOOKBACK_DAYS}-day sentiment window. "
            f"These articles are {NEWS_HISTORICAL_CONTEXT_START_DAYS}–{NEWS_HISTORICAL_CONTEXT_END_DAYS} days old and are shown only as background; "
            "they do not affect the News Sentiment result.",
            icon="⚠️",
        )
        for start in range(0, len(historical_articles), 2):
            columns = st.columns(2, gap="medium")
            for column, article in zip(columns, historical_articles[start : start + 2]):
                with column:
                    _render_article_card_design_b(article, show_tone=False)


def _render_interpretation(analysis: DashboardAnalysisResult) -> None:
    with st.expander("How to interpret this screen"):
        st.markdown(
            "The stock model and the news model are separate evidence sources. "
            "The application does not combine them into a recommendation or tell you to buy or sell."
        )
        st.markdown(
            "Stock probability is the model's estimated probability of an upward move for the selected "
            "horizon. News sentiment is a recency- and confidence-weighted score from -1 (more negative) "
            "to +1 (more positive). Low or unavailable coverage should be treated as weak evidence."
        )
        st.caption(
            "This is an educational decision-support prototype, not financial advice. "
            "Always verify the underlying information before making an investment decision."
        )


def _page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _remember_ticker(ticker: str) -> None:
    st.session_state["selected_ticker"] = ticker


def _default_ticker(metadata: dict[str, str]) -> str:
    selected = st.session_state.get("selected_ticker")
    return selected if selected in metadata else next(iter(metadata))


def _render_analysis_result(analysis: DashboardAnalysisResult) -> None:
    stock_column, news_column = st.columns(2)
    with stock_column:
        _render_stock_signal(analysis)
    with news_column:
        _render_news_signal(analysis)
    _render_news_articles(analysis)
    _render_interpretation(analysis)
    st.caption(f"Screen generated at {analysis.generated_at}.")


def _stock_outlook_page() -> None:
    metadata = _load_company_metadata()
    tickers = list(metadata)
    _page_header(
        "Analyze",
        "Stock Outlook",
        "See the modelled direction and recent news evidence for one Bursa Malaysia company.",
    )

    default_ticker = _default_ticker(metadata)
    selected_ticker = st.session_state.get("outlook_ticker", default_ticker)
    if selected_ticker not in tickers:
        selected_ticker = default_ticker
    selected_horizon = int(st.session_state.get("outlook_horizon", 1))
    logo_column, panel_column = st.columns([0.58, 4.42])
    with logo_column:
        _render_selector_logo(selected_ticker, metadata)
    with panel_column:
        with st.container(border=True):
            ticker_column, horizon_column, action_column = st.columns([2.2, 1.7, 1])
            with ticker_column:
                ticker = _render_company_selector(
                    "Company", tickers, metadata, selected_ticker, "outlook_ticker"
                )
            with horizon_column:
                horizon_days = st.radio(
                    "Prediction horizon",
                    [1, 5, 10],
                    index=[1, 5, 10].index(selected_horizon) if selected_horizon in [1, 5, 10] else 0,
                    format_func=lambda value: f"{value} trading day(s)",
                    horizontal=True,
                    key="outlook_horizon",
                )
            with action_column:
                st.markdown("<div style='height:1.72rem'></div>", unsafe_allow_html=True)
                submitted = st.button("Analyze", type="primary", use_container_width=True, key="outlook_analyze")
    _remember_ticker(ticker)

    if submitted:
        with st.spinner("Retrieving the latest stock and news signals..."):
            try:
                st.session_state["dashboard_analysis"] = run_dashboard_analysis(ticker, int(horizon_days))
                st.session_state.pop("dashboard_error", None)
            except Exception:
                st.session_state.pop("dashboard_analysis", None)
                st.session_state["dashboard_error"] = "The selected analysis could not be started."

    error = st.session_state.get("dashboard_error")
    if error:
        st.error(error)

    analysis = st.session_state.get("dashboard_analysis")
    if not isinstance(analysis, DashboardAnalysisResult):
        return
    if analysis.ticker != ticker or analysis.horizon_days != int(horizon_days):
        return
    _render_analysis_result(analysis)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_market_snapshot(ticker: str) -> MarketSnapshot:
    return fetch_market_snapshot(ticker)


def _format_price(value: float | None) -> str:
    return "Unavailable" if value is None else f"RM {value:,.2f}"


def _format_percent(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:+.2%}"


def _render_snapshot(snapshot: MarketSnapshot) -> None:
    tone = "positive" if (snapshot.change_pct or 0) > 0 else "negative" if (snapshot.change_pct or 0) < 0 else "neutral"
    st.markdown(
        f"""
        <div class="signal-card {tone}">
            <div class="signal-title-row">
                <span class="eyebrow">Latest available price</span>
                <span class="source-pill live">Daily close</span>
            </div>
            <div class="snapshot-price">{_format_price(snapshot.price)}<span class="snapshot-currency">MYR</span></div>
            <div class="signal-subtitle">Observation date {html.escape(snapshot.as_of_date)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if snapshot.stale_data_warning:
        st.warning("The latest available market observation is more than five days old.")
    st.caption("This is a daily market snapshot, not a guaranteed real-time quote. Verify the timestamp before acting.")

    change_column, range_column, volume_column = st.columns(3)
    change_column.metric("Daily change", _format_price(snapshot.change), _format_percent(snapshot.change_pct))
    range_text = "Unavailable" if snapshot.day_low is None or snapshot.day_high is None else f"RM {snapshot.day_low:,.2f} – RM {snapshot.day_high:,.2f}"
    range_column.metric("Day range", range_text)
    volume_column.metric("Volume", "Unavailable" if snapshot.volume is None else f"{snapshot.volume:,}")

    history = pd.DataFrame(
        {"Close": snapshot.history_prices},
        index=pd.to_datetime(snapshot.history_dates),
    )
    st.markdown('<div class="section-heading">Recent closing prices</div>', unsafe_allow_html=True)
    st.line_chart(history, height=250, color="#0071e3")


def _market_snapshot_page() -> None:
    metadata = _load_company_metadata()
    tickers = list(metadata)
    _page_header(
        "Analyze",
        "Market Snapshot",
        "Check the latest available daily price, range, volume and recent closing-price trend.",
    )
    default_ticker = _default_ticker(metadata)
    selected_ticker = st.session_state.get("snapshot_ticker", default_ticker)
    logo_column, panel_column = st.columns([0.58, 4.42])
    with logo_column:
        _render_selector_logo(selected_ticker, metadata)
    with panel_column:
        with st.container(border=True):
            ticker = _render_company_selector(
                "Company", tickers, metadata, selected_ticker, "snapshot_ticker"
            )
            st.caption(
                "The snapshot updates automatically whenever you choose a company. "
                "Cached data may be reused for up to five minutes."
            )
    _remember_ticker(ticker)
    with st.spinner("Loading the latest available market snapshot..."):
        try:
            snapshot = _cached_market_snapshot(ticker)
        except Exception:
            snapshot = None
    if not isinstance(snapshot, MarketSnapshot) or snapshot.ticker != ticker:
        st.error("The market snapshot could not be loaded for this company.")
        return
    _render_snapshot(snapshot)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_prediction(ticker: str, horizon_days: int) -> PredictionResult:
    return predict_stock_trend(ticker, horizon_days)


def _comparison_detail(ticker: str, horizon_days: int) -> dict[str, Any]:
    try:
        return {
            "snapshot": _cached_market_snapshot(ticker),
            "prediction": _cached_prediction(ticker, horizon_days),
            "error": None,
        }
    except Exception as exc:
        return {"snapshot": None, "prediction": None, "error": str(exc)}


def _render_comparison_product(detail: dict[str, Any]) -> None:
    snapshot = detail.get("snapshot")
    prediction = detail.get("prediction")
    if not isinstance(snapshot, MarketSnapshot) or not isinstance(prediction, PredictionResult):
        st.error("This ticker could not be loaded.")
        return

    tone, display_label, icon_direction, _ = _stock_signal_presentation(prediction)
    series = pd.Series(snapshot.history_prices, dtype=float)
    period_return = (series.iloc[-1] / series.iloc[0] - 1.0) if len(series) > 1 and series.iloc[0] else None
    volatility = series.pct_change().dropna().std() * (252**0.5) if len(series) > 2 else None
    st.markdown(
        f"""
        <div class="compare-product">
            <div class="compare-product-header">
                <div class="compare-product-heading">
                    <div class="compare-product-title">{html.escape(snapshot.company_name)}</div>
                    <div class="compare-ticker">{html.escape(snapshot.ticker)} · {html.escape(snapshot.as_of_date)}</div>
                </div>
            </div>
            {_company_logo_markup(snapshot.ticker, snapshot.company_name, "compare-logo")}
            <div class="compare-trend-badge {tone}">{_graph_icon(icon_direction)}</div>
            <div class="compare-price">RM {snapshot.price:,.2f}</div>
            <div class="compare-delta {tone}">{_format_percent(snapshot.change_pct)} today</div>
            <div class="compare-caption">Latest available daily close</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sparkline = pd.DataFrame({"Close": snapshot.history_prices})
    st.line_chart(sparkline, height=135, color="#0071e3")
    st.markdown(
        f"""
        <div class="compare-stat-list {tone}">
            <div class="compare-stat-row"><span>30-observation return</span><strong>{_format_percent(period_return)}</strong></div>
            <div class="compare-stat-row"><span>Annualized volatility</span><strong>{_format_percent(volatility)}</strong></div>
            <div class="compare-stat-row"><span>Model outlook</span><strong class="semantic-color">{html.escape(display_label)}</strong></div>
            <div class="compare-stat-row"><span>Probability Up</span><strong>{prediction.probability_up:.1%}</strong></div>
            <div class="compare-stat-row"><span>Evidence</span><strong>{html.escape('Accepted' if prediction.directional_signal_available else 'Research only')}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _compare_tickers_page() -> None:
    metadata = _load_company_metadata()
    tickers = list(metadata)
    _page_header(
        "Analyze",
        "Compare Tickers",
        "Compare a small group of companies using the same market and model evidence.",
    )
    if st.session_state.get("comparison_state_version") != COMPARISON_STATE_VERSION:
        st.session_state.pop("comparison_tickers", None)
        for index in range(COMPARISON_MAX_TICKERS):
            st.session_state.pop(f"comparison_slot_{index}", None)
        st.session_state["comparison_state_version"] = COMPARISON_STATE_VERSION
    default_ticker = _default_ticker(metadata)
    default_slots = [default_ticker] + [
        ticker for ticker in tickers if ticker != default_ticker
    ][: COMPARISON_MIN_TICKERS - 1]
    selected_tickers = _comparison_values_from_state(tickers, default_slots)
    st.session_state["comparison_tickers"] = selected_tickers

    # Keep the company columns aligned across logos, selectors and result cards.
    # The add action is intentionally outside the bordered selector panel so it
    # stays visually separate and never changes the panel's column composition.
    comparison_column_widths = [1.0] * len(selected_tickers) + [0.42]
    logo_columns = st.columns(comparison_column_widths)
    for logo_column, current_value in zip(logo_columns, selected_tickers):
        with logo_column:
            _render_selector_logo(current_value, metadata)

    st.markdown('<div class="comparison-selector-gap"></div>', unsafe_allow_html=True)

    slot_values: list[str] = []
    selector_panel, add_action = st.columns(
        [sum(comparison_column_widths[:-1]), comparison_column_widths[-1]],
        gap="small",
    )
    with selector_panel:
        with st.container(border=True):
            selector_columns = st.columns(len(selected_tickers))
            for index, selector_column in enumerate(selector_columns):
                with selector_column:
                    current_value = selected_tickers[index]
                    locked_values = set(slot_values) | set(selected_tickers[index + 1 :])
                    selected_value = _render_locked_company_selector(
                        f"Ticker {index + 1}",
                        tickers,
                        metadata,
                        current_value,
                        locked_values,
                        f"comparison_slot_{index}",
                    )
                    if selected_value in locked_values:
                        selected_value = current_value
                    slot_values.append(selected_value)
                    if index >= COMPARISON_MIN_TICKERS:
                        with st.container(key=f"remove_ticker_action_{index}"):
                            st.button(
                                "Remove",
                                key=f"comparison_remove_{index}",
                                on_click=_remove_comparison_ticker,
                                args=(index,),
                                use_container_width=True,
                            )

            st.session_state["comparison_tickers"] = slot_values

            horizon_days = st.radio(
                "Prediction horizon",
                [1, 5, 10],
                format_func=lambda value: f"{value} trading day(s)",
                horizontal=True,
                key="comparison_horizon",
            )

    with add_action:
        st.markdown('<div class="comparison-add-action-spacer"></div>', unsafe_allow_html=True)
        st.button(
            "+",
            key="comparison_add_ticker",
            on_click=_add_comparison_ticker,
            args=(tickers,),
            disabled=len(selected_tickers) >= COMPARISON_MAX_TICKERS,
            use_container_width=True,
            help="Add another ticker to the comparison",
        )

    unique_slots = list(dict.fromkeys(slot_values))
    with st.spinner("Updating the comparison..."):
        details = [_comparison_detail(ticker, int(horizon_days)) for ticker in unique_slots]
    if all(detail.get("snapshot") is None for detail in details):
        st.error("None of the selected tickers could be loaded.")
        return

    product_columns = st.columns(comparison_column_widths)
    for product_column, detail in zip(product_columns, details):
        with product_column:
            _render_comparison_product(detail)

    valid_details = [detail for detail in details if isinstance(detail.get("snapshot"), MarketSnapshot)]
    if len(valid_details) >= 2:
        normalized_history: dict[str, list[float]] = {}
        for detail in valid_details:
            snapshot = detail["snapshot"]
            prices = pd.Series(snapshot.history_prices, dtype=float)
            if not prices.empty and prices.iloc[0] != 0:
                normalized_history[snapshot.ticker] = [float(value / prices.iloc[0] * 100.0) for value in prices]
        if normalized_history:
            st.markdown('<div class="section-heading">Relative performance</div>', unsafe_allow_html=True)
            st.caption("Each line starts at 100 for the first displayed observation, so the shape—not the share price—can be compared.")
            st.line_chart(pd.DataFrame(normalized_history), height=320)

    st.markdown(
        '<div class="comparison-note"><strong>How to read this view:</strong> use the price, return, volatility and model evidence together. A higher return or probability is not automatically a better choice, and experimental models deserve extra caution.</div>',
        unsafe_allow_html=True,
    )


def _news_page() -> None:
    metadata = _load_company_metadata()
    tickers = list(metadata)
    _page_header(
        "Analyze",
        "News & Sentiment",
        "Read the supporting articles and understand how sentiment coverage affects the signal.",
    )
    default_ticker = _default_ticker(metadata)
    selected_ticker = st.session_state.get("news_ticker", default_ticker)
    logo_column, panel_column = st.columns([0.58, 4.42])
    with logo_column:
        _render_selector_logo(selected_ticker, metadata)
    with panel_column:
        with st.container(border=True):
            ticker = _render_company_selector(
                "Company", tickers, metadata, selected_ticker, "news_ticker"
            )
            horizon_days = st.radio("Context horizon", [1, 5, 10], horizontal=True, key="news_horizon")
            refresh_news = st.button("Refresh news", type="primary", key="news_refresh")
    _remember_ticker(ticker)
    if refresh_news:
        with st.spinner("Retrieving recent company news and sentiment..."):
            try:
                st.session_state["news_result"] = retrieve_news_with_marketaux_fallback(ticker, horizon_days)
                st.session_state.pop("news_error", None)
            except Exception:
                st.session_state.pop("news_result", None)
                st.session_state["news_error"] = "News could not be retrieved. Check the NewsAPI configuration."
    if st.session_state.get("news_error"):
        st.error(st.session_state["news_error"])
    news = st.session_state.get("news_result")
    if isinstance(news, NewsContextResult) and int(
        (news.provider_metadata or {}).get("news_context_config_version", 0) or 0
    ) != NEWS_CONTEXT_CONFIG_VERSION:
        st.session_state.pop("news_result", None)
        news = None
    if not isinstance(news, NewsContextResult) or news.ticker != ticker:
        st.info("Select a company and refresh news to load the sentiment context.")
        return
    analysis = DashboardAnalysisResult(
        ticker=news.ticker,
        company_name=news.company_name,
        horizon_days=int(horizon_days),
        stock=None,
        stock_source="unavailable",
        stock_captured_at=None,
        stock_warning=None,
        news=news,
        news_source=("fallback" if news.provider_metadata.get("retrieval_source") == "fallback" else "live"),
        news_captured_at=news.retrieved_at,
        news_warning=None,
        generated_at=news.retrieved_at,
    )
    _render_news_signal(analysis)
    source_filter = st.multiselect(
        "Filter sources",
        sorted({str(article.get("source_name")) for article in news.articles if article.get("source_name")}),
        key="news_sources",
    )
    sentiment_filter = st.multiselect("Filter sentiment", ["positive", "neutral", "negative"], key="news_sentiment")
    filtered_articles = [
        article
        for article in news.articles
        if (not source_filter or article.get("source_name") in source_filter)
        and (not sentiment_filter or article.get("sentiment_label") in sentiment_filter)
    ]
    filtered_news = NewsContextResult(
        ticker=news.ticker,
        company_name=news.company_name,
        query_text=news.query_text,
        as_of_date=news.as_of_date,
        retrieved_at=news.retrieved_at,
        raw_article_count=news.raw_article_count,
        relevant_article_count=len(filtered_articles),
        results_truncated=news.results_truncated,
        contexts=news.contexts,
        articles=filtered_articles,
        output_paths=news.output_paths,
        provider_mode=news.provider_mode,
        provider_article_counts=news.provider_article_counts,
        provider_status=news.provider_status,
        provider_metadata=news.provider_metadata,
        historical_articles=news.historical_articles,
    )
    _render_news_articles(
        DashboardAnalysisResult(
            ticker=analysis.ticker,
            company_name=analysis.company_name,
            horizon_days=analysis.horizon_days,
            stock=None,
            stock_source="unavailable",
            stock_captured_at=None,
            stock_warning=None,
            news=filtered_news,
            news_source=("fallback" if filtered_news.provider_metadata.get("retrieval_source") == "fallback" else "live"),
            news_captured_at=news.retrieved_at,
            news_warning=None,
            generated_at=news.retrieved_at,
        )
    )


ASK_SYSTEM_PROMPT = """
You are StockSense's educational Ask & Learn tutor. Answer beginner questions in clear, friendly language.
Use the supplied StockSense context as evidence, distinguish model outputs from facts, and mention timestamps or missing data when relevant.
Never give personalized financial advice and never tell the user to buy, sell, or hold a security.
If the supplied context is missing or insufficient, say so instead of inventing values. Explain one concept at a time, use short sections or bullets, and stay under 350 words.
""".strip()


def _ask_grounding_context(ticker: str, horizon_days: int) -> str:
    lines = [
        f"Selected company: {ticker}",
        f"Selected prediction horizon: {horizon_days} trading day(s)",
    ]
    analysis = st.session_state.get("dashboard_analysis")
    if not isinstance(analysis, DashboardAnalysisResult) or analysis.ticker != ticker or analysis.horizon_days != int(horizon_days):
        lines.append("No matching Stock Outlook analysis is loaded in this session. Do not invent stock or news values.")
        return "\n".join(lines)

    lines.append(f"Analysis generated at: {analysis.generated_at}")
    if analysis.stock is not None:
        stock = analysis.stock
        validation = "accepted" if stock.directional_signal_available else "experimental"
        stock_context = (
            f"direction={stock.prediction_label}; probability_up={stock.probability_up:.3f}; "
            f"decision_threshold={stock.decision_threshold:.3f}"
        )
        lines.append(
            "Stock signal: "
            f"{stock_context}; "
            f"as_of={stock.as_of_date}; validation={validation}; source={analysis.stock_source}."
        )
        supporting = ", ".join(factor["label"] for factor in stock.supporting_factors) or "none identified"
        opposing = ", ".join(factor["label"] for factor in stock.opposing_factors) or "none identified"
        lines.append(
            "Local model reasons (sensitivity versus development medians, not causal): "
            f"supports={supporting}; opposes={opposing}; "
            f"threshold_margin_strength={stock.evidence_strength}."
        )
    else:
        lines.append("Stock signal: unavailable.")

    if analysis.news is not None:
        context = analysis.news.contexts.get(str(horizon_days), {})
        score = context.get("weighted_sentiment_score")
        score_text = "unavailable" if score is None else f"{float(score):+.2f}"
        provider_metadata = analysis.news.provider_metadata or {}
        lines.append(
            "News context: "
            f"weighted_score={score_text}; coverage={context.get('coverage_confidence', 'Unavailable')}; "
            f"usable_articles={context.get('usable_article_count', 0)}; "
            f"context_window_days={context.get('window_days', 'unknown')}; "
            f"retrieval_lookback_days={provider_metadata.get('retrieval_lookback_days', NEWS_RETRIEVAL_LOOKBACK_DAYS)}; "
            f"historical_background_articles={provider_metadata.get('historical_article_count', 0)} "
            "(excluded from sentiment); "
            f"source={analysis.news_source}."
        )
    else:
        lines.append("News context: unavailable.")
    return "\n".join(lines)


def _ask_chat_messages(ticker: str, horizon_days: int) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": f"{ASK_SYSTEM_PROMPT}\n\nStockSense context:\n{_ask_grounding_context(ticker, horizon_days)}"}]
    for message in st.session_state.get("ask_messages", []):
        if message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str):
            messages.append({"role": message["role"], "content": message["content"]})
    return messages


def _ask_learn_page() -> None:
    metadata = _load_company_metadata()
    tickers = list(metadata)
    _page_header(
        "Learn",
        "Ask & Learn",
        "A conversational space for understanding the model, the data and investing vocabulary.",
    )
    st.markdown(
        f'<div class="chat-intro"><strong>Ask StockSense.</strong><br>Gemma 4 answers in a beginner-friendly conversational style using the selected ticker, horizon and any loaded dashboard evidence.</div>',
        unsafe_allow_html=True,
    )
    if get_openrouter_api_key():
        st.caption(f"Primary: {OPENROUTER_MODEL} · fallback: {OPENROUTER_FALLBACK_MODEL}")
    else:
        st.warning("OpenRouter is not configured. Add a key to config/openrouter_api_key.txt to enable Ask & Learn responses.")
    default_ticker = _default_ticker(metadata)
    selected_ticker = st.session_state.get("ask_ticker", default_ticker)
    logo_column, panel_column = st.columns([0.58, 4.42])
    with logo_column:
        _render_selector_logo(selected_ticker, metadata)
    with panel_column:
        with st.container(border=True):
            context_column, horizon_column, clear_column = st.columns([2.2, 1.6, 1])
            with context_column:
                ticker = _render_company_selector(
                    "Conversation ticker",
                    tickers,
                    metadata,
                    selected_ticker,
                    "ask_ticker",
                )
            with horizon_column:
                horizon_days = st.selectbox("Conversation horizon", [1, 5, 10], key="ask_horizon")
            with clear_column:
                st.markdown("<div style='height:1.72rem'></div>", unsafe_allow_html=True)
                if st.button("Clear chat", key="ask_clear"):
                    st.session_state.pop("ask_messages", None)
                    st.rerun()
    _remember_ticker(ticker)

    if "ask_messages" not in st.session_state:
        st.session_state["ask_messages"] = [
            {
                "role": "assistant",
                "content": "Hi — I’m StockSense’s tutor. I can explain probabilities, sentiment, data freshness and model limitations. What would you like to learn?",
            }
        ]
    for message in st.session_state["ask_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.caption("Suggested questions")
    suggestion_columns = st.columns(3)
    suggestions = [
        "What does probability of Up mean?",
        "Why can news sentiment be unavailable?",
        "What does experimental model mean?",
    ]
    for index, suggestion in enumerate(suggestions):
        with suggestion_columns[index]:
            if st.button(suggestion, key=f"ask_suggestion_{index}", use_container_width=True):
                st.session_state["ask_pending_prompt"] = suggestion
                st.rerun()

    prompt = st.chat_input("Ask about the prediction, news or investing terms...")
    pending_prompt = st.session_state.pop("ask_pending_prompt", None)
    prompt = prompt or pending_prompt
    if prompt:
        st.session_state["ask_messages"].append({"role": "user", "content": prompt})
        try:
            with st.spinner("StockSense is thinking..."):
                answer = complete_chat(_ask_chat_messages(ticker, int(horizon_days)))
                response = answer.content
                if answer.model != OPENROUTER_MODEL:
                    response = f"_Answered by fallback model `{answer.model}`._\n\n{response}"
        except OpenRouterConfigurationError as exc:
            response = f"**OpenRouter is not connected yet.**\n\n{exc}"
        except OpenRouterError as exc:
            response = f"**Gemma 4 could not answer right now.**\n\n{exc}"
        st.session_state["ask_messages"].append({"role": "assistant", "content": response})
        st.rerun()


def _learning_center_page() -> None:
    _page_header(
        "Learn",
        "Learning Center",
        "Short explanations for reading the dashboard with appropriate caution.",
    )
    with st.expander("How to read a Stock Outlook", expanded=True):
        st.markdown(
            "The direction label is produced from the model probability and its locked decision threshold. "
            "A green or red label communicates the model output, not certainty or a recommendation."
        )
    with st.expander("What does the news sentiment score mean?"):
        st.markdown(
            "The score ranges from -1 (more negative) to +1 (more positive). Recent, higher-confidence articles receive more weight. "
            "Coverage and source diversity matter: an extreme score from very few articles is weak evidence."
        )
    with st.expander("Accepted versus experimental"):
        st.markdown(
            "Accepted horizons passed the project's frozen 2025 holdout rule. Experimental horizons remain visible for transparency, "
            "but should not be treated as equally validated."
        )
    with st.expander("Why data freshness matters"):
        st.markdown(
            "Market and news data can be delayed, incomplete or unavailable. Every page should show an observation or retrieval timestamp. "
            "A saved fallback result is clearly labeled so it is not mistaken for current information."
        )
    st.markdown('<div class="section-heading">A beginner workflow</div>', unsafe_allow_html=True)
    st.markdown(
        "1. Start with **Stock Outlook** and choose one company.\n"
        "2. Check the model status and observation date.\n"
        "3. Open **News & Sentiment** to inspect supporting articles.\n"
        "4. Use **Compare Tickers** to understand differences, not to chase a winner.\n"
        "5. Ask **Ask & Learn** to clarify any unfamiliar term."
    )


def _overview_page() -> None:
    metadata = _load_company_metadata()
    _page_header(
        "Start",
        "Overview",
        "A calm starting point for learning how market data, model evidence and news context fit together.",
    )
    st.markdown(
        f"""
        <div class="apple-hero">
            <div class="brand-row">
                {_brand_logo_markup()}
                <div class="brand-copy">
                    <div class="brand-wordmark">StockSense</div>
                </div>
            </div>
            <h1>Start with one company.</h1>
            <p>Use a small, explainable workflow: inspect the outlook, verify the evidence, then learn why the signals look the way they do.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Supported companies", len(metadata))
    metric_columns[1].metric("Prediction horizons", "1 / 5 / 10 days")
    metric_columns[2].metric("Separate evidence streams", "2")
    st.markdown('<div class="section-heading">Recommended path</div>', unsafe_allow_html=True)
    path_columns = st.columns(3)
    path_columns[0].markdown("**1 · Stock Outlook**\n\nChoose a ticker and horizon to see the model output.")
    path_columns[1].markdown("**2 · News & Sentiment**\n\nRead the articles and check coverage before interpreting the tone.")
    path_columns[2].markdown("**3 · Ask & Learn**\n\nUse the conversation template to clarify concepts in plain language.")
    st.markdown('<div class="learn-panel"><strong>Educational boundary:</strong> this prototype is decision support, not personalized financial advice. Use timestamps, evidence status and source coverage to keep your interpretation grounded.</div>', unsafe_allow_html=True)


def _alerts_page() -> None:
    _page_header(
        "Workspace",
        "Alerts",
        "A calm place for future reminders about market observations and model evidence.",
    )
    with st.container(border=True):
        st.markdown('<div class="section-heading">Alerts are coming next</div>', unsafe_allow_html=True)
        st.write(
            "Alert rules are not active in this research prototype yet. The planned design will let you be notified when a selected ticker has a new observation, a material direction change or fresh news coverage."
        )
        st.caption("No alerts are currently configured.")


def _reports_page() -> None:
    metadata = _load_company_metadata()
    _page_header(
        "Workspace",
        "Reports",
        "Review the evidence behind StockSense outputs before drawing conclusions.",
    )
    report_columns = st.columns(3)
    report_columns[0].metric("Companies covered", len(metadata))
    report_columns[1].metric("Horizon views", "1 / 5 / 10 days")
    report_columns[2].metric("Evidence streams", "Market + news")
    with st.container(border=True):
        st.markdown('<div class="section-heading">Report workspace</div>', unsafe_allow_html=True)
        st.write(
            "Exportable reports are planned for a later iteration. For now, use Stock Outlook, News & Sentiment and Compare to inspect the live evidence and timestamps directly."
        )


def _settings_page() -> None:
    _page_header(
        "System",
        "Settings",
        "Adjust the workspace appearance and review the data-source boundaries.",
    )
    with st.container(border=True):
        st.markdown('<div class="section-heading">Appearance</div>', unsafe_allow_html=True)
        st.write("StockSense currently uses a light workspace theme designed for clear reading of model evidence and charts.")
    with st.container(border=True):
        st.markdown('<div class="section-heading">Data sources</div>', unsafe_allow_html=True)
        st.write("Market data is sourced from Yahoo Finance and news coverage uses NewsAPI when configured. Each page surfaces timestamps and fallback states.")


def _help_page() -> None:
    _page_header(
        "Support",
        "Help",
        "Learn how to move through StockSense and interpret its evidence carefully.",
    )
    with st.expander("Where should I start?", expanded=True):
        st.write("Open Stock Outlook, select a company and prediction horizon, then choose Analyze to load the current market and news evidence.")
    with st.expander("How should I read a trend result?"):
        st.write("Treat the trend as a model signal rather than a guarantee. Check the observation date, evidence status and probability together.")
    with st.expander("Why is some information unavailable?"):
        st.write("Market and news providers can be delayed, rate-limited or missing coverage. StockSense labels unavailable or research-only evidence instead of filling gaps with guesses.")
    with st.expander("Where can I learn the investing terms?"):
        st.write("Use Ask & Learn for conversational explanations, or open Learning Center for short guided notes about probabilities, sentiment and model limitations.")


def main() -> None:
    _inject_theme()
    overview_page = st.Page(
        _overview_page,
        title="Overview",
        icon=":material/home:",
        url_path="overview",
        default=True,
    )
    stock_outlook_page = st.Page(
        _stock_outlook_page,
        title="Stock Outlook",
        icon=":material/insights:",
        url_path="stock-outlook",
    )
    market_snapshot_page = st.Page(
        _market_snapshot_page,
        title="Market Snapshot",
        icon=":material/show_chart:",
        url_path="market-snapshot",
    )
    compare_page = st.Page(
        _compare_tickers_page,
        title="Compare Tickers",
        icon=":material/compare_arrows:",
        url_path="compare",
    )
    news_page = st.Page(
        _news_page,
        title="News & Sentiment",
        icon=":material/newspaper:",
        url_path="news",
    )
    ask_page = st.Page(
        _ask_learn_page,
        title="Ask & Learn",
        icon=":material/forum:",
        url_path="ask-learn",
    )
    learning_center_page = st.Page(
        _learning_center_page,
        title="Learning Center",
        icon=":material/school:",
        url_path="learning-center",
    )
    settings_page = st.Page(
        _settings_page,
        title="Settings",
        icon=":material/settings:",
        url_path="settings",
    )
    help_page = st.Page(
        _help_page,
        title="Help",
        icon=":material/help_outline:",
        url_path="help",
    )

    pages = {
        "Start": [overview_page],
        "Analyze": [stock_outlook_page, market_snapshot_page, compare_page, news_page],
        "Learn": [ask_page, learning_center_page],
        "Support": [settings_page, help_page],
    }
    navigation = st.navigation(pages, position="hidden")
    active_href = navigation.url_path or ""
    active_nav_css = f"""
<style>
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][href="{html.escape(active_href, quote=True)}"] {{
    background: rgba(0, 113, 227, 0.13) !important;
    color: var(--blue) !important;
}}
</style>
"""
    with st.sidebar:
        st.markdown(active_nav_css, unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="brand-row sidebar-brand" style="margin-bottom:1rem;">
                {_brand_logo_markup()}
                <div class="brand-copy">
                    <div class="brand-wordmark">StockSense</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="sidebar_main_navigation"):
            st.markdown('<div class="sidebar-section-label">Start</div>', unsafe_allow_html=True)
            st.page_link(overview_page, label="Overview", icon=":material/home:")

            st.markdown('<div class="sidebar-section-label">Analyze</div>', unsafe_allow_html=True)
            st.page_link(stock_outlook_page, label="Stock Outlook", icon=":material/insights:")
            st.page_link(market_snapshot_page, label="Market Snapshot", icon=":material/show_chart:")
            st.page_link(compare_page, label="Compare Tickers", icon=":material/compare_arrows:")
            st.page_link(news_page, label="News & Sentiment", icon=":material/newspaper:")

            st.markdown('<div class="sidebar-section-label">Learn</div>', unsafe_allow_html=True)
            st.page_link(ask_page, label="Ask & Learn", icon=":material/forum:")
            st.page_link(learning_center_page, label="Learning Center", icon=":material/school:")
        with st.container(key="sidebar_utility_navigation"):
            st.page_link(settings_page, label="Settings", icon=":material/settings:")
            st.page_link(help_page, label="Help", icon=":material/help_outline:")
    navigation.run()


if __name__ == "__main__":
    main()
