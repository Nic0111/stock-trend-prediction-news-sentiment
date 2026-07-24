from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.finbert_sentiment_model import FinBERTDependencyError
from pipeline.download_financial_news_data import NewsContextResult
from pipeline.download_financial_news_data import build_news_context_metadata
from pipeline.marketaux_news_provider import (
    MARKETAUX_CACHE_DIR,
    MARKETAUX_QUERY_CONFIG_VERSION,
    MarketauxConfigurationError,
    MarketauxRequestError,
    _read_marketaux_cache,
    _write_marketaux_cache,
    build_marketaux_request_url,
    build_marketaux_search_query,
    get_marketaux_api_token,
    normalize_marketaux_rows,
    probe_marketaux_entity,
    retrieve_marketaux_financial_news,
    retrieve_news_with_marketaux_fallback,
)


class FakeSentimentModel:
    classes_ = np.array(["negative", "neutral", "positive"])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[0.05, 0.10, 0.85]]), (len(texts), 1))


class FakeResponse:
    def __init__(self, payload: dict, headers: dict | None = None) -> None:
        self.payload = payload
        self.headers = headers or {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class MarketauxProviderTests(unittest.TestCase):
    @staticmethod
    def _news_result(*, coverage: str = "Low", provider: str = "newsapi", source: str = "Example Source") -> NewsContextResult:
        article = {
            "provider": provider,
            "title": f"{provider} Maybank earnings update",
            "description": "Maybank reported stronger earnings.",
            "url": f"https://{source.casefold().replace(' ', '')}.example/story",
            "source_name": source,
            "source_domain": f"{source.casefold().replace(' ', '')}.example",
            "published_at": "2026-07-10T08:00:00Z",
            "sentiment_label": "positive",
            "sentiment_score": 1,
            "sentiment_probability": 0.9,
            "sentiment_uncertain": False,
        }
        context = {
            "horizon_days": 5,
            "window_days": 14,
            "half_life_days": 7.0,
            "candidate_article_count": 1,
            "usable_article_count": 1,
            "uncertain_article_count": 0,
            "unique_source_count": 1,
            "weighted_sentiment_score": 1.0,
            "positive_count": 1,
            "neutral_count": 0,
            "negative_count": 0,
            "coverage_confidence": coverage,
            "low_confidence_flag": coverage in {"Low", "Unavailable"},
        }
        return NewsContextResult(
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            query_text="Maybank",
            as_of_date="2026-07-10",
            retrieved_at="2026-07-10T08:00:00+00:00",
            raw_article_count=1,
            relevant_article_count=1,
            results_truncated=False,
            contexts={"5": context},
            articles=[article],
            output_paths={},
            provider_mode=provider,
            provider_article_counts={provider: 1},
            provider_status={provider: "ok"},
            provider_metadata={
                **build_news_context_metadata(),
                **(
                    {
                        "marketaux_query_config_version": MARKETAUX_QUERY_CONFIG_VERSION,
                        "query_mode": "alias_search",
                    }
                    if provider == "marketaux"
                    else {}
                ),
                **({"marketaux_headers": {}} if provider == "marketaux" else {}),
            },
        )

    def test_request_url_has_bounded_window_and_entity_filters(self) -> None:
        url = build_marketaux_request_url(
            "token",
            symbols=["1155.KL"],
            as_of_date=date(2026, 7, 10),
        )
        self.assertIn("symbols=1155.KL", url)
        self.assertIn("published_after=2026-06-10", url)
        self.assertIn("published_before=2026-07-11", url)
        self.assertIn("filter_entities=true", url)
        self.assertIn("must_have_entities=true", url)
        self.assertIn("limit=3", url)

    def test_historical_request_url_accepts_explicit_date_bounds(self) -> None:
        url = build_marketaux_request_url(
            "token",
            symbols=["1155.KL"],
            as_of_date=date(2026, 7, 10),
            published_after=date(2026, 4, 11),
            published_before=date(2026, 6, 10),
        )
        self.assertIn("published_after=2026-04-11", url)
        self.assertIn("published_before=2026-06-10", url)

    def test_experimental_alias_search_uses_marketaux_or_syntax_without_entity_gate(self) -> None:
        query = build_marketaux_search_query(["Malayan Banking Berhad", "Maybank"])
        self.assertEqual(query, '"Malayan Banking Berhad" | "Maybank"')
        url = build_marketaux_request_url(
            "token",
            search=query,
            as_of_date=date(2026, 7, 10),
            require_entities=False,
        )
        self.assertIn("search=%22Malayan+Banking+Berhad%22+%7C+%22Maybank%22", url)
        self.assertIn("filter_entities=false", url)
        self.assertIn("must_have_entities=false", url)

    def test_token_file_precedes_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marketaux_api_key.txt"
            path.write_text("file-token\n", encoding="utf-8")
            with patch("pipeline.marketaux_news_provider.MARKETAUX_API_KEY_FILE", path), patch.dict(
                "os.environ", {"MARKETAUX_API_TOKEN": "environment-token"}
            ):
                self.assertEqual(get_marketaux_api_token(), "file-token")

    def test_missing_token_is_explicit(self) -> None:
        with patch("pipeline.marketaux_news_provider.MARKETAUX_API_KEY_FILE", Path("does-not-exist")), patch.dict(
            "os.environ", {}, clear=True
        ):
            with self.assertRaises(MarketauxConfigurationError):
                get_marketaux_api_token()

    def test_rows_normalize_entity_metadata_and_deduplicate(self) -> None:
        payload = {
            "data": [
                {
                    "title": "Maybank reports growth",
                    "description": "Malayan Banking Berhad reported stronger earnings.",
                    "snippet": "Earnings improved.",
                    "url": "https://example.com/story?utm_source=marketaux",
                    "published_at": "2026-07-10T08:00:00Z",
                    "source": "example.com",
                    "entities": [
                        {
                            "symbol": "1155.KL",
                            "name": "Malayan Banking Berhad",
                            "match_score": 91.0,
                            "sentiment_score": 0.4,
                        }
                    ],
                },
                {
                    "title": "Maybank reports growth",
                    "description": "Malayan Banking Berhad reported stronger earnings.",
                    "snippet": "Earnings improved.",
                    "url": "https://example.com/story",
                    "published_at": "2026-07-10T08:00:00Z",
                    "source": "example.com",
                    "entities": [{"symbol": "1155.KL", "name": "Malayan Banking Berhad"}],
                },
            ]
        }
        rows = normalize_marketaux_rows(
            payload,
            ["Malayan Banking Berhad", "Maybank"],
            entity_mapping={"symbol": "1155.KL"},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "marketaux")
        self.assertEqual(rows[0]["text_completeness"], "title_description_snippet")
        self.assertEqual(rows[0]["provider_entity_symbol"], "1155.KL")

    def test_package_registry_is_excluded_even_with_exact_entity_match(self) -> None:
        payload = {
            "data": [
                {
                    "title": "python-mytnb 0.1.0",
                    "description": "Python library for the Tenaga Nasional Berhad API.",
                    "snippet": "Package metadata",
                    "url": "https://pypi.org/project/python-mytnb/",
                    "published_at": "2026-07-10T08:00:00Z",
                    "source": "pypi.org",
                    "entities": [{"symbol": "1155.KL", "name": "Malayan Banking Berhad"}],
                }
            ]
        }
        diagnostics: dict[str, int] = {}
        rows = normalize_marketaux_rows(
            payload,
            ["Malayan Banking Berhad", "Maybank"],
            entity_mapping={"symbol": "1155.KL"},
            diagnostics=diagnostics,
        )
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics, {"package_registry": 1})

    def test_retrieval_reuses_shared_sentiment_and_context_contract(self) -> None:
        response = {
            "meta": {"found": 1, "returned": 1},
            "data": [
                {
                    "title": "Maybank reports growth",
                    "description": "Maybank reported stronger earnings.",
                    "snippet": "Earnings improved.",
                    "url": "https://example.com/maybank",
                    "published_at": "2026-07-10T08:00:00Z",
                    "source": "example.com",
                    "entities": [{"symbol": "1155.KL", "name": "Malayan Banking Berhad"}],
                }
            ],
        }
        with patch("pipeline.marketaux_news_provider.load_stock_metadata", return_value={"1155.KL": "Malayan Banking Berhad"}):
            result = retrieve_marketaux_financial_news(
                "1155.KL",
                as_of_date=date(2026, 7, 10),
                token="token",
                entity_mapping={"symbol": "1155.KL", "status": "supported"},
                response_data=response,
                model=FakeSentimentModel(),
            )
        self.assertIsInstance(result, NewsContextResult)
        self.assertEqual(result.provider_mode, "marketaux")
        self.assertEqual(result.relevant_article_count, 1)
        self.assertEqual(result.articles[0]["sentiment_label"], "positive")
        self.assertEqual(result.provider_metadata["coverage_mode"], "entity")

    def test_alias_retrieval_uses_marketaux_query_language_and_disables_entity_gate(self) -> None:
        response = {"meta": {"found": 0, "returned": 0}, "data": []}
        with patch(
            "pipeline.marketaux_news_provider.fetch_marketaux_response",
            return_value=(response, {}),
        ) as fetch:
            with patch(
                "pipeline.marketaux_news_provider.load_stock_metadata",
                return_value={"1155.KL": "Malayan Banking Berhad"},
            ):
                result = retrieve_marketaux_financial_news(
                    "1155.KL",
                    as_of_date=date(2026, 7, 10),
                    token="token",
                    model=FakeSentimentModel(),
                )
        kwargs = fetch.call_args.kwargs
        self.assertEqual(
            kwargs["search"],
            '"Malayan Banking Berhad" | "Malayan Banking" | "Maybank" | "Maybank Group"',
        )
        self.assertIsNone(kwargs["symbols"])
        self.assertFalse(kwargs["require_entities"])
        self.assertEqual(result.provider_metadata["query_mode"], "alias_search")

    def test_symbol_retrieval_keeps_marketaux_entity_gate(self) -> None:
        response = {"meta": {"found": 0, "returned": 0}, "data": []}
        with patch(
            "pipeline.marketaux_news_provider.fetch_marketaux_response",
            return_value=(response, {}),
        ) as fetch:
            with patch(
                "pipeline.marketaux_news_provider.load_stock_metadata",
                return_value={"1155.KL": "Malayan Banking Berhad"},
            ):
                result = retrieve_marketaux_financial_news(
                    "1155.KL",
                    as_of_date=date(2026, 7, 10),
                    token="token",
                    entity_mapping={"symbol": "1155.KL", "status": "supported"},
                    model=FakeSentimentModel(),
                )
        kwargs = fetch.call_args.kwargs
        self.assertEqual(kwargs["symbols"], ["1155.KL"])
        self.assertIsNone(kwargs["search"])
        self.assertTrue(kwargs["require_entities"])
        self.assertEqual(result.provider_metadata["query_mode"], "symbol")

    def test_foreign_same_name_listing_is_not_exact_bursa_coverage(self) -> None:
        payload = {
            "data": [
                {
                    "symbol": "MLYNF",
                    "name": "Malayan Banking Berhad",
                    "exchange": None,
                    "exchange_long": "OTC Markets",
                    "country": "us",
                }
            ]
        }

        def opener(*args: object, **kwargs: object) -> FakeResponse:
            return FakeResponse(payload)

        match = probe_marketaux_entity(
            "1155.KL",
            "Malayan Banking Berhad",
            ["Maybank"],
            "token",
            opener=opener,
        )
        self.assertEqual(match.status, "unsupported")
        self.assertIsNone(match.symbol)

    def test_high_newsapi_coverage_skips_marketaux(self) -> None:
        newsapi = self._news_result(coverage="High")
        with patch("pipeline.marketaux_news_provider.retrieve_marketaux_financial_news") as marketaux_call:
            result = retrieve_news_with_marketaux_fallback("1155.KL", 5, newsapi_result=newsapi)
        marketaux_call.assert_not_called()
        self.assertEqual(result.provider_mode, "newsapi")
        self.assertEqual(result.provider_status["marketaux"], "skipped")
        self.assertFalse(result.provider_metadata["marketaux_used"])

    def test_shared_finbert_failure_does_not_spend_marketaux_quota(self) -> None:
        runtime_error = FinBERTDependencyError("FinBERT runtime is unavailable")
        with (
            patch(
                "pipeline.download_financial_news_data.retrieve_financial_news",
                side_effect=runtime_error,
            ),
            patch("pipeline.marketaux_news_provider.retrieve_marketaux_financial_news") as marketaux_call,
        ):
            with self.assertRaises(FinBERTDependencyError):
                retrieve_news_with_marketaux_fallback(
                    "1295.KL", 1, marketaux_token="token"
                )
        marketaux_call.assert_not_called()

    def test_low_newsapi_coverage_merges_marketaux_and_records_provenance(self) -> None:
        newsapi = self._news_result(coverage="Low", source="NewsAPI Source")
        marketaux = self._news_result(coverage="Low", provider="marketaux", source="Marketaux Source")
        with tempfile.TemporaryDirectory() as directory, patch(
            "pipeline.marketaux_news_provider.MARKETAUX_CACHE_DIR", Path(directory)
        ), patch(
            "pipeline.marketaux_news_provider.retrieve_marketaux_financial_news", return_value=marketaux
        ) as marketaux_call:
            result = retrieve_news_with_marketaux_fallback(
                "1155.KL", 5, as_of_date="2026-07-10", newsapi_result=newsapi, marketaux_token="token"
            )
        marketaux_call.assert_called_once()
        self.assertEqual(result.provider_mode, "newsapi_marketaux_complement")
        self.assertTrue(result.provider_metadata["marketaux_used"])
        self.assertEqual(result.provider_article_counts, {"newsapi": 1, "marketaux": 1})
        self.assertEqual(result.contexts["5"]["unique_source_count"], 2)

    def test_newsapi_failure_is_not_reported_as_success_for_marketaux_fallback(self) -> None:
        marketaux = self._news_result(provider="marketaux", source="Marketaux Source")
        with tempfile.TemporaryDirectory() as directory, \
            patch("pipeline.marketaux_news_provider.MARKETAUX_CACHE_DIR", Path(directory)), \
            patch(
                "pipeline.download_financial_news_data.retrieve_financial_news",
                side_effect=RuntimeError("NewsAPI HTTP 429"),
            ), \
            patch("pipeline.marketaux_news_provider.retrieve_marketaux_financial_news", return_value=marketaux):
            result = retrieve_news_with_marketaux_fallback(
                "1155.KL",
                5,
                as_of_date="2026-07-10",
                marketaux_token="token",
            )
        self.assertEqual(result.provider_status["newsapi"], "unavailable")
        self.assertEqual(result.provider_status["marketaux"], "ok")
        self.assertIn("HTTP 429", result.provider_metadata["newsapi_error"])

    def test_fresh_marketaux_cache_prevents_second_request(self) -> None:
        newsapi = self._news_result(coverage="Low")
        marketaux = self._news_result(coverage="Low", provider="marketaux", source="Marketaux Source")
        with tempfile.TemporaryDirectory() as directory, patch(
            "pipeline.marketaux_news_provider.MARKETAUX_CACHE_DIR", Path(directory)
        ), patch(
            "pipeline.marketaux_news_provider.retrieve_marketaux_financial_news", return_value=marketaux
        ) as marketaux_call:
            retrieve_news_with_marketaux_fallback("1155.KL", 5, as_of_date="2026-07-10", newsapi_result=newsapi, marketaux_token="token")
            retrieve_news_with_marketaux_fallback("1155.KL", 5, as_of_date="2026-07-10", newsapi_result=newsapi, marketaux_token="token")
        self.assertEqual(marketaux_call.call_count, 1)

    def test_legacy_marketaux_cache_is_not_used(self) -> None:
        result = self._news_result(provider="marketaux")
        result.provider_metadata["news_context_config_version"] = 1
        with tempfile.TemporaryDirectory() as directory, patch(
            "pipeline.marketaux_news_provider.MARKETAUX_CACHE_DIR", Path(directory)
        ):
            _write_marketaux_cache(result, result.retrieved_at)
            loaded = _read_marketaux_cache("1155.KL", date(2026, 7, 10))
        self.assertIsNone(loaded)

    def test_old_marketaux_query_cache_is_not_used(self) -> None:
        result = self._news_result(provider="marketaux")
        result.provider_metadata.pop("marketaux_query_config_version", None)
        with tempfile.TemporaryDirectory() as directory, patch(
            "pipeline.marketaux_news_provider.MARKETAUX_CACHE_DIR", Path(directory)
        ):
            _write_marketaux_cache(result, result.retrieved_at)
            loaded = _read_marketaux_cache("1155.KL", date(2026, 7, 10))
        self.assertIsNone(loaded)

    def test_empty_recent_result_triggers_separate_historical_context(self) -> None:
        newsapi = self._news_result(coverage="Unavailable")
        newsapi.articles = []
        newsapi.relevant_article_count = 0
        newsapi.contexts["5"]["candidate_article_count"] = 0
        newsapi.contexts["5"]["usable_article_count"] = 0
        newsapi.contexts["5"]["weighted_sentiment_score"] = None
        recent = self._news_result(provider="marketaux", coverage="Unavailable")
        recent.articles = []
        recent.relevant_article_count = 0
        historical = NewsContextResult(
            ticker="1155.KL",
            company_name="Malayan Banking Berhad",
            query_text="Maybank",
            as_of_date="2026-07-10",
            retrieved_at="2026-07-10T08:00:01+00:00",
            raw_article_count=1,
            relevant_article_count=0,
            results_truncated=False,
            contexts={},
            articles=[],
            output_paths={},
            provider_mode="marketaux_historical_context",
            provider_article_counts={"marketaux_historical": 1},
            provider_status={"marketaux": "ok"},
            provider_metadata={
                **build_news_context_metadata(),
                "marketaux_query_config_version": MARKETAUX_QUERY_CONFIG_VERSION,
                "query_mode": "alias_search",
            },
            historical_articles=[
                {
                    "provider": "marketaux",
                    "source_name": "Historical Source",
                    "source_domain": "historical.example",
                    "title": "Maybank historical update",
                    "description": "An older company background article.",
                    "url": "https://historical.example/story",
                    "published_at": "2026-05-01T08:00:00Z",
                }
            ],
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "pipeline.marketaux_news_provider.MARKETAUX_CACHE_DIR", Path(directory)
        ), patch(
            "pipeline.marketaux_news_provider.retrieve_marketaux_financial_news",
            side_effect=[recent, historical],
        ) as marketaux_call:
            result = retrieve_news_with_marketaux_fallback(
                "1155.KL",
                5,
                as_of_date="2026-07-10",
                newsapi_result=newsapi,
                marketaux_token="token",
            )
            second_result = retrieve_news_with_marketaux_fallback(
                "1155.KL",
                5,
                as_of_date="2026-07-10",
                newsapi_result=newsapi,
                marketaux_token="token",
            )
        self.assertEqual(marketaux_call.call_count, 2)
        self.assertEqual(result.relevant_article_count, 0)
        self.assertEqual(len(result.articles), 0)
        self.assertEqual(len(result.historical_articles), 1)
        self.assertEqual(result.provider_metadata["historical_article_count"], 1)
        self.assertEqual(len(second_result.historical_articles), 1)

    def test_historical_articles_do_not_change_sentiment_context(self) -> None:
        base = self._news_result(coverage="Unavailable")
        base.articles = []
        base.relevant_article_count = 0
        historical = NewsContextResult(
            ticker=base.ticker,
            company_name=base.company_name,
            query_text=base.query_text,
            as_of_date=base.as_of_date,
            retrieved_at=base.retrieved_at,
            raw_article_count=1,
            relevant_article_count=0,
            results_truncated=False,
            contexts={},
            articles=[],
            output_paths={},
            provider_metadata=build_news_context_metadata(),
            historical_articles=[{"title": "old", "published_at": "2026-05-01T00:00:00Z"}],
        )
        from pipeline.marketaux_news_provider import _with_historical_context

        result = _with_historical_context(base, historical, cache_state="fresh")
        self.assertEqual(result.contexts, base.contexts)
        self.assertEqual(result.articles, base.articles)
        self.assertEqual(result.relevant_article_count, base.relevant_article_count)


if __name__ == "__main__":
    unittest.main()
