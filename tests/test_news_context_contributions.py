import unittest
from datetime import date

from pipeline.download_financial_news_data import CONTEXT_WINDOWS, build_news_contexts


class NewsContextContributionTests(unittest.TestCase):
    def test_horizon_contributions_reproduce_weighted_score(self) -> None:
        rows = [
            {
                "article_id": "recent-positive",
                "published_at": "2026-07-24T09:00:00Z",
                "sentiment_uncertain": False,
                "sentiment_score": 1,
                "sentiment_label": "positive",
                "sentiment_probability": 0.9,
                "source_domain": "example-one.com",
            },
            {
                "article_id": "older-negative",
                "published_at": "2026-07-17T09:00:00Z",
                "sentiment_uncertain": False,
                "sentiment_score": -1,
                "sentiment_label": "negative",
                "sentiment_probability": 0.8,
                "source_domain": "example-two.com",
            },
        ]

        contexts = build_news_contexts(rows, date(2026, 7, 25))
        context = contexts["10"]
        contributions = context["article_contributions"]

        self.assertEqual(len(contributions), 2)
        self.assertAlmostEqual(sum(row["normalized_weight"] for row in contributions), 1.0)
        self.assertAlmostEqual(
            sum(row["signed_contribution"] for row in contributions),
            context["weighted_sentiment_score"],
        )
        self.assertEqual(context["window_days"], CONTEXT_WINDOWS[10])

    def test_unusable_articles_do_not_receive_contributions(self) -> None:
        rows = [
            {
                "article_id": "uncertain",
                "published_at": "2026-07-24T09:00:00Z",
                "sentiment_uncertain": True,
                "sentiment_score": 1,
                "sentiment_label": "positive",
                "sentiment_probability": 0.9,
                "source_domain": "example.com",
            }
        ]

        contexts = build_news_contexts(rows, date(2026, 7, 25))
        self.assertEqual(contexts["1"]["article_contributions"], [])
        self.assertIsNone(contexts["1"]["weighted_sentiment_score"])

    def test_syndicated_story_does_not_overstate_coverage(self) -> None:
        rows = [
            {
                "article_id": f"copy-{index}",
                "event_id": "event-one",
                "published_at": "2026-07-24T09:00:00Z",
                "sentiment_uncertain": False,
                "sentiment_score": 1,
                "sentiment_label": "positive",
                "sentiment_probability": 0.9,
                "source_domain": f"source-{index}.com",
            }
            for index in range(5)
        ]
        context = build_news_contexts(rows, date(2026, 7, 25))["10"]
        self.assertEqual(context["independent_event_count"], 1)
        self.assertEqual(context["coverage_confidence"], "Low")

    def test_syndicated_copies_do_not_outvote_an_independent_event(self) -> None:
        rows = [
            {
                "article_id": f"positive-copy-{index}",
                "event_id": "positive-event",
                "published_at": "2026-07-24T09:00:00Z",
                "sentiment_uncertain": False,
                "sentiment_score": 1,
                "sentiment_label": "positive",
                "sentiment_probability": 0.9,
                "source_domain": f"wire-copy-{index}.example",
            }
            for index in range(10)
        ]
        rows.append(
            {
                "article_id": "independent-negative",
                "event_id": "negative-event",
                "published_at": "2026-07-24T09:00:00Z",
                "sentiment_uncertain": False,
                "sentiment_score": -1,
                "sentiment_label": "negative",
                "sentiment_probability": 0.9,
                "source_domain": "independent.example",
            }
        )
        context = build_news_contexts(rows, date(2026, 7, 25))["10"]

        self.assertEqual(context["independent_event_count"], 2)
        self.assertAlmostEqual(context["weighted_sentiment_score"], 0.0)
        positive_weight = sum(
            item["normalized_weight"]
            for item in context["article_contributions"]
            if item["event_id"] == "positive-event"
        )
        negative_weight = sum(
            item["normalized_weight"]
            for item in context["article_contributions"]
            if item["event_id"] == "negative-event"
        )
        self.assertAlmostEqual(positive_weight, negative_weight)


if __name__ == "__main__":
    unittest.main()
