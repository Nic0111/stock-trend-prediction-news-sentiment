import { buildNewsContributionData } from "./NewsEvidence";

test("maps server-authoritative contribution rows back to article headlines", () => {
  const articles = [
    { article_id: "a", title: "Recent positive", source_domain: "one.example" },
    { article_id: "b", title: "Older negative", source_domain: "two.example" },
  ];
  const context = {
    half_life_days: 7,
    article_contributions: [
      { article_id: "b", age_days: 8, recency_weight: 0.45, sentiment_confidence: 0.8, raw_weight: 0.36, normalized_weight: 0.35, sentiment_score: -1, sentiment_label: "negative", signed_contribution: -0.35 },
      { article_id: "a", age_days: 1, recency_weight: 0.9, sentiment_confidence: 0.9, raw_weight: 0.81, normalized_weight: 0.65, sentiment_score: 1, sentiment_label: "positive", signed_contribution: 0.65 },
    ],
  };

  const data = buildNewsContributionData(articles, context);
  expect(data.map((row) => row.articleId)).toEqual(["a", "b"]);
  expect(data[0].title).toBe("Recent positive");
  expect(data[1].tone).toBe("negative");
  expect(data[0].signedContribution + data[1].signedContribution).toBeCloseTo(0.3);
});
