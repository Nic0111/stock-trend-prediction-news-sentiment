import { isArticleContributingToContext } from "./NewsCard";

test("marks only confident articles inside the selected context window", () => {
  const context = { window_days: 7 };
  const usable = { published_at: "2026-07-15T09:00:00Z", sentiment_uncertain: false, sentiment_score: 1 };
  const old = { ...usable, published_at: "2026-07-01T09:00:00Z" };
  const uncertain = { ...usable, sentiment_uncertain: true };
  expect(isArticleContributingToContext(usable, context, "2026-07-17")).toBe(true);
  expect(isArticleContributingToContext(old, context, "2026-07-17")).toBe(false);
  expect(isArticleContributingToContext(uncertain, context, "2026-07-17")).toBe(false);
});
