import { describe, expect, it } from "vitest";
import type { ComparisonItem, MarketSnapshot } from "../types";
import { buildComparisonSeries, buildRiskReturnPoints, periodReturn } from "./comparison";

function snapshot(ticker: string, prices: number[], overrides: Partial<MarketSnapshot> = {}): MarketSnapshot {
  const history = prices.map((price, index) => ({
    date: `2026-01-${String(index + 1).padStart(2, "0")}`,
    open: price,
    high: price,
    low: price,
    close: price,
    adjusted_close: price,
    volume: 100,
    dividend: 0,
    split: 0,
    ma20: null,
    ma60: null,
    rsi14: null,
    macd: null,
    macd_signal: null,
  }));
  return {
    ticker,
    company_name: ticker,
    price: prices.at(-1) || 0,
    previous_close: prices.at(-2) || null,
    change: null,
    change_pct: null,
    day_high: null,
    day_low: null,
    volume: 100,
    as_of_date: history.at(-1)?.date || "",
    data_source: "test",
    stale_data_warning: false,
    history_dates: history.map((point) => point.date),
    history_prices: prices,
    history,
    performance: { one_day: null, five_day: null, one_month: null, three_month: null, ytd: null, one_year: null },
    range_52_week: { low: null, high: null, position_pct: null },
    activity: { average_volume_20d: null, volume_ratio: null },
    risk: { annualized_volatility_20d: 0.2, atr14_pct: null, daily_range_pct: null },
    market_context: { history: [], excess_return_one_month: null, excess_return_three_month: null, beta20: null, beta60: null, correlation20: null, correlation60: null, klci_return_one_month: null, klci_trend: "Unavailable", klci_volatility_20d: null },
    ...overrides,
  };
}

function item(ticker: string, prices: number[], overrides: Partial<MarketSnapshot> = {}): ComparisonItem {
  return { ticker, snapshot: snapshot(ticker, prices, overrides), prediction: null, error: null };
}

describe("comparison data utilities", () => {
  it("rebases selected stocks from a shared first session", () => {
    const result = buildComparisonSeries([item("AAA", [10, 11, 12]), item("BBB", [20, 18, 22])], "1M", false);
    expect(result.data[0]).toMatchObject({ AAA: 100, BBB: 100 });
    expect(result.data.at(-1)?.AAA).toBeCloseTo(120);
    expect(result.data.at(-1)?.BBB).toBeCloseTo(110);
  });

  it("keeps the selected lanes aligned and rebases KLCI from the same first session", () => {
    const klci = [
      { date: "2026-01-01", close: 100, adjusted_close: 100 },
      { date: "2026-01-02", close: 101, adjusted_close: 101 },
      { date: "2026-01-03", close: 99, adjusted_close: 99 },
    ];
    const result = buildComparisonSeries([
      item("AAA", [10, 11, 12], { market_context: { history: klci, excess_return_one_month: null, excess_return_three_month: null, beta20: null, beta60: null, correlation20: null, correlation60: null, klci_return_one_month: null, klci_trend: "Unavailable", klci_volatility_20d: null } }),
      item("BBB", [20, 18, 22], { market_context: { history: klci, excess_return_one_month: null, excess_return_three_month: null, beta20: null, beta60: null, correlation20: null, correlation60: null, klci_return_one_month: null, klci_trend: "Unavailable", klci_volatility_20d: null } }),
    ], "1M", true);
    expect(result.data[0]).toMatchObject({ AAA: 100, BBB: 100, KLCI: 100 });
    expect(result.data.at(-1)?.KLCI).toBeCloseTo(99);
  });

  it("uses enriched performance fields and computes six-month movement from history", () => {
    const oneMonth = snapshot("AAA", [10, 11, 12], { performance: { one_day: null, five_day: null, one_month: 0.25, three_month: null, ytd: null, one_year: null } });
    expect(periodReturn(oneMonth, "1M")).toBe(0.25);
    const sixMonth = snapshot("BBB", Array.from({ length: 130 }, (_, index) => index + 1));
    expect(periodReturn(sixMonth, "6M")).toBeCloseTo(130 / 4 - 1);
  });

  it("omits risk-return points when either metric is unavailable", () => {
    const points = buildRiskReturnPoints([item("AAA", [10, 11, 12]), item("BBB", [10, 11, 12], { risk: { annualized_volatility_20d: null, atr14_pct: null, daily_range_pct: null } })], "1M");
    expect(points).toHaveLength(0);
  });
});
