import type { BenchmarkPoint, ComparisonItem, MarketHistoryPoint, MarketSnapshot } from "../types";

export type ComparisonRange = "1M" | "3M" | "6M" | "1Y";

export const COMPARISON_RANGE_LENGTHS: Record<ComparisonRange, number> = {
  "1M": 21,
  "3M": 63,
  "6M": 126,
  "1Y": 252,
};

export const COMPARISON_PALETTE = ["#0071e3", "#7b61ff", "#0f8b8d", "#c77d16", "#d14d72"];

export interface ComparisonSeriesPoint {
  date: string;
  [ticker: string]: string | number | null;
}

export interface ComparisonSeries {
  ticker: string;
  name: string;
  color: string;
}

export interface ComparisonRiskPoint {
  ticker: string;
  name: string;
  color: string;
  risk: number;
  returnValue: number;
}

function historyFor(snapshot: MarketSnapshot): MarketHistoryPoint[] {
  if (snapshot.history?.length) return snapshot.history;
  return snapshot.history_dates.map((date, index) => ({
    date,
    open: null,
    high: null,
    low: null,
    close: snapshot.history_prices[index] ?? null,
    adjusted_close: snapshot.history_prices[index] ?? null,
    volume: null,
    dividend: null,
    split: null,
    ma20: null,
    ma60: null,
    rsi14: null,
    macd: null,
    macd_signal: null,
  }));
}

export function getHistory(item: ComparisonItem): MarketHistoryPoint[] {
  return item.snapshot ? historyFor(item.snapshot) : [];
}

function adjustedValue(point: MarketHistoryPoint): number | null {
  return point.adjusted_close ?? point.close;
}

function benchmarkValue(point: BenchmarkPoint): number | null {
  return point.adjusted_close ?? point.close;
}

export function buildComparisonSeries(
  items: ComparisonItem[],
  range: ComparisonRange,
  includeKlci: boolean,
): { data: ComparisonSeriesPoint[]; series: ComparisonSeries[] } {
  const validItems = items.filter((item) => item.snapshot && getHistory(item).some((point) => adjustedValue(point) !== null));
  const historyMaps = validItems.map((item) => new Map(getHistory(item).map((point) => [point.date, adjustedValue(point)])));
  if (!historyMaps.length) return { data: [], series: [] };

  const commonDates = [...historyMaps[0].keys()]
    .filter((date) => historyMaps.every((map) => typeof map.get(date) === "number"))
    .sort()
    .slice(-COMPARISON_RANGE_LENGTHS[range]);
  if (!commonDates.length) return { data: [], series: [] };

  const series: ComparisonSeries[] = validItems.map((item, index) => ({
    ticker: item.ticker,
    name: item.snapshot?.company_name || item.ticker,
    color: COMPARISON_PALETTE[index % COMPARISON_PALETTE.length],
  }));
  const bases = new Map(series.map((entry, index) => [entry.ticker, historyMaps[index].get(commonDates[0]) as number]));
  const data = commonDates.map((date, index) => {
    const row: ComparisonSeriesPoint = { date };
    validItems.forEach((item, itemIndex) => {
      const value = historyMaps[itemIndex].get(date);
      const base = bases.get(item.ticker);
      row[item.ticker] = typeof value === "number" && base ? (value / base) * 100 : null;
    });

    if (includeKlci) {
      const benchmark = validItems[0].snapshot?.market_context?.history ?? [];
      const benchmarkMap = new Map(benchmark.map((point) => [point.date, benchmarkValue(point)]));
      const benchmarkDates = commonDates.filter((commonDate) => typeof benchmarkMap.get(commonDate) === "number");
      const benchmarkBase = benchmarkDates.length ? benchmarkMap.get(benchmarkDates[0]) : null;
      const market = benchmarkMap.get(date);
      row.KLCI = typeof market === "number" && typeof benchmarkBase === "number" && benchmarkBase !== 0 ? (market / benchmarkBase) * 100 : null;
    }
    return row;
  });
  return { data, series };
}

export function periodReturn(snapshot: MarketSnapshot | null, range: ComparisonRange): number | null {
  if (!snapshot) return null;
  const performance = snapshot.performance;
  if (range === "1M") return performance?.one_month ?? null;
  if (range === "3M") return performance?.three_month ?? null;
  if (range === "1Y") return performance?.one_year ?? null;

  const history = historyFor(snapshot).filter((point) => adjustedValue(point) !== null);
  if (history.length <= COMPARISON_RANGE_LENGTHS[range]) return null;
  const start = adjustedValue(history[history.length - COMPARISON_RANGE_LENGTHS[range] - 1]);
  const end = adjustedValue(history.at(-1)!);
  return start && end ? end / start - 1 : null;
}

export function buildRiskReturnPoints(items: ComparisonItem[], range: ComparisonRange): ComparisonRiskPoint[] {
  return items.flatMap((item, index) => {
    const risk = item.snapshot?.risk?.annualized_volatility_20d;
    const returnValue = periodReturn(item.snapshot, range);
    if (typeof risk !== "number" || typeof returnValue !== "number") return [];
    return [{
      ticker: item.ticker,
      name: item.snapshot?.company_name || item.ticker,
      color: COMPARISON_PALETTE[index % COMPARISON_PALETTE.length],
      risk,
      returnValue,
    }];
  });
}

export function lastHistoryPoint(snapshot: MarketSnapshot | null): MarketHistoryPoint | null {
  const history = snapshot ? historyFor(snapshot) : [];
  return history.at(-1) ?? null;
}

export function displayName(snapshot: MarketSnapshot | null, fallback: string): string {
  return snapshot?.company_name || fallback;
}
