import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { Activity, BarChart3, CalendarDays, Gauge, Info, Scale, ShieldCheck, Target, TrendingDown, TrendingUp, Volume2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { createComparison, getCompanies } from "../api/client";
import { LoadingState, ErrorState } from "../components/AsyncState";
import { CompanyLogo } from "../components/CompanyLogo";
import { CompanySelector } from "../components/CompanySelector";
import { HorizonSelector } from "../components/HorizonSelector";
import { ComparisonPerformanceChart, ComparisonReturnChart, ComparisonRiskReturnChart } from "../components/Chart";
import { PageHeader } from "../components/PageHeader";
import { StockModelEvidence } from "../components/ModelDiagnostics";
import { useAppState } from "../state/AppState";
import type { ComparisonItem, Company, MarketSnapshot, Prediction } from "../types";
import { buildComparisonSeries, buildRiskReturnPoints, COMPARISON_PALETTE, displayName, lastHistoryPoint, periodReturn, type ComparisonRange } from "../utils/comparison";

const RETURN_OPTIONS: Array<{ key: ComparisonRange; label: string }> = [
  { key: "1M", label: "1 month" },
  { key: "3M", label: "3 months" },
  { key: "6M", label: "6 months" },
  { key: "1Y", label: "1 year" },
];

function money(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `RM ${value.toFixed(2)}` : "Unavailable";
}

function percent(value: number | null | undefined, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%` : "Unavailable";
}

function number(value: number | null | undefined, digits = 1) {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "Unavailable";
}

function dateLabel(value: string | undefined) {
  if (!value) return "Unavailable";
  const parsed = new Date(`${value}T00:00:00`);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function itemColor(item: ComparisonItem, order: string[]) {
  return COMPARISON_PALETTE[Math.max(0, order.indexOf(item.ticker)) % COMPARISON_PALETTE.length];
}

function successfulItems(items: ComparisonItem[]) {
  return items.filter((item) => item.snapshot && item.prediction);
}

function evidenceLabel(prediction: Prediction | null) {
  if (!prediction) return "Unavailable";
  if (prediction.evidence_status === "accepted") return "Accepted";
  if (prediction.evidence_status === "limited") return "Limited evidence";
  return "Research only";
}

function evidenceClass(prediction: Prediction | null) {
  if (!prediction) return "unknown";
  if (prediction.evidence_status === "accepted") return "accepted";
  if (prediction.evidence_status === "limited") return "limited";
  return "experimental";
}

function SnapshotMetric({ label, value, note, tone }: { label: string; value: string; note?: string; tone?: "positive" | "negative" | "neutral" }) {
  return <div className="comparison-metric-cell"><span>{label}</span><strong className={tone ? `value-${tone}` : ""}>{value}</strong>{note && <small>{note}</small>}</div>;
}

function RangeTrack({ snapshot }: { snapshot: MarketSnapshot }) {
  const position = snapshot.range_52_week?.position_pct;
  const positionAvailable = typeof position === "number" && Number.isFinite(position);
  const safePosition = positionAvailable ? Math.max(0, Math.min(1, position as number)) : 0.5;
  return <div className="comparison-range-track" aria-label={`52-week position ${positionAvailable ? `${(safePosition * 100).toFixed(0)} percent` : "unavailable"}`}><div className="comparison-range-line"><i className={positionAvailable ? "" : "unavailable"} style={{ left: `${safePosition * 100}%` }} /></div><div className="comparison-range-values"><span>{money(snapshot.range_52_week?.low)}</span><strong>{positionAvailable ? `${(safePosition * 100).toFixed(0)}%` : "Unavailable"}</strong><span>{money(snapshot.range_52_week?.high)}</span></div></div>;
}

function MetricBar({ value, max = 1, tone = "blue" }: { value: number | null | undefined; max?: number; tone?: "blue" | "green" | "amber" }) {
  const width = typeof value === "number" && Number.isFinite(value) ? Math.max(4, Math.min(100, (Math.abs(value) / max) * 100)) : 0;
  const sign = typeof value === "number" && value < 0 ? "negative" : "";
  return <div className={`comparison-meter ${tone} ${sign}`}><i style={{ width: `${width}%` }} /></div>;
}

function ProbabilityTrack({ prediction }: { prediction: Prediction }) {
  const probability = Math.max(0, Math.min(1, prediction.probability_up));
  const threshold = Math.max(0, Math.min(1, prediction.decision_threshold));
  const direction = prediction.prediction_label.toLowerCase() === "down" ? "Down" : "Up";
  return <div className="probability-track-wrap"><div className="probability-track" aria-label={`${direction}, ${percent(probability)} probability of upward movement`}><i className={direction === "Down" ? "negative" : "positive"} style={{ left: `${probability * 100}%` }} /><b style={{ left: `${threshold * 100}%` }} /></div><div className="probability-labels"><span>0% Up</span><strong>{percent(probability)} Up probability</strong><span>100% Up</span></div><small>Cutoff {percent(threshold)} · marker shows the model’s Up probability</small></div>;
}

function ComparisonOverview({ items, order, companies, range }: { items: ComparisonItem[]; order: string[]; companies: Company[]; range: ComparisonRange }) {
  return <section className="comparison-section comparison-overview-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><BarChart3 size={18} /></div><div><span className="panel-kicker">At a glance</span><h2>See the differences before diving deeper</h2><p>Each column keeps one company’s evidence together, so price, activity, risk and model output are easy to compare.</p></div></div><div className="comparison-overview-grid" style={{ "--comparison-count": order.length } as CSSProperties}>{items.map((item) => { const snapshot = item.snapshot; const prediction = item.prediction; const company = companies.find((candidate) => candidate.ticker === item.ticker); if (!snapshot || !prediction) return <article className="comparison-overview-column comparison-overview-unavailable" key={item.ticker}><div className="comparison-company-heading"><CompanyLogo company={company} className="comparison-overview-logo" /><div><h3>{company?.company_name || item.ticker}</h3><span>{item.ticker}</span></div></div><div className="notice error">{item.error || "Evidence unavailable for this ticker."}</div></article>; const color = itemColor(item, order); const selectedReturn = periodReturn(snapshot, range); const dayTone = typeof snapshot.change_pct === "number" ? snapshot.change_pct >= 0 ? "positive" : "negative" : "neutral"; const returnTone = typeof selectedReturn === "number" ? selectedReturn >= 0 ? "positive" : "negative" : "neutral"; return <article className="comparison-overview-column" key={item.ticker} style={{ "--ticker-color": color } as CSSProperties}><div className="comparison-company-heading"><CompanyLogo company={company} className="comparison-overview-logo" /><div><h3>{snapshot.company_name}</h3><span>{item.ticker}</span></div></div><SnapshotMetric label="Latest price" value={money(snapshot.price)} /><SnapshotMetric label="Today" value={snapshot.change_pct === null ? "Unavailable" : `${percent(snapshot.change_pct, 2)}`} tone={dayTone} /><SnapshotMetric label="Selected period" value={percent(selectedReturn)} note={`${range} return`} tone={returnTone} /><SnapshotMetric label="Volume activity" value={snapshot.activity?.volume_ratio == null ? "Unavailable" : `${snapshot.activity.volume_ratio.toFixed(1)}×`} note="versus 20-day average" /><SnapshotMetric label="Recent volatility" value={percent(snapshot.risk?.annualized_volatility_20d)} /><div className={`comparison-evidence-chip ${evidenceClass(prediction)}`}><ShieldCheck size={14} /> {evidenceLabel(prediction)}</div></article>; })}</div></section>;
}

function ProbabilitySection({ items, order }: { items: ComparisonItem[]; order: string[] }) {
  return <section className="comparison-section comparison-model-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><Target size={18} /></div><div><span className="panel-kicker">Model evidence</span><h2>How strong is each model signal?</h2><p>Read the probability and cutoff together. A research-only result is visible, but it is not presented as an accepted conclusion.</p></div></div><div className="comparison-model-grid" style={{ "--comparison-count": items.length } as CSSProperties}>{items.map((item) => { if (!item.snapshot || !item.prediction) return null; const prediction = item.prediction; const down = prediction.prediction_label.toLowerCase() === "down"; return <article className={`comparison-model-column ${down ? "negative" : "positive"}`} key={item.ticker} style={{ "--ticker-color": itemColor(item, order) } as CSSProperties }><div className="model-column-title"><strong>{item.ticker}</strong><span className={`direction-badge ${down ? "negative" : "positive"}`}>{down ? <TrendingDown size={14} /> : <TrendingUp size={14} />} {prediction.prediction_label}</span></div><ProbabilityTrack prediction={prediction} /><div className="model-column-meta"><span>{prediction.horizon_days} trading day(s)</span><span>{dateLabel(prediction.as_of_date)}</span></div><span className={`evidence-label ${evidenceClass(prediction)}`}>{evidenceLabel(prediction)}</span></article>; })}</div></section>;
}

function buildInsights(items: ComparisonItem[], range: ComparisonRange) {
  const valid = items.filter((item) => item.snapshot).map((item) => ({ item, value: periodReturn(item.snapshot, range), risk: item.snapshot?.risk?.annualized_volatility_20d, volume: item.snapshot?.activity?.volume_ratio })).filter((entry) => typeof entry.value === "number" || typeof entry.risk === "number" || typeof entry.volume === "number");
  if (!valid.length) return ["Comparison insights will appear when enough market history is available."];
  const insights: string[] = [];
  const returns = valid.filter((entry) => typeof entry.value === "number");
  if (returns.length) { const high = returns.slice().sort((a, b) => (b.value as number) - (a.value as number))[0]; const low = returns.slice().sort((a, b) => (a.value as number) - (b.value as number))[0]; insights.push(`${displayName(high.item.snapshot, high.item.ticker)} has the strongest ${range} return at ${percent(high.value)}; ${displayName(low.item.snapshot, low.item.ticker)} is at ${percent(low.value)}.`); }
  const risks = valid.filter((entry) => typeof entry.risk === "number");
  if (risks.length) { const lowRisk = risks.slice().sort((a, b) => (a.risk as number) - (b.risk as number))[0]; insights.push(`${displayName(lowRisk.item.snapshot, lowRisk.item.ticker)} has the lowest recent volatility at ${percent(lowRisk.risk)}. Lower volatility describes smaller recent moves, not guaranteed safety.`); }
  const active = valid.filter((entry) => typeof entry.volume === "number").sort((a, b) => (b.volume as number) - (a.volume as number))[0];
  if (active?.volume !== undefined && active.volume !== null) insights.push(`${displayName(active.item.snapshot, active.item.ticker)} is the most active today relative to its own 20-day volume average at ${active.volume.toFixed(1)}×.`);
  const directions = items.filter((item) => item.prediction).map((item) => item.prediction?.prediction_label);
  if (directions.length > 1) insights.push(new Set(directions).size === 1 ? "The selected model directions agree for this horizon; check the evidence status beside each signal." : "The selected model directions differ for this horizon, so compare the probabilities and evidence labels before drawing a conclusion.");
  return insights.slice(0, 4);
}

function ComparisonErrorCard({ item, company }: { item: ComparisonItem; company?: Company }) {
  return <article className="comparison-error-card"><CompanyLogo company={company} className="comparison-card-logo" /><strong>{company?.company_name || item.ticker}</strong><span>{item.ticker}</span><div className="notice error">{item.error || "This ticker could not be loaded."}</div></article>;
}

export function ComparePage() {
  const companies = useQuery({ queryKey: ["companies"], queryFn: ({ signal }) => getCompanies(signal), staleTime: 86400000 });
  const { comparisonTickers, setComparisonTickers, comparisonHorizon, setComparisonHorizon } = useAppState();
  const [range, setRange] = useState<ComparisonRange>("3M");
  const [includeKlci, setIncludeKlci] = useState(true);

  useEffect(() => {
    if (!companies.data?.length) return;
    const valid = comparisonTickers.filter((ticker) => companies.data.some((company) => company.ticker === ticker));
    if (valid.length < 2) setComparisonTickers([companies.data[0].ticker, companies.data[1]?.ticker || companies.data[0].ticker]);
    else if (valid.length !== comparisonTickers.length) setComparisonTickers(valid);
  }, [companies.data, comparisonTickers, setComparisonTickers]);

  const comparison = useQuery({ queryKey: ["comparison", comparisonTickers, comparisonHorizon], queryFn: ({ signal }) => createComparison(comparisonTickers, comparisonHorizon, signal), enabled: comparisonTickers.length >= 2, staleTime: 900000, retry: false });
  const addTicker = () => { const next = companies.data?.find((company) => !comparisonTickers.includes(company.ticker)); if (next) setComparisonTickers([...comparisonTickers, next.ticker]); };
  const removeTicker = (index: number) => { if (index >= 2) setComparisonTickers(comparisonTickers.filter((_, itemIndex) => itemIndex !== index)); };
  const items = comparison.data?.items ?? [];
  const validItems = useMemo(() => successfulItems(items), [items]);
  const series = useMemo(() => buildComparisonSeries(items, range, includeKlci), [items, range, includeKlci]);
  const riskPoints = useMemo(() => buildRiskReturnPoints(items, range), [items, range]);
  const returnData = useMemo(() => validItems.flatMap((item, index) => { const value = periodReturn(item.snapshot, range); return typeof value === "number" ? [{ ticker: item.ticker, name: item.snapshot?.company_name || item.ticker, value, color: COMPARISON_PALETTE[index % COMPARISON_PALETTE.length] }] : []; }), [validItems, range]);
  const insights = useMemo(() => buildInsights(items, range), [items, range]);
  const layoutStyle = { "--comparison-count": Math.max(2, comparisonTickers.length) } as CSSProperties;
  const observationDates = validItems.map((item) => item.snapshot?.as_of_date).filter(Boolean) as string[];
  const sharedDate = observationDates.length > 0 && observationDates.every((date) => date === observationDates[0]);

  if (companies.isLoading) return <><PageHeader eyebrow="Analyze" title="Compare Tickers" description="Compare a small group of companies using the same market and model evidence." /><LoadingState /></>;
  if (companies.isError || !companies.data) return <><PageHeader eyebrow="Analyze" title="Compare Tickers" description="Compare a small group of companies using the same market and model evidence." /><ErrorState message={companies.error?.message || "Companies could not be loaded."} /></>;

  return <>
    <PageHeader eyebrow="Analyze" title="Compare Tickers" description="Follow the same market evidence across a small group of Bursa Malaysia companies." />
    <section className="comparison-selector" style={layoutStyle}>
      <div className="comparison-selector-heading"><span className="panel-kicker">Companies</span><button className="comparison-add-button" onClick={addTicker} disabled={comparisonTickers.length >= 5} aria-label={comparisonTickers.length >= 5 ? "Maximum of five tickers selected" : "Add ticker"}>{comparisonTickers.length >= 5 ? "5 of 5 selected" : "+ Add ticker"}</button></div>
      <div className="comparison-lanes">
        {comparisonTickers.map((ticker, index) => <div className="comparison-lane" key={ticker}>
          <CompanyLogo company={companies.data.find((item) => item.ticker === ticker)} className="comparison-logo" />
          <CompanySelector label={`Ticker ${index + 1}`} companies={companies.data} value={ticker} locked={comparisonTickers.filter((item) => item !== ticker)} onChange={(value) => setComparisonTickers(comparisonTickers.map((item, itemIndex) => itemIndex === index ? value : item))} />
          {index >= 2 && <button className="remove-button" onClick={() => removeTicker(index)}>Remove</button>}
        </div>)}
      </div>
      <HorizonSelector value={comparisonHorizon} onChange={setComparisonHorizon} />
    </section>
    {comparison.isLoading && <LoadingState label="Updating the comparison…" />}
    {comparison.isError && <ErrorState message={comparison.error.message} />}
    {comparison.data && <div className="comparison-runway" style={layoutStyle}>
      <div className="comparison-observation-strip"><div><CalendarDays size={16} /><span>{sharedDate ? `All companies observed ${dateLabel(observationDates[0])}` : "Observation dates differ between companies"}</span></div><span className="comparison-observation-note">Historical evidence · no ranking or buy/sell recommendation</span></div>
      {items.length > 0 && <ComparisonOverview items={items} order={comparisonTickers} companies={companies.data} range={range} />}
      <StockModelEvidence horizon={comparisonHorizon} compact />
      {items.some((item) => !item.snapshot || !item.prediction) && <section className="comparison-section comparison-errors-section"><span className="panel-kicker">Partial result</span><p>Some selected companies could not be loaded. The available lanes remain comparable.</p><div className="comparison-error-grid">{items.filter((item) => !item.snapshot || !item.prediction).map((item) => <ComparisonErrorCard key={item.ticker} item={item} company={companies.data.find((company) => company.ticker === item.ticker)} />)}</div></section>}
      <section className="comparison-section comparison-performance-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><TrendingUp size={18} /></div><div><span className="panel-kicker">Price journey</span><h2>Which price moved more?</h2><p>Every line starts at 100 on the first shared session. The height shows percentage movement, not the share-price size.</p></div></div><div className="comparison-chart-toolbar"><div className="segmented-control" aria-label="Comparison window">{RETURN_OPTIONS.map((option) => <button key={option.key} className={range === option.key ? "selected" : ""} onClick={() => setRange(option.key)}>{option.key}</button>)}</div><button className={`benchmark-toggle ${includeKlci ? "selected" : ""}`} onClick={() => setIncludeKlci((value) => !value)} aria-pressed={includeKlci}><Scale size={15} /> {includeKlci ? "KLCI shown" : "Show KLCI"}</button></div><ComparisonPerformanceChart data={series.data} series={series.series} includeKlci={includeKlci} /><p className="chart-caption"><Info size={15} /> Adjusted close includes distributions such as dividends. Use the date and legend to follow one company through the comparison.</p></section>
      <div className="comparison-two-column"><section className="comparison-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><BarChart3 size={18} /></div><div><span className="panel-kicker">Return map</span><h2>How did the selected window perform?</h2><p>Positive and negative bars show historical movement for the same period.</p></div></div><ComparisonReturnChart data={returnData} /></section><section className="comparison-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><Gauge size={18} /></div><div><span className="panel-kicker">Risk and return</span><h2>How large were the moves?</h2><p>Points farther right moved more recently; higher points returned more in the selected window.</p></div></div><ComparisonRiskReturnChart points={riskPoints} /></section></div>
      <section className="comparison-section comparison-range-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><Activity size={18} /></div><div><span className="panel-kicker">Position and activity</span><h2>Where is each company in its own range?</h2><p>Range position and volume are company-relative signals, so they should not be read as a league table.</p></div></div><div className="comparison-aligned-rows">{validItems.map((item) => { if (!item.snapshot) return null; const last = lastHistoryPoint(item.snapshot); return <div className="comparison-aligned-row" key={item.ticker} style={{ "--ticker-color": itemColor(item, comparisonTickers) } as CSSProperties}><div className="comparison-row-label"><span className="ticker-swatch" /><strong>{item.ticker}</strong><small>{item.snapshot.company_name}</small></div><RangeTrack snapshot={item.snapshot} /><div className="comparison-row-metric"><Volume2 size={15} /><strong>{item.snapshot.activity?.volume_ratio == null ? "Unavailable" : `${item.snapshot.activity.volume_ratio.toFixed(1)}×`}</strong><span>volume</span></div><div className="comparison-row-metric"><Gauge size={15} /><strong>{last?.rsi14 == null ? "Unavailable" : last.rsi14.toFixed(1)}</strong><span>RSI-14</span></div></div>; })}</div></section>
      <div className="comparison-two-column"><section className="comparison-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><ShieldCheck size={18} /></div><div><span className="panel-kicker">Market behaviour</span><h2>How does each stock move with KLCI?</h2><p>Beta describes sensitivity; correlation describes how consistently the directions have matched.</p></div></div><div className="comparison-behaviour-grid">{validItems.map((item) => { if (!item.snapshot) return null; const context = item.snapshot.market_context; return <div className="comparison-behaviour-card" key={item.ticker} style={{ "--ticker-color": itemColor(item, comparisonTickers) } as CSSProperties}><strong>{item.ticker}</strong><SnapshotMetric label="60-day beta" value={number(context?.beta60, 2)} note="market sensitivity" /><MetricBar value={context?.beta60} max={2} tone="blue" /><SnapshotMetric label="60-day correlation" value={number(context?.correlation60, 2)} note="shared direction" /><MetricBar value={context?.correlation60} max={1} tone="green" /></div>; })}</div></section><section className="comparison-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><Volume2 size={18} /></div><div><span className="panel-kicker">Risk details</span><h2>What should a beginner check?</h2><p>These measures describe recent movement size, not the direction of the next move.</p></div></div><div className="comparison-risk-list">{validItems.map((item) => { if (!item.snapshot) return null; return <div className="comparison-risk-row" key={item.ticker}><strong>{item.ticker}</strong><span>Volatility</span><b>{percent(item.snapshot.risk?.annualized_volatility_20d)}</b><span>ATR-14</span><b>{percent(item.snapshot.risk?.atr14_pct)}</b></div>; })}</div></section></div>
      <ProbabilitySection items={validItems} order={comparisonTickers} />
      <section className="comparison-section comparison-insights-section"><div className="comparison-section-heading"><div className="comparison-section-icon"><Target size={18} /></div><div><span className="panel-kicker">Beginner reading guide</span><h2>What stands out in this group?</h2><p>These notes are descriptive comparisons generated from the selected window; they are not investment advice.</p></div></div><div className="comparison-insights-grid">{insights.map((insight, index) => <div className="comparison-insight" key={insight}><span>0{index + 1}</span><p>{insight}</p></div>)}</div><div className="comparison-reading-order"><strong>Suggested reading order</strong><span>1. Price journey</span><span>2. Return and risk</span><span>3. KLCI relationship</span><span>4. Model evidence</span></div></section>
    </div>}
  </>;
}
