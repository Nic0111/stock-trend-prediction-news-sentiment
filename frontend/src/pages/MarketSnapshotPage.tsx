import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, CalendarDays, Gauge, Info, Layers3, LineChart as LineChartIcon, Scale, ShieldCheck, TrendingDown, TrendingUp, Volume2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getCompanies, getMarketSnapshot } from "../api/client";
import { LoadingState, ErrorState } from "../components/AsyncState";
import { CompanyLogo } from "../components/CompanyLogo";
import { CompanySelector } from "../components/CompanySelector";
import { MarketComparisonChart, MarketPriceChart, MomentumChart } from "../components/Chart";
import { PageHeader } from "../components/PageHeader";
import { useAppState } from "../state/AppState";
import type { MarketHistoryPoint, MarketSnapshot } from "../types";

type TimeRange = "1M" | "3M" | "6M" | "1Y";
type ChartMode = "line" | "candlestick";

const RANGE_LENGTHS: Record<TimeRange, number> = { "1M": 21, "3M": 63, "6M": 126, "1Y": 252 };

function money(value: number | null | undefined) {
  return value === null || value === undefined || !Number.isFinite(value) ? "Unavailable" : `RM ${value.toFixed(2)}`;
}

function percent(value: number | null | undefined, digits = 1) {
  return value === null || value === undefined || !Number.isFinite(value) ? "Unavailable" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

function number(value: number | null | undefined, digits = 0) {
  return value === null || value === undefined || !Number.isFinite(value) ? "Unavailable" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function dateLabel(value: string) {
  const parsed = new Date(`${value}T00:00:00`);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function snapshotPoints(snapshot: MarketSnapshot): MarketHistoryPoint[] {
  if (snapshot.history?.length) return snapshot.history;
  return snapshot.history_dates.map((date, index) => ({ date, open: null, high: null, low: null, close: snapshot.history_prices[index] ?? null, adjusted_close: snapshot.history_prices[index] ?? null, volume: null, dividend: null, split: null, ma20: null, ma60: null, rsi14: null, macd: null, macd_signal: null }));
}

function ValuePair({ label, value, note, tone }: { label: string; value: string; note?: string; tone?: "positive" | "negative" | "neutral" }) {
  return <div className="snapshot-value-pair"><span>{label}</span><strong className={tone ? `value-${tone}` : ""}>{value}</strong>{note && <small>{note}</small>}</div>;
}

function SectionIntro({ icon: Icon, eyebrow, title, copy }: { icon: typeof Activity; eyebrow: string; title: string; copy: string }) {
  return <div className="snapshot-section-intro"><div className="snapshot-section-icon"><Icon size={18} /></div><div><span className="panel-kicker">{eyebrow}</span><h2>{title}</h2><p>{copy}</p></div></div>;
}

function RangeMeter({ low, high, position }: { low: number | null | undefined; high: number | null | undefined; position: number | null | undefined }) {
  const safePosition = Math.max(0, Math.min(1, position ?? 0));
  return <div className="range-meter" aria-label={`52-week range from ${money(low)} to ${money(high)}`}><div className="range-meter-track"><i style={{ left: `${safePosition * 100}%` }} /></div><div className="range-meter-labels"><span>{money(low)}</span><strong>{position === null || position === undefined ? "Unavailable" : `${(safePosition * 100).toFixed(0)}% of range`}</strong><span>{money(high)}</span></div></div>;
}

function RsiGauge({ value }: { value: number | null | undefined }) {
  const safeValue = Math.max(0, Math.min(100, value ?? 50));
  return <div className="rsi-gauge" aria-label={`RSI 14 ${value === null || value === undefined ? "unavailable" : value.toFixed(1)}`}><div className="rsi-track"><i style={{ left: `${safeValue}%` }} /></div><div className="rsi-labels"><span>Oversold<br /><b>&lt; 30</b></span><span>Balanced<br /><b>30–70</b></span><span>Overbought<br /><b>&gt; 70</b></span></div><strong className="rsi-value">{value === null || value === undefined ? "Unavailable" : value.toFixed(1)}</strong></div>;
}

export function MarketSnapshotPage() {
  const companies = useQuery({ queryKey: ["companies"], queryFn: ({ signal }) => getCompanies(signal), staleTime: 86400000 });
  const { snapshotTicker, setSnapshotTicker } = useAppState();
  const [range, setRange] = useState<TimeRange>("3M");
  const [chartMode, setChartMode] = useState<ChartMode>("line");
  useEffect(() => { if (companies.data?.length && !companies.data.some((company) => company.ticker === snapshotTicker)) setSnapshotTicker(companies.data[0].ticker); }, [companies.data, snapshotTicker, setSnapshotTicker]);
  const snapshot = useQuery({ queryKey: ["market-snapshot", snapshotTicker], queryFn: ({ signal }) => getMarketSnapshot(snapshotTicker, signal), enabled: Boolean(snapshotTicker), staleTime: 300000, retry: 1 });
  const company = companies.data?.find((item) => item.ticker === snapshotTicker);
  const points = useMemo(() => snapshot.data ? snapshotPoints(snapshot.data) : [], [snapshot.data]);
  const visiblePoints = useMemo(() => points.slice(-RANGE_LENGTHS[range]), [points, range]);
  const contextHistory = snapshot.data?.market_context?.history ?? [];
  const visibleContext = useMemo(() => contextHistory.slice(-RANGE_LENGTHS[range]), [contextHistory, range]);
  const latest = snapshot.data;
  const chartIdentity = company ? { ticker: company.ticker, companyName: company.company_name } : latest ? { ticker: latest.ticker, companyName: latest.company_name } : null;
  const changeTone = latest?.change_pct === null || latest?.change_pct === undefined ? "neutral" : latest.change_pct >= 0 ? "positive" : "negative";
  const rsi = latest?.history?.at(-1)?.rsi14;
  const macdPoints = visiblePoints.filter((point) => point.macd !== null || point.macd_signal !== null);

  if (companies.isLoading) return <><PageHeader eyebrow="Analyze" title="Market Snapshot" description="Read price, activity, trend, risk, and Malaysian market context in one guided view." /><LoadingState /></>;
  if (companies.isError || !companies.data) return <><PageHeader eyebrow="Analyze" title="Market Snapshot" description="Read price, activity, trend, risk, and Malaysian market context in one guided view." /><ErrorState message={companies.error?.message || "Companies could not be loaded."} /></>;

  return <>
    <PageHeader eyebrow="Analyze" title="Market Snapshot" description="Read price, activity, trend, risk, and Malaysian market context in one guided view." />
    <section className="selector-panel compact market-selector"><CompanyLogo company={company} className="selector-logo" /><CompanySelector label="Company" companies={companies.data} value={snapshotTicker} onChange={setSnapshotTicker} /><div className="snapshot-selector-status"><span className="panel-kicker">Market evidence</span><strong>{latest?.sector || "Bursa Malaysia"}</strong><small>Updates automatically when you choose a company.</small></div></section>
    {snapshot.isLoading && <LoadingState label="Loading the latest available market snapshot…" />}
    {snapshot.isError && <ErrorState message={snapshot.error.message} />}
    {latest && <div className="snapshot-story">
      <section className={`snapshot-overview-panel ${changeTone}`}>
        <div className="snapshot-overview-heading"><div><span className="panel-kicker">{latest.company_name} · {latest.ticker}</span><h2>What is happening today?</h2><p>Latest available daily close, shown with the session range and market freshness.</p></div><span className={`freshness-pill ${latest.freshness_status || "unknown"}`}><span className="live-dot" /> {latest.freshness_status === "current" ? "Current session" : latest.freshness_status === "delayed" ? "Delayed session" : "Check timestamp"}</span></div>
        <div className="snapshot-price-line"><div className="snapshot-price-icon">{changeTone === "positive" ? <TrendingUp size={24} /> : changeTone === "negative" ? <TrendingDown size={24} /> : <Activity size={24} />}</div><div><strong>{money(latest.price)}</strong><span className={`value-${changeTone}`}>{latest.change === null ? "Unavailable" : `${latest.change >= 0 ? "+" : ""}${money(Math.abs(latest.change))} today`} · {percent(latest.change_pct, 2)}</span></div><div className="snapshot-observation"><CalendarDays size={16} /><span>Observed {dateLabel(latest.as_of_date)}</span></div></div>
        <div className="today-range"><div className="today-range-label"><span>Day low</span><strong>{money(latest.day_low)}</strong></div><div className="today-range-track"><i style={{ left: `${latest.day_low !== null && latest.day_high !== null && latest.day_high !== latest.day_low ? Math.max(0, Math.min(100, ((latest.price - latest.day_low) / (latest.day_high - latest.day_low)) * 100)) : 50}%` }} /><span className="range-open" style={{ left: `${latest.day_low !== null && latest.day_high !== null && latest.day_high !== latest.day_low && latest.open != null ? Math.max(0, Math.min(100, ((latest.open - latest.day_low) / (latest.day_high - latest.day_low)) * 100)) : 50}%` }} /></div><div className="today-range-label align-right"><span>Day high</span><strong>{money(latest.day_high)}</strong></div></div>
        <div className="metric-row snapshot-metrics"><ValuePair label="Previous close" value={money(latest.previous_close)} /><ValuePair label="Open" value={money(latest.open)} /><ValuePair label="Volume" value={number(latest.volume)} note="Shares traded" /></div>
      </section>
      {latest.stale_data_warning && <div className="notice warning"><strong>Check the timestamp.</strong> {latest.freshness_reason || "The latest stock session is behind the latest market session."}</div>}

      <section className="snapshot-section chart-story-section"><SectionIntro icon={LineChartIcon} eyebrow="Price journey" title="How has the stock moved?" copy="The line shows adjusted closing prices. Moving averages smooth the daily noise so a beginner can see the broader direction." /><div className="snapshot-chart-controls"><div className="segmented-control">{(Object.keys(RANGE_LENGTHS) as TimeRange[]).map((option) => <button key={option} className={range === option ? "selected" : ""} onClick={() => setRange(option)}>{option}</button>)}</div><div className="segmented-control"><button className={chartMode === "line" ? "selected" : ""} onClick={() => setChartMode("line")}>Line</button><button className={chartMode === "candlestick" ? "selected" : ""} onClick={() => setChartMode("candlestick")}>Candles</button></div></div><MarketPriceChart points={visiblePoints} identity={chartIdentity} mode={chartMode} /><p className="chart-caption"><Info size={15} /> Adjusted close accounts for distributions such as dividends, making periods easier to compare. Candles show each session’s open, high, low, and close.</p></section>

      <section className="snapshot-section performance-section"><SectionIntro icon={BarChart3} eyebrow="Return map" title="What has the price returned?" copy="These are historical changes over different trading windows, not promises about the future." /><div className="performance-grid">{[["1 day", latest.performance?.one_day], ["5 days", latest.performance?.five_day], ["1 month", latest.performance?.one_month], ["3 months", latest.performance?.three_month], ["YTD", latest.performance?.ytd], ["1 year", latest.performance?.one_year]].map(([label, value]) => <div className="performance-item" key={String(label)}><span>{String(label)}</span><strong className={typeof value === "number" ? value >= 0 ? "value-positive" : "value-negative" : ""}>{percent(value as number | null)}</strong><div className="performance-bar"><i className={typeof value === "number" && value < 0 ? "negative" : ""} style={{ width: `${Math.min(100, Math.max(8, Math.abs((value as number || 0) * 180)))}%` }} /></div></div>)}</div></section>

      <div className="snapshot-two-column"><section className="snapshot-section"><SectionIntro icon={Activity} eyebrow="Position" title="Where is it in its yearly range?" copy="A position near the top or bottom of the range describes location, not direction." /><RangeMeter low={latest.range_52_week?.low} high={latest.range_52_week?.high} position={latest.range_52_week?.position_pct} /><div className="snapshot-mini-grid"><ValuePair label="52-week low" value={money(latest.range_52_week?.low)} /><ValuePair label="52-week high" value={money(latest.range_52_week?.high)} /><ValuePair label="Current position" value={latest.range_52_week?.position_pct === null || latest.range_52_week?.position_pct === undefined ? "Unavailable" : `${(latest.range_52_week.position_pct * 100).toFixed(0)}%`} /></div></section><section className="snapshot-section"><SectionIntro icon={Volume2} eyebrow="Activity" title="How active was trading?" copy="Volume helps you see whether a move happened during relatively busy or quiet trading." /><div className="activity-visual"><div className="activity-column"><strong>{number(latest.volume)}</strong><span>Latest volume</span></div><div className="activity-column"><strong>{number(latest.activity?.average_volume_20d)}</strong><span>20-day average</span></div><div className="activity-ratio"><div className="activity-track"><i style={{ width: `${Math.min(100, Math.max(5, (latest.activity?.volume_ratio || 0) * 35))}%` }} /></div><strong>{latest.activity?.volume_ratio === null || latest.activity?.volume_ratio === undefined ? "Unavailable" : `${latest.activity.volume_ratio.toFixed(1)}×`}</strong><span>versus average</span></div></div></section></div>

      <div className="snapshot-two-column"><section className="snapshot-section"><SectionIntro icon={Gauge} eyebrow="Momentum" title="Is momentum stretched?" copy="RSI compares recent upward and downward closes. It is a context signal, not a buy or sell rule." /><RsiGauge value={rsi} /><div className="indicator-callout"><strong>{rsi === null || rsi === undefined ? "RSI unavailable" : rsi < 30 ? "Lower momentum zone" : rsi > 70 ? "Higher momentum zone" : "Balanced momentum zone"}</strong><span>RSI-14 · 30 to 70 is commonly interpreted as the middle zone.</span></div></section><section className="snapshot-section"><SectionIntro icon={ShieldCheck} eyebrow="Risk" title="How much does it move?" copy="Volatility and ATR describe the size of recent moves; they do not predict whether the next move is up or down." /><div className="risk-metric-list"><ValuePair label="Annualized 20-day volatility" value={percent(latest.risk?.annualized_volatility_20d)} /><ValuePair label="ATR-14 as % of price" value={percent(latest.risk?.atr14_pct)} /><ValuePair label="Latest daily range" value={percent(latest.risk?.daily_range_pct)} /></div></section></div>

      <section className="snapshot-section momentum-section"><SectionIntro icon={Activity} eyebrow="Momentum path" title="MACD and signal line" copy="The lines compare two smoothed price averages. A crossover can describe changing momentum, but it is not a standalone forecast." /><MomentumChart points={macdPoints} identity={chartIdentity} /><div className="chart-caption"><Info size={15} /> Values are normalized by adjusted close to keep the chart comparable across companies.</div></section>

      <section className="snapshot-section market-context-section"><SectionIntro icon={Scale} eyebrow="Market context" title="How did it move versus KLCI?" copy="Both lines start at 100 so the relative journey is easier to compare with Malaysia’s broad market index." /><MarketComparisonChart points={visiblePoints} benchmark={visibleContext} identity={chartIdentity} /><div className="context-metric-grid"><ValuePair label="1-month excess return" value={percent(latest.market_context?.excess_return_one_month)} /><ValuePair label="60-day beta" value={number(latest.market_context?.beta60, 2)} note="Movement relative to KLCI" /><ValuePair label="60-day correlation" value={number(latest.market_context?.correlation60, 2)} note="Shared direction" /><ValuePair label="KLCI trend" value={latest.market_context?.klci_trend || "Unavailable"} /><ValuePair label="KLCI 1-month return" value={percent(latest.market_context?.klci_return_one_month)} /><ValuePair label="KLCI 20-day volatility" value={percent(latest.market_context?.klci_volatility_20d)} /></div></section>

      <section className="snapshot-section events-section"><SectionIntro icon={Layers3} eyebrow="Events" title="Dividends and stock splits" copy="Corporate actions can change how a historical chart should be read. Markers appear when the data source reports them." /><div className="event-list">{points.filter((point) => (point.dividend ?? 0) > 0 || (point.split ?? 0) > 0).slice(-8).reverse().map((point) => <div className="event-row" key={point.date}><CalendarDays size={16} /><strong>{dateLabel(point.date)}</strong><span>{(point.dividend ?? 0) > 0 ? `Dividend ${money(point.dividend)}` : ""}{(point.split ?? 0) > 0 ? ` Split ${point.split}` : ""}</span></div>)}{!points.some((point) => (point.dividend ?? 0) > 0 || (point.split ?? 0) > 0) && <p className="empty-event">No recorded corporate actions in this period.</p>}</div></section>

      <section className="snapshot-reading-guide"><div><span className="panel-kicker">Beginner reading order</span><h2>Five questions to ask before moving on</h2></div><ol><li><strong>When?</strong><span>Check the observation date and freshness.</span></li><li><strong>What?</strong><span>Read the price and session range.</span></li><li><strong>How active?</strong><span>Compare today’s volume with its average.</span></li><li><strong>How risky?</strong><span>Review trend, momentum, and volatility together.</span></li><li><strong>Compared with what?</strong><span>Use KLCI as the Malaysian market reference.</span></li></ol><p className="muted-note">Market Snapshot explains observed evidence. Stock Outlook and News &amp; Sentiment provide separate model and news views.</p></section>
    </div>}
  </>;
}
