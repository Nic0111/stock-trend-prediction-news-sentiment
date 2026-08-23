import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { BenchmarkPoint, MarketHistoryPoint } from "../types";
import type { ComparisonRiskPoint, ComparisonSeries, ComparisonSeriesPoint } from "../utils/comparison";

export interface ChartCompanyIdentity {
  ticker: string;
  companyName: string;
}

type TooltipEntry = {
  value?: unknown;
  name?: unknown;
  dataKey?: unknown;
  color?: string;
  stroke?: string;
  fill?: string;
  payload?: Record<string, unknown>;
};

type CompanyTooltipProps = {
  active?: boolean;
  label?: unknown;
  payload?: readonly TooltipEntry[];
  identity?: ChartCompanyIdentity | null;
  seriesIdentity?: Record<string, ChartCompanyIdentity>;
  formatLabel?: (label: unknown) => string;
  formatValue?: (value: unknown, name: string, entry: TooltipEntry) => { value: string; label: string };
};

function identityForEntry(entry: TooltipEntry, seriesIdentity?: Record<string, ChartCompanyIdentity>) {
  if (!seriesIdentity) return undefined;
  const payloadTicker = entry.payload?.ticker;
  const keys = [entry.dataKey, entry.name, payloadTicker].filter((value) => value !== null && value !== undefined).map(String);
  return keys.map((key) => seriesIdentity[key]).find(Boolean);
}

function defaultTooltipValue(value: unknown, name: string) {
  return { value: value === null || value === undefined ? "Unavailable" : String(value), label: name };
}

export function CompanyTooltip({ active, label, payload, identity, seriesIdentity, formatLabel, formatValue }: CompanyTooltipProps) {
  if (!active || !payload?.length) return null;
  const rows = payload.filter((entry) => entry.value !== null && entry.value !== undefined);
  if (!rows.length) return null;
  const hasSeriesIdentity = Boolean(seriesIdentity && Object.keys(seriesIdentity).length);
  return <div className="company-chart-tooltip" role="status">
    {!hasSeriesIdentity && identity && <div className="company-chart-tooltip-identity"><strong>{identity.companyName}</strong><span>{identity.ticker}</span></div>}
    <div className="company-chart-tooltip-date">{formatLabel ? formatLabel(label) : String(label ?? "")}</div>
    <div className="company-chart-tooltip-rows">
      {rows.map((entry, index) => {
        const key = String(entry.dataKey ?? entry.name ?? index);
        const rowIdentity = hasSeriesIdentity ? identityForEntry(entry, seriesIdentity) : undefined;
        const formatted = formatValue ? formatValue(entry.value, String(entry.name ?? entry.dataKey ?? "Value"), entry) : defaultTooltipValue(entry.value, String(entry.name ?? entry.dataKey ?? "Value"));
        return <div className="company-chart-tooltip-row" key={`${key}-${index}`}>
          <span className="company-chart-tooltip-row-label">
            <i style={{ background: entry.color || entry.stroke || entry.fill || "#687383" }} />
            {rowIdentity && <span className="company-chart-tooltip-series"><strong>{rowIdentity.companyName}</strong><em>{rowIdentity.ticker}</em></span>}
            {!rowIdentity && <span>{formatted.label}</span>}
          </span>
          <strong>{formatted.value}</strong>
        </div>;
      })}
    </div>
  </div>;
}

export function PriceChart({ dates, prices, identity, height = 250 }: { dates: string[]; prices: number[]; identity?: ChartCompanyIdentity | null; height?: number }) {
  const data = dates.map((date, index) => ({ date, value: prices[index] })).filter((item) => typeof item.value === "number");
  return <div style={{ width: "100%", height }}><ResponsiveContainer><LineChart data={data}><XAxis dataKey="date" hide /><YAxis hide domain={["auto", "auto"]} /><Tooltip content={<CompanyTooltip identity={identity} formatValue={(value) => ({ value: typeof value === "number" ? `RM ${value.toFixed(2)}` : String(value), label: "Close" })} />} /><Line type="monotone" dataKey="value" name="Close" stroke="#0071e3" strokeWidth={2.3} dot={false} /></LineChart></ResponsiveContainer></div>;
}

export function MarketPriceChart({ points, identity, mode = "line", height = 390 }: { points: MarketHistoryPoint[]; identity?: ChartCompanyIdentity | null; mode?: "line" | "candlestick"; height?: number }) {
  if (mode === "candlestick") return <CandlestickChart points={points} identity={identity} height={height} />;
  const data = points.map((point) => ({
    ...point,
    value: point.adjusted_close ?? point.close,
    volume_millions: point.volume === null ? null : point.volume / 1_000_000,
  }));
  return <div className="market-chart" style={{ width: "100%", height }}><ResponsiveContainer><ComposedChart data={data} margin={{ top: 12, right: 10, left: 0, bottom: 6 }}>
    <CartesianGrid vertical={false} stroke="rgba(61,93,133,0.12)" />
    <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#687383" }} minTickGap={38} tickFormatter={(value) => String(value).slice(5)} />
    <YAxis yAxisId="price" orientation="right" tick={{ fontSize: 11, fill: "#687383" }} tickFormatter={(value) => `RM ${Number(value).toFixed(1)}`} domain={["auto", "auto"]} width={58} />
    <YAxis yAxisId="volume" hide domain={[0, "auto"]} />
    <Tooltip content={<CompanyTooltip identity={identity} formatLabel={(label) => `Session ${String(label)}`} formatValue={(value, name) => {
      if (name === "volume_millions") return { value: `${Number(value).toFixed(2)}M`, label: "Volume" };
      return { value: `RM ${Number(value).toFixed(2)}`, label: name === "value" ? "Adjusted close" : String(name) };
    }} />} />
    <Legend verticalAlign="top" align="left" height={32} wrapperStyle={{ fontSize: 12 }} />
    <Bar yAxisId="volume" dataKey="volume_millions" name="Volume" fill="rgba(0,113,227,0.15)" barSize={8} />
    <Area yAxisId="price" type="monotone" dataKey="value" name="Adjusted close" stroke="#0071e3" fill="rgba(0,113,227,0.08)" strokeWidth={2.5} dot={false} connectNulls />
    <Line yAxisId="price" type="monotone" dataKey="ma20" name="MA20" stroke="#248a3d" strokeWidth={1.8} dot={false} connectNulls />
    <Line yAxisId="price" type="monotone" dataKey="ma60" name="MA60" stroke="#9a6700" strokeWidth={1.8} dot={false} connectNulls />
  </ComposedChart></ResponsiveContainer></div>;
}

function CandlestickChart({ points, identity, height }: { points: MarketHistoryPoint[]; identity?: ChartCompanyIdentity | null; height: number }) {
  const width = 1000;
  const top = 20;
  const bottom = 38;
  const chartHeight = 290;
  const values = points.flatMap((point) => [point.high, point.low, point.ma20, point.ma60].filter((value): value is number => typeof value === "number"));
  const minimum = Math.min(...values, 0);
  const maximum = Math.max(...values, 1);
  const y = (value: number) => top + ((maximum - value) / Math.max(maximum - minimum, 0.0001)) * chartHeight;
  const step = width / Math.max(points.length, 1);
  const candleWidth = Math.max(2, step * 0.55);
  const linePath = (key: "ma20" | "ma60") => points.map((point, index) => point[key] === null ? "" : `${index * step + step / 2},${y(point[key] as number)}`).filter(Boolean).join(" ");
  return <div className="market-chart candle-chart" style={{ width: "100%", height }}><svg viewBox={`0 0 ${width} ${top + chartHeight + bottom}`} role="img" aria-label="Daily candlestick price chart with moving averages">
    <line x1="0" x2={width} y1={top + chartHeight} y2={top + chartHeight} stroke="rgba(61,93,133,0.18)" />
    {points.map((point, index) => {
      if (point.high === null || point.low === null || point.open === null || point.close === null) return null;
      const x = index * step + step / 2;
      const positive = point.close >= point.open;
      const color = positive ? "#248a3d" : "#d70015";
      const bodyTop = Math.min(y(point.open), y(point.close));
      const bodyHeight = Math.max(1.5, Math.abs(y(point.close) - y(point.open)));
      return <g key={point.date}><title>{`${identity ? `${identity.companyName} (${identity.ticker}) · ` : ""}${point.date}: Open RM ${point.open.toFixed(2)}, high RM ${point.high.toFixed(2)}, low RM ${point.low.toFixed(2)}, close RM ${point.close.toFixed(2)}`}</title><line x1={x} x2={x} y1={y(point.high)} y2={y(point.low)} stroke={color} strokeWidth={1.2} /><rect x={x - candleWidth / 2} y={bodyTop} width={candleWidth} height={bodyHeight} fill={positive ? "rgba(36,138,61,0.55)" : "rgba(215,0,21,0.55)"} stroke={color} /></g>;
    })}
    <polyline points={linePath("ma20")} fill="none" stroke="#248a3d" strokeWidth="2" />
    <polyline points={linePath("ma60")} fill="none" stroke="#9a6700" strokeWidth="2" />
    <text x="8" y={top + chartHeight + 26} fill="#687383" fontSize="14">Older sessions</text><text x={width - 100} y={top + chartHeight + 26} fill="#687383" fontSize="14">Latest</text>
  </svg></div>;
}

export function MarketComparisonChart({ points, benchmark, identity, height = 320 }: { points: MarketHistoryPoint[]; benchmark: BenchmarkPoint[]; identity?: ChartCompanyIdentity | null; height?: number }) {
  const benchmarkMap = new Map(benchmark.map((point) => [point.date, point.adjusted_close ?? point.close]));
  const stockBase = points.find((point) => point.adjusted_close !== null)?.adjusted_close;
  const benchmarkBase = benchmark.find((point) => benchmarkMap.get(point.date) !== null)?.adjusted_close ?? benchmark.find((point) => benchmarkMap.get(point.date) !== null)?.close;
  const data = points.map((point) => {
    const stock = point.adjusted_close;
    const market = benchmarkMap.get(point.date) ?? null;
    return { date: point.date, stock: stockBase && stock ? (stock / stockBase) * 100 : null, klci: benchmarkBase && market ? (market / benchmarkBase) * 100 : null };
  });
  return <div className="market-chart" style={{ width: "100%", height }}><ResponsiveContainer><LineChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 8 }}>
    <CartesianGrid vertical={false} stroke="rgba(61,93,133,0.12)" /><XAxis dataKey="date" tick={{ fontSize: 11, fill: "#687383" }} minTickGap={38} tickFormatter={(value) => String(value).slice(5)} /><YAxis tick={{ fontSize: 11, fill: "#687383" }} domain={["auto", "auto"]} width={42} /><Tooltip content={<CompanyTooltip seriesIdentity={{ stock: identity || { ticker: "Selected ticker", companyName: "Selected company" }, klci: { ticker: "^KLSE", companyName: "FTSE Bursa Malaysia KLCI" } }} formatLabel={(label) => `Session ${String(label)}`} formatValue={(value, name) => ({ value: `${Number(value).toFixed(1)}`, label: name === "stock" ? "Stock (base 100)" : "KLCI (base 100)" })} />} /><Legend verticalAlign="top" align="left" height={28} wrapperStyle={{ fontSize: 12 }} /><ReferenceLine y={100} stroke="rgba(61,93,133,0.3)" strokeDasharray="4 4" /><Line type="monotone" dataKey="stock" name="Stock" stroke="#0071e3" strokeWidth={2.5} dot={false} connectNulls /><Line type="monotone" dataKey="klci" name="KLCI" stroke="#687383" strokeWidth={2} dot={false} connectNulls />
  </LineChart></ResponsiveContainer></div>;
}

export function MomentumChart({ points, identity, height = 220 }: { points: MarketHistoryPoint[]; identity?: ChartCompanyIdentity | null; height?: number }) {
  return <div className="market-chart" style={{ width: "100%", height }}><ResponsiveContainer><LineChart data={points} margin={{ top: 10, right: 12, left: 0, bottom: 5 }}>
    <CartesianGrid vertical={false} stroke="rgba(61,93,133,0.12)" /><XAxis dataKey="date" hide /><YAxis tick={{ fontSize: 11, fill: "#687383" }} width={36} domain={[-0.04, 0.04]} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} /><Tooltip content={<CompanyTooltip identity={identity} formatLabel={(label) => `Session ${String(label)}`} formatValue={(value, name) => ({ value: `${(Number(value) * 100).toFixed(2)}%`, label: String(name) })} />} /><ReferenceLine y={0} stroke="rgba(61,93,133,0.36)" /><Line type="monotone" dataKey="macd" name="MACD" stroke="#0071e3" strokeWidth={2} dot={false} connectNulls /><Line type="monotone" dataKey="macd_signal" name="Signal" stroke="#9a6700" strokeWidth={1.8} dot={false} connectNulls />
  </LineChart></ResponsiveContainer></div>;
}

export function RelativeChart({ series, identities = {}, height = 320 }: { series: Record<string, number[]>; identities?: Record<string, ChartCompanyIdentity>; height?: number }) {
  const length = Math.max(...Object.values(series).map((values) => values.length), 0);
  const data = Array.from({ length }, (_, index) => Object.fromEntries([["index", index], ...Object.entries(series).map(([ticker, values]) => [ticker, values[index]])]));
  const colors = ["#0071e3", "#248a3d", "#d70015", "#9a6700", "#6e6e73"];
  return <div style={{ width: "100%", height }}><ResponsiveContainer><LineChart data={data}><XAxis dataKey="index" hide /><YAxis domain={["auto", "auto"]} tick={{ fontSize: 12 }} /><Tooltip content={<CompanyTooltip seriesIdentity={identities} />} /><LineChartLines keys={Object.keys(series)} colors={colors} /></LineChart></ResponsiveContainer></div>;
}

export function ComparisonPerformanceChart({ data, series, includeKlci = false, height = 360 }: { data: ComparisonSeriesPoint[]; series: ComparisonSeries[]; includeKlci?: boolean; height?: number }) {
  if (!data.length || !series.length) return <div className="chart-empty">There is not enough shared history to draw this comparison.</div>;
  return <div className="market-chart comparison-performance-chart" style={{ width: "100%", height }}>
    <ResponsiveContainer>
      <LineChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 8 }}>
        <CartesianGrid vertical={false} stroke="rgba(61,93,133,0.12)" />
        <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#687383" }} minTickGap={38} tickFormatter={(value) => String(value).slice(5)} />
        <YAxis tick={{ fontSize: 11, fill: "#687383" }} tickFormatter={(value) => `${Number(value).toFixed(0)}`} domain={["auto", "auto"]} width={42} />
        <Tooltip content={<CompanyTooltip seriesIdentity={{ ...Object.fromEntries(series.map((entry) => [entry.ticker, { ticker: entry.ticker, companyName: entry.name }])), ...(includeKlci ? { KLCI: { ticker: "^KLSE", companyName: "FTSE Bursa Malaysia KLCI" } } : {}) }} formatLabel={(label) => `Session ${String(label)}`} formatValue={(value, name) => ({ value: `${Number(value).toFixed(1)}`, label: name === "KLCI" ? "KLCI (base 100)" : `${String(name)} (base 100)` })} />} />
        <Legend verticalAlign="top" align="left" height={30} wrapperStyle={{ fontSize: 12 }} />
        <ReferenceLine y={100} stroke="rgba(61,93,133,0.34)" strokeDasharray="4 4" />
        {series.map((entry) => <Line key={entry.ticker} type="monotone" dataKey={entry.ticker} name={entry.ticker} stroke={entry.color} strokeWidth={2.4} dot={false} connectNulls={false} />)}
        {includeKlci && <Line type="monotone" dataKey="KLCI" name="KLCI" stroke="#687383" strokeWidth={1.8} strokeDasharray="5 4" dot={false} connectNulls={false} />}
      </LineChart>
    </ResponsiveContainer>
  </div>;
}

export function ComparisonReturnChart({ data, height = 300 }: { data: Array<{ ticker: string; name: string; value: number; color: string }>; height?: number }) {
  if (!data.length) return <div className="chart-empty">Return data is unavailable for this comparison.</div>;
  return <div className="market-chart" style={{ width: "100%", height }}>
    <ResponsiveContainer>
      <BarChart data={data} layout="vertical" margin={{ top: 12, right: 18, left: 12, bottom: 8 }}>
        <CartesianGrid horizontal={false} stroke="rgba(61,93,133,0.12)" />
        <XAxis type="number" tick={{ fontSize: 11, fill: "#687383" }} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} domain={["auto", "auto"]} />
        <YAxis type="category" dataKey="ticker" tick={{ fontSize: 11, fill: "#233e61", fontWeight: 650 }} width={58} />
        <Tooltip content={<CompanyTooltip seriesIdentity={Object.fromEntries(data.map((item) => [item.ticker, { ticker: item.ticker, companyName: item.name }]))} formatLabel={(label) => `${String(label)} selected-period return`} formatValue={(value) => ({ value: `${(Number(value) * 100).toFixed(1)}%`, label: "Return" })} />} />
        <ReferenceLine x={0} stroke="rgba(61,93,133,0.38)" />
        <Bar dataKey="value" name="Return" radius={[0, 7, 7, 0]} minPointSize={3}>
          {data.map((entry) => <Cell key={entry.ticker} fill={entry.value >= 0 ? entry.color : "#d14d72"} fillOpacity={0.88} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>;
}

export function ComparisonRiskReturnChart({ points, height = 320 }: { points: ComparisonRiskPoint[]; height?: number }) {
  if (!points.length) return <div className="chart-empty">Risk and return data is unavailable for this comparison.</div>;
  const medianRisk = points.slice().sort((a, b) => a.risk - b.risk)[Math.floor(points.length / 2)]?.risk;
  const medianReturn = points.slice().sort((a, b) => a.returnValue - b.returnValue)[Math.floor(points.length / 2)]?.returnValue;
  return <div className="market-chart" style={{ width: "100%", height }}>
    <ResponsiveContainer>
      <ScatterChart margin={{ top: 16, right: 18, left: 6, bottom: 18 }}>
        <CartesianGrid stroke="rgba(61,93,133,0.12)" />
        <XAxis type="number" dataKey="risk" name="Annualized volatility" tick={{ fontSize: 11, fill: "#687383" }} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} domain={["auto", "auto"]} label={{ value: "Recent movement size", position: "insideBottom", offset: -8, fontSize: 11, fill: "#687383" }} />
        <YAxis type="number" dataKey="returnValue" name="Return" tick={{ fontSize: 11, fill: "#687383" }} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`} domain={["auto", "auto"]} width={44} label={{ value: "Selected return", angle: -90, position: "insideLeft", fontSize: 11, fill: "#687383" }} />
        <ZAxis range={[110, 110]} />
        <Tooltip cursor={{ strokeDasharray: "4 4" }} content={<CompanyTooltip seriesIdentity={Object.fromEntries(points.map((point) => [point.ticker, { ticker: point.ticker, companyName: point.name }]))} formatLabel={(label) => String(label || "Company")} formatValue={(value, name) => ({ value: `${(Number(value) * 100).toFixed(1)}%`, label: name === "risk" ? "Volatility" : "Return" })} />} />
        {typeof medianRisk === "number" && <ReferenceLine x={medianRisk} stroke="rgba(61,93,133,0.32)" strokeDasharray="4 4" />}
        {typeof medianReturn === "number" && <ReferenceLine y={medianReturn} stroke="rgba(61,93,133,0.32)" strokeDasharray="4 4" />}
        {points.map((point) => <Scatter key={point.ticker} name={point.ticker} data={[point]} fill={point.color} />)}
      </ScatterChart>
    </ResponsiveContainer>
  </div>;
}

function LineChartLines({ keys, colors }: { keys: string[]; colors: string[] }) {
  return <>{keys.map((key, index) => <Line key={key} type="monotone" dataKey={key} stroke={colors[index % colors.length]} strokeWidth={2} dot={false} />)}</>;
}
