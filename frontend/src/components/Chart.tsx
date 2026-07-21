import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function PriceChart({ dates, prices, height = 250 }: { dates: string[]; prices: number[]; height?: number }) {
  const data = dates.map((date, index) => ({ date, value: prices[index] })).filter((item) => typeof item.value === "number");
  return <div style={{ width: "100%", height }}><ResponsiveContainer><LineChart data={data}><XAxis dataKey="date" hide /><YAxis hide domain={["auto", "auto"]} /><Tooltip formatter={(value) => typeof value === "number" ? [`RM ${value.toFixed(2)}`, "Close"] : [String(value), "Close"]} labelFormatter={(label) => String(label)} /><Line type="monotone" dataKey="value" stroke="#0071e3" strokeWidth={2.3} dot={false} /></LineChart></ResponsiveContainer></div>;
}

export function RelativeChart({ series, height = 320 }: { series: Record<string, number[]>; height?: number }) {
  const length = Math.max(...Object.values(series).map((values) => values.length), 0);
  const data = Array.from({ length }, (_, index) => Object.fromEntries([ ["index", index], ...Object.entries(series).map(([ticker, values]) => [ticker, values[index]]) ]));
  const colors = ["#0071e3", "#248a3d", "#d70015", "#9a6700", "#6e6e73"];
  return <div style={{ width: "100%", height }}><ResponsiveContainer><LineChart data={data}><XAxis dataKey="index" hide /><YAxis domain={["auto", "auto"]} tick={{ fontSize: 12 }} /><Tooltip /><LineChartLines keys={Object.keys(series)} colors={colors} /></LineChart></ResponsiveContainer></div>;
}

function LineChartLines({ keys, colors }: { keys: string[]; colors: string[] }) {
  return <>{keys.map((key, index) => <Line key={key} type="monotone" dataKey={key} stroke={colors[index % colors.length]} strokeWidth={2} dot={false} />)}</>;
}
