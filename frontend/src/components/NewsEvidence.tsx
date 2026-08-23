import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { BarChart3, Clock3, Info, Newspaper, ShieldCheck, Sparkles } from "lucide-react";
import type { NewsArticle, NewsArticleContribution, NewsHorizonContext } from "../types";

const TONE_COLORS = {
  positive: "#248a3d",
  neutral: "#8994a3",
  negative: "#d70015",
};

type NewsChartIdentity = { ticker: string; companyName: string };
type NewsTooltipEntry = { value?: unknown; name?: unknown; color?: string; payload?: unknown };

function NewsChartTooltip({ active, payload, label, identity, kind }: { active?: boolean; payload?: readonly NewsTooltipEntry[]; label?: unknown; identity?: NewsChartIdentity | null; kind: "influence" | "timeline" | "mix" | "publisher" }) {
  if (!active || !payload?.length) return null;
  const first = payload[0]?.payload as Record<string, unknown> | undefined;
  if (!first) return null;
  const title = String(first.title || first.publisher || label || "News evidence");
  const source = first.source ? String(first.source) : "";
  const tone = typeof first.y === "number" ? (first.y === 1 ? "Positive" : first.y === -1 ? "Negative" : "Neutral") : "";
  return <div className="company-chart-tooltip news-chart-tooltip" role="status">
    {identity && <div className="company-chart-tooltip-identity"><strong>{identity.companyName}</strong><span>{identity.ticker}</span></div>}
    <div className="company-chart-tooltip-date">{kind === "timeline" ? dateLabel(String(first.publishedAt || "")) : kind === "mix" ? "Selected context" : kind === "publisher" ? "Usable article sources" : "Tone contribution"}</div>
    {kind === "influence" && <><strong className="news-tooltip-title">{title}</strong>{source && <span className="news-tooltip-source">{source}</span>}<div className="company-chart-tooltip-row"><span>Tone contribution</span><strong>{percent(Number(first.signedContribution || 0), 1)}</strong></div><div className="company-chart-tooltip-row"><span>FinBERT confidence</span><strong>{confidence(Number(first.confidence || 0))}</strong></div></>}
    {kind === "timeline" && <><strong className="news-tooltip-title">{title}</strong>{source && <span className="news-tooltip-source">{source}</span>}<div className="company-chart-tooltip-row"><span>Tone</span><strong>{tone}</strong></div><div className="company-chart-tooltip-row"><span>FinBERT confidence</span><strong>{confidence(Number(first.confidence || 0))}</strong></div></>}
    {kind === "mix" && <div className="company-chart-tooltip-rows">{payload.filter((entry) => entry.value !== null && entry.value !== undefined).map((entry, index) => <div className="company-chart-tooltip-row" key={`${String(entry.name || "tone")}-${index}`}><span className="company-chart-tooltip-row-label"><i style={{ background: entry.color || "#687383" }} /><span>{String(entry.name || "Tone")}</span></span><strong>{String(entry.value)}</strong></div>)}</div>}
    {kind === "publisher" && <div className="company-chart-tooltip-row"><span>{title}</span><strong>{String(first.count || 0)} usable</strong></div>}
  </div>;
}

function articleId(article: NewsArticle) {
  return String(article.article_id || article.url || article.title || "article");
}

function toneForScore(score: number) {
  if (score > 0) return "positive";
  if (score < 0) return "negative";
  return "neutral";
}

function dateValue(value?: string) {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed.getTime();
}

function dateLabel(value?: string) {
  const parsed = value ? new Date(value) : null;
  return !parsed || Number.isNaN(parsed.getTime())
    ? "Date unavailable"
    : parsed.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

function percent(value: number | null | undefined, digits = 0) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`
    : "Unavailable";
}

function confidence(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "Unavailable";
}

export interface NewsContributionDatum {
  articleId: string;
  title: string;
  source: string;
  publishedAt?: string;
  signedContribution: number;
  normalizedWeight: number;
  confidence: number;
  ageDays: number;
  tone: "positive" | "neutral" | "negative";
}

export function buildNewsContributionData(
  articles: NewsArticle[],
  context: NewsHorizonContext,
): NewsContributionDatum[] {
  const rows = context.article_contributions || [];
  const articlesById = new Map(articles.map((article) => [articleId(article), article]));
  return rows
    .map((row: NewsArticleContribution) => {
      const article = articlesById.get(row.article_id);
      return {
        articleId: row.article_id,
        title: article?.title || "Untitled article",
        source: article?.source_domain || article?.source_name || "Unknown publisher",
        publishedAt: article?.published_at,
        signedContribution: Number(row.signed_contribution || 0),
        normalizedWeight: Number(row.normalized_weight || 0),
        confidence: Number(row.sentiment_confidence || 0),
        ageDays: Number(row.age_days || 0),
        tone: toneForScore(Number(row.sentiment_score || 0)) as NewsContributionDatum["tone"],
      };
    })
    .sort((left, right) => Math.abs(right.signedContribution) - Math.abs(left.signedContribution));
}

export function NewsInfluenceChart({
  articles,
  context,
  identity,
  onSelectArticle,
}: {
  articles: NewsArticle[];
  context: NewsHorizonContext;
  identity?: NewsChartIdentity | null;
  onSelectArticle?: (articleId: string) => void;
}) {
  const data = buildNewsContributionData(articles, context);
  if (!data.length) {
    return <div className="news-chart-empty"><Info size={17} /><span>No usable articles are contributing to this horizon yet.</span></div>;
  }
  const magnitude = Math.max(...data.map((item) => Math.abs(item.signedContribution)), 0.03) * 1.2;
  return <div className="news-influence-chart">
    <div className="news-chart-axis-note"><span>Pushes tone negative</span><span>Balanced</span><span>Pushes tone positive</span></div>
    <div className="news-chart-frame" style={{ height: Math.max(220, data.length * 46) }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 12, left: 4, bottom: 8 }}>
          <CartesianGrid horizontal={false} stroke="rgba(61,93,133,0.12)" />
          <XAxis type="number" domain={[-magnitude, magnitude]} tick={{ fontSize: 10, fill: "#687383" }} tickFormatter={(value) => percent(Number(value))} />
          <YAxis type="category" dataKey="source" width={106} tick={{ fontSize: 11, fill: "#233e61", fontWeight: 650 }} />
          <ReferenceLine x={0} stroke="rgba(61,93,133,0.52)" />
          <Tooltip cursor={{ fill: "rgba(0,113,227,0.05)" }} content={<NewsChartTooltip identity={identity} kind="influence" />} />
          <Bar dataKey="signedContribution" name="Tone contribution" radius={5} minPointSize={3} onClick={(entry) => onSelectArticle?.(String(((entry as unknown) as NewsContributionDatum).articleId))}>
            {data.map((entry) => <Cell key={entry.articleId} fill={TONE_COLORS[entry.tone]} fillOpacity={0.86} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
    <div className="news-influence-list">
      {data.map((item) => <button key={item.articleId} className="news-influence-item" type="button" onClick={() => onSelectArticle?.(item.articleId)}>
        <span className={`news-influence-dot ${item.tone}`} />
        <span className="news-influence-title">{item.title}</span>
        <span className="news-influence-value">{percent(item.signedContribution, 1)}</span>
      </button>)}
    </div>
  </div>;
}

interface TimelinePoint {
  x: number;
  y: number;
  confidence: number;
  ageDays: number;
  title: string;
  source: string;
  publishedAt?: string;
}

function timelinePoints(articles: NewsArticle[], asOfDate: string): TimelinePoint[] {
  const asOf = new Date(`${asOfDate}T23:59:59Z`).getTime();
  return articles.flatMap((article) => {
    const x = dateValue(article.published_at);
    if (x === null) return [];
    const score = Number(article.sentiment_score);
    const y = score === 1 ? 1 : score === -1 ? -1 : 0;
    return [{
      x,
      y,
      confidence: Number(article.sentiment_probability || 0.5),
      ageDays: Math.max(0, (asOf - x) / 86400000),
      title: article.title || "Untitled article",
      source: article.source_domain || article.source_name || "Unknown publisher",
      publishedAt: article.published_at,
    }];
  });
}

export function NewsToneTimeline({
  articles,
  historicalArticles = [],
  context,
  asOfDate,
  identity,
}: {
  articles: NewsArticle[];
  historicalArticles?: NewsArticle[];
  context: NewsHorizonContext;
  asOfDate: string;
  identity?: NewsChartIdentity | null;
}) {
  const active = timelinePoints(articles, asOfDate);
  const historical = timelinePoints(historicalArticles, asOfDate);
  const points = [...active, ...historical];
  if (!points.length) return <div className="news-chart-empty"><Clock3 size={17} /><span>No dated articles are available for this timeline.</span></div>;
  const end = new Date(`${asOfDate}T23:59:59Z`).getTime();
  const start = end - Number(context.window_days || 0) * 86400000;
  const grouped = (tone: number) => active.filter((point) => point.y === tone);
  return <div className="news-timeline-chart">
    <div className="news-timeline-legend"><span><i className="positive" />Positive</span><span><i className="neutral" />Neutral</span><span><i className="negative" />Negative</span><span><i className="historical" />Older context</span></div>
    <div className="news-chart-frame" style={{ height: 285 }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 12, right: 16, left: 2, bottom: 8 }}>
          <CartesianGrid stroke="rgba(61,93,133,0.12)" />
          <XAxis type="number" dataKey="x" domain={[Math.min(start, ...points.map((point) => point.x)), end]} tick={{ fontSize: 10, fill: "#687383" }} tickFormatter={(value) => dateLabel(new Date(Number(value)).toISOString())} />
          <YAxis type="number" dataKey="y" domain={[-1.2, 1.2]} ticks={[-1, 0, 1]} tick={{ fontSize: 11, fill: "#687383" }} tickFormatter={(value) => value === 1 ? "Positive" : value === -1 ? "Negative" : "Neutral"} width={62} />
          <ZAxis type="number" dataKey="confidence" range={[65, 260]} />
          <ReferenceArea x1={start} x2={end} fill="rgba(0,113,227,0.06)" strokeOpacity={0} />
          <ReferenceLine y={0} stroke="rgba(61,93,133,0.38)" strokeDasharray="4 4" />
          <Tooltip cursor={{ strokeDasharray: "4 4" }} content={<NewsChartTooltip identity={identity} kind="timeline" />} />
          <Scatter name="Positive" data={grouped(1)} fill={TONE_COLORS.positive} />
          <Scatter name="Neutral" data={grouped(0)} fill={TONE_COLORS.neutral} />
          <Scatter name="Negative" data={grouped(-1)} fill={TONE_COLORS.negative} />
          {historical.length > 0 && <Scatter name="Older context" data={historical} fill="#a7b2c0" opacity={0.62} />}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
    <p className="chart-caption"><Info size={15} /> Bubble size shows FinBERT confidence. The blue band is the selected {context.window_days || "current"}-day context; older context is shown for background only.</p>
  </div>;
}

export function SentimentMixChart({ context, identity }: { context: NewsHorizonContext; identity?: NewsChartIdentity | null }) {
  const data = [{
    name: "Articles",
    positive: Number(context.positive_count || 0),
    neutral: Number(context.neutral_count || 0),
    negative: Number(context.negative_count || 0),
  }];
  const total = data[0].positive + data[0].neutral + data[0].negative;
  if (!total) return <div className="news-chart-empty"><Sparkles size={17} /><span>There are no usable sentiment labels in this context.</span></div>;
  return <div className="sentiment-mix-chart"><ResponsiveContainer width="100%" height={92}><BarChart data={data} layout="vertical" margin={{ top: 12, right: 8, left: 0, bottom: 12 }}><XAxis type="number" hide domain={[0, total]} /><YAxis type="category" dataKey="name" hide /><Tooltip content={<NewsChartTooltip identity={identity} kind="mix" />} /><Bar dataKey="negative" stackId="tone" fill={TONE_COLORS.negative} name="Negative" radius={[7, 0, 0, 7]} /><Bar dataKey="neutral" stackId="tone" fill={TONE_COLORS.neutral} name="Neutral" /><Bar dataKey="positive" stackId="tone" fill={TONE_COLORS.positive} name="Positive" radius={[0, 7, 7, 0]} /></BarChart></ResponsiveContainer><div className="sentiment-mix-labels"><span><i className="negative" />Negative {data[0].negative}</span><span><i className="neutral" />Neutral {data[0].neutral}</span><span><i className="positive" />Positive {data[0].positive}</span></div></div>;
}

export function EvidenceFunnel({ rawCount, relevantCount, context }: { rawCount: number; relevantCount: number; context: NewsHorizonContext }) {
  const rows = [
    ["Retrieved", rawCount],
    ["Relevant after entity filter", relevantCount],
    ["Inside this horizon", Number(context.candidate_article_count || 0)],
    ["Usable for sentiment", Number(context.usable_article_count || 0)],
    ["Independent news events", Number(context.independent_event_count || 0)],
  ] as const;
  const maximum = Math.max(rows[0][1], 1);
  return <div className="evidence-funnel">{rows.map(([label, value]) => <div className="evidence-funnel-row" key={label}><div className="evidence-funnel-label"><span>{label}</span><strong>{value}</strong></div><div className="evidence-funnel-track"><i style={{ width: `${Math.max(value > 0 ? 6 : 0, (value / maximum) * 100)}%` }} /></div></div>)}</div>;
}

export function PublisherBreakdown({ articles, context, identity }: { articles: NewsArticle[]; context: NewsHorizonContext; identity?: NewsChartIdentity | null }) {
  const contributingIds = new Set((context.article_contributions || []).map((row) => row.article_id));
  const counts = new Map<string, number>();
  articles.forEach((article) => {
    if (!contributingIds.has(articleId(article))) return;
    const label = article.source_domain || article.source_name || "Unknown publisher";
    counts.set(label, (counts.get(label) || 0) + 1);
  });
  const data = [...counts.entries()].sort((left, right) => right[1] - left[1]).slice(0, 5).map(([publisher, count]) => ({ publisher, count }));
  if (!data.length) return <div className="news-chart-empty"><Newspaper size={17} /><span>Publisher diversity is unavailable until usable articles are found.</span></div>;
  return <div className="publisher-breakdown"><ResponsiveContainer width="100%" height={Math.max(140, data.length * 34)}><BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, left: 4, bottom: 4 }}><CartesianGrid horizontal={false} stroke="rgba(61,93,133,0.12)" /><XAxis type="number" allowDecimals={false} tick={{ fontSize: 10, fill: "#687383" }} /><YAxis type="category" dataKey="publisher" width={104} tick={{ fontSize: 10, fill: "#233e61" }} /><Tooltip content={<NewsChartTooltip identity={identity} kind="publisher" />} /><Bar dataKey="count" fill="#0071e3" radius={[0, 6, 6, 0]} /></BarChart></ResponsiveContainer></div>;
}

export function NewsCalculationGuide({ context }: { context: NewsHorizonContext }) {
  return <details className="news-calculation-guide"><summary><ShieldCheck size={17} /> How this sentiment score is calculated</summary><div className="news-calculation-content"><p>StockSense first checks whether an article clearly refers to the selected company. FinBERT then assigns a positive, neutral or negative tone and a confidence estimate.</p><div className="news-formula"><code>recency weight = exp(−ln(2) × article age ÷ {Number(context.half_life_days || 0).toFixed(1)}-day half-life)</code><code>article weight = recency weight × FinBERT confidence</code><code>weighted tone = Σ(tone × article weight) ÷ Σ(article weight)</code></div><p>Recent, confident articles influence the score more. A confidence value describes the model's certainty about the wording; it is not proof that the article is factually correct. Low coverage means the evidence should be treated cautiously.</p></div></details>;
}
