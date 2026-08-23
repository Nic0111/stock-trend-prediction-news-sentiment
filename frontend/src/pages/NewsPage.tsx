import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { BarChart3, CalendarDays, Filter, Gauge, Info, Layers3, Newspaper, RefreshCw, Search, ShieldCheck } from "lucide-react";
import { createAnalysis, getCompanies, resetNewsOverrides, updateNewsOverride } from "../api/client";
import { LoadingState, ErrorState } from "../components/AsyncState";
import { CompanyLogo } from "../components/CompanyLogo";
import { CompanySelector } from "../components/CompanySelector";
import { HorizonSelector } from "../components/HorizonSelector";
import { EvidenceFunnel, NewsCalculationGuide, NewsInfluenceChart, NewsToneTimeline, PublisherBreakdown, SentimentMixChart } from "../components/NewsEvidence";
import { isArticleContributingToContext, NewsCard } from "../components/NewsCard";
import { PageHeader } from "../components/PageHeader";
import { SentimentModelEvidence } from "../components/ModelDiagnostics";
import { useAppState } from "../state/AppState";
import type { DashboardAnalysis, NewsArticle, NewsHorizonContext } from "../types";

function articleKey(article: NewsArticle) {
  return String(article.article_id || article.url || article.title || "article");
}

function providerNames(article: NewsArticle) {
  return (article.provider_sources || [article.provider || "newsapi"]).map((provider) => provider.toLowerCase());
}

function providerLabel(provider: string) {
  return provider === "newsapi" ? "NewsAPI" : provider === "marketaux" ? "Marketaux" : provider;
}

function toneLabel(value: string) {
  const tone = value.toLowerCase();
  return tone === "positive" || tone === "negative" ? tone : "neutral";
}

function scoreLabel(score: number | null | undefined) {
  if (typeof score !== "number" || !Number.isFinite(score)) return "Unavailable";
  return `${score >= 0 ? "+" : ""}${score.toFixed(2)}`;
}

function coverageCopy(context: NewsHorizonContext) {
  const coverage = String(context.coverage_confidence || "Unavailable");
  const usable = Number(context.usable_article_count || 0);
  const sources = Number(context.unique_source_count || 0);
  if (coverage === "Unavailable") return "No confident, company-specific articles are available for this context.";
  if (coverage === "Low") return `${usable} usable article${usable === 1 ? "" : "s"} from ${sources} publisher${sources === 1 ? "" : "s"}; treat this as a limited evidence signal.`;
  return `${usable} usable articles from ${sources} publishers contribute to this context.`;
}

export function NewsPage() {
  const companies = useQuery({ queryKey: ["companies"], queryFn: ({ signal }) => getCompanies(signal), staleTime: 86400000 });
  const { newsTicker, setNewsTicker, newsHorizon, setNewsHorizon } = useAppState();
  const [analysis, setAnalysis] = useState<DashboardAnalysis | null>(null);
  const [publisherFilter, setPublisherFilter] = useState<string[]>([]);
  const [providerFilter, setProviderFilter] = useState<string[]>([]);
  const [toneFilter, setToneFilter] = useState<string[]>([]);
  const [usedOnly, setUsedOnly] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedArticleId, setSelectedArticleId] = useState<string | null>(null);
  const [correctionNotice, setCorrectionNotice] = useState("");

  useEffect(() => {
    if (companies.data?.length && !companies.data.some((company) => company.ticker === newsTicker)) setNewsTicker(companies.data[0].ticker);
  }, [companies.data, newsTicker, setNewsTicker]);

  const mutation = useMutation({
    mutationFn: () => createAnalysis({ ticker: newsTicker, horizon: newsHorizon }),
    onSuccess: (result) => {
      setAnalysis(result);
      setPublisherFilter([]);
      setProviderFilter([]);
      setToneFilter([]);
      setUsedOnly(false);
      setQuery("");
      setSelectedArticleId(null);
      setCorrectionNotice("");
    },
  });
  const correctionMutation = useMutation({
    mutationFn: ({ articleId: selectedArticle, action }: { articleId: string; action: "include" | "exclude" }) => updateNewsOverride(analysis!.analysis_id, selectedArticle, action),
    onMutate: () => setCorrectionNotice("Recalculating the news evidence…"),
    onSuccess: (result) => {
      setAnalysis(result);
      setCorrectionNotice("Sentiment recalculated from your article selection.");
    },
  });
  const resetMutation = useMutation({
    mutationFn: () => resetNewsOverrides(analysis!.analysis_id),
    onMutate: () => setCorrectionNotice("Restoring the automatic article set…"),
    onSuccess: (result) => {
      setAnalysis(result);
      setCorrectionNotice("Automatic article selection restored.");
    },
  });

  const news = analysis?.news || null;
  const context = news?.contexts[String(newsHorizon)] || {};
  const company = companies.data?.find((item) => item.ticker === newsTicker);
  const chartIdentity = company ? { ticker: company.ticker, companyName: company.company_name } : news ? { ticker: news.ticker, companyName: news.company_name } : null;

  const publisherOptions = useMemo(() => Array.from(new Set((news?.articles || []).map((article) => String(article.source_domain || article.source_name || "")).filter(Boolean))).sort(), [news]);
  const providerOptions = useMemo(() => Array.from(new Set((news?.articles || []).flatMap(providerNames))).sort(), [news]);
  const contributingIds = useMemo(() => new Set((context.article_contributions || []).map((row) => row.article_id)), [context]);
  const filteredArticles = useMemo(() => (news?.articles || []).filter((article) => {
    const key = articleKey(article);
    const publisher = String(article.source_domain || article.source_name || "");
    const tone = toneLabel(String(article.sentiment_label || "neutral"));
    const providerMatch = !providerFilter.length || providerNames(article).some((provider) => providerFilter.includes(provider));
    const searchMatch = !query.trim() || `${article.title || ""} ${article.description || ""} ${publisher}`.toLowerCase().includes(query.trim().toLowerCase());
    return (!publisherFilter.length || publisherFilter.includes(publisher)) && (!toneFilter.length || toneFilter.includes(tone)) && providerMatch && (!usedOnly || contributingIds.has(key)) && searchMatch;
  }).sort((left, right) => {
    const leftContribution = Math.abs(Number((context.article_contributions || []).find((row) => row.article_id === articleKey(left))?.signed_contribution || 0));
    const rightContribution = Math.abs(Number((context.article_contributions || []).find((row) => row.article_id === articleKey(right))?.signed_contribution || 0));
    return usedOnly ? rightContribution - leftContribution : (new Date(right.published_at || 0).getTime() - new Date(left.published_at || 0).getTime());
  }), [news, publisherFilter, providerFilter, toneFilter, usedOnly, query, contributingIds, context]);

  useEffect(() => {
    if (!selectedArticleId) return;
    const target = document.getElementById(`news-article-${encodeURIComponent(selectedArticleId)}`);
    target?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [selectedArticleId, filteredArticles.length]);

  function selectArticle(id: string) {
    setPublisherFilter([]);
    setProviderFilter([]);
    setToneFilter([]);
    setUsedOnly(false);
    setQuery("");
    setSelectedArticleId(id);
  }

  function toggleValue(value: string, setter: React.Dispatch<React.SetStateAction<string[]>>) {
    setter((current) => current.includes(value) ? current.filter((item) => item !== value) : [...current, value]);
  }

  if (companies.isLoading) return <><PageHeader eyebrow="Analyze" title="News & Sentiment" description="Read the supporting articles and understand how sentiment coverage affects the signal." /><LoadingState /></>;
  if (companies.isError || !companies.data) return <><PageHeader eyebrow="Analyze" title="News & Sentiment" description="Read the supporting articles and understand how sentiment coverage affects the signal." /><ErrorState message={companies.error?.message || "Companies could not be loaded."} /></>;

  const score = typeof context.weighted_sentiment_score === "number" ? context.weighted_sentiment_score : null;
  const tone = score === null ? "neutral" : score > 0.05 ? "positive" : score < -0.05 ? "negative" : "neutral";
  const sourceSummary = Object.entries(news?.provider_article_counts || {}).filter(([name]) => name !== "effective" && !name.endsWith("_historical"));

  return <>
    <PageHeader eyebrow="Analyze" title="News & Sentiment" description="Follow the evidence trail from retrieved headlines to the weighted tone used in the selected horizon." />
    <section className="selector-panel stacked-selector news-selector">
      <div className="selector-primary-row">
        <CompanyLogo company={company} className="selector-logo" />
        <CompanySelector label="Company" companies={companies.data} value={newsTicker} onChange={(value) => { setNewsTicker(value); setAnalysis(null); setCorrectionNotice(""); }} />
        <button className="primary-button" onClick={() => mutation.mutate()} disabled={mutation.isPending}>{mutation.isPending ? "Retrieving…" : news ? "Refresh news" : "Analyze news"}</button>
      </div>
      <HorizonSelector value={newsHorizon} onChange={(value) => { setNewsHorizon(value); setSelectedArticleId(null); }} />
    </section>
    {mutation.isError && <ErrorState message={mutation.error.message} />}
    {!news && !mutation.isPending && <section className="news-empty-state"><div className="news-empty-icon"><Newspaper size={28} /></div><div><span className="panel-kicker">Beginner reading flow</span><h2>Start with the evidence, not a headline.</h2><p>Choose a company and horizon, then StockSense will retrieve, filter and classify the available articles before explaining how much evidence is usable.</p></div></section>}
    {news && <>
      <section className={`news-pulse tone-${tone}`}>
        <div className="news-pulse-heading"><div><span className="panel-kicker">News pulse · {context.window_days || "—"}-day context</span><h2>{tone === "positive" ? "Positive tone" : tone === "negative" ? "Negative tone" : score === null ? "Tone unavailable" : "Balanced tone"}</h2><p>{coverageCopy(context)}</p></div><span className={`coverage-badge ${String(context.coverage_confidence || "unavailable").toLowerCase()}`}>{String(context.coverage_confidence || "Unavailable")} coverage</span></div>
        <div className="news-pulse-score"><strong>{scoreLabel(score)}</strong><span>Weighted sentiment score</span><div className="news-score-track"><i style={{ left: `${Math.min(100, Math.max(0, ((score ?? 0) + 1) * 50))}%` }} /></div><div className="news-score-labels"><span>More negative</span><span>Balanced</span><span>More positive</span></div></div>
        <div className="news-pulse-facts"><div><span>Usable articles</span><strong>{Number(context.usable_article_count || 0)}</strong></div><div><span>Publishers</span><strong>{Number(context.unique_source_count || 0)}</strong></div><div><span>Newest article</span><strong>{context.newest_article_age_days == null ? "Unavailable" : `${Number(context.newest_article_age_days).toFixed(1)}d ago`}</strong></div><div><span>Retrieved</span><strong>{news.retrieved_at ? new Date(news.retrieved_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "Unavailable"}</strong></div></div>
      </section>

      {correctionNotice && <div className="news-correction-status" role="status"><RefreshCw size={15} className={correctionMutation.isPending || resetMutation.isPending ? "is-spinning" : ""} />{correctionNotice}</div>}
      {news.results_truncated && <div className="notice warning"><strong>Some search results were truncated.</strong> The charts and score use the articles that passed the available retrieval limit.</div>}

      <section className="news-evidence-section">
        <div className="news-section-intro"><div className="news-section-icon"><BarChart3 size={19} /></div><div><span className="panel-kicker">Evidence trail</span><h2>What shaped the tone?</h2><p>Each bar shows an article's signed contribution after confidence and recency weighting. Select one to inspect it below.</p></div></div>
         <NewsInfluenceChart articles={news.articles} context={context} identity={chartIdentity} onSelectArticle={selectArticle} />
      </section>

      <div className="news-evidence-grid">
         <section className="news-evidence-section news-timeline-section"><div className="news-section-intro"><div className="news-section-icon"><CalendarDays size={19} /></div><div><span className="panel-kicker">Across the context window</span><h2>When did the tone appear?</h2><p>Newer evidence is more influential because the weighting gives recent articles more emphasis.</p></div></div><NewsToneTimeline articles={news.articles} historicalArticles={news.historical_articles} context={context} asOfDate={news.as_of_date} identity={chartIdentity} /></section>
         <section className="news-evidence-section"><div className="news-section-intro"><div className="news-section-icon"><Gauge size={19} /></div><div><span className="panel-kicker">Signal mix</span><h2>What kind of headlines are present?</h2><p>A balanced score can come from neutral headlines or from positive and negative headlines offsetting each other.</p></div></div><SentimentMixChart context={context} identity={chartIdentity} /><div className="news-mix-note"><Info size={15} /> Neutral does not mean no news; it means FinBERT found no directional tone strong enough to move the mix.</div></section>
      </div>

       <section className="news-evidence-section news-quality-section"><div className="news-section-intro"><div className="news-section-icon"><Layers3 size={19} /></div><div><span className="panel-kicker">Evidence quality</span><h2>How much evidence survived the checks?</h2><p>Retrieval volume and sentiment confidence are different things. This funnel makes the distinction visible.</p></div></div><div className="news-quality-grid"><div><h3>Evidence funnel</h3><EvidenceFunnel rawCount={news.raw_article_count} relevantCount={news.relevant_article_count} context={context} /></div><div><h3>Usable publishers</h3><PublisherBreakdown articles={news.articles} context={context} identity={chartIdentity} /></div><div className="news-provider-summary"><h3>Data providers</h3>{sourceSummary.length ? sourceSummary.map(([provider, count]) => <div className="provider-status-row" key={provider}><span>{providerLabel(provider)}</span><strong>{count}</strong><small>{news.provider_status?.[provider] || "status unknown"}</small></div>) : <div className="news-chart-empty"><Info size={16} /><span>Provider information unavailable.</span></div>}<p className="chart-caption"><ShieldCheck size={15} /> Provider count describes retrieval coverage; publisher count describes independent article sources.</p></div></div></section>

      <SentimentModelEvidence compact />

      <section className="news-article-section">
        <div className="section-heading-row news-articles-heading"><div><span className="panel-kicker">Inspect and correct</span><h2 className="section-heading">Relevant news articles</h2><p className="muted-note">Articles marked “Used for sentiment” are inside the selected context and passed the confidence checks.</p></div>{Object.keys(news.user_overrides || {}).length > 0 && <button className="secondary-button" type="button" onClick={() => resetMutation.mutate()} disabled={resetMutation.isPending}>Reset</button>}</div>
        <div className="news-filter-bar"><div className="news-filter-search"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search headlines or publishers" aria-label="Search headlines or publishers" /></div><div className="news-filter-label"><Filter size={15} /> Filter</div><button className={`news-filter-chip${usedOnly ? " selected" : ""}`} type="button" onClick={() => setUsedOnly((value) => !value)}>Used in score</button>{toneFilterOptions.map((option) => <button className={`news-filter-chip ${option}${toneFilter.includes(option) ? " selected" : ""}`} type="button" key={option} onClick={() => toggleValue(option, setToneFilter)}>{option[0].toUpperCase() + option.slice(1)}</button>)}{providerOptions.map((option) => <button className={`news-filter-chip${providerFilter.includes(option) ? " selected" : ""}`} type="button" key={option} onClick={() => toggleValue(option, setProviderFilter)}>{providerLabel(option)}</button>)}{publisherOptions.slice(0, 4).map((option) => <button className={`news-filter-chip${publisherFilter.includes(option) ? " selected" : ""}`} type="button" key={option} onClick={() => toggleValue(option, setPublisherFilter)}>{option}</button>)}</div>
        {filteredArticles.length ? <div className="article-grid news-article-grid">{filteredArticles.map((article) => { const id = articleKey(article); return <div id={`news-article-${encodeURIComponent(id)}`} className={selectedArticleId === id ? "news-article-highlight" : ""} key={id}><NewsCard article={article} contributingHorizon={article.contributes_to_horizons?.includes(newsHorizon) ? newsHorizon : (isArticleContributingToContext(article, context, news.as_of_date) ? newsHorizon : undefined)} onExclude={article.article_id ? () => correctionMutation.mutate({ articleId: article.article_id!, action: "exclude" }) : undefined} busy={correctionMutation.isPending} /></div>; })}</div> : <div className="notice">No articles match these filters. Clear a filter to see the available evidence.</div>}
      </section>

      <details className="review-articles"><summary>Review excluded articles ({news.review_articles?.length || 0})</summary><p className="review-articles-help">These articles matched the search but were not specific enough to influence sentiment automatically. Include only if they clearly refer to the selected company.</p>{news.review_articles?.length ? <div className="article-grid">{news.review_articles.map((article, index) => <NewsCard review article={article} onInclude={article.article_id ? () => correctionMutation.mutate({ articleId: article.article_id!, action: "include" }) : undefined} busy={correctionMutation.isPending} key={`${articleKey(article)}-review-${index}`} />)}</div> : <div className="notice">No excluded candidates for this analysis.</div>}</details>
      {news.historical_articles.length > 0 && <section className="news-older-section"><div className="section-heading-row"><div><span className="panel-kicker">Background only</span><h2 className="section-heading">Older company context</h2></div></div><div className="notice warning">Older articles help explain the broader news landscape but do not affect the selected News Sentiment result.</div><div className="article-grid">{news.historical_articles.map((article, index) => <NewsCard key={`historical-${articleKey(article)}-${index}`} article={article} showTone={false} />)}</div></section>}
      <NewsCalculationGuide context={context} />
    </>}
  </>;
}

const toneFilterOptions = ["positive", "neutral", "negative"];
