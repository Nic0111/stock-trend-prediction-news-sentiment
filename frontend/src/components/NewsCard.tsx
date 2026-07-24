import { CalendarDays, ExternalLink, Globe2, Newspaper, ShieldCheck, TrendingDown, TrendingUp, Minus, X, Check } from "lucide-react";
import type { NewsArticle } from "../types";

export function isArticleContributingToContext(
  article: NewsArticle,
  context: Record<string, unknown>,
  asOfDate: string,
) {
  const windowDays = Number(context.window_days);
  const publishedAt = article.published_at ? new Date(article.published_at) : null;
  const asOf = new Date(`${asOfDate}T00:00:00Z`);
  if (!Number.isFinite(windowDays) || windowDays < 0 || !publishedAt || Number.isNaN(publishedAt.getTime()) || Number.isNaN(asOf.getTime())) return false;
  if (article.sentiment_uncertain !== false || ![-1, 0, 1].includes(Number(article.sentiment_score))) return false;
  asOf.setUTCDate(asOf.getUTCDate() + 1);
  const ageDays = Math.max(0, (asOf.getTime() - publishedAt.getTime()) / 86400000);
  return ageDays <= windowDays;
}

function dateLabel(value?: string) {
  if (!value) return "Date unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Date unavailable" : date.toLocaleString([], { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

function websiteLabel(article: NewsArticle) {
  if (article.source_domain) return article.source_domain;
  if (!article.url) return article.source_name || "Unknown website";
  try {
    return new URL(article.url).hostname.replace(/^www\./, "");
  } catch {
    return article.source_name || "Unknown website";
  }
}

function inclusionLabel(article: NewsArticle, review: boolean) {
  if (!review) return article.user_override === "include" ? "Included by You" : "Included Automatically";
  if (article.user_override === "exclude") return "Excluded by You";
  return "Review candidate";
}

export function NewsCard({ article, showTone = true, contributingHorizon, review = false, onExclude, onInclude, busy = false }: { article: NewsArticle; showTone?: boolean; contributingHorizon?: number; review?: boolean; onExclude?: () => void; onInclude?: () => void; busy?: boolean }) {
  const sentiment = String(article.sentiment_label || "neutral").toLowerCase();
  const tone = sentiment === "positive" || sentiment === "negative" ? sentiment : "neutral";
  const StatusIcon = tone === "positive" ? TrendingUp : tone === "negative" ? TrendingDown : Minus;
  const source = (article.provider_sources || [article.provider || "newsapi"]).map((item) => item === "newsapi" ? "NewsAPI" : item === "marketaux" ? "Marketaux" : item).join(" / ");
  const website = websiteLabel(article);
  const confidence = typeof article.sentiment_probability === "number" ? `Confidence ${Math.round(article.sentiment_probability * 100)}%` : "Unavailable";
  const status = inclusionLabel(article, review);
  const statusClass = review ? (article.user_override === "exclude" ? "excluded" : "review") : article.user_override === "include" ? "manual" : "automatic";
  return <article className={`article-card has-inclusion-badge${contributingHorizon ? " has-contributor" : ""}${review ? " article-card-review" : ""}`}><span className={`article-inclusion-badge ${statusClass}`}>{status}</span>{contributingHorizon && <span className="article-contributor-badge">Used for {contributingHorizon}-day sentiment</span>}<div className="article-meta"><div className="article-source-block"><div className="source-line"><Newspaper size={15} /><strong>Sources:</strong> {source}</div><div className="website-line"><Globe2 size={17} /><strong>{website}</strong></div><div className="date-line"><CalendarDays size={15} />{dateLabel(article.published_at)}</div></div>{showTone && <div className="article-status"><span className={`tone-pill ${tone}`}><StatusIcon size={16} />{tone[0].toUpperCase() + tone.slice(1)}</span><span className="confidence-pill"><ShieldCheck size={17} />{confidence}</span></div>}</div><div className="article-body"><h3>{article.title || "Untitled article"}</h3>{(article.description || article.content_snippet) && <p>{article.description || article.content_snippet}</p>}<div className="article-relevance-reason">{review ? (article.user_override === "exclude" ? "You excluded this article from the analysis." : (article.relevance_reasons?.join(" · ") || "The company match is uncertain.")) : article.user_override ? "Adjusted by you" : "Automatically included"}</div></div><div className="article-actions">{article.url && <a className="read-button" href={article.url} target="_blank" rel="noreferrer"><span>Read article</span><ExternalLink size={15} /></a>}{!review && onExclude && <button className="article-action-button article-action-exclude" type="button" onClick={onExclude} disabled={busy} aria-label="Exclude article from this analysis"><X size={15} />Exclude</button>}{review && onInclude && <button className="article-action-button article-action-include" type="button" onClick={onInclude} disabled={busy} aria-label="Include article in this analysis"><Check size={15} />Include</button>}</div></article>;
}
