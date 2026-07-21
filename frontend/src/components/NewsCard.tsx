import { CalendarDays, ExternalLink, Globe2, Newspaper, ShieldCheck, TrendingDown, TrendingUp, Minus } from "lucide-react";
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

export function NewsCard({ article, showTone = true, contributingHorizon }: { article: NewsArticle; showTone?: boolean; contributingHorizon?: number }) {
  const sentiment = String(article.sentiment_label || "neutral").toLowerCase();
  const tone = sentiment === "positive" || sentiment === "negative" ? sentiment : "neutral";
  const StatusIcon = tone === "positive" ? TrendingUp : tone === "negative" ? TrendingDown : Minus;
  const source = (article.provider_sources || [article.provider || "newsapi"]).map((item) => item === "newsapi" ? "NewsAPI" : item === "marketaux" ? "Marketaux" : item).join(" / ");
  const website = websiteLabel(article);
  const confidence = typeof article.sentiment_probability === "number" ? `${Math.round(article.sentiment_probability * 100)}%` : "Unavailable";
  return <article className={`article-card${contributingHorizon ? " has-contributor" : ""}`}>{contributingHorizon && <span className="article-contributor-badge">Used for {contributingHorizon}-day sentiment</span>}<div className="article-meta"><div className="article-source-block"><div className="source-line"><Newspaper size={15} /><strong>Sources:</strong> {source}</div><div className="website-line"><Globe2 size={17} /><strong>{website}</strong></div><div className="date-line"><CalendarDays size={15} />{dateLabel(article.published_at)}</div></div>{showTone && <div className="article-status"><span className={`tone-pill ${tone}`}><StatusIcon size={16} />{tone[0].toUpperCase() + tone.slice(1)}</span><span className="confidence-pill"><ShieldCheck size={17} />Confidence {confidence}</span></div>}</div><div className="article-body"><h3>{article.title || "Untitled article"}</h3>{(article.description || article.content_snippet) && <p>{article.description || article.content_snippet}</p>} </div>{article.url && <a className="read-button" href={article.url} target="_blank" rel="noreferrer"><span>Read article</span><ExternalLink size={15} /></a>}</article>;
}
