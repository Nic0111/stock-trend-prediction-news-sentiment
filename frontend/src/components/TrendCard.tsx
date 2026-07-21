import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import type { DashboardAnalysis, Prediction } from "../types";

function directionPresentation(prediction: Prediction) {
  const isDown = prediction.prediction_label.toLowerCase() === "down";
  return {
    tone: isDown ? "negative" : "positive",
    label: isDown ? "Down" : "Up",
    subtitle: isDown
      ? prediction.probability_up >= 0.5
        ? `Up probability ${(prediction.probability_up * 100).toFixed(1)}% · below the ${(prediction.decision_threshold * 100).toFixed(1)}% Up cutoff`
        : `${((1 - prediction.probability_up) * 100).toFixed(1)}% probability of downward movement`
      : `${(prediction.probability_up * 100).toFixed(1)}% probability of upward movement`,
    Icon: isDown ? ArrowDownRight : ArrowUpRight,
  };
}

export function TrendCard({ analysis }: { analysis: DashboardAnalysis }) {
  if (!analysis.stock) return <div className="signal-panel"><div className="notice error">Stock signal unavailable.</div></div>;
  const presentation = directionPresentation(analysis.stock);
  const Icon = presentation.Icon;
  const freshness = analysis.stock.freshness_status === "unknown"
    ? (analysis.stock.stale_data_warning ? "stale" : "fresh")
    : (analysis.stock.freshness_status || (analysis.stock.stale_data_warning ? "stale" : "fresh"));
  const sourceLabel = analysis.stock_source === "fallback"
    ? "Saved fallback"
    : freshness === "stale"
      ? "Stale data"
      : freshness === "delayed"
        ? "Delayed"
        : "Live";
  return (
    <section className={`signal-panel ${presentation.tone}`}>
      <div className="panel-kicker">Model outlook <span className={`live-dot ${freshness !== "fresh" || analysis.stock_source !== "live" ? "stale" : ""}`}>{sourceLabel}</span></div>
      <div className="signal-main"><div className="signal-icon"><Icon size={30} /></div><div><h2>{presentation.label}</h2><p>{presentation.subtitle}</p></div></div>
      <div className="fact-grid"><div><span>Market observation</span><strong>{analysis.stock.as_of_date}</strong></div><div><span>Horizon</span><strong>{analysis.horizon_days} trading day(s)</strong></div><div><span>Evidence</span><strong>{analysis.stock.directional_signal_available ? "Accepted" : "Experimental · research only"}</strong></div></div>
      {!analysis.stock.directional_signal_available && <div className="notice warning">This is an experimental model direction. It has not passed the frozen acceptance benchmark and is for research transparency.</div>}
      {analysis.stock_warning && <div className="notice warning">{analysis.stock_warning}</div>}
      <details className="details"><summary>Stock model details</summary><div className="detail-copy"><p>Model: {analysis.stock.model_name}</p><p>Probability Up: {(analysis.stock.probability_up * 100).toFixed(1)}% · Probability Down: {((1 - analysis.stock.probability_up) * 100).toFixed(1)}%</p><p>Decision threshold: {(analysis.stock.decision_threshold * 100).toFixed(1)}% Up · Evidence strength: {analysis.stock.evidence_strength}</p><p>Supporting factors: {analysis.stock.supporting_factors.map((factor) => factor.label).join(", ") || "None identified"}</p><p>Opposing factors: {analysis.stock.opposing_factors.map((factor) => factor.label).join(", ") || "None identified"}</p><p>Latest valid market session: {analysis.stock.market_reference_date || "Unavailable"} · Session lag: {analysis.stock.observation_lag_sessions ?? "Unavailable"}</p></div></details>
    </section>
  );
}

export function NewsSummaryCard({ analysis }: { analysis: DashboardAnalysis }) {
  const news = analysis.news;
  if (!news) return <section className="signal-panel"><div className="notice error">News sentiment unavailable.</div></section>;
  const context = news.contexts[String(analysis.horizon_days)] || {};
  const score = typeof context.weighted_sentiment_score === "number" ? context.weighted_sentiment_score : null;
  const tone = score === null ? "neutral" : score > 0.05 ? "positive" : score < -0.05 ? "negative" : "neutral";
  return <section className={`signal-panel ${tone}`}><div className="panel-kicker">Weighted news tone <span className="live-dot">{analysis.news_source === "fallback" ? "Saved fallback" : "Live"}</span></div><div className="signal-main"><div className="signal-icon"><Minus size={30} /></div><div><h2>{score === null ? "Unavailable" : `${score >= 0 ? "+" : ""}${score.toFixed(2)}`}</h2><p>{String(context.coverage_confidence || "Coverage unavailable")}</p></div></div><div className="sentiment-scale"><span /> <i style={{ left: `${Math.min(100, Math.max(0, ((score ?? 0) + 1) * 50))}%` }} /></div><div className="fact-grid"><div><span>Coverage</span><strong>{String(context.coverage_confidence || "Unavailable")}</strong></div><div><span>Usable articles</span><strong>{String(context.usable_article_count ?? 0)}</strong></div><div><span>Sources</span><strong>{Object.keys(news.provider_article_counts || {}).length}</strong></div></div>{analysis.news_warning && <div className="notice warning">{analysis.news_warning}</div>}</section>;
}
