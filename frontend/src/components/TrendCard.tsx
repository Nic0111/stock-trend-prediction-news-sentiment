import { ArrowDownRight, ArrowUpRight, Check, CircleAlert, Minus, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";
import type { DashboardAnalysis, Prediction, SynthesisRelationship } from "../types";

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
  const evidenceStatus = analysis.stock.evidence_status || (analysis.stock.directional_signal_available ? "accepted" : "experimental");
  const evidenceLabel = evidenceStatus === "accepted"
    ? "Accepted"
    : evidenceStatus === "limited"
      ? "Limited evidence"
      : "Experimental · research only";
  return (
    <section className={`signal-panel ${presentation.tone}`}>
      <div className="panel-kicker">Model outlook <span className={`live-dot ${freshness !== "fresh" || analysis.stock_source !== "live" ? "stale" : ""}`}>{sourceLabel}</span></div>
      <div className="signal-main"><div className="signal-icon"><Icon size={30} /></div><div><h2>{presentation.label}</h2><p>{presentation.subtitle}</p></div></div>
      <div className="fact-grid"><div><span>Market observation</span><strong>{analysis.stock.as_of_date}</strong></div><div><span>Horizon</span><strong>{analysis.horizon_days} trading day(s)</strong></div><div><span>Evidence</span><strong>{evidenceLabel}</strong></div></div>
      {evidenceStatus === "experimental" && <div className="notice warning">This is an experimental model direction. It has not passed the frozen acceptance benchmark and is for research transparency.</div>}
      {analysis.stock_warning && <div className="notice warning">{analysis.stock_warning}</div>}
    </section>
  );
}

const relationshipLabels: Record<SynthesisRelationship, string> = {
  reinforcing: "Reinforcing signals",
  conflicting: "Conflicting signals",
  balanced: "Balanced news tone",
  news_unavailable: "News unavailable",
};

const relationshipTone: Record<SynthesisRelationship, string> = {
  reinforcing: "positive",
  conflicting: "warning",
  balanced: "neutral",
  news_unavailable: "neutral",
};

export function CombinedOutlookCard({ analysis, children }: { analysis: DashboardAnalysis; children?: ReactNode }) {
  const outlook = analysis.combined_outlook;
  if (!outlook) return null;
  const direction = outlook.direction?.toLowerCase();
  const isDown = direction === "down";
  const Icon = isDown ? ArrowDownRight : ArrowUpRight;
  const directionTone = outlook.direction ? (isDown ? "negative" : "positive") : "neutral";
  const probabilityText = typeof outlook.probability_up === "number"
    ? outlook.probability_up >= 0.5
      ? `${(outlook.probability_up * 100).toFixed(1)}% probability of upward movement`
      : `${((1 - outlook.probability_up) * 100).toFixed(1)}% probability of downward movement`
      : "No stock probability is available";
  const RelationshipIcon = outlook.relationship === "reinforcing"
    ? Check
    : outlook.relationship === "conflicting"
      ? CircleAlert
      : ShieldCheck;
  const newsAdjustment = outlook.component_contributions?.news_adjustment_probability_points;
  const fusionMethod = outlook.fusion_method === "transparent_log_odds"
    ? "Transparent log-odds fusion"
      : outlook.fusion_method === "late_fusion_logistic"
        ? "Validated late-fusion model"
        : outlook.fusion_method === "anchored_incremental_news"
          ? "Anchored news adjustment"
      : null;

  return (
    <section className={`combined-outlook-card ${directionTone}`}>
      <div className="combined-outlook-heading">
        <div>
          <span className="panel-kicker">Aggregate Result</span>
        </div>
        <span className={`combined-validation ${relationshipTone[outlook.relationship]}`}>
          <RelationshipIcon size={16} />
          {relationshipLabels[outlook.relationship]}
        </span>
      </div>
      <div className="combined-outlook-main signal-main">
        <div className="combined-direction-icon signal-icon"><Icon size={30} /></div>
        <div>
          <h2>{outlook.direction || "Unavailable"}</h2>
          <p>{probabilityText}</p>
        </div>
      </div>
      <div className="combined-fact-grid">
        <div><span>Observation</span><strong>{outlook.as_of_date || "Unavailable"}</strong></div>
        <div><span>Horizon</span><strong>{analysis.horizon_days} trading day(s)</strong></div>
        <div><span>Usable articles</span><strong>{outlook.usable_article_count} · {outlook.news_coverage}</strong></div>
      </div>
      {outlook.probability_kind === "combined" && typeof newsAdjustment === "number" && (
        <div className="combined-contribution-grid">
          <div><span>Stock input</span><strong>{((outlook.stock_probability_up ?? 0) * 100).toFixed(1)}% Up</strong></div>
          <div><span>News adjustment</span><strong>{newsAdjustment >= 0 ? "+" : ""}{newsAdjustment.toFixed(1)} pts</strong></div>
          <div><span>Fusion method</span><strong>{fusionMethod}</strong></div>
        </div>
      )}
      <p className="combined-explanation">{outlook.explanation}</p>
      {outlook.mode === "stock_led_synthesis" && outlook.fallback_reason && (
        <p className="combined-fallback-note">Numerical fusion was not applied: {outlook.fallback_reason}.</p>
      )}
      {children}
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
