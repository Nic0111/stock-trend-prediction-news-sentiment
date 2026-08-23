import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  CheckCircle2,
  CircleAlert,
  FlaskConical,
  Gauge,
  Info,
  Scale,
  ShieldCheck,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Link } from "react-router-dom";
import { getModelDiagnostics } from "../api/client";
import type { Horizon, SentimentModelDiagnostic, StockHorizonDiagnostic } from "../types";
import { ErrorState, LoadingState } from "./AsyncState";

const METRIC_COLORS = {
  primary: "#126fde",
  secondary: "#8ca0b8",
  accent: "#178a59",
  caution: "#ca8a04",
  negative: "#d9485f",
};

function percentage(value: number | null | undefined, digits = 1) {
  return typeof value === "number" ? `${(value * 100).toFixed(digits)}%` : "Unavailable";
}

function metricLabel(value: unknown) {
  return String(value || "Unavailable").replaceAll("_", " ");
}

function metricTooltip({ active, payload, label }: { active?: boolean; payload?: ReadonlyArray<{ name?: unknown; value?: unknown; color?: string }>; label?: unknown }) {
  if (!active || !payload?.length) return null;
  return <div className="evaluation-tooltip"><strong>{String(label ?? "Metric")}</strong>{payload.map((entry, index) => <span key={`${String(entry.name)}-${index}`} style={{ "--tooltip-color": entry.color } as React.CSSProperties}><i />{String(entry.name ?? "Value")}: {percentage(typeof entry.value === "number" ? entry.value : null)}</span>)}</div>;
}

function EvidenceHeader({ eyebrow, title, description, icon }: { eyebrow: string; title: string; description: string; icon: React.ReactNode }) {
  return <div className="evaluation-heading"><div className="evaluation-heading-icon">{icon}</div><div><span className="panel-kicker">{eyebrow}</span><h2>{title}</h2><p>{description}</p></div></div>;
}

function EvaluationProtocol() {
  return <div className="evaluation-protocol" role="note"><Info size={16} /><div><strong>How to read this evidence</strong><span>These are recorded evaluation results, not a live accuracy score and not a guarantee for the next forecast.</span></div></div>;
}

function StockLedgerBody({ horizon }: { horizon: StockHorizonDiagnostic }) {
  const comparison = [
    { metric: "Balanced acc.", model: horizon.metrics.balanced_accuracy, baseline: horizon.baseline_metrics.balanced_accuracy },
    { metric: "Macro-F1", model: horizon.metrics.macro_f1, baseline: horizon.baseline_metrics.macro_f1 },
  ];
  const foldData = horizon.cv_fold_balanced_accuracy.map((score, index) => ({ fold: `Fold ${index + 1}`, score }));
  const failed = [...horizon.observed_failed_checks, ...horizon.shadow_failed_checks.filter((item) => !horizon.observed_failed_checks.includes(item))];
  const accepted = horizon.status.toLowerCase() === "accepted";
  return <>
    <div className="evaluation-ledger-strip">
      <div><span>Evaluation</span><strong>2025 retrospective holdout</strong></div>
      <div><span>Selected model</span><strong>{metricLabel(horizon.model_name)}</strong></div>
      <div><span>Decision cutoff</span><strong>{percentage(horizon.decision_threshold)}</strong></div>
      <div><span>Evidence status</span><strong className={accepted ? "status-pass" : "status-caution"}>{accepted ? <CheckCircle2 size={14} /> : <FlaskConical size={14} />}{metricLabel(horizon.status)}</strong></div>
    </div>
    <div className="evaluation-chart-grid">
      <article className="evaluation-chart-panel">
        <div className="evaluation-chart-title"><div><span>Model versus baseline</span><strong>Does it add predictive skill?</strong></div><Scale size={18} /></div>
        <div className="evaluation-chart-canvas">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={comparison} margin={{ top: 10, right: 8, left: -12, bottom: 2 }}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="metric" tick={{ fontSize: 11 }} interval={0} /><YAxis domain={[0, 0.8]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fontSize: 11 }} /><Tooltip content={metricTooltip} /><Legend wrapperStyle={{ fontSize: 12 }} /><Bar dataKey="model" name="Selected model" fill={METRIC_COLORS.primary} radius={[5, 5, 0, 0]} /><Bar dataKey="baseline" name="Strongest baseline" fill={METRIC_COLORS.secondary} radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer>
        </div>
        <p className="evaluation-chart-note">Balanced accuracy gives Up and Down equal importance; Macro-F1 also checks whether both classes are handled consistently.</p>
      </article>
      <article className="evaluation-chart-panel">
        <div className="evaluation-chart-title"><div><span>Validation stability</span><strong>Does performance survive time folds?</strong></div><BarChart3 size={18} /></div>
        <div className="evaluation-chart-canvas">
          {foldData.length ? <ResponsiveContainer width="100%" height="100%"><LineChart data={foldData} margin={{ top: 10, right: 14, left: -12, bottom: 2 }}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="fold" tick={{ fontSize: 11 }} /><YAxis domain={[0.4, 0.7]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fontSize: 11 }} /><Tooltip content={metricTooltip} /><Line type="monotone" dataKey="score" name="Balanced accuracy" stroke={METRIC_COLORS.accent} strokeWidth={2.5} dot={{ r: 4, fill: METRIC_COLORS.accent }} /></LineChart></ResponsiveContainer> : <div className="evaluation-empty">Fold-level evidence is unavailable.</div>}
        </div>
        <p className="evaluation-chart-note">Each point is a later chronological validation block. Large swings mean the relationship may depend on the market period.</p>
      </article>
    </div>
    <div className="evaluation-reading-row">
      <div><Gauge size={17} /><span><strong>{percentage(horizon.metrics.balanced_accuracy)}</strong> holdout balanced accuracy</span></div>
      <div><ShieldCheck size={17} /><span><strong>{horizon.balanced_accuracy_gain == null ? "Unavailable" : `${horizon.balanced_accuracy_gain >= 0 ? "+" : ""}${(horizon.balanced_accuracy_gain * 100).toFixed(1)} pp`}</strong> versus {horizon.strongest_baseline}</span></div>
      <div><Scale size={17} /><span><strong>{typeof horizon.metrics.mcc === "number" ? horizon.metrics.mcc.toFixed(3) : "Unavailable"}</strong> Matthews correlation</span></div>
      <div className={failed.length ? "has-caution" : ""}>{failed.length ? <CircleAlert size={17} /> : <CheckCircle2 size={17} />}<span><strong>{failed.length}</strong> recorded acceptance check{failed.length === 1 ? "" : "s"} not passed</span></div>
    </div>
  </>;
}

export function StockModelEvidence({ horizon, compact = false }: { horizon: Horizon; compact?: boolean }) {
  const query = useQuery({ queryKey: ["model-diagnostics"], queryFn: ({ signal }) => getModelDiagnostics(signal), staleTime: 300000 });
  if (query.isLoading) return <LoadingState label="Loading recorded model evidence…" />;
  if (query.isError || !query.data?.stock.available) return <ErrorState message={query.error?.message || "Recorded stock-model evidence is unavailable."} />;
  const selected = query.data.stock.horizons.find((item) => item.horizon_days === horizon);
  if (!selected) return null;
  return <section className={`evaluation-ledger ${compact ? "compact" : ""}`}>
    <EvidenceHeader eyebrow={`${horizon}-trading-day model`} title="Recorded model evidence" description="A transparent view of the selected horizon's frozen evaluation, baseline and time-fold stability." icon={<Gauge size={20} />} />
    <StockLedgerBody horizon={selected} />
    {compact ? <div className="evaluation-compact-footer"><EvaluationProtocol /><Link className="evaluation-dashboard-link" to={`/model-performance?horizon=${horizon}`}>Open full performance dashboard</Link></div> : <EvaluationProtocol />}
  </section>;
}

function SentimentComparison({ models }: { models: SentimentModelDiagnostic[] }) {
  const data = models.map((model) => ({ name: model.label, balancedAccuracy: model.balanced_accuracy, macroF1: model.macro_f1, calibration: model.ece }));
  return <article className="evaluation-chart-panel">
    <div className="evaluation-chart-title"><div><span>Model comparison</span><strong>Which language model generalizes better?</strong></div><Scale size={18} /></div>
    <div className="evaluation-chart-canvas evaluation-chart-wide"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 10, right: 8, left: -12, bottom: 8 }}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} /><YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fontSize: 11 }} /><Tooltip content={metricTooltip} /><Legend wrapperStyle={{ fontSize: 12 }} /><Bar dataKey="balancedAccuracy" name="Balanced accuracy" fill={METRIC_COLORS.primary} radius={[5, 5, 0, 0]} /><Bar dataKey="macroF1" name="Macro-F1" fill={METRIC_COLORS.accent} radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div>
    <p className="evaluation-chart-note">The active ensemble is compared with both component models on the same frozen 2025 retrospective set.</p>
  </article>;
}

function SentimentClassChart({ model }: { model?: SentimentModelDiagnostic }) {
  const data = Object.entries(model?.class_f1 || {}).map(([label, value]) => ({ label: label[0].toUpperCase() + label.slice(1), value }));
  const colors = [METRIC_COLORS.negative, METRIC_COLORS.caution, METRIC_COLORS.accent];
  return <article className="evaluation-chart-panel">
    <div className="evaluation-chart-title"><div><span>Class-level reliability</span><strong>Which tone is hardest to recognize?</strong></div><BarChart3 size={18} /></div>
    <div className="evaluation-chart-canvas evaluation-chart-wide"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} layout="vertical" margin={{ top: 8, right: 18, left: 8, bottom: 2 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fontSize: 11 }} /><YAxis type="category" dataKey="label" width={62} tick={{ fontSize: 11 }} /><Tooltip content={metricTooltip} /><Bar dataKey="value" name="F1 score" radius={[0, 6, 6, 0]}>{data.map((entry, index) => <Cell key={entry.label} fill={colors[index % colors.length]} />)}</Bar></BarChart></ResponsiveContainer></div>
    <p className="evaluation-chart-note">Class F1 combines precision and recall. A lower Neutral bar shows why balanced headlines remain the hardest category.</p>
  </article>;
}

export function SentimentModelEvidence({ compact = false }: { compact?: boolean }) {
  const query = useQuery({ queryKey: ["model-diagnostics"], queryFn: ({ signal }) => getModelDiagnostics(signal), staleTime: 300000 });
  if (query.isLoading) return <LoadingState label="Loading recorded sentiment evidence…" />;
  if (query.isError || !query.data?.sentiment.available) return <ErrorState message={query.error?.message || "Recorded sentiment-model evidence is unavailable."} />;
  const sentiment = query.data.sentiment;
  const ensemble = sentiment.models.find((model) => model.key === "ensemble");
  const failed = sentiment.acceptance.failed_checks.length;
  return <section className={`evaluation-ledger sentiment-ledger ${compact ? "compact" : ""}`}>
    <EvidenceHeader eyebrow="FinBERT evidence" title="Recorded sentiment-model performance" description="The active Bursa + PhraseBank ensemble is evaluated independently from today's retrieved articles." icon={<FlaskConical size={20} />} />
    <div className="evaluation-ledger-strip">
      <div><span>Evaluation</span><strong>{sentiment.evaluation_year} retrospective holdout</strong></div>
      <div><span>Active mixture</span><strong>{percentage(sentiment.weights.bursa, 0)} Bursa · {percentage(sentiment.weights.phrasebank, 0)} PhraseBank</strong></div>
      <div><span>Rows evaluated</span><strong>{sentiment.evaluation_rows ?? "Unavailable"}</strong></div>
      <div><span>Governance</span><strong className={sentiment.acceptance.eligible_for_promotion ? "status-pass" : "status-caution"}>{sentiment.acceptance.eligible_for_promotion ? <CheckCircle2 size={14} /> : <CircleAlert size={14} />}{sentiment.manual_override_active ? "Manual production override" : metricLabel(sentiment.promotion_status)}</strong></div>
    </div>
    <div className="evaluation-chart-grid"><SentimentComparison models={sentiment.models} /><SentimentClassChart model={ensemble} /></div>
    <div className="evaluation-reading-row">
      <div><Gauge size={17} /><span><strong>{percentage(ensemble?.balanced_accuracy)}</strong> ensemble balanced accuracy</span></div>
      <div><ShieldCheck size={17} /><span><strong>{percentage(ensemble?.ece)}</strong> calibration error (lower is better)</span></div>
      <div className={failed ? "has-caution" : ""}>{failed ? <CircleAlert size={17} /> : <CheckCircle2 size={17} />}<span><strong>{failed}</strong> promotion gate{failed === 1 ? "" : "s"} not passed</span></div>
    </div>
    {!compact && <EvaluationProtocol />}
  </section>;
}

export function ModelDiagnosticsDashboard() {
  return <section className="diagnostics-dashboard" aria-labelledby="diagnostics-dashboard-title">
    <div className="diagnostics-dashboard-heading"><span className="panel-kicker">Model evidence lab</span><h2 id="diagnostics-dashboard-title">How StockSense tests its models</h2><p>Explore the frozen evaluation evidence behind the interface. These charts make model quality visible without mixing historical performance with live predictions.</p></div>
    <div className="diagnostics-horizon-stack">{([1, 5, 10] as Horizon[]).map((horizon) => <StockModelEvidence key={horizon} horizon={horizon} compact />)}</div>
    <SentimentModelEvidence />
  </section>;
}
