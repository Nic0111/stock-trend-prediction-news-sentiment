import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  BarChart3,
  FlaskConical,
  Gauge,
  Info,
  Scale,
  TrendingUp,
} from "lucide-react";
import { useId, useMemo, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
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
  Tooltip,
  XAxis,
  YAxis,
  type LabelProps,
} from "recharts";
import { getModelPerformance } from "../api/client";
import { ErrorState, LoadingState } from "../components/AsyncState";
import { PageHeader } from "../components/PageHeader";
import {
  formatPercentagePoints,
  formatPercent,
  formatPerformanceValue,
} from "../utils/metrics";
import type {
  FusionCompanyDiagnostic,
  FusionSubgroupDiagnostic,
  Horizon,
  ModelPerformance,
  StockPerformanceHorizon,
} from "../types";

const COLORS = {
  blue: "#0876df",
  navy: "#153653",
  green: "#198754",
  red: "#d9485f",
  amber: "#c98b09",
  slate: "#8da0b5",
  violet: "#7257d7",
};

// Keep the candidate-family legend in the same horizon order as the bars.
// Recharts otherwise sorts legend labels alphabetically (10-day before 5-day).
const CANDIDATE_HORIZON_LEGEND = [
  { value: "1-day balanced accuracy", type: "square", color: COLORS.blue },
  { value: "5-day balanced accuracy", type: "square", color: COLORS.green },
  { value: "10-day balanced accuracy", type: "square", color: COLORS.violet },
] as const;

// Keep sentiment classes visually consistent everywhere in the diagnostics:
// positive is constructive (green), neutral is watchful (amber), and
// negative is adverse (red). The explicit order also avoids relying on the
// key order returned by a benchmark JSON file.
const SENTIMENT_CLASS_ORDER = ["positive", "neutral", "negative"] as const;

function sentimentClassColor(label: string) {
  switch (label.toLowerCase()) {
    case "positive": return COLORS.green;
    case "neutral": return COLORS.amber;
    case "negative": return COLORS.red;
    default: return COLORS.slate;
  }
}
const HORIZONS: Horizon[] = [1, 5, 10];

function percent(value: number | null | undefined, digits = 1) {
  return formatPercent(value, digits);
}

function plain(value: string | null | undefined) {
  return String(value || "Unavailable").replaceAll("_", " ");
}

// Percentage-point units are already stated in these chart y-axis titles.
// Keep the signed value/precision from the shared formatter, but avoid
// repeating the `pp` suffix on every tick label.
function formatPercentagePointsAxis(
  value: number | null | undefined,
  options: { signed?: boolean } = {},
) {
  return formatPercentagePoints(value, options).replace(/\s+pp$/, "");
}

// Keep chart axis titles in dedicated, readable bands. The x-axis title is
// centered below the plot. Y-axis titles are rendered from the axis viewBox,
// so each chart positions its own title just outside the tick-label band
// instead of inheriting a position from the surrounding panel. Tooltips remain
// responsible for exact point values.
function xAxisLabel(value: string) {
  return {
    value,
    position: "bottom" as const,
    angle: 0,
    offset: 18,
    fill: "#607890",
    fontSize: 12,
    fontWeight: 650,
  };
}

function yAxisLabel(value: string) {
  return {
    content: (props: LabelProps) => {
      const viewBox = props?.viewBox as { x?: number; y?: number; width?: number; height?: number } | undefined;
      // Recharts gives a Y-axis label the axis' tick-label band as its
      // viewBox.  Centering the title in that band puts it on top of the
      // tick values (especially at 100% browser zoom). Keep the title in a
      // dedicated strip just to the left of the tick labels instead. Every
      // model-performance chart uses this helper, so the spacing stays
      // consistent for both numeric and categorical Y axes.
      const x = (viewBox?.x ?? 0) - 22;
      const y = (viewBox?.y ?? 0) + (viewBox?.height ?? 0) / 2;
      return (
        <text
          x={x}
          y={y}
          transform={`rotate(-90 ${x} ${y})`}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#607890"
          fontSize={11}
          fontWeight={650}
        >
          {value}
        </text>
      );
    },
  };
}

// Improvement metrics are stored as probability deltas (for example 0.001
// means +0.10 percentage points). Recharts' automatic tick generator can
// choose awkward values such as 0.065 pp, which makes a small improvement
// harder to read. Use a short list of human-friendly 0.05/0.10-style steps,
// while still including zero and preserving negative improvements.
function improvementAxis(values: Array<number | null | undefined>) {
  const finiteValues = values.filter(
    (value): value is number => typeof value === "number" && Number.isFinite(value),
  );
  const minimum = Math.min(0, ...(finiteValues.length ? finiteValues : [0]));
  const maximum = Math.max(0, ...(finiteValues.length ? finiteValues : [0]));
  const span = Math.max(maximum - minimum, 0.0005);
  const preferredSteps = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
  const targetStep = span / 5;
  const step = preferredSteps.find((candidate) => candidate >= targetStep) || preferredSteps[preferredSteps.length - 1];
  const lower = Math.floor((minimum + 1e-12) / step) * step;
  const upper = Math.ceil((maximum - 1e-12) / step) * step;
  const safeUpper = upper <= lower ? lower + step : upper;
  const count = Math.max(1, Math.round((safeUpper - lower) / step));
  const ticks = Array.from({ length: count + 1 }, (_, index) =>
    Number((lower + index * step).toFixed(10)),
  );

  return {
    domain: [lower, safeUpper] as [number, number],
    ticks,
  };
}

const CHART_GUIDES = {
  baseline: {
    purpose: "Compares the selected champion with the strongest simple baseline on the same retrospective metrics.",
    read: "Higher balanced accuracy and Macro-F1 are better. A clear champion lead suggests the model learned more than the baseline.",
  },
  learning: {
    purpose: "Shows whether adding more chronological training observations improves later validation performance.",
    read: "A healthy curve has validation rising or stabilising while the training-validation gap stays reasonably small. A wide gap can indicate overfitting.",
  },
  foldStability: {
    purpose: "Shows balanced accuracy in each later time fold, so performance can be checked across market regimes.",
    read: "Points close to the mean indicate steadier behaviour. Large swings mean the result may depend on a particular market period.",
  },
  calibration: {
    purpose: "Checks whether predicted probabilities match the frequency of outcomes observed in the evaluation data.",
    read: "Points nearer the diagonal are better. A low ECE and a line close to the diagonal mean the probabilities are more trustworthy.",
  },
  ranking: {
    purpose: "Tests how well probabilities rank Up cases above Down cases across decision thresholds.",
    read: "A ROC curve above the no-skill line and a strong precision-recall curve indicate useful ranking. Higher AUC is better.",
  },
  threshold: {
    purpose: "Shows how changing the probability cutoff changes balanced accuracy and Up/Down recall.",
    read: "Prefer a cutoff with balanced recalls and a broad, stable high-performing region rather than a sharp isolated peak.",
  },
  importance: {
    purpose: "Measures how much balanced accuracy falls when each feature is shuffled in the fitted model.",
    read: "A larger positive decrease means the model relied more on that feature. This is a diagnostic, not proof of causation.",
  },
  seedStability: {
    purpose: "Refits the same selected configuration with fixed random seeds to test reproducibility.",
    read: "Tightly grouped balanced-accuracy and Macro-F1 lines are better; wide spread means results depend on the random seed.",
  },
  confusion: {
    purpose: "Counts correct and incorrect Up/Down classifications by comparing actual rows with predicted columns.",
    read: "The diagonal cells (TN and TP) should dominate. Off-diagonal FP and FN cells are the direction errors to reduce.",
  },
  candidates: {
    purpose: "Compares every evaluated model family across the one-, five- and ten-trading-day development horizons.",
    read: "Higher balanced accuracy is better. Dark outlines identify the selected champion for each horizon; missing bars mean no result was recorded.",
  },
  fusionArms: {
    purpose: "Compares stock-only and news-integrated study arms using the same retrospective rows.",
    read: "News is useful only when it improves probability and direction metrics without worsening calibration or stability.",
  },
  fusionGain: {
    purpose: "Summarises how much news changes Brier probability error and balanced accuracy compared with stock-only.",
    read: "Positive improvement is better: it means fusion reduced probability error or increased balanced accuracy.",
  },
  fusionRolling: {
    purpose: "Tracks fusion improvement across separate retrospective evaluation years.",
    read: "Repeated gains above zero are stronger evidence than one favourable year; negative years show when news hurt.",
  },
  fusionCalibration: {
    purpose: "Compares stock-only and fusion probabilities with the observed outcome frequency in each calibration bin.",
    read: "Lines close to the ideal diagonal are better. Fusion should stay close to the diagonal while improving other metrics.",
  },
  placebo: {
    purpose: "Creates a null distribution by shuffling news within company and month while preserving timing and availability.",
    read: "The real news gain should exceed nearly all shuffled trials. A low placebo percentile weakens the case for genuine news value.",
  },
  companyGain: {
    purpose: "Shows whether news improves or harms balanced accuracy for each company rather than only in aggregate.",
    read: "More non-negative bars and no large negative company losses are better. A single average can hide uneven performance.",
  },
  transitions: {
    purpose: "Counts cases where news corrected a stock error, harmed a correct stock signal, or agreed with it.",
    read: "Corrections should outweigh harms for news to add value. Always consider the number of observations behind each bar.",
  },
  coverage: {
    purpose: "Compares stock-only and fusion performance at different levels of usable news coverage.",
    read: "Fusion should help when reliable news is present and converge to stock-only when news is unavailable.",
  },
  confidence: {
    purpose: "Checks whether accuracy changes across stock-model probability confidence buckets.",
    read: "Accuracy that rises with confidence is reassuring, but small buckets should not be overinterpreted.",
  },
  reliability: {
    purpose: "Checks whether news reliability or coverage levels correspond to different fusion performance.",
    read: "Higher-reliability buckets should be at least as useful as low-reliability buckets; missing news should fall back to stock-only.",
  },
  corrections: {
    purpose: "Shows the largest news corrections and whether their coefficients keep a consistent sign across purged folds.",
    read: "Stable direction and small fold dispersion are better. Unstable corrections are weak evidence for promotion.",
  },
  sentimentModels: {
    purpose: "Compares the sentiment model variants on identical retrospective labelled rows.",
    read: "Higher balanced accuracy and Macro-F1 are better; compare all models on the same split rather than raw scores from different datasets.",
  },
  sentimentClassF1: {
    purpose: "Shows F1 separately for positive, neutral and negative sentiment classes.",
    read: "Higher and more even bars are better. A weak neutral bar means overall accuracy may hide a class-specific weakness.",
  },
  sentimentCoverage: {
    purpose: "Shows how many eligible Bursa news articles are available in each development year.",
    read: "Coverage should be sufficient across the timeline; large gaps or tiny years limit how confidently the model can generalise.",
  },
  labelDistribution: {
    purpose: "Shows how many labelled articles belong to each sentiment class.",
    read: "More balanced class counts make macro metrics more informative. A dominant class can make accuracy look better than minority-class performance.",
  },
} as const;

type ChartGuideKey = keyof typeof CHART_GUIDES;

function ChartTitle({
  guide,
  children,
}: {
  guide: ChartGuideKey;
  children: ReactNode;
}) {
  const tooltipId = useId();
  const content = CHART_GUIDES[guide];
  return (
    <div
      className="chart-title-guide"
      tabIndex={0}
      aria-describedby={tooltipId}
    >
      <h3>{children}</h3>
      <Info size={13} aria-hidden="true" />
      <div className="chart-title-tooltip" id={tooltipId} role="tooltip">
        <strong>What this shows</strong>
        <span>{content.purpose}</span>
        <strong>How to read it</strong>
        <span>{content.read}</span>
      </div>
    </div>
  );
}

function EvaluationTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: ReadonlyArray<{
    name?: unknown;
    value?: unknown;
    color?: string;
    payload?: Record<string, unknown>;
  }>;
  label?: unknown;
}) {
  if (!active || !payload?.length) return null;
  const context = payload[0]?.payload || {};
  return (
    <div className="performance-tooltip">
      <strong>
        {String(
          context.company_name ||
            context.model_name ||
            label ||
            "Recorded value",
        )}
      </strong>
      {Boolean(context.ticker) && <span>{String(context.ticker)}</span>}
      {payload
        .filter(
          (entry) =>
            typeof entry.value === "number" &&
            !String(entry.name).toLowerCase().includes("band"),
        )
        .map((entry, index) => (
          <span key={`${String(entry.name)}-${index}`}>
            <i style={{ background: entry.color }} />
            {String(entry.name)}:{" "}
            {context.value_format === "count"
              ? Number(entry.value).toLocaleString()
              : formatPerformanceValue(Number(entry.value), String(entry.name))}
          </span>
        ))}
    </div>
  );
}

function ChartEmpty({ message = "No recorded values are available for this view." }: { message?: string }) {
  return (
    <div className="performance-empty" role="status">
      <Info size={20} />
      <strong>No chart data</strong>
      <span>{message}</span>
    </div>
  );
}

function hasNumericValues(
  rows: readonly unknown[],
  keys: readonly string[],
) {
  return rows.some((row) => {
    if (!row || typeof row !== "object") return false;
    const record = row as Record<string, unknown>;
    return keys.some(
      (key) =>
        typeof record[key] === "number" && Number.isFinite(record[key]),
    );
  });
}

function subgroupBalancedAccuracy(
  subgroup: FusionSubgroupDiagnostic | undefined,
  side: "stock" | "fusion",
) {
  if (!subgroup) return null;
  const record = subgroup as unknown as Record<string, unknown>;
  const flatKey = side === "stock"
    ? "stock_balanced_accuracy"
    : "fusion_balanced_accuracy";
  const flat = record[flatKey];
  if (typeof flat === "number" && Number.isFinite(flat)) return flat;

  const armKeys = side === "stock"
    ? ["stock_only_meta", "stock_only", "stock_meta", "stock"]
    : ["compact_full", "full_fusion", "stock_plus_news", "fusion"];
  for (const armKey of armKeys) {
    const arm = record[armKey];
    if (!arm || typeof arm !== "object") continue;
    const armRecord = arm as Record<string, unknown>;
    const score =
      armRecord.balanced_accuracy ??
      (armRecord.metrics && typeof armRecord.metrics === "object"
        ? (armRecord.metrics as Record<string, unknown>).balanced_accuracy
        : undefined);
    if (typeof score === "number" && Number.isFinite(score)) return score;
  }
  return null;
}

const EVIDENCE_BUCKET_ORDER = ["high", "medium", "low", "unavailable"] as const;

function orderedEvidenceBuckets<T>(entries: Array<[string, T]>) {
  const rank = new Map<string, number>(
    EVIDENCE_BUCKET_ORDER.map((bucket, index) => [bucket, index]),
  );

  return [...entries]
    .sort(([left], [right]) => {
      const leftRank = rank.get(left.trim().toLowerCase()) ?? Number.MAX_SAFE_INTEGER;
      const rightRank = rank.get(right.trim().toLowerCase()) ?? Number.MAX_SAFE_INTEGER;
      return leftRank - rightRank || left.localeCompare(right);
    })
    .map(([bucket, value]) => {
      const normalized = bucket.trim().toLowerCase();
      const label = EVIDENCE_BUCKET_ORDER.includes(
        normalized as (typeof EVIDENCE_BUCKET_ORDER)[number],
      )
        ? normalized.charAt(0).toUpperCase() + normalized.slice(1)
        : plain(bucket);
      return [label, value] as [string, T];
    });
}

function MetricScoreboard({ horizon }: { horizon: StockPerformanceHorizon }) {
  const items = [
    {
      label: "Balanced accuracy",
      value: horizon.metrics.balanced_accuracy,
      note: "Equal weight for Up and Down",
      format: "percent",
    },
    {
      label: "Macro-F1",
      value: horizon.metrics.macro_f1,
      note: "Consistency across both directions",
      format: "percent",
    },
    {
      label: "MCC",
      value: horizon.metrics.mcc,
      note: "Correlation from -1 to +1",
      format: "number",
    },
    {
      label: "ROC-AUC",
      value: horizon.metrics.roc_auc,
      note: "Probability ranking skill",
      format: "percent",
    },
    {
      label: "Brier score",
      value: horizon.metrics.brier_score,
      note: "Probability error - lower is better",
      format: "number",
    },
    {
      label: "ECE",
      value: horizon.metrics.ece,
      note: "Calibration gap - lower is better",
      format: "percent",
    },
    {
      label: "Baseline uplift (pp)",
      value: horizon.balanced_accuracy_gain,
      note: `Against ${plain(horizon.strongest_baseline)} · percentage-point difference`,
      format: "delta",
    },
  ] as const;
  return (
    <div className="performance-scoreboard">
      {items.map(({ label, value, note, format }) => (
        <div key={label}>
          <span>{label}</span>
          <strong>
            {typeof value === "number"
              ? format === "delta"
                ? formatPercentagePoints(value)
                : format === "percent"
                  ? percent(value)
                  : value.toFixed(3)
              : "Unavailable"}
          </strong>
          <small>{note}</small>
        </div>
      ))}
    </div>
  );
}

function ProtocolRail({ performance }: { performance: ModelPerformance }) {
  const stock = performance.stock;
  return (
    <section className="performance-protocol" aria-labelledby="protocol-title">
      <div className="performance-section-heading">
        <Info size={19} />
        <div>
          <span>Evaluation protocol</span>
          <h2 id="protocol-title">Two periods, two different jobs</h2>
        </div>
      </div>
      <div className="protocol-rail protocol-rail-two-periods">
        <div>
          <b>Develop</b>
          <strong>
            {stock.development_period.start?.slice(0, 4)}-
            {stock.development_period.end?.slice(0, 4)}
          </strong>
          <span>
            Train models and validate through purged chronological folds.
          </span>
        </div>
        <i className="protocol-connector" aria-hidden="true" />
        <div>
          <b>Retrospective test</b>
          <strong>{stock.retrospective_test_period.start?.slice(0, 4)}</strong>
          <span>
            Compare the frozen champion with baselines on later historical data.
          </span>
        </div>
      </div>
    </section>
  );
}

function BaselinePanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const data = [
    {
      metric: "Balanced accuracy",
      Champion: horizon.metrics.balanced_accuracy,
      Baseline: horizon.baseline_metrics.balanced_accuracy,
    },
    {
      metric: "Macro-F1",
      Champion: horizon.metrics.macro_f1,
      Baseline: horizon.baseline_metrics.macro_f1,
    },
  ];
  return (
    <article className="performance-panel">
      <header>
        <div>
          <span>Retrospective</span>
          <ChartTitle guide="baseline">Champion versus strongest baseline</ChartTitle>
        </div>
        <Scale size={19} />
      </header>
      <div className="performance-chart">
        {hasNumericValues(data, ["Champion", "Baseline"]) ? (
          <ResponsiveContainer width="100%" height="100%">
              <BarChart
              data={data}
              margin={{ top: 14, right: 12, left: 72, bottom: 48 }}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fontSize: 11 }} label={xAxisLabel("Metric")} />
              <YAxis
                domain={[0, 0.8]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={yAxisLabel("Score")}
              />
              <Tooltip content={<EvaluationTooltip />} />
              <Legend />
              <Bar dataKey="Champion" fill={COLORS.blue} radius={[6, 6, 0, 0]} />
              <Bar dataKey="Baseline" fill={COLORS.slate} radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <ChartEmpty message="Champion and baseline scores were not recorded for this horizon." />}
      </div>
      <p>
        Balanced accuracy is the primary comparison because Up and Down receive
        equal importance.
      </p>
    </article>
  );
}

function LearningCurvePanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const data = horizon.learning_curve.map((point) => ({
    ...point,
    rows: point.mean_training_rows,
    trainingLower: Math.max(0, point.training_mean - point.training_std),
    trainingBand: point.training_std * 2,
    validationLower: Math.max(0, point.validation_mean - point.validation_std),
    validationBand: point.validation_std * 2,
  }));
  return (
    <article className="performance-panel">
      <header>
        <div>
          <span>Development</span>
          <ChartTitle guide="learning">Learning curve</ChartTitle>
        </div>
        <TrendingUp size={19} />
      </header>
      <div className="performance-chart">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={data}
              margin={{ top: 14, right: 12, left: 72, bottom: 48 }}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis type="number" dataKey="rows" domain={[0, "auto"]} tick={{ fontSize: 11 }} label={xAxisLabel("Training rows")} />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={yAxisLabel("Balanced accuracy")}
              />
              <Tooltip content={<EvaluationTooltip />} />
              <Legend />
              <Area
                dataKey="trainingLower"
                stackId="training-band"
                stroke="none"
                fill="transparent"
                name="Training band base"
                legendType="none"
              />
              <Area
                dataKey="trainingBand"
                stackId="training-band"
                stroke="#496b8d"
                strokeOpacity={0.62}
                strokeWidth={1.2}
                strokeDasharray="4 3"
                fill="#496b8d"
                fillOpacity={0.24}
                name="Training fold variability"
              />
              <Area
                dataKey="validationLower"
                stackId="validation-band"
                stroke="none"
                fill="transparent"
                name="Validation band base"
                legendType="none"
              />
              <Area
                dataKey="validationBand"
                stackId="validation-band"
                stroke="#2f86d1"
                strokeOpacity={0.7}
                strokeWidth={1.2}
                strokeDasharray="4 3"
                fill="#2f86d1"
                fillOpacity={0.2}
                name="Validation fold variability"
              />
              <Line
                dataKey="training_mean"
                name="Training balanced accuracy"
                stroke={COLORS.navy}
                strokeWidth={2.2}
                dot={{ r: 3 }}
              />
              <Line
                dataKey="validation_mean"
                name="Validation balanced accuracy"
                stroke={COLORS.blue}
                strokeWidth={2.8}
                dot={{ r: 4 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="performance-empty">
            <FlaskConical size={22} />
            <strong>Learning curve is being prepared</strong>
            <span>
              The dashboard will not substitute fold scores or fabricated
              points.
            </span>
          </div>
        )}
      </div>
      <p>
        Increasing training rows should narrow the gap between training and
        later validation performance.
      </p>
    </article>
  );
}

function StabilityPanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const data = horizon.cv_fold_balanced_accuracy.map((score, index) => ({
    fold: `Fold ${index + 1}`,
    score,
  }));
  const mean = data.length
    ? data.reduce((sum, row) => sum + row.score, 0) / data.length
    : 0;
  return (
    <article className="performance-panel">
      <header>
        <div>
          <span>Development</span>
          <ChartTitle guide="foldStability">Time-fold stability</ChartTitle>
        </div>
        <Activity size={19} />
      </header>
      <div className="performance-chart">
        {hasNumericValues(data, ["score"]) ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 14, right: 12, left: 72, bottom: 48 }}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="fold" label={xAxisLabel("Chronological fold")} />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={yAxisLabel("Balanced accuracy")}
              />
              <Tooltip content={<EvaluationTooltip />} />
              <ReferenceLine
                y={mean}
                stroke={COLORS.slate}
                strokeDasharray="5 5"
                label="Mean"
              />
              <Line
                dataKey="score"
                name="Balanced accuracy"
                stroke={COLORS.green}
                strokeWidth={2.6}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : <ChartEmpty message="Chronological fold scores are unavailable for this run." />}
      </div>
      <p>
        Each point is a later chronological block. Large swings warn that
        performance may depend on the market regime.
      </p>
    </article>
  );
}

function CalibrationPanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const data = horizon.calibration_curve.map((point) => ({
    ...point,
    perfect: point.predicted,
  }));
  return (
    <article className="performance-panel">
      <header>
        <div>
          <span>Retrospective</span>
          <ChartTitle guide="calibration">Probability reliability</ChartTitle>
        </div>
        <Gauge size={19} />
      </header>
      <div className="performance-chart">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 14, right: 12, left: 72, bottom: 48 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="predicted"
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={xAxisLabel("Predicted Up probability")}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={yAxisLabel("Observed Up frequency")}
              />
              <Tooltip content={<EvaluationTooltip />} />
              <Line
                dataKey="perfect"
                stroke={COLORS.slate}
                dot={false}
                strokeDasharray="5 5"
                name="Perfect calibration"
              />
              <Line
                dataKey="observed"
                name="Observed Up frequency"
                stroke={COLORS.violet}
                strokeWidth={2.7}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="performance-empty">
            Reliability bins are unavailable.
          </div>
        )}
      </div>
      <p>
        Points near the diagonal mean a stated probability behaved similarly in
        the holdout data. ECE: {percent(horizon.metrics.ece)}.
      </p>
    </article>
  );
}

function DiscriminationPanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const roc = horizon.roc_curve.map((point) => ({
    x: point.false_positive_rate,
    y: point.true_positive_rate,
    reference: point.false_positive_rate,
  }));
  const precisionRecall = horizon.precision_recall_curve.map((point) => ({
    x: point.recall,
    y: point.precision,
  }));
  return (
    <article className="performance-panel performance-panel-wide">
      <header><div><span>Retrospective</span><ChartTitle guide="ranking">Probability ranking: ROC and precision-recall</ChartTitle></div><Activity size={19} /></header>
      <div className="performance-chart">
        {hasNumericValues(roc, ["y"]) || hasNumericValues(precisionRecall, ["y"]) ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 14, right: 12, left: 72, bottom: 48 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} label={xAxisLabel("False-positive rate / recall")} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} label={yAxisLabel("Rate")} />
              <Tooltip content={<EvaluationTooltip />} />
              <Legend />
              <Line data={roc} dataKey="reference" name="No-skill ROC" stroke={COLORS.slate} strokeDasharray="5 5" dot={false} />
              <Line data={roc} dataKey="y" name="ROC curve" stroke={COLORS.blue} strokeWidth={2.5} dot={false} />
              <Line data={precisionRecall} dataKey="y" name="Precision-recall curve" stroke={COLORS.green} strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <ChartEmpty message="ROC and precision-recall points are unavailable for this horizon." />}
      </div>
      <p>ROC shows ranking across thresholds; precision–recall makes the trade-off between finding Up cases and false alarms visible.</p>
    </article>
  );
}

function ThresholdPanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const selected = horizon.threshold_curve.find((row) => row.selected)?.threshold;
  return (
    <article className="performance-panel">
      <header><div><span>Retrospective</span><ChartTitle guide="threshold">Decision-threshold sensitivity</ChartTitle></div><Gauge size={19} /></header>
      <div className="performance-chart">
        {hasNumericValues(horizon.threshold_curve, ["balanced_accuracy", "up_recall", "down_recall"]) ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={horizon.threshold_curve} margin={{ top: 14, right: 12, left: 72, bottom: 48 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis type="number" dataKey="threshold" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} label={xAxisLabel("Decision threshold")} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} label={yAxisLabel("Rate")} />
              <Tooltip content={<EvaluationTooltip />} /><Legend />
              {typeof selected === "number" && <ReferenceLine x={selected} stroke={COLORS.amber} label="Selected" />}
              <Line dataKey="balanced_accuracy" name="Balanced accuracy" stroke={COLORS.blue} dot={false} />
              <Line dataKey="up_recall" name="Up recall" stroke={COLORS.green} dot={false} />
              <Line dataKey="down_recall" name="Down recall" stroke={COLORS.red} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <ChartEmpty message="Threshold-sensitivity scores are unavailable for this horizon." />}
      </div>
      <p>The selected cutoff is locked before the 2025 evaluation; this plot shows how the two direction recalls trade off around it.</p>
    </article>
  );
}

function ImportancePanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const data = horizon.permutation_importance.slice(0, 12).map((row) => ({ ...row, label: plain(row.feature) })).reverse();
  return (
    <article className="performance-panel performance-panel-wide permutation-importance-panel">
      <header><div><span>Retrospective diagnostic</span><ChartTitle guide="importance">Permutation importance</ChartTitle></div><Scale size={19} /></header>
      <div className="performance-chart performance-chart-tall permutation-importance-chart">
        {hasNumericValues(data, ["importance_mean"]) ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 8, right: 12, left: 84, bottom: 48 }}>
              <CartesianGrid horizontal={false} strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, "auto"]} tickFormatter={(v) => formatPercentagePoints(v, { signed: false })} label={xAxisLabel("Balanced-accuracy decrease (pp)")} />
              <YAxis type="category" dataKey="label" width={155} interval={0} tick={{ fontSize: 11 }} label={yAxisLabel("Feature")} />
              <Tooltip content={<EvaluationTooltip />} />
              <Bar dataKey="importance_mean" name="Balanced-accuracy decrease" fill={COLORS.violet} radius={[0, 5, 5, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <ChartEmpty message="Permutation importance was not recorded for this run." />}
      </div>
      <p>A larger decrease after shuffling means the fitted model depended more on that feature. It is not a causal claim.</p>
    </article>
  );
}

function SeedPanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  return (
    <article className="performance-panel">
      <header><div><span>Development</span><ChartTitle guide="seedStability">Fixed-manifest seed stability</ChartTitle></div><Activity size={19} /></header>
      <div className="performance-chart">
        {horizon.seed_stability.length ? <ResponsiveContainer width="100%" height="100%">
          <LineChart data={horizon.seed_stability} margin={{ top: 14, right: 12, left: 72, bottom: 48 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis type="number" dataKey="seed" domain={[0, "auto"]} label={xAxisLabel("Random seed")} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} label={yAxisLabel("Score")} />
            <Tooltip content={<EvaluationTooltip />} /><Legend />
            <Line dataKey="balanced_accuracy" name="Balanced accuracy" stroke={COLORS.blue} strokeWidth={2.5} />
            <Line dataKey="macro_f1" name="Macro-F1" stroke={COLORS.green} strokeWidth={2.2} />
          </LineChart>
        </ResponsiveContainer> : <div className="performance-empty">Seed-panel evidence is available after an Official v3 run.</div>}
      </div>
      <p>The same selected hyperparameters are refitted with five frozen seeds. Stable values reduce dependence on one lucky random state.</p>
    </article>
  );
}

function ConfusionPanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const matrix = horizon.confusion_matrix;
  const labelIndex = (label: string) =>
    matrix?.labels.findIndex(
      (candidate) => candidate.toLowerCase() === label.toLowerCase(),
    ) ?? -1;
  const orderedLabels = ["Down", "Up"];
  return (
    <article className="performance-panel">
      <header>
        <div>
          <span>Retrospective</span>
          <ChartTitle guide="confusion">Direction errors</ChartTitle>
        </div>
        <BarChart3 size={19} />
      </header>
      {matrix ? (
        <div className="confusion-ledger">
          <div className="confusion-axis-corner">
            <span>Actual rows</span>
            <span>Predicted columns</span>
          </div>
          {orderedLabels.map((label) => (
            <b key={`predicted-${label}`}>Predicted {label}</b>
          ))}
          {orderedLabels.flatMap((actualLabel) => {
            const rowIndex = labelIndex(actualLabel);
            return [
              <b key={`actual-${actualLabel}`}>Actual {actualLabel}</b>,
              ...orderedLabels.map((predictedLabel) => {
                const columnIndex = labelIndex(predictedLabel);
                const value = matrix.values[rowIndex]?.[columnIndex];
                const normalized = matrix.normalized[rowIndex]?.[columnIndex];
                const cellName =
                  actualLabel === "Down"
                    ? predictedLabel === "Down"
                      ? "TN"
                      : "FP"
                    : predictedLabel === "Up"
                      ? "TP"
                      : "FN";
                const cellDescription =
                  cellName === "TN"
                    ? "True negative: Down correctly predicted"
                    : cellName === "FP"
                      ? "False positive: Down incorrectly predicted as Up"
                      : cellName === "FN"
                        ? "False negative: Up incorrectly predicted as Down"
                        : "True positive: Up correctly predicted";
                return (
                  <div
                    className={
                      actualLabel === predictedLabel ? "correct" : "error"
                    }
                    key={`${actualLabel}-${predictedLabel}`}
                    title={cellDescription}
                  >
                    <b className="confusion-cell-code">{cellName}</b>
                    <strong>{value ?? "—"}</strong>
                    <span>{percent(normalized)}</span>
                  </div>
                );
              }),
            ];
          })}
        </div>
      ) : (
        <div className="performance-empty">
          Confusion matrix is unavailable.
        </div>
      )}
      <p>
        Positive means Up and negative means Down. Percentages are normalized
        within each actual-direction row.
      </p>
    </article>
  );
}

function CandidatesPanel({
  horizons,
}: {
  horizons: StockPerformanceHorizon[];
}) {
  const horizonByDays = new Map(
    horizons.map((item) => [item.horizon_days, item]),
  );
  const modelNames = [
    ...new Set(
      horizons.flatMap((item) =>
        item.candidate_families.map((row) => row.model_name),
      ),
    ),
  ];
  const scoreFor = (modelName: string, days: Horizon) =>
    horizonByDays
      .get(days)
      ?.candidate_families.find((row) => row.model_name === modelName)
      ?.balanced_accuracy ?? null;
  const data = modelNames
    .map((modelName) => ({
      model_name: modelName,
      label: plain(modelName),
      day1: scoreFor(modelName, 1),
      day5: scoreFor(modelName, 5),
      day10: scoreFor(modelName, 10),
    }))
    .sort(
      (left, right) =>
        Math.max(right.day1 ?? -1, right.day5 ?? -1, right.day10 ?? -1) -
        Math.max(left.day1 ?? -1, left.day5 ?? -1, left.day10 ?? -1),
    );
  const championFor = (days: Horizon) => horizonByDays.get(days)?.model_name;
  const chartHeight = Math.max(340, data.length * 44 + 76);
  return (
    <article className="performance-panel performance-panel-wide">
      <header>
        <div>
          <span>Development</span>
          <ChartTitle guide="candidates">All evaluated candidate families</ChartTitle>
        </div>
        <Scale size={19} />
      </header>
      <div
        className="performance-chart performance-candidate-chart"
        style={{ height: chartHeight }}
      >
        {hasNumericValues(data, ["day1", "day5", "day10"]) ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              barCategoryGap="24%"
              margin={{ top: 8, right: 24, left: 80, bottom: 48 }}
            >
              <CartesianGrid horizontal={false} strokeDasharray="3 3" />
              <XAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                label={xAxisLabel("Balanced accuracy")}
              />
              <YAxis
                type="category"
                dataKey="label"
                width={145}
                tick={{ fontSize: 10 }}
                label={yAxisLabel("Model family")}
              />
              <Tooltip content={<EvaluationTooltip />} />
              <Legend
                itemSorter={(entry) => {
                  const index = CANDIDATE_HORIZON_LEGEND.findIndex(
                    (item) => item.value === entry.value,
                  );
                  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
                }}
              />
              <Bar
                dataKey="day1"
                name="1-day balanced accuracy"
                fill={COLORS.blue}
                radius={[0, 4, 4, 0]}
              >
                {data.map((item) => (
                  <Cell
                    key={`1-${item.model_name}`}
                    stroke={
                      item.model_name === championFor(1) ? COLORS.navy : "none"
                    }
                    strokeWidth={item.model_name === championFor(1) ? 2 : 0}
                  />
                ))}
              </Bar>
              <Bar
                dataKey="day5"
                name="5-day balanced accuracy"
                fill={COLORS.green}
                radius={[0, 4, 4, 0]}
              >
                {data.map((item) => (
                  <Cell
                    key={`5-${item.model_name}`}
                    stroke={
                      item.model_name === championFor(5) ? COLORS.navy : "none"
                    }
                    strokeWidth={item.model_name === championFor(5) ? 2 : 0}
                  />
                ))}
              </Bar>
              <Bar
                dataKey="day10"
                name="10-day balanced accuracy"
                fill={COLORS.violet}
                radius={[0, 4, 4, 0]}
              >
                {data.map((item) => (
                  <Cell
                    key={`10-${item.model_name}`}
                    stroke={
                      item.model_name === championFor(10) ? COLORS.navy : "none"
                    }
                    strokeWidth={item.model_name === championFor(10) ? 2 : 0}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <ChartEmpty message="Candidate-family scores are unavailable for the selected stock run." />}
      </div>
      <p>
        Every family evaluated in the active stock-model run is shown. Each bar
        is that family's best purged chronological development result for
        the stated horizon; an absent bar means the family was not evaluated for
        that horizon. Dark outlines mark the selected champions.
      </p>
    </article>
  );
}

function CompanyTable({
  rows,
  horizon,
  sentiment = false,
}: {
  rows:
    | ModelPerformance["sentiment"]["per_company"]
    | StockPerformanceHorizon["per_company"];
  horizon?: Horizon;
  sentiment?: boolean;
}) {
  return (
    <div className="performance-table-wrap">
      <table className="performance-table company-metrics-table">
        <thead>
          <tr>
            <th>Company</th>
            <th>Rows</th>
            <th>Balanced accuracy</th>
            <th>Macro-F1</th>
            <th>MCC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.ticker}
              title={`${row.company_name || row.ticker} (${row.ticker})${horizon ? ` - ${horizon}-trading-day horizon - Retrospective` : " - 2025 sentiment retrospective"}`}
            >
              <td>
                <strong>{row.company_name || row.ticker}</strong>
                <span>{row.ticker}</span>
              </td>
              <td>{row.rows}</td>
              <td>
                {(() => {
                  const balancedAccuracy = row.balanced_accuracy;
                  const chanceBaseline = sentiment ? 1 / 3 : 0.5;
                  const normalizedScore =
                    typeof balancedAccuracy === "number" &&
                    Number.isFinite(balancedAccuracy)
                      ? Math.max(
                          0,
                          Math.min(
                            1,
                            (balancedAccuracy - chanceBaseline) /
                              (1 - chanceBaseline),
                          ),
                        )
                      : null;
                  const metricStyle = {
                    "--metric-hue":
                      normalizedScore === null
                        ? "210deg"
                        : `${Math.round(normalizedScore * 120)}deg`,
                  } as React.CSSProperties;
                  return (
                <span
                  className="metric-heat metric-heat-gradient"
                  style={metricStyle}
                  aria-label={`Balanced accuracy: ${percent(balancedAccuracy)}`}
                  title={`Balanced accuracy: ${percent(balancedAccuracy)}`}
                >
                  {percent(balancedAccuracy)}
                </span>
                  );
                })()}
              </td>
              <td>{percent(row.macro_f1)}</td>
              <td>
                {typeof row.mcc === "number"
                  ? row.mcc.toFixed(3)
                  : "Unavailable"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FusionAblationPanel({ performance, horizon }: { performance: ModelPerformance; horizon: Horizon }) {
  const result = performance.fusion_ablation?.horizons?.[String(horizon)];
  if (!result) return null;
  const labels: Record<string, string> = {
    frozen_stock: "Frozen stock probability",
    stock_only_meta: "Stock-only meta-model",
    stock_meta: "Stock-only meta-model",
    stock_only: "Stock only",
    news_only: "News only",
    stock_plus_news: "Stock + news",
    stock_plus_shuffled_news: "Stock + shuffled news",
    permuted_news_placebo: "Permuted-news placebo",
    transparent_weighted_formula: "Transparent weighted diagnostic",
    occurrence_attention: "News occurrence / attention",
    sentiment_tone: "Sentiment tone",
    material_event_category: "Material event category",
    compact_full: "Full stock + news fusion",
    full_fusion: "Full stock + news fusion",
    legacy_17_feature_stacker: "Legacy 17-feature stacker · diagnostic",
  };
  const data = Object.entries(result.arms)
    .filter(([, value]) => Boolean(value.metrics))
    .map(([key, value]) => ({
      arm: labels[key] || plain(key),
      balanced_accuracy: value.metrics?.balanced_accuracy,
      macro_f1: value.metrics?.macro_f1,
      brier_score: value.metrics?.brier_score,
    }));
  const isControlled = [
    "stocksense-fusion-evaluation-v2",
    "stocksense-fusion-evaluation-v3",
    "stocksense-fusion-evaluation-v4",
  ].includes(performance.fusion_ablation?.schema_version || "");
  const benchmarkRows = result.benchmark_rows ?? result.matched_benchmark_rows ?? 0;
  const baGain = result.balanced_accuracy_gain ?? result.stock_plus_news_gain ?? 0;
  const baInterval = result.bootstrap?.balanced_accuracy_gain ?? result.block_bootstrap_gain_ci;
  const rolling = result.rolling_origins || [];
  const companyData = Object.entries(result.per_company || {}).map(([ticker, value]) => ({
    ticker,
    gain: value.gain,
    rows: value.rows,
  }));
  const transitionData = Object.entries(result.error_transitions || {}).map(([name, value]) => ({
    name: plain(name),
    value,
  }));
  const coverageData = orderedEvidenceBuckets(
    Object.entries(result.subgroups?.news_coverage || result.subgroups?.coverage || {}),
  ).map(([coverage, value]) => ({
    coverage,
    stock: subgroupBalancedAccuracy(value, "stock"),
    fusion: subgroupBalancedAccuracy(value, "fusion"),
    rows: value.rows,
  }));
  const stockConfidence =
    result.confidence_accuracy?.stock_only_meta ||
    result.confidence_accuracy?.stock_only ||
    [];
  const fusionConfidence =
    result.confidence_accuracy?.compact_full ||
    result.confidence_accuracy?.full_fusion ||
    result.confidence_accuracy?.stock_plus_news ||
    [];
  const confidenceData = stockConfidence.map((value, index) => ({
    confidence: `Q${value.bucket}`,
    stock: value.accuracy,
    fusion: fusionConfidence[index]?.accuracy,
    rows: value.rows,
  }));
  const reliabilityData = orderedEvidenceBuckets(
    Object.entries(result.subgroups?.news_reliability || {}),
  ).map(([reliability, value]) => ({
    reliability,
    stock: subgroupBalancedAccuracy(value, "stock"),
    fusion: subgroupBalancedAccuracy(value, "fusion"),
    rows: value.rows,
  }));
  const coefficientData = Object.entries(result.coefficient_stability?.coefficients || {})
    .map(([feature, value]) => ({
      feature: plain(feature.replace(/^news_/, "")),
      coefficient: value.mean,
      sign_consistency: value.sign_consistency,
      standard_deviation: value.standard_deviation,
    }))
    .sort((left, right) => Math.abs(right.coefficient) - Math.abs(left.coefficient))
    .slice(0, 8);
  const stockCalibration = result.calibration?.stock_meta?.curve || result.calibration?.stock_only?.curve || [];
  const fusionCalibration =
    result.calibration?.compact_full?.curve ||
    result.calibration?.stock_plus_news?.curve ||
    result.calibration?.full_fusion?.curve ||
    [];
  const calibration = Array.from({ length: Math.max(stockCalibration.length, fusionCalibration.length) }, (_, index) => ({
    bin: index + 1,
    ideal: (index + 0.5) / Math.max(stockCalibration.length, fusionCalibration.length, 1),
    stock_observed: stockCalibration[index]?.observed,
    fusion_observed: fusionCalibration[index]?.observed,
  }));
  const placeboDistribution = result.placebo?.complete_event?.distribution || result.placebo?.distribution || [];
  const placeboPercentile = Math.min(
    result.placebo?.content_only?.actual_percentile ?? result.placebo?.actual_percentile ?? 0,
    result.placebo?.complete_event?.actual_percentile ?? result.placebo?.actual_percentile ?? 0,
  );
  const placeboBins = Array.from({ length: 12 }, (_, index) => {
    if (!placeboDistribution.length) return { bin: index, count: 0, gain: 0 };
    const minimum = Math.min(...placeboDistribution);
    const maximum = Math.max(...placeboDistribution);
    const width = Math.max((maximum - minimum) / 12, 1e-6);
    const lower = minimum + index * width;
    const upper = index === 11 ? maximum + 1e-9 : lower + width;
    return {
      bin: formatPercentagePoints(lower, { signed: false }),
      gain: lower,
      count: placeboDistribution.filter((value) => value >= lower && value < upper).length,
    };
  });
  const accepted = Boolean(result.accepted);
  return (
    <section className="performance-evaluation-group fusion-study">
      <div className="performance-section-heading"><FlaskConical size={19} /><div>
        <span>Controlled integration study</span>
        <h2>Does news add directional information?</h2>
        <p>{isControlled ? "Retrospective evidence, 2019–2025. " : ""}All arms use the same {benchmarkRows} target-complete 2025 rows. Missing news has zero numerical influence.</p>
      </div></div>
      <article className="performance-panel performance-panel-wide">
         <header><div><span>Retrospective ablation</span><ChartTitle guide="fusionArms">Stock signal versus news integration</ChartTitle></div><Scale size={19} /></header>
        <div className="performance-chart">
          {hasNumericValues(data, ["balanced_accuracy", "macro_f1"]) ? (
            <ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 14, right: 12, left: 72, bottom: 48 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="arm" tick={{ fontSize: 10 }} label={xAxisLabel("Evaluation arm")} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} label={yAxisLabel("Score")} />
              <Tooltip content={<EvaluationTooltip />} /><Legend />
              <Bar dataKey="balanced_accuracy" name="Balanced accuracy" fill={COLORS.blue} radius={[6, 6, 0, 0]} />
              <Bar dataKey="macro_f1" name="Macro-F1" fill={COLORS.green} radius={[6, 6, 0, 0]} />
            </BarChart></ResponsiveContainer>
          ) : <ChartEmpty message="No fusion arms contain usable directional scores." />}
        </div>
        <div className={`fusion-study-verdict ${accepted ? "accepted" : "experimental"}`}>
          <strong>{accepted ? "Incremental value established" : "Incremental value not established"}</strong>
          <span>{result.teaching_conclusion || (accepted ? "Every frozen all-horizon gate passed." : "Stock-led runtime remains active; this is retrospective research evidence.")}</span>
           <small>Balanced-accuracy gain: {formatPercentagePoints(baGain)}{baInterval ? ` · 95% moving-block interval ${formatPercentagePoints(baInterval.lower)} to ${formatPercentagePoints(baInterval.upper)}` : ""}{typeof result.holm_adjusted_clark_west_p_value === "number" ? ` · Holm-adjusted p=${result.holm_adjusted_clark_west_p_value.toFixed(3)}` : ""}</small>
        </div>
      </article>
      {isControlled && (
        <div className="performance-grid performance-grid-two fusion-diagnostics-grid">
          <article className="performance-panel">
            <header><div><span>Probability improvement</span><ChartTitle guide="fusionGain">Brier and directional gain</ChartTitle></div><TrendingUp size={19} /></header>
            <div className="fusion-interval-summary">
               <div><strong>{formatPercentagePoints(result.brier_improvement)}</strong><span>Brier improvement (pp)</span></div>
               <div><strong>{formatPercentagePoints(baGain)}</strong><span>Balanced-accuracy gain (pp)</span></div>
              <div><strong>{percent(placeboPercentile)}</strong><span>Worst-case placebo percentile</span></div>
            </div>
            <p className="performance-note">Positive Brier improvement means the combined probability was closer to the observed outcome. Intervals use moving blocks of trading sessions.</p>
          </article>
          <article className="performance-panel">
            <header><div><span>Rolling origin</span><ChartTitle guide="fusionRolling">Gain by evaluation year</ChartTitle></div><Activity size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(rolling, ["brier_improvement", "balanced_accuracy_gain"]) ? (
                <ResponsiveContainer width="100%" height="100%"><LineChart data={rolling} margin={{ top: 12, right: 12, left: 72, bottom: 48 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="evaluation_year" label={xAxisLabel("Evaluation year")} /><YAxis tickFormatter={(value) => formatPercentagePointsAxis(value, { signed: false })} label={yAxisLabel("Improvement (pp)")} /><Tooltip content={<EvaluationTooltip />} /><Legend /><ReferenceLine y={0} stroke={COLORS.slate} />
                  <Line type="monotone" dataKey="brier_improvement" name="Brier improvement" stroke={COLORS.blue} strokeWidth={2} />
                  <Line type="monotone" dataKey="balanced_accuracy_gain" name="Balanced-accuracy gain" stroke={COLORS.green} strokeWidth={2} />
                </LineChart></ResponsiveContainer>
              ) : <ChartEmpty message="Rolling-origin evidence is unavailable." />}
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Calibration</span><ChartTitle guide="fusionCalibration">Predicted versus observed probability</ChartTitle></div><Gauge size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(calibration, ["stock_observed", "fusion_observed"]) ? (
                <ResponsiveContainer width="100%" height="100%"><LineChart data={calibration} margin={{ top: 12, right: 12, left: 72, bottom: 48 }}>
                  <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="bin" name="Probability bin" label={xAxisLabel("Predicted probability bin")} /><YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} label={yAxisLabel("Observed frequency")} /><Tooltip content={<EvaluationTooltip />} /><Legend />
                  <Line dataKey="ideal" name="Ideal calibration" stroke={COLORS.slate} strokeDasharray="4 4" dot={false} />
                  <Line dataKey="stock_observed" name="Stock only" stroke={COLORS.navy} strokeWidth={2} />
                  <Line dataKey="fusion_observed" name="Stock + news" stroke={COLORS.blue} strokeWidth={2} />
                </LineChart></ResponsiveContainer>
              ) : <ChartEmpty message="Calibration bins are unavailable." />}
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Placebo test</span><ChartTitle guide="placebo">Within-company/month shuffled news</ChartTitle></div><FlaskConical size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(placeboBins, ["count"]) ? (
                <ResponsiveContainer width="100%" height="100%"><BarChart data={placeboBins} margin={{ top: 12, right: 12, left: 72, bottom: 48 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="bin" tick={{ fontSize: 9 }} label={xAxisLabel("Permuted Brier improvement (pp)")} /><YAxis domain={[0, "auto"]} label={yAxisLabel("Trial count")} /><Tooltip content={<EvaluationTooltip />} /><Bar dataKey="count" name="Permuted trials" fill={COLORS.violet} radius={[4, 4, 0, 0]} />
                </BarChart></ResponsiveContainer>
              ) : <ChartEmpty message="Placebo trials are unavailable." />}
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Company stability</span><ChartTitle guide="companyGain">Balanced-accuracy gain by ticker</ChartTitle></div><BarChart3 size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(companyData, ["gain"]) ? (
                <ResponsiveContainer width="100%" height="100%"><BarChart data={companyData} layout="vertical" margin={{ top: 8, right: 15, left: 84, bottom: 48 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" /><XAxis type="number" tickFormatter={(value) => formatPercentagePoints(value, { signed: false })} label={xAxisLabel("Balanced-accuracy gain (pp)")} /><YAxis dataKey="ticker" type="category" width={62} tick={{ fontSize: 10 }} label={yAxisLabel("Ticker")} /><ReferenceLine x={0} stroke={COLORS.slate} /><Tooltip content={<EvaluationTooltip />} /><Bar dataKey="gain" name="Balanced-accuracy gain" radius={[0, 5, 5, 0]}>{companyData.map((entry) => <Cell key={entry.ticker} fill={entry.gain >= 0 ? COLORS.green : COLORS.red} />)}</Bar>
                </BarChart></ResponsiveContainer>
              ) : <ChartEmpty message="Per-company gain values are unavailable." />}
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Error transitions</span><ChartTitle guide="transitions">When news corrected or harmed</ChartTitle></div><Scale size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(transitionData, ["value"]) ? (
                <ResponsiveContainer width="100%" height="100%"><BarChart data={transitionData} margin={{ top: 12, right: 12, left: 72, bottom: 48 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 9 }} label={xAxisLabel("Prediction transition")} /><YAxis domain={[0, "auto"]} label={yAxisLabel("Observations")} /><Tooltip content={<EvaluationTooltip />} /><Bar dataKey="value" name="Observations" fill={COLORS.navy} radius={[5, 5, 0, 0]} />
                </BarChart></ResponsiveContainer>
              ) : <ChartEmpty message="Error-transition counts are unavailable." />}
            </div>
          </article>
          <article className="performance-panel performance-panel-wide">
            <header><div><span>Evidence coverage</span><ChartTitle guide="coverage">Performance conditioned on news availability</ChartTitle></div><Info size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(coverageData, ["stock", "fusion"]) ? (
                <ResponsiveContainer width="100%" height="100%"><BarChart data={coverageData} margin={{ top: 12, right: 12, left: 72, bottom: 48 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="coverage" label={xAxisLabel("News coverage")}/><YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} label={yAxisLabel("Accuracy")} /><Tooltip content={<EvaluationTooltip />} /><Legend /><Bar dataKey="stock" name="Stock only" fill={COLORS.slate} radius={[5, 5, 0, 0]} /><Bar dataKey="fusion" name="Stock + news" fill={COLORS.blue} radius={[5, 5, 0, 0]} />
                </BarChart></ResponsiveContainer>
              ) : <ChartEmpty message="This evaluation artifact did not retain row-level news-coverage groups." />}
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Prediction confidence</span><ChartTitle guide="confidence">Accuracy by stock-confidence bucket</ChartTitle></div><Gauge size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(confidenceData, ["stock", "fusion"]) ? (
                <ResponsiveContainer width="100%" height="100%"><LineChart data={confidenceData} margin={{ top: 12, right: 12, left: 72, bottom: 48 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="confidence" label={xAxisLabel("Stock-confidence bucket")} /><YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} label={yAxisLabel("Accuracy")} /><Tooltip content={<EvaluationTooltip />} /><Legend /><Line dataKey="stock" name="Stock only" stroke={COLORS.slate} strokeWidth={2} /><Line dataKey="fusion" name="Stock + news" stroke={COLORS.blue} strokeWidth={2} />
                </LineChart></ResponsiveContainer>
              ) : <ChartEmpty message="This evaluation artifact did not retain row-level stock-confidence buckets." />}
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Evidence reliability</span><ChartTitle guide="reliability">Accuracy by news-reliability bucket</ChartTitle></div><Info size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(reliabilityData, ["stock", "fusion"]) ? (
                <ResponsiveContainer width="100%" height="100%"><LineChart data={reliabilityData} margin={{ top: 12, right: 12, left: 72, bottom: 48 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="reliability" label={xAxisLabel("News-reliability bucket")} /><YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} label={yAxisLabel("Accuracy")} /><Tooltip content={<EvaluationTooltip />} /><Legend /><Line dataKey="stock" name="Stock only" stroke={COLORS.slate} strokeWidth={2} /><Line dataKey="fusion" name="Stock + news" stroke={COLORS.blue} strokeWidth={2} />
                </LineChart></ResponsiveContainer>
              ) : <ChartEmpty message="This evaluation artifact did not retain row-level news-reliability buckets." />}
            </div>
          </article>
          {coefficientData.length > 0 && <article className="performance-panel performance-panel-wide">
            <header><div><span>Development stability</span><ChartTitle guide="corrections">Largest news corrections across purged folds</ChartTitle></div><Scale size={19} /></header>
            <div className="performance-chart compact">
              {hasNumericValues(coefficientData, ["coefficient"]) ? (
                <ResponsiveContainer width="100%" height="100%"><BarChart data={coefficientData} layout="vertical" margin={{ top: 8, right: 18, left: 72, bottom: 48 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" /><XAxis type="number" label={xAxisLabel("Mean standardized coefficient")} /><YAxis dataKey="feature" type="category" width={145} tick={{ fontSize: 10 }} label={yAxisLabel("News feature")} /><ReferenceLine x={0} stroke={COLORS.slate} /><Tooltip content={<EvaluationTooltip />} /><Bar dataKey="coefficient" name="Mean standardized coefficient" radius={[0, 5, 5, 0]}>{coefficientData.map((entry) => <Cell key={entry.feature} fill={entry.coefficient >= 0 ? COLORS.green : COLORS.red} />)}</Bar>
                </BarChart></ResponsiveContainer>
              ) : <ChartEmpty message="News-coefficient stability is unavailable." />}
            </div>
            <p className="performance-note">Only the eight largest standardized news corrections are shown. Hover to inspect fold dispersion and sign consistency; instability weakens the case for incremental news value.</p>
          </article>}
        </div>
      )}
    </section>
  );
}

type FusionHorizonResult = NonNullable<
  NonNullable<ModelPerformance["fusion_ablation"]>["horizons"][string]
>;

function FusionCompanyTable({
  result,
  horizon,
}: {
  result: FusionHorizonResult;
  horizon: Horizon;
}) {
  const rows = Object.entries(result.per_company || {})
    .map(([ticker, value]) => ({
      ...(value as FusionCompanyDiagnostic),
      ticker: value.ticker || ticker,
      company_name: value.company_name || value.ticker || ticker,
    }))
    .sort((left, right) => (left.company_name || left.ticker || "").localeCompare(right.company_name || right.ticker || ""));

  const metricBadge = (value: number | null | undefined, label: string) => {
    const normalizedScore =
      typeof value === "number" && Number.isFinite(value)
        ? Math.max(0, Math.min(1, (value - 0.5) / 0.5))
        : null;
    const metricStyle = {
      "--metric-hue":
        normalizedScore === null
          ? "210deg"
          : `${Math.round(normalizedScore * 120)}deg`,
    } as React.CSSProperties;
    return (
      <span
        className="metric-heat metric-heat-gradient"
        style={metricStyle}
        aria-label={`${label}: ${percent(value)}`}
        title={`${label}: ${percent(value)}`}
      >
        {percent(value)}
      </span>
    );
  };

  return (
    <section className="performance-table-section fusion-company-table-section">
      <div className="performance-section-heading">
        <BarChart3 size={19} />
        <div>
          <span>Retrospective</span>
          <h2>Fusion performance by company</h2>
          <p>
            Compare the stock-only baseline with the news-integrated result for
            each company at the selected {horizon}-trading-day horizon. Small
            company samples are diagnostic, not a ranking.
          </p>
          <div className="metric-gradient-legend" aria-label="Balanced accuracy colour scale">
            <span>Lower</span>
            <i aria-hidden="true" />
            <span>Higher</span>
          </div>
        </div>
      </div>
      {rows.length ? (
        <div className="performance-table-wrap">
          <table className="performance-table fusion-company-table">
            <thead>
              <tr>
                <th>Company</th>
                <th>Rows</th>
                <th>Stock-only BA</th>
                <th>Fusion BA</th>
                <th>BA gain (pp)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const gainClass =
                  row.gain > 0 ? "positive" : row.gain < 0 ? "negative" : "neutral";
                const ticker = row.ticker || "Unavailable";
                const companyName = row.company_name || ticker;
                return (
                  <tr
                    key={ticker}
                    title={`${companyName} (${ticker}) - ${horizon}-trading-day horizon - Retrospective`}
                  >
                    <td>
                      <strong>{companyName}</strong>
                      <span>{ticker}</span>
                    </td>
                    <td>{row.rows.toLocaleString()}</td>
                    <td>{metricBadge(row.stock_balanced_accuracy, "Stock-only balanced accuracy")}</td>
                    <td>{metricBadge(row.fusion_balanced_accuracy, "Fusion balanced accuracy")}</td>
                    <td>
                      <span
                        className={`fusion-gain-badge ${gainClass}`}
                        title={`Balanced-accuracy gain: ${formatPercentagePoints(row.gain)}`}
                      >
                        {formatPercentagePoints(row.gain)}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <ChartEmpty message="Per-company fusion metrics are not available in this evaluation artifact." />
      )}
      <p className="performance-note fusion-company-table-note">
        BA gain is Fusion minus Stock-only, shown in percentage points. Positive
        values mean the news correction improved balanced accuracy for that company; the artifact
        does not retain per-company Macro-F1 or MCC for this study.
      </p>
    </section>
  );
}

function fusionArmMetrics(
  result: FusionHorizonResult | undefined,
  preferredKeys: string[],
) {
  for (const key of preferredKeys) {
    const metrics = result?.arms?.[key]?.metrics;
    if (metrics) return metrics;
  }
  return undefined;
}

function FusionPerformanceView({
  performance,
  horizonValue,
  setHorizon,
}: {
  performance: ModelPerformance;
  horizonValue: Horizon;
  setHorizon: (value: Horizon) => void;
}) {
  const fusion = performance.fusion_ablation;
  const horizonRecords = HORIZONS.map((days) => ({
    days,
    result: fusion?.horizons?.[String(days)],
  }));
  const horizonData = horizonRecords
    .map(({ days, result }) => {
      const stock = fusionArmMetrics(result, ["stock_only_meta", "stock_only"]);
      const combined = fusionArmMetrics(result, ["compact_full", "full_fusion", "stock_plus_news"]);
      return {
        horizon: `${days}-day`,
        stock_ba: stock?.balanced_accuracy,
        fusion_ba: combined?.balanced_accuracy,
        stock_brier: stock?.brier_score,
        fusion_brier: combined?.brier_score,
        brier_gain: result?.brier_improvement,
        ba_gain: result?.balanced_accuracy_gain,
        rows: result?.benchmark_rows,
        news_rows: result?.news_present_rows,
      };
    })
    .filter((row) => hasNumericValues([row], ["stock_ba", "fusion_ba", "brier_gain"]));
  const selected = fusion?.horizons?.[String(horizonValue)];
  const stockMetrics = fusionArmMetrics(selected, ["stock_only_meta", "stock_only"]);
  const combinedMetrics = fusionArmMetrics(selected, ["compact_full", "full_fusion", "stock_plus_news"]);
  const selectedMetricData = [
    { metric: "Balanced accuracy", stock: stockMetrics?.balanced_accuracy, fusion: combinedMetrics?.balanced_accuracy },
    { metric: "Macro-F1", stock: stockMetrics?.macro_f1, fusion: combinedMetrics?.macro_f1 },
    { metric: "ECE", stock: stockMetrics?.ece, fusion: combinedMetrics?.ece },
  ];
  const rolling = selected?.rolling_origins || [];
  const brierAxis = improvementAxis(horizonData.map((row) => row.brier_gain));
  const rollingImprovementAxis = improvementAxis(
    rolling.flatMap((row) => [row.brier_improvement, row.balanced_accuracy_gain]),
  );
  const hasFusionEvidence = Boolean(fusion && (horizonData.length || selected));

  return (
    <>
      <div
        className="performance-horizon-switcher"
        role="group"
        aria-label="Fusion prediction horizon"
      >
        {HORIZONS.map((value) => (
          <button
            className={value === horizonValue ? "active" : ""}
            type="button"
            onClick={() => setHorizon(value)}
            key={value}
          >
            <strong>
              {value} trading {value === 1 ? "day" : "days"}
            </strong>
            <span>
              {value === 1
                ? "Near term"
                : value === 5
                  ? "About one market week"
                  : "About two market weeks"}
            </span>
          </button>
        ))}
      </div>
      {!hasFusionEvidence ? (
        <section className="performance-panel fusion-performance-empty">
          <div className="performance-section-heading">
            <FlaskConical size={19} />
            <div>
              <span>Fusion</span>
              <h2>Fusion diagnostics are not available</h2>
            </div>
          </div>
          <ChartEmpty message="The API did not expose a provenance-matched fusion evaluation for the selected stock run." />
        </section>
      ) : (
        <>
          <section className="sentiment-production-hero fusion-performance-hero">
            <div>
              <span className="panel-kicker">Fusion model performance</span>
              <h2>Stock Trend + News Evidence</h2>
              <p>
                Compare the capacity-matched stock baseline with the incremental
                news correction without mixing development and retrospective evidence.
              </p>
            </div>
          </section>
          <div className="performance-scoreboard fusion-scoreboard">
            <div>
              <span>Selected horizon</span>
              <strong>{horizonValue} days</strong>
              <small>{selected?.benchmark_rows?.toLocaleString() || "Unavailable"} target-complete rows</small>
            </div>
            <div>
              <span>News-present rows</span>
              <strong>{selected?.news_present_rows?.toLocaleString() || "0"}</strong>
              <small>Rows with eligible news evidence</small>
            </div>
            <div>
              <span>Balanced-accuracy gain (pp)</span>
              <strong>{formatPercentagePoints(selected?.balanced_accuracy_gain)}</strong>
              <small>Fusion minus stock baseline (pp)</small>
            </div>
            <div>
              <span>Brier improvement (pp)</span>
              <strong>{formatPercentagePoints(selected?.brier_improvement)}</strong>
              <small>Positive means lower probability error (pp)</small>
            </div>
          </div>
          <section className="performance-evaluation-group fusion-overview-group">
            <div className="performance-section-heading">
              <Activity size={19} />
              <div>
                <span>Retrospective evidence</span>
                <h2>Fusion performance at a glance</h2>
                <p>Hover each series to see the horizon, rows and exact stock-versus-fusion metric.</p>
              </div>
            </div>
            <div className="performance-grid">
              <article className="performance-panel performance-panel-wide">
                <header><div><span>All horizons</span><ChartTitle guide="fusionArms">Balanced accuracy: stock versus fusion</ChartTitle></div><TrendingUp size={19} /></header>
                <div className="performance-chart">
                  {hasNumericValues(horizonData, ["stock_ba", "fusion_ba"]) ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={horizonData} margin={{ top: 14, right: 12, left: 72, bottom: 48 }}>
                        <CartesianGrid vertical={false} strokeDasharray="3 3" />
                        <XAxis dataKey="horizon" label={xAxisLabel("Prediction horizon")} />
                        <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} label={yAxisLabel("Balanced accuracy")} />
                        <Tooltip content={<EvaluationTooltip />} /><Legend />
                        <Bar dataKey="stock_ba" name="Stock only" fill={COLORS.slate} radius={[5, 5, 0, 0]} />
                        <Bar dataKey="fusion_ba" name="Fusion" fill={COLORS.blue} radius={[5, 5, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : <ChartEmpty message="Balanced-accuracy values are unavailable for the fusion horizons." />}
                </div>
                <p>Balanced accuracy gives Up and Down equal importance; bars are retrospective, not prospective evidence.</p>
              </article>
              <article className="performance-panel">
                <header><div><span>All horizons</span><ChartTitle guide="fusionGain">Brier improvement</ChartTitle></div><Gauge size={19} /></header>
                <div className="performance-chart">
                  {hasNumericValues(horizonData, ["brier_gain"]) ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={horizonData} margin={{ top: 14, right: 12, left: 72, bottom: 48 }}>
                        <CartesianGrid vertical={false} strokeDasharray="3 3" />
                        <XAxis dataKey="horizon" label={xAxisLabel("Prediction horizon")} />
                        <YAxis domain={brierAxis.domain} ticks={brierAxis.ticks} tickFormatter={(value) => formatPercentagePointsAxis(value, { signed: true })} label={yAxisLabel("Brier improvement (pp)")} />
                        <Tooltip content={<EvaluationTooltip />} /><ReferenceLine y={0} stroke={COLORS.slate} />
                        <Bar dataKey="brier_gain" name="Brier improvement" radius={[5, 5, 0, 0]}>
                          {horizonData.map((row) => <Cell key={row.horizon} fill={(row.brier_gain || 0) >= 0 ? COLORS.green : COLORS.red} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : <ChartEmpty message="Brier improvement is unavailable for the fusion horizons." />}
                </div>
                <p>Higher is better: a positive value means fusion reduced probability error.</p>
              </article>
              <article className="performance-panel">
                <header><div><span>Selected horizon</span><ChartTitle guide="fusionArms">Metric comparison</ChartTitle></div><Scale size={19} /></header>
                <div className="performance-chart">
                  {hasNumericValues(selectedMetricData, ["stock", "fusion"]) ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={selectedMetricData} margin={{ top: 14, right: 12, left: 72, bottom: 48 }}>
                        <CartesianGrid vertical={false} strokeDasharray="3 3" />
                        <XAxis dataKey="metric" tick={{ fontSize: 10 }} label={xAxisLabel("Metric")} />
                        <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} label={yAxisLabel("Score")} />
                        <Tooltip content={<EvaluationTooltip />} /><Legend />
                        <Bar dataKey="stock" name="Stock only" fill={COLORS.slate} radius={[5, 5, 0, 0]} />
                        <Bar dataKey="fusion" name="Fusion" fill={COLORS.blue} radius={[5, 5, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : <ChartEmpty message="Selected-horizon fusion metrics are unavailable." />}
                </div>
                <p>ECE is calibration error and lower is better; the same chart keeps it visible beside directional metrics.</p>
              </article>
              <article className="performance-panel performance-panel-wide">
                <header><div><span>Rolling origins</span><ChartTitle guide="fusionRolling">Gain by retrospective evaluation year</ChartTitle></div><Activity size={19} /></header>
                <div className="performance-chart">
                  {hasNumericValues(rolling, ["brier_improvement", "balanced_accuracy_gain"]) ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={rolling} margin={{ top: 14, right: 12, left: 72, bottom: 48 }}>
                        <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="evaluation_year" label={xAxisLabel("Evaluation year")} /><YAxis domain={rollingImprovementAxis.domain} ticks={rollingImprovementAxis.ticks} tickFormatter={(value) => formatPercentagePointsAxis(value, { signed: true })} label={yAxisLabel("Improvement (pp)")} /><Tooltip content={<EvaluationTooltip />} /><Legend /><ReferenceLine y={0} stroke={COLORS.slate} />
                        <Line dataKey="brier_improvement" name="Brier improvement" stroke={COLORS.blue} strokeWidth={2.4} dot={{ r: 3 }} />
                        <Line dataKey="balanced_accuracy_gain" name="Balanced-accuracy gain" stroke={COLORS.green} strokeWidth={2.4} dot={{ r: 3 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : <ChartEmpty message="Rolling-origin gains are unavailable for the selected horizon." />}
                </div>
                <p>Rolling-origin results show whether the incremental signal repeats across different market years.</p>
              </article>
            </div>
          </section>
          <FusionAblationPanel performance={performance} horizon={horizonValue} />
          {selected && <FusionCompanyTable result={selected} horizon={horizonValue} />}
        </>
      )}
    </>
  );
}

function StockPerformanceView({
  performance,
  horizonValue,
  setHorizon,
}: {
  performance: ModelPerformance;
  horizonValue: Horizon;
  setHorizon: (value: Horizon) => void;
}) {
  const horizon = performance.stock.horizons.find(
    (item) => item.horizon_days === horizonValue,
  );
  if (!horizon)
    return (
      <ErrorState message="The selected horizon has no recorded model evidence." />
    );
  return (
    <>
      <div
        className="performance-horizon-switcher"
        role="group"
        aria-label="Prediction horizon"
      >
        {HORIZONS.map((value) => (
          <button
            className={value === horizonValue ? "active" : ""}
            type="button"
            onClick={() => setHorizon(value)}
            key={value}
          >
            <strong>
              {value} trading {value === 1 ? "day" : "days"}
            </strong>
            <span>
              {value === 1
                ? "Near term"
                : value === 5
                  ? "About one market week"
                  : "About two market weeks"}
            </span>
          </button>
        ))}
      </div>
      <ProtocolRail performance={performance} />
      <section className="performance-ledger-hero">
        <div>
          <span className="panel-kicker">
            Champion record - {horizonValue}-day
          </span>
          <h2>{plain(horizon.model_name)}</h2>
          <p>Frozen model evidence from the active stock registry.</p>
        </div>
      </section>
      <MetricScoreboard horizon={horizon} />
      {horizon.observed_bootstrap_gain_ci && (
        <div className="notice performance-ci-note">
          <strong>Baseline-uplift uncertainty:</strong>{" "}
          {formatPercentagePoints(horizon.observed_bootstrap_gain_ci.lower)} to{" "}
          {formatPercentagePoints(horizon.observed_bootstrap_gain_ci.upper)} 95% date-block
          interval (median {formatPercentagePoints(horizon.observed_bootstrap_gain_ci.median)}).
          {horizon.observed_bootstrap_gain_ci.repetitions
            ? ` Based on ${horizon.observed_bootstrap_gain_ci.repetitions.toLocaleString()} resamples`
            : ""}
          {horizon.observed_bootstrap_gain_ci.block_size
            ? ` with ${horizon.observed_bootstrap_gain_ci.block_size}-session blocks.`
            : "."}
        </div>
      )}
      <section className="performance-evaluation-group development">
        <div className="performance-section-heading">
          <Activity size={19} />
          <div>
            <span>Development</span>
            <h2>Model selection and time-fold validation</h2>
            <p>
              Evidence used to compare candidates and inspect stability before
              the retrospective test.
            </p>
          </div>
        </div>
        <div className="performance-grid">
          <LearningCurvePanel horizon={horizon} />
          <StabilityPanel horizon={horizon} />
          <SeedPanel horizon={horizon} />
          <CandidatesPanel horizons={performance.stock.horizons} />
        </div>
      </section>
      <section className="performance-evaluation-group retrospective">
        <div className="performance-section-heading">
          <BarChart3 size={19} />
          <div>
            <span>Retrospective</span>
            <h2>Frozen historical evaluation</h2>
            <p>
              Later data used to measure baseline uplift, probability
              reliability and direction errors.
            </p>
          </div>
        </div>
        <div className="performance-grid">
          <BaselinePanel horizon={horizon} />
          <CalibrationPanel horizon={horizon} />
          <ConfusionPanel horizon={horizon} />
          <ThresholdPanel horizon={horizon} />
          <ImportancePanel horizon={horizon} />
          <DiscriminationPanel horizon={horizon} />
        </div>
      </section>
      <section className="performance-table-section">
        <div className="performance-section-heading">
          <BarChart3 size={19} />
          <div>
            <span>Retrospective</span>
            <h2>Stock performance by company</h2>
            <p>
              Hover or focus a row to keep the company, ticker, horizon and
              evaluation period together.
            </p>
            <div className="metric-gradient-legend" aria-label="Balanced accuracy colour scale">
              <span>Lower</span>
              <i aria-hidden="true" />
              <span>Higher</span>
            </div>
          </div>
        </div>
        <CompanyTable rows={horizon.per_company} horizon={horizonValue} />
      </section>
    </>
  );
}

function SentimentPerformanceView({
  performance,
}: {
  performance: ModelPerformance;
}) {
  const sentiment = performance.sentiment;
  const ensemble = sentiment.models.find((item) => item.key === "ensemble");
  const comparison = sentiment.models.map((item) => ({
    ...item,
    name: item.label,
  }));
  const classData = SENTIMENT_CLASS_ORDER.map((label) => ({
    label: label[0].toUpperCase() + label.slice(1),
    value: ensemble?.class_f1?.[label] ?? null,
  }));
  const yearlyCoverage = Object.entries(sentiment.dataset.year_counts).map(
    ([year, rows]) => ({ year, rows, value_format: "count" }),
  );
  const classCoverage = SENTIMENT_CLASS_ORDER.map((label) => ({
    label: label[0].toUpperCase() + label.slice(1),
    rows: sentiment.dataset.class_counts[label] ?? 0,
    value_format: "count",
  }));
  return (
    <>
      <section className="sentiment-production-hero">
        <div>
          <span className="panel-kicker">Active production configuration</span>
          <h2>Bursa + PhraseBank FinBERT ensemble</h2>
          <p>
            {percent(sentiment.weights.bursa, 0)} Bursa-adapted model -{" "}
            {percent(sentiment.weights.phrasebank, 0)} PhraseBank model -
            temperature{" "}
            {sentiment.ensemble_temperature?.toFixed(2) || "Unavailable"}
          </p>
        </div>
      </section>
      <div className="protocol-rail sentiment-period">
        <div>
          <b>Development and selection</b>
          <strong>2015-2024</strong>
          <span>
            Chronological Bursa evidence and PhraseBank retention checks.
          </span>
        </div>
        <i className="protocol-connector" aria-hidden="true" />
        <div>
          <b>Retrospective test</b>
          <strong>2025</strong>
          <span>
            Locked headline performance for the current 2015-2025 scope.
          </span>
        </div>
        <i />
        <div>
          <b>Runtime use</b>
          <strong>
            {percent(sentiment.weights.bursa, 0)} /{" "}
            {percent(sentiment.weights.phrasebank, 0)}
          </strong>
          <span>
            Weighted probabilities are calibrated before producing article tone.
          </span>
        </div>
      </div>
      <div className="performance-grid">
        <article className="performance-panel sentiment-retrospective">
          <header>
            <div>
              <span>Retrospective</span>
              <ChartTitle guide="sentimentModels">Model comparison</ChartTitle>
            </div>
            <Scale size={19} />
          </header>
          <div className="performance-chart">
            {hasNumericValues(comparison, ["balanced_accuracy", "macro_f1"]) ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={comparison}
                  margin={{ top: 14, right: 12, left: 72, bottom: 48 }}
                >
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} label={xAxisLabel("Sentiment model")} />
                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                    label={yAxisLabel("Score")}
                  />
                  <Tooltip content={<EvaluationTooltip />} />
                  <Legend />
                  <Bar
                    dataKey="balanced_accuracy"
                    name="Balanced accuracy"
                    fill={COLORS.blue}
                    radius={[5, 5, 0, 0]}
                  />
                  <Bar
                    dataKey="macro_f1"
                    name="Macro-F1"
                    fill={COLORS.green}
                    radius={[5, 5, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : <ChartEmpty message="Sentiment model scores are unavailable for the active benchmark." />}
          </div>
          <p>
            All three approaches are compared on the same Retrospective rows.
          </p>
        </article>
        <article className="performance-panel sentiment-retrospective">
          <header>
            <div>
              <span>Retrospective</span>
              <ChartTitle guide="sentimentClassF1">Ensemble class F1</ChartTitle>
            </div>
            <BarChart3 size={19} />
          </header>
          <div className="performance-chart">
            {hasNumericValues(classData, ["value"]) ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={classData}
                  layout="vertical"
                  margin={{ top: 14, right: 16, left: 72, bottom: 48 }}
                >
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    domain={[0, 1]}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                    label={xAxisLabel("F1 score")}
                  />
                  <YAxis type="category" dataKey="label" width={60} label={yAxisLabel("Sentiment class")} />
                  <Tooltip content={<EvaluationTooltip />} />
                  <Bar dataKey="value" name="F1 score" radius={[0, 6, 6, 0]}>
                    {classData.map((item) => (
                      <Cell
                        key={item.label}
                        fill={sentimentClassColor(item.label)}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <ChartEmpty message="Per-class F1 values are unavailable for the ensemble." />}
          </div>
          <p>
            Neutral remains visible as its own class; it is not hidden inside a
            single overall accuracy number.
          </p>
        </article>
        <article className="performance-panel sentiment-development">
          <header>
            <div>
              <span>Development</span>
              <ChartTitle guide="sentimentCoverage">Bursa coverage by year</ChartTitle>
            </div>
            <BarChart3 size={19} />
          </header>
          <div className="performance-chart">
            {hasNumericValues(yearlyCoverage, ["rows"]) ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={yearlyCoverage}
                  margin={{ top: 14, right: 8, left: 72, bottom: 48 }}
                >
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="year" tick={{ fontSize: 10 }} label={xAxisLabel("Publication year")} />
                  <YAxis domain={[0, "auto"]} allowDecimals={false} label={yAxisLabel("Eligible articles")} />
                  <Tooltip content={<EvaluationTooltip />} />
                  <Bar
                    dataKey="rows"
                    name="Eligible articles"
                    fill={COLORS.blue}
                    radius={[5, 5, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : <ChartEmpty message="Year-level Bursa coverage is unavailable." />}
          </div>
          <p>
            {sentiment.dataset.eligible_rows?.toLocaleString() || "Unavailable"}{" "}
            eligible Bursa articles are retained across the locked production
            scope.
          </p>
        </article>
        <article className="performance-panel sentiment-development">
          <header>
            <div>
              <span>Development</span>
              <ChartTitle guide="labelDistribution">Label distribution</ChartTitle>
            </div>
            <Scale size={19} />
          </header>
          <div className="performance-chart">
            {hasNumericValues(classCoverage, ["rows"]) ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={classCoverage}
                  layout="vertical"
                  margin={{ top: 14, right: 16, left: 86, bottom: 48 }}
                >
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, "auto"]} allowDecimals={false} label={xAxisLabel("Article count")} />
                  <YAxis type="category" dataKey="label" width={62} label={yAxisLabel("Sentiment class")} />
                  <Tooltip content={<EvaluationTooltip />} />
                  <Bar
                    dataKey="rows"
                    name="Eligible articles"
                    radius={[0, 6, 6, 0]}
                  >
                    {classCoverage.map((item) => (
                      <Cell
                        key={item.label}
                        fill={sentimentClassColor(item.label)}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <ChartEmpty message="Sentiment label counts are unavailable." />}
          </div>
          <p>
            Class imbalance is shown explicitly because it can make overall
            accuracy look stronger than minority-class performance.
          </p>
        </article>
      </div>
      <div className="performance-scoreboard sentiment-scoreboard">
        <div>
          <span>Accuracy</span>
          <strong>{percent(ensemble?.accuracy)}</strong>
          <small>All correct labels</small>
        </div>
        <div>
          <span>Balanced accuracy</span>
          <strong>{percent(ensemble?.balanced_accuracy)}</strong>
          <small>Equal class importance</small>
        </div>
        <div>
          <span>Macro-F1</span>
          <strong>{percent(ensemble?.macro_f1)}</strong>
          <small>Average class consistency</small>
        </div>
        <div>
          <span>Log loss</span>
          <strong>{ensemble?.log_loss?.toFixed(3) || "Unavailable"}</strong>
          <small>Lower is better</small>
        </div>
        <div>
          <span>ECE</span>
          <strong>{percent(ensemble?.ece)}</strong>
          <small>Calibration gap</small>
        </div>
      </div>
      <section className="performance-table-section">
        <div className="performance-section-heading">
          <BarChart3 size={19} />
          <div>
            <span>Retrospective</span>
            <h2>Sentiment performance by company</h2>
            <p>
              Small company samples should be interpreted as diagnostic
              evidence, not a stable ranking.
            </p>
            <div className="metric-gradient-legend" aria-label="Balanced accuracy colour scale">
              <span>Lower</span>
              <i aria-hidden="true" />
              <span>Higher</span>
            </div>
          </div>
        </div>
        <CompanyTable rows={sentiment.per_company} sentiment />
      </section>
    </>
  );
}

export function ModelPerformancePage() {
  const [params, setParams] = useSearchParams();
  const initialHorizon = Number(params.get("horizon"));
  const [horizon, setHorizonState] = useState<Horizon>(
    HORIZONS.includes(initialHorizon as Horizon)
      ? (initialHorizon as Horizon)
      : 1,
  );
  const [view, setView] = useState<"stock" | "sentiment" | "fusion">(
    params.get("view") === "sentiment"
      ? "sentiment"
      : params.get("view") === "fusion"
        ? "fusion"
        : "stock",
  );
  const [runId, setRunId] = useState(params.get("run") || "");
  const query = useQuery({
    queryKey: ["model-performance", runId],
    queryFn: ({ signal }) => getModelPerformance(signal, runId || undefined),
    staleTime: 300000,
  });
  const setHorizon = (value: Horizon) => {
    setHorizonState(value);
    setParams({ view, horizon: String(value) });
  };
  const setActiveView = (value: "stock" | "sentiment" | "fusion") => {
    setView(value);
    setParams({ view: value, horizon: String(horizon) });
  };
  const generatedLabel = useMemo(
    () =>
      query.data?.stock.artifact_generated_at
        ? new Date(query.data.stock.artifact_generated_at).toLocaleString()
        : "Detailed artifact unavailable",
    [query.data],
  );
  return (
    <div className="model-performance-page">
      <PageHeader
        eyebrow="Learn"
        title="Model Performance"
        description="Inspect how StockSense was evaluated before interpreting a live prediction. Historical scores, time folds and acceptance checks remain visibly separated."
      />
      <div className="performance-view-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={view === "stock"}
          className={view === "stock" ? "active" : ""}
          onClick={() => setActiveView("stock")}
        >
          <TrendingUp size={17} />
          Stock Trend
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={view === "sentiment"}
          className={view === "sentiment" ? "active" : ""}
          onClick={() => setActiveView("sentiment")}
        >
          <FlaskConical size={17} />
          News Sentiment
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={view === "fusion"}
          className={view === "fusion" ? "active fusion-tab" : "fusion-tab"}
          onClick={() => setActiveView("fusion")}
        >
          <Scale size={17} />
          Fusion
        </button>
        <span>Evidence generated {generatedLabel}</span>
        <label className="performance-run-selector">
          <span>Recorded run</span>
          <select value={runId} onChange={(event) => {
            const value = event.target.value;
            setRunId(value);
            const next = new URLSearchParams(params);
            value ? next.set("run", value) : next.delete("run");
            setParams(next);
          }}>
            <option value="">Active production</option>
            {(query.data?.available_runs || []).filter((run) => !run.active).map((run) => (
              <option key={run.run_id} value={run.run_id} disabled={!run.artifact_available}>
                {run.run_id} · {run.label}{run.artifact_available ? "" : " · generate diagnostics first"}
              </option>
            ))}
          </select>
        </label>
      </div>
      {!query.isLoading && query.data?.stock.artifact_warning && (
        <div className="notice warning performance-artifact-warning">
          {query.data.stock.artifact_warning}
        </div>
      )}
      {query.isLoading ? (
        <LoadingState label="Loading model evidence..." />
      ) : query.isError || !query.data ? (
        <ErrorState
          message={query.error?.message || "Model performance is unavailable."}
        />
      ) : (
        <>
          {view === "stock" ? (
            <StockPerformanceView
              performance={query.data}
              horizonValue={horizon}
              setHorizon={setHorizon}
            />
          ) : view === "sentiment" ? (
            <SentimentPerformanceView performance={query.data} />
          ) : (
            <FusionPerformanceView
              performance={query.data}
              horizonValue={horizon}
              setHorizon={setHorizon}
            />
          )}
          <div className="performance-disclosure">
            <Info size={16} />
            <span>{query.data.disclosure}</span>
          </div>
        </>
      )}
    </div>
  );
}
