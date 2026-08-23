import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  BarChart3,
  Check,
  FlaskConical,
  Gauge,
  Info,
  Scale,
  ShieldCheck,
  TrendingUp,
} from "lucide-react";
import { useMemo, useState } from "react";
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
} from "recharts";
import { getModelPerformance } from "../api/client";
import { ErrorState, LoadingState } from "../components/AsyncState";
import { PageHeader } from "../components/PageHeader";
import type {
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
const HORIZONS: Horizon[] = [1, 5, 10];

function percent(value: number | null | undefined, digits = 1) {
  return typeof value === "number"
    ? `${(value * 100).toFixed(digits)}%`
    : "Unavailable";
}

function plain(value: string | null | undefined) {
  return String(value || "Unavailable").replaceAll("_", " ");
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
              : percent(Number(entry.value))}
          </span>
        ))}
    </div>
  );
}

function MetricScoreboard({ horizon }: { horizon: StockPerformanceHorizon }) {
  const items = [
    [
      "Balanced accuracy",
      horizon.metrics.balanced_accuracy,
      "Equal weight for Up and Down",
    ],
    [
      "Macro-F1",
      horizon.metrics.macro_f1,
      "Consistency across both directions",
    ],
    ["MCC", horizon.metrics.mcc, "Correlation from Ã¢Ë†â€™1 to +1", false],
    ["ROC-AUC", horizon.metrics.roc_auc, "Probability ranking skill"],
    [
      "Brier score",
      horizon.metrics.brier_score,
      "Probability error Â· lower is better",
      false,
    ],
    ["ECE", horizon.metrics.ece, "Calibration gap Â· lower is better"],
    [
      "Baseline uplift",
      horizon.balanced_accuracy_gain,
      `Against ${plain(horizon.strongest_baseline)}`,
    ],
  ] as const;
  return (
    <div className="performance-scoreboard">
      {items.map(([label, value, note, percentage = true]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>
            {typeof value === "number"
              ? percentage
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
  const shadow = stock.shadow_period as {
    start_date?: string;
    end_date?: string;
  };
  return (
    <section className="performance-protocol" aria-labelledby="protocol-title">
      <div className="performance-section-heading">
        <Info size={19} />
        <div>
          <span>Evaluation protocol</span>
          <h2 id="protocol-title">Three periods, three different jobs</h2>
        </div>
      </div>
      <div className="protocol-rail">
        <div>
          <b>Develop</b>
          <strong>
            {stock.development_period.start?.slice(0, 4)}â€“
            {stock.development_period.end?.slice(0, 4)}
          </strong>
          <span>
            Train models and validate through purged chronological folds.
          </span>
        </div>
        <i />
        <div>
          <b>Retrospective test</b>
          <strong>{stock.retrospective_test_period.start?.slice(0, 4)}</strong>
          <span>
            Compare the frozen champion with baselines on later historical data.
          </span>
        </div>
        <i />
        <div>
          <b>Stock shadow</b>
          <strong>{shadow.start_date?.slice(0, 4) || "Unavailable"}</strong>
          <span>
            Monitor a later stock period without merging it into the
            retrospective score.
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
          <h3>Champion versus strongest baseline</h3>
        </div>
        <Scale size={19} />
      </header>
      <div className="performance-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 14, right: 12, left: -6, bottom: 2 }}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
            <YAxis
              domain={[0, 0.8]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
            />
            <Tooltip content={<EvaluationTooltip />} />
            <Legend />
            <Bar dataKey="Champion" fill={COLORS.blue} radius={[6, 6, 0, 0]} />
            <Bar dataKey="Baseline" fill={COLORS.slate} radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
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
          <h3>Learning curve</h3>
        </div>
        <TrendingUp size={19} />
      </header>
      <div className="performance-chart">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={data}
              margin={{ top: 14, right: 12, left: -5, bottom: 2 }}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="rows" tick={{ fontSize: 11 }} />
              <YAxis
                domain={[0.4, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
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
                stroke="none"
                fill="rgba(9,42,77,0.10)"
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
                stroke="none"
                fill="rgba(8,118,223,0.14)"
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
          <h3>Time-fold stability</h3>
        </div>
        <Activity size={19} />
      </header>
      <div className="performance-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 14, right: 12, left: -5, bottom: 2 }}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="fold" />
            <YAxis
              domain={[0.4, 0.7]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
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
          <h3>Probability reliability</h3>
        </div>
        <Gauge size={19} />
      </header>
      <div className="performance-chart">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 14, right: 12, left: -5, bottom: 2 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="predicted"
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
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
      <header><div><span>Retrospective</span><h3>Probability ranking: ROC and precision–recall</h3></div><Activity size={19} /></header>
      <div className="performance-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 14, right: 18, left: -4, bottom: 2 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip content={<EvaluationTooltip />} />
            <Legend />
            <Line data={roc} dataKey="reference" name="No-skill ROC" stroke={COLORS.slate} strokeDasharray="5 5" dot={false} />
            <Line data={roc} dataKey="y" name="ROC curve" stroke={COLORS.blue} strokeWidth={2.5} dot={false} />
            <Line data={precisionRecall} dataKey="y" name="Precision–recall curve" stroke={COLORS.green} strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p>ROC shows ranking across thresholds; precision–recall makes the trade-off between finding Up cases and false alarms visible.</p>
    </article>
  );
}

function ThresholdPanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const selected = horizon.threshold_curve.find((row) => row.selected)?.threshold;
  return (
    <article className="performance-panel">
      <header><div><span>Retrospective</span><h3>Decision-threshold sensitivity</h3></div><Gauge size={19} /></header>
      <div className="performance-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={horizon.threshold_curve} margin={{ top: 14, right: 12, left: -4, bottom: 2 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="threshold" tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <YAxis domain={[0.3, 0.8]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip content={<EvaluationTooltip />} /><Legend />
            {typeof selected === "number" && <ReferenceLine x={selected} stroke={COLORS.amber} label="Selected" />}
            <Line dataKey="balanced_accuracy" name="Balanced accuracy" stroke={COLORS.blue} dot={false} />
            <Line dataKey="up_recall" name="Up recall" stroke={COLORS.green} dot={false} />
            <Line dataKey="down_recall" name="Down recall" stroke={COLORS.red} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p>The selected cutoff is locked before the 2025 evaluation; this plot shows how the two direction recalls trade off around it.</p>
    </article>
  );
}

function ImportancePanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  const data = horizon.permutation_importance.slice(0, 12).map((row) => ({ ...row, label: plain(row.feature) })).reverse();
  return (
    <article className="performance-panel">
      <header><div><span>Retrospective diagnostic</span><h3>Permutation importance</h3></div><Scale size={19} /></header>
      <div className="performance-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 8, right: 12, left: 72, bottom: 2 }}>
            <CartesianGrid horizontal={false} strokeDasharray="3 3" />
            <XAxis type="number" tickFormatter={(v) => `${(v * 100).toFixed(1)} pp`} />
            <YAxis type="category" dataKey="label" width={125} tick={{ fontSize: 10 }} />
            <Tooltip content={<EvaluationTooltip />} />
            <Bar dataKey="importance_mean" name="Balanced-accuracy decrease" fill={COLORS.violet} radius={[0, 5, 5, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p>A larger decrease after shuffling means the fitted model depended more on that feature. It is not a causal claim.</p>
    </article>
  );
}

function SeedPanel({ horizon }: { horizon: StockPerformanceHorizon }) {
  return (
    <article className="performance-panel">
      <header><div><span>Development</span><h3>Fixed-manifest seed stability</h3></div><Activity size={19} /></header>
      <div className="performance-chart">
        {horizon.seed_stability.length ? <ResponsiveContainer width="100%" height="100%">
          <LineChart data={horizon.seed_stability} margin={{ top: 14, right: 12, left: -4, bottom: 2 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="seed" />
            <YAxis domain={[0.4, 0.7]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
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
          <h3>Direction errors</h3>
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
          <h3>All evaluated candidate families</h3>
        </div>
        <Scale size={19} />
      </header>
      <div
        className="performance-chart performance-candidate-chart"
        style={{ height: chartHeight }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            barCategoryGap="24%"
            margin={{ top: 8, right: 24, left: 52, bottom: 8 }}
          >
            <CartesianGrid horizontal={false} strokeDasharray="3 3" />
            <XAxis
              type="number"
              domain={[0.4, 0.65]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={145}
              tick={{ fontSize: 10 }}
            />
            <Tooltip content={<EvaluationTooltip />} />
            <Legend />
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
      </div>
      <p>
        Every family evaluated in the active stock-model run is shown. Each bar
        is that familyÃ¢â‚¬â„¢s best purged chronological development result for
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
      <table className="performance-table">
        <thead>
          <tr>
            <th>Company</th>
            <th>Rows</th>
            <th>Balanced accuracy</th>
            <th>Macro-F1</th>
            <th>MCC</th>
            {!sentiment && <th>Status</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.ticker}
              title={`${row.company_name || row.ticker} (${row.ticker})${horizon ? ` Â· ${horizon}-trading-day horizon Â· Retrospective` : " Â· 2025 sentiment retrospective"}`}
            >
              <td>
                <strong>{row.company_name || row.ticker}</strong>
                <span>{row.ticker}</span>
              </td>
              <td>{row.rows}</td>
              <td>
                <span
                  className="metric-heat"
                  style={
                    {
                      "--metric-strength": Math.max(
                        0,
                        Math.min(
                          1,
                          ((row.balanced_accuracy || 0.4) - 0.4) / 0.3,
                        ),
                      ),
                    } as React.CSSProperties
                  }
                >
                  {percent(row.balanced_accuracy)}
                </span>
              </td>
              <td>{percent(row.macro_f1)}</td>
              <td>
                {typeof row.mcc === "number"
                  ? row.mcc.toFixed(3)
                  : "Unavailable"}
              </td>
              {!sentiment && <td>{plain(row.status)}</td>}
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
  const isControlled = ["stocksense-fusion-evaluation-v2", "stocksense-fusion-evaluation-v3"].includes(performance.fusion_ablation?.schema_version || "");
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
  const coverageData = Object.entries(result.subgroups?.news_coverage || result.subgroups?.coverage || {}).map(([coverage, value]) => ({
    coverage,
    stock: value.stock_balanced_accuracy,
    fusion: value.fusion_balanced_accuracy,
    rows: value.rows,
  }));
  const stockConfidence = result.confidence_accuracy?.stock_only || [];
  const fusionConfidence = result.confidence_accuracy?.full_fusion || [];
  const confidenceData = stockConfidence.map((value, index) => ({
    confidence: `Q${value.bucket}`,
    stock: value.accuracy,
    fusion: fusionConfidence[index]?.accuracy,
    rows: value.rows,
  }));
  const reliabilityData = Object.entries(result.subgroups?.news_reliability || {}).map(([reliability, value]) => ({
    reliability,
    stock: value.stock_balanced_accuracy,
    fusion: value.fusion_balanced_accuracy,
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
  const fusionCalibration = result.calibration?.stock_plus_news?.curve || result.calibration?.full_fusion?.curve || [];
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
      bin: `${(lower * 100).toFixed(2)}%`,
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
        <header><div><span>Retrospective ablation</span><h3>Stock signal versus news integration</h3></div><Scale size={19} /></header>
        <div className="performance-chart">
          <ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{ top: 14, right: 12, left: -5, bottom: 2 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="arm" tick={{ fontSize: 10 }} />
            <YAxis domain={[0.3, 0.8]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip content={<EvaluationTooltip />} /><Legend />
            <Bar dataKey="balanced_accuracy" name="Balanced accuracy" fill={COLORS.blue} radius={[6, 6, 0, 0]} />
            <Bar dataKey="macro_f1" name="Macro-F1" fill={COLORS.green} radius={[6, 6, 0, 0]} />
          </BarChart></ResponsiveContainer>
        </div>
        <div className={`fusion-study-verdict ${accepted ? "accepted" : "experimental"}`}>
          <strong>{accepted ? "Incremental value established" : "Incremental value not established"}</strong>
          <span>{result.teaching_conclusion || (accepted ? "Every frozen all-horizon gate passed." : "Stock-led runtime remains active; this is retrospective research evidence.")}</span>
          <small>Balanced-accuracy gain: {percent(baGain)}{baInterval ? ` · 95% moving-block interval ${percent(baInterval.lower)} to ${percent(baInterval.upper)}` : ""}{typeof result.holm_adjusted_clark_west_p_value === "number" ? ` · Holm-adjusted p=${result.holm_adjusted_clark_west_p_value.toFixed(3)}` : ""}</small>
        </div>
      </article>
      {isControlled && (
        <div className="performance-grid performance-grid-two fusion-diagnostics-grid">
          <article className="performance-panel">
            <header><div><span>Probability improvement</span><h3>Brier and directional gain</h3></div><TrendingUp size={19} /></header>
            <div className="fusion-interval-summary">
              <div><strong>{percent(result.brier_improvement)}</strong><span>Brier improvement</span></div>
              <div><strong>{percent(baGain)}</strong><span>Balanced-accuracy gain</span></div>
              <div><strong>{percent(placeboPercentile)}</strong><span>Worst-case placebo percentile</span></div>
            </div>
            <p className="performance-note">Positive Brier improvement means the combined probability was closer to the observed outcome. Intervals use moving blocks of trading sessions.</p>
          </article>
          <article className="performance-panel">
            <header><div><span>Rolling origin</span><h3>Gain by evaluation year</h3></div><Activity size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><LineChart data={rolling} margin={{ top: 12, right: 12, left: 0, bottom: 2 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="evaluation_year" /><YAxis tickFormatter={(value) => `${(value * 100).toFixed(1)}%`} /><Tooltip content={<EvaluationTooltip />} /><Legend /><ReferenceLine y={0} stroke={COLORS.slate} />
                <Line type="monotone" dataKey="brier_improvement" name="Brier improvement" stroke={COLORS.blue} strokeWidth={2} />
                <Line type="monotone" dataKey="balanced_accuracy_gain" name="Balanced-accuracy gain" stroke={COLORS.green} strokeWidth={2} />
              </LineChart></ResponsiveContainer>
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Calibration</span><h3>Predicted versus observed probability</h3></div><Gauge size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><LineChart data={calibration} margin={{ top: 12, right: 12, left: 0, bottom: 2 }}>
                <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="bin" name="Probability bin" /><YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} /><Tooltip content={<EvaluationTooltip />} /><Legend />
                <Line dataKey="ideal" name="Ideal calibration" stroke={COLORS.slate} strokeDasharray="4 4" dot={false} />
                <Line dataKey="stock_observed" name="Stock only" stroke={COLORS.navy} strokeWidth={2} />
                <Line dataKey="fusion_observed" name="Stock + news" stroke={COLORS.blue} strokeWidth={2} />
              </LineChart></ResponsiveContainer>
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Placebo test</span><h3>Within-company/month shuffled news</h3></div><FlaskConical size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><BarChart data={placeboBins} margin={{ top: 12, right: 12, left: 0, bottom: 2 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="bin" tick={{ fontSize: 9 }} /><YAxis /><Tooltip content={<EvaluationTooltip />} /><Bar dataKey="count" name="Permuted trials" fill={COLORS.violet} radius={[4, 4, 0, 0]} />
              </BarChart></ResponsiveContainer>
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Company stability</span><h3>Balanced-accuracy gain by ticker</h3></div><BarChart3 size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><BarChart data={companyData} layout="vertical" margin={{ top: 8, right: 15, left: 18, bottom: 2 }}>
                <CartesianGrid horizontal={false} strokeDasharray="3 3" /><XAxis type="number" tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} /><YAxis dataKey="ticker" type="category" width={62} tick={{ fontSize: 10 }} /><ReferenceLine x={0} stroke={COLORS.slate} /><Tooltip content={<EvaluationTooltip />} /><Bar dataKey="gain" name="Balanced-accuracy gain" radius={[0, 5, 5, 0]}>{companyData.map((entry) => <Cell key={entry.ticker} fill={entry.gain >= 0 ? COLORS.green : COLORS.red} />)}</Bar>
              </BarChart></ResponsiveContainer>
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Error transitions</span><h3>When news corrected or harmed</h3></div><Scale size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><BarChart data={transitionData} margin={{ top: 12, right: 12, left: 0, bottom: 2 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 9 }} /><YAxis /><Tooltip content={<EvaluationTooltip />} /><Bar dataKey="value" name="Observations" fill={COLORS.navy} radius={[5, 5, 0, 0]} />
              </BarChart></ResponsiveContainer>
            </div>
          </article>
          <article className="performance-panel performance-panel-wide">
            <header><div><span>Evidence coverage</span><h3>Performance conditioned on news availability</h3></div><Info size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><BarChart data={coverageData} margin={{ top: 12, right: 12, left: 0, bottom: 2 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="coverage" /><YAxis domain={[0.3, 0.8]} tickFormatter={(value) => `${Math.round(value * 100)}%`} /><Tooltip content={<EvaluationTooltip />} /><Legend /><Bar dataKey="stock" name="Stock only" fill={COLORS.slate} radius={[5, 5, 0, 0]} /><Bar dataKey="fusion" name="Stock + news" fill={COLORS.blue} radius={[5, 5, 0, 0]} />
              </BarChart></ResponsiveContainer>
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Prediction confidence</span><h3>Accuracy by stock-confidence bucket</h3></div><Gauge size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><LineChart data={confidenceData} margin={{ top: 12, right: 12, left: 0, bottom: 2 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="confidence" /><YAxis domain={[0.3, 0.8]} tickFormatter={(value) => `${Math.round(value * 100)}%`} /><Tooltip content={<EvaluationTooltip />} /><Legend /><Line dataKey="stock" name="Stock only" stroke={COLORS.slate} strokeWidth={2} /><Line dataKey="fusion" name="Stock + news" stroke={COLORS.blue} strokeWidth={2} />
              </LineChart></ResponsiveContainer>
            </div>
          </article>
          <article className="performance-panel">
            <header><div><span>Evidence reliability</span><h3>Accuracy by news-reliability bucket</h3></div><Info size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><LineChart data={reliabilityData} margin={{ top: 12, right: 12, left: 0, bottom: 2 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="reliability" /><YAxis domain={[0.3, 0.8]} tickFormatter={(value) => `${Math.round(value * 100)}%`} /><Tooltip content={<EvaluationTooltip />} /><Legend /><Line dataKey="stock" name="Stock only" stroke={COLORS.slate} strokeWidth={2} /><Line dataKey="fusion" name="Stock + news" stroke={COLORS.blue} strokeWidth={2} />
              </LineChart></ResponsiveContainer>
            </div>
          </article>
          {coefficientData.length > 0 && <article className="performance-panel performance-panel-wide">
            <header><div><span>Development stability</span><h3>Largest news corrections across purged folds</h3></div><Scale size={19} /></header>
            <div className="performance-chart compact">
              <ResponsiveContainer width="100%" height="100%"><BarChart data={coefficientData} layout="vertical" margin={{ top: 8, right: 18, left: 42, bottom: 2 }}>
                <CartesianGrid horizontal={false} strokeDasharray="3 3" /><XAxis type="number" /><YAxis dataKey="feature" type="category" width={145} tick={{ fontSize: 10 }} /><ReferenceLine x={0} stroke={COLORS.slate} /><Tooltip content={<EvaluationTooltip />} /><Bar dataKey="coefficient" name="Mean standardized coefficient" radius={[0, 5, 5, 0]}>{coefficientData.map((entry) => <Cell key={entry.feature} fill={entry.coefficient >= 0 ? COLORS.green : COLORS.red} />)}</Bar>
              </BarChart></ResponsiveContainer>
            </div>
            <p className="performance-note">Only the eight largest standardized news corrections are shown. Hover to inspect fold dispersion and sign consistency; instability weakens the case for incremental news value.</p>
          </article>}
        </div>
      )}
    </section>
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
            Champion record Â· {horizonValue}-day
          </span>
          <h2>{plain(horizon.model_name)}</h2>
          <p>Frozen model evidence from the active stock registry.</p>
        </div>
        <div
          className={`performance-status ${horizon.status.toLowerCase() === "accepted" ? "accepted" : "experimental"}`}
        >
          {horizon.status.toLowerCase() === "accepted" ? (
            <Check size={17} />
          ) : (
            <FlaskConical size={17} />
          )}
          {plain(horizon.status)}
        </div>
      </section>
      <MetricScoreboard horizon={horizon} />
      {horizon.observed_bootstrap_gain_ci && (
        <div className="notice performance-ci-note">
          <strong>Baseline-uplift uncertainty:</strong>{" "}
          {percent(horizon.observed_bootstrap_gain_ci.lower)} to{" "}
          {percent(horizon.observed_bootstrap_gain_ci.upper)} 95% date-block
          interval (median {percent(horizon.observed_bootstrap_gain_ci.median)}).
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
            <h2>Performance by company</h2>
            <p>
              Hover or focus a row to keep the company, ticker, horizon and
              evaluation period together.
            </p>
          </div>
        </div>
        <CompanyTable rows={horizon.per_company} horizon={horizonValue} />
      </section>
      <FusionAblationPanel performance={performance} horizon={horizonValue} />
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
  const classData = Object.entries(ensemble?.class_f1 || {}).map(
    ([label, value]) => ({
      label: label[0].toUpperCase() + label.slice(1),
      value,
    }),
  );
  const yearlyCoverage = Object.entries(sentiment.dataset.year_counts).map(
    ([year, rows]) => ({ year, rows, value_format: "count" }),
  );
  const classCoverage = Object.entries(sentiment.dataset.class_counts).map(
    ([label, rows]) => ({
      label: label[0].toUpperCase() + label.slice(1),
      rows,
      value_format: "count",
    }),
  );
  return (
    <>
      <section className="sentiment-production-hero">
        <div>
          <span className="panel-kicker">Active production configuration</span>
          <h2>Bursa + PhraseBank FinBERT ensemble</h2>
          <p>
            {percent(sentiment.weights.bursa, 0)} Bursa-adapted model Â·{" "}
            {percent(sentiment.weights.phrasebank, 0)} PhraseBank model Â·
            temperature{" "}
            {sentiment.ensemble_temperature?.toFixed(2) || "Unavailable"}
          </p>
        </div>
        <div className="performance-status accepted">
          <ShieldCheck size={17} />
          Production override
        </div>
      </section>
      <div className="protocol-rail sentiment-period">
        <div>
          <b>Development and selection</b>
          <strong>2015â€“2024</strong>
          <span>
            Chronological Bursa evidence and PhraseBank retention checks.
          </span>
        </div>
        <i />
        <div>
          <b>Retrospective test</b>
          <strong>2025</strong>
          <span>
            Locked headline performance for the current 2015â€“2025 scope.
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
              <h3>Model comparison</h3>
            </div>
            <Scale size={19} />
          </header>
          <div className="performance-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparison}
                margin={{ top: 14, right: 8, left: -6, bottom: 2 }}
              >
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(v) => `${Math.round(v * 100)}%`}
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
          </div>
          <p>
            All three approaches are compared on the same Retrospective rows.
          </p>
        </article>
        <article className="performance-panel sentiment-retrospective">
          <header>
            <div>
              <span>Retrospective</span>
              <h3>Ensemble class F1</h3>
            </div>
            <BarChart3 size={19} />
          </header>
          <div className="performance-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={classData}
                layout="vertical"
                margin={{ top: 14, right: 16, left: 10, bottom: 2 }}
              >
                <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(v) => `${Math.round(v * 100)}%`}
                />
                <YAxis type="category" dataKey="label" width={60} />
                <Tooltip content={<EvaluationTooltip />} />
                <Bar dataKey="value" name="F1 score" radius={[0, 6, 6, 0]}>
                  {classData.map((item, index) => (
                    <Cell
                      key={item.label}
                      fill={[COLORS.red, COLORS.amber, COLORS.green][index]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
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
              <h3>Bursa coverage by year</h3>
            </div>
            <BarChart3 size={19} />
          </header>
          <div className="performance-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={yearlyCoverage}
                margin={{ top: 14, right: 8, left: -4, bottom: 2 }}
              >
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="year" tick={{ fontSize: 10 }} />
                <YAxis allowDecimals={false} />
                <Tooltip content={<EvaluationTooltip />} />
                <Bar
                  dataKey="rows"
                  name="Eligible articles"
                  fill={COLORS.blue}
                  radius={[5, 5, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
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
              <h3>Label distribution</h3>
            </div>
            <Scale size={19} />
          </header>
          <div className="performance-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={classCoverage}
                layout="vertical"
                margin={{ top: 14, right: 16, left: 8, bottom: 2 }}
              >
                <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="label" width={62} />
                <Tooltip content={<EvaluationTooltip />} />
                <Bar
                  dataKey="rows"
                  name="Eligible articles"
                  radius={[0, 6, 6, 0]}
                >
                  {classCoverage.map((item, index) => (
                    <Cell
                      key={item.label}
                      fill={[COLORS.red, COLORS.amber, COLORS.green][index]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
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
  const [view, setView] = useState<"stock" | "sentiment">(
    params.get("view") === "sentiment" ? "sentiment" : "stock",
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
  const setActiveView = (value: "stock" | "sentiment") => {
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
        <LoadingState label="Loading model evidenceÃ¢â‚¬Â¦" />
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
          ) : (
            <SentimentPerformanceView performance={query.data} />
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
