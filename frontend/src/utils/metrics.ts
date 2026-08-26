/**
 * Format a probability-like metric for presentation. Accuracy, F1 and ECE are
 * intentionally kept at one decimal place: these are level metrics and extra
 * digits would suggest more certainty than the evaluation supports.
 */
export function formatPercent(
  value: number | null | undefined,
  digits = 1,
) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "Unavailable";
}

/**
 * Format an absolute change between two probability metrics as percentage
 * points. Small but non-zero changes get more precision so they cannot be
 * mistaken for zero after rounding (for example 0.000021 -> +0.0021 pp).
 */
export function formatPercentagePoints(
  value: number | null | undefined,
  options: { signed?: boolean } = {},
) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }
  const percentagePoints = value * 100;
  const magnitude = Math.abs(percentagePoints);
  const digits =
    magnitude === 0
      ? 1
      : magnitude < 0.01
        ? 4
        : magnitude < 0.1
          ? 3
          : magnitude < 1
            ? 2
            : 1;
  const sign = options.signed !== false && percentagePoints > 0 ? "+" : "";
  return `${sign}${percentagePoints.toFixed(digits)} pp`;
}

/**
 * Use the delta formatter for chart tooltips while preserving the normal
 * percentage formatter for level metrics.
 */
export function formatPerformanceValue(
  value: number | null | undefined,
  label: string | null | undefined,
) {
  const metric = String(label || "").toLowerCase();
  return /gain|improvement|uplift|decrease|difference|delta/.test(metric)
    ? formatPercentagePoints(value)
    : formatPercent(value);
}
