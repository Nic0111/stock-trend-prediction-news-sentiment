import { describe, expect, it } from "vitest";
import {
  formatPercent,
  formatPercentagePoints,
  formatPerformanceValue,
} from "./metrics";

describe("performance metric formatting", () => {
  it("keeps level metrics at the normal one-decimal percentage precision", () => {
    expect(formatPercent(0.786)).toBe("78.6%");
  });

  it("keeps a small non-zero improvement visible as percentage points", () => {
    expect(formatPercentagePoints(0.000021)).toBe("+0.0021 pp");
    expect(formatPercentagePoints(-0.000021)).toBe("-0.0021 pp");
  });

  it("uses the delta formatter for improvement tooltips", () => {
    expect(formatPerformanceValue(0.000021, "Brier improvement")).toBe(
      "+0.0021 pp",
    );
    expect(formatPerformanceValue(0.786, "Balanced accuracy")).toBe("78.6%");
  });
});
