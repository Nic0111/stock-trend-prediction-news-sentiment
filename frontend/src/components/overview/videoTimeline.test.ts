import { applyLinger, buildTimelineRanges, locateTimelinePosition } from "./videoTimeline";

describe("video timeline", () => {
  it("builds normalized ranges from scene weights", () => {
    const ranges = buildTimelineRanges([{ scrollWeight: 1 }, { scrollWeight: 2 }, { scrollWeight: 1 }]);
    expect(ranges).toEqual([
      { start: 0, end: 0.25 },
      { start: 0.25, end: 0.75 },
      { start: 0.75, end: 1 },
    ]);
  });

  it("crossfades adjacent scenes around their shared boundary", () => {
    const ranges = buildTimelineRanges([{ scrollWeight: 1 }, { scrollWeight: 1 }]);
    const position = locateTimelinePosition(0.5, ranges, 0.05);
    expect(position.opacities[0]).toBeCloseTo(0.5);
    expect(position.opacities[1]).toBeCloseTo(0.5);
  });

  it("covers mismatched seams with a reversible branded veil", () => {
    const ranges = buildTimelineRanges([{ scrollWeight: 1 }, { scrollWeight: 1 }]);
    const position = locateTimelinePosition(0.5, ranges, {
      seamTransitions: ["veil"],
      veilWidth: 0.1,
    });
    expect(position.transition?.kind).toBe("veil");
    expect(position.transition?.opacity).toBeCloseTo(1);
    expect(position.opacities).toEqual([0, 1]);

    const before = locateTimelinePosition(0.45, ranges, {
      seamTransitions: ["veil"],
      veilWidth: 0.1,
    });
    const after = locateTimelinePosition(0.55, ranges, {
      seamTransitions: ["veil"],
      veilWidth: 0.1,
    });
    expect(before.transition?.opacity).toBeCloseTo(after.transition?.opacity ?? 0);
    expect(before.opacities).toEqual([1, 0]);
    expect(after.opacities).toEqual([0, 1]);
  });

  it("keeps linger monotonic and preserves both seam frames", () => {
    expect(applyLinger(0, 0.5)).toBe(0);
    expect(applyLinger(1, 0.5)).toBe(1);
    expect(applyLinger(0.35, 0.5)).toBeLessThan(applyLinger(0.65, 0.5));
  });
});
