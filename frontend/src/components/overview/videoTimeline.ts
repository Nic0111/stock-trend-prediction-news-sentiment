export interface TimelineItem {
  scrollWeight: number;
}

export interface TimelineRange {
  start: number;
  end: number;
}

export interface TimelinePosition {
  activeIndex: number;
  localProgress: number[];
  opacities: number[];
  transition: {
    index: number;
    kind: "crossfade" | "veil";
    progress: number;
    opacity: number;
  } | null;
}

export interface TimelineOptions {
  crossfadeWidth?: number;
  veilWidth?: number;
  seamTransitions?: readonly ("crossfade" | "veil")[];
}

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const smoothstep = (value: number) => {
  const x = clamp(value);
  return x * x * (3 - 2 * x);
};

export function buildTimelineRanges(items: readonly TimelineItem[]): TimelineRange[] {
  const total = items.reduce((sum, item) => sum + Math.max(0.01, item.scrollWeight), 0);
  let cursor = 0;
  return items.map((item, index) => {
    const start = cursor / total;
    cursor += Math.max(0.01, item.scrollWeight);
    return { start, end: index === items.length - 1 ? 1 : cursor / total };
  });
}

export function locateTimelinePosition(
  progress: number,
  ranges: readonly TimelineRange[],
  options: number | TimelineOptions = 0.014,
): TimelinePosition {
  if (ranges.length === 0) {
    return { activeIndex: 0, localProgress: [], opacities: [], transition: null };
  }
  const resolved: TimelineOptions = typeof options === "number"
    ? { crossfadeWidth: options }
    : options;
  const crossfadeWidth = resolved.crossfadeWidth ?? 0.014;
  const veilWidth = resolved.veilWidth ?? 0.04;
  const seamTransitions = resolved.seamTransitions ?? [];
  const value = clamp(progress);
  const localProgress = ranges.map((range) =>
    clamp((value - range.start) / Math.max(0.0001, range.end - range.start)),
  );
  const opacities = ranges.map(() => 0);
  let transition: TimelinePosition["transition"] = null;
  let activeIndex = ranges.findIndex((range) => value < range.end);
  if (activeIndex < 0) activeIndex = ranges.length - 1;
  opacities[activeIndex] = 1;

  for (let index = 0; index < ranges.length - 1; index += 1) {
    const boundary = ranges[index].end;
    const kind = seamTransitions[index] ?? "crossfade";
    const width = kind === "veil" ? veilWidth : crossfadeWidth;
    if (value < boundary - width || value > boundary + width) continue;
    const phase = clamp((value - (boundary - width)) / Math.max(width * 2, 0.0001));
    if (kind === "veil") {
      const veilOpacity = phase <= 0.5
        ? smoothstep(phase * 2)
        : smoothstep((1 - phase) * 2);
      opacities.fill(0);
      if (value < boundary) {
        opacities[index] = 1;
        activeIndex = index;
      } else {
        opacities[index + 1] = 1;
        activeIndex = index + 1;
      }
      transition = { index, kind, progress: phase, opacity: veilOpacity };
    } else {
      const blend = smoothstep(phase);
      opacities.fill(0);
      opacities[index] = 1 - blend;
      opacities[index + 1] = blend;
      activeIndex = blend < 0.5 ? index : index + 1;
      transition = { index, kind, progress: phase, opacity: 0 };
    }
    break;
  }

  return { activeIndex, localProgress, opacities, transition };
}

export function applyLinger(progress: number, linger: number) {
  const x = clamp(progress);
  const amount = clamp(linger, 0, 0.6);
  const centered = x - 0.5;
  return (1 - amount) * x + amount * (4 * centered * centered * centered + 0.5);
}
