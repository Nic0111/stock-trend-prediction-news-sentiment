import { useLayoutEffect, useRef, useState } from "react";
import type { OverviewCameraStop } from "../data/overviewJourney";

interface ScrollCameraOptions {
  stops: readonly OverviewCameraStop[];
  worldWidth: number;
  worldHeight: number;
}

interface ScrollCameraResult {
  sectionRef: React.RefObject<HTMLElement | null>;
  viewportRef: React.RefObject<HTMLDivElement | null>;
  worldRef: React.RefObject<HTMLDivElement | null>;
  activeIndex: number;
  progress: number;
}

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const smoothstep = (value: number) => value * value * (3 - 2 * value);

function interpolate(a: number, b: number, progress: number) {
  return a + (b - a) * progress;
}

/**
 * Converts document scroll into a camera transform without causing a React
 * render for every wheel event. The camera is written directly to the world
 * layer, which keeps the interaction smooth even when the artwork is detailed.
 */
export function useScrollCamera({ stops, worldWidth, worldHeight }: ScrollCameraOptions): ScrollCameraResult {
  const sectionRef = useRef<HTMLElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const worldRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef<number | null>(null);
  const latestProgressRef = useRef(0);
  const [activeIndex, setActiveIndex] = useState(0);
  const [progress, setProgress] = useState(0);

  useLayoutEffect(() => {
    const updateCamera = () => {
      frameRef.current = null;
      const section = sectionRef.current;
      const viewport = viewportRef.current;
      const world = worldRef.current;
      if (!section || !viewport || !world || stops.length === 0) return;

      const rect = section.getBoundingClientRect();
      const scrollableDistance = Math.max(1, rect.height - window.innerHeight);
      const nextProgress = clamp(-rect.top / scrollableDistance);
      latestProgressRef.current = nextProgress;

      const segmentCount = Math.max(1, stops.length - 1);
      const position = nextProgress * segmentCount;
      const segmentIndex = Math.min(segmentCount - 1, Math.floor(position));
      const segmentProgress = position - segmentIndex;
      const current = stops[segmentIndex] || stops[0];
      const next = stops[Math.min(segmentIndex + 1, stops.length - 1)] || current;
      const eased = smoothstep(segmentProgress);
      const x = interpolate(current.x, next.x, eased);
      const y = interpolate(current.y, next.y, eased);
      const scale = interpolate(current.scale, next.scale, eased);
      const active = Math.min(stops.length - 1, Math.round(position));
      const viewportRect = viewport.getBoundingClientRect();
      const tx = viewportRect.width / 2 - x * scale;
      const ty = viewportRect.height / 2 - y * scale;
      world.style.transform = `translate3d(${tx}px, ${ty}px, 0) scale(${scale})`;
      world.style.setProperty("--camera-progress", String(nextProgress));
      world.style.setProperty("--camera-scale", String(scale));

      setProgress((previous) => (Math.abs(previous - nextProgress) > 0.01 ? nextProgress : previous));
      setActiveIndex((previous) => (previous === active ? previous : active));
    };

    const requestUpdate = () => {
      // Apply the transform immediately so the camera still responds in
      // throttled/background browser tabs. The queued frame keeps repeated
      // wheel events inexpensive during normal interaction.
      updateCamera();
      if (frameRef.current === null) frameRef.current = window.requestAnimationFrame(updateCamera);
    };

    const handleResize = () => requestUpdate();
    const handleScroll = () => requestUpdate();
    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleResize);
    const resizeObserver = typeof ResizeObserver !== "undefined" ? new ResizeObserver(requestUpdate) : null;
    const sectionElement = sectionRef.current;
    if (resizeObserver && sectionElement) resizeObserver.observe(sectionElement);
    requestUpdate();

    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleResize);
      resizeObserver?.disconnect();
      if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
    };
  }, [stops, worldHeight, worldWidth]);

  return { sectionRef, viewportRef, worldRef, activeIndex, progress };
}
