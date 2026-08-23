import { useEffect } from "react";

export interface StockSensePerformanceSnapshot {
  largestContentfulPaintMs: number | null;
  longTaskCount: number;
  longestTaskMs: number;
  overview?: {
    issued: number;
    coalesced: number;
    presented: number;
    lastLatencyMs: number;
    maxLatencyMs: number;
  };
}

declare global {
  interface Window {
    __stocksensePerformance?: StockSensePerformanceSnapshot;
  }
}

export function usePerformanceMonitor() {
  useEffect(() => {
    if (!import.meta.env.DEV || typeof PerformanceObserver === "undefined") return;
    const snapshot: StockSensePerformanceSnapshot = {
      largestContentfulPaintMs: null,
      longTaskCount: 0,
      longestTaskMs: 0,
    };
    window.__stocksensePerformance = snapshot;
    const observers: PerformanceObserver[] = [];

    try {
      const observer = new PerformanceObserver((list) => {
        const last = list.getEntries().at(-1);
        if (last) snapshot.largestContentfulPaintMs = Math.round(last.startTime);
      });
      observer.observe({ type: "largest-contentful-paint", buffered: true });
      observers.push(observer);
    } catch {
      // Diagnostic-only feature; unsupported browsers continue normally.
    }

    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          snapshot.longTaskCount += 1;
          snapshot.longestTaskMs = Math.max(snapshot.longestTaskMs, Math.round(entry.duration));
        }
      });
      observer.observe({ type: "longtask", buffered: true });
      observers.push(observer);
    } catch {
      // Diagnostic-only feature; unsupported browsers continue normally.
    }

    return () => observers.forEach((observer) => observer.disconnect());
  }, []);
}
