import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CinematicOverviewScene } from "../data/overviewJourney";
import {
  applyLinger,
  buildTimelineRanges,
  locateTimelinePosition,
  type TimelinePosition,
} from "../components/overview/videoTimeline";

export type VideoQuality = "1080p60";

interface VideoScrollWorldOptions {
  scenes: readonly CinematicOverviewScene[];
}

interface SeekDiagnostics {
  issued: number;
  coalesced: number;
  presented: number;
  lastLatencyMs: number;
  maxLatencyMs: number;
}

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const FRAME_TOLERANCE = 0.55;

export function useVideoScrollWorld({ scenes }: VideoScrollWorldOptions) {
  const sectionRef = useRef<HTMLElement | null>(null);
  const layerRefs = useRef<Array<HTMLDivElement | null>>([]);
  const videoRefs = useRef<Array<HTMLVideoElement | null>>([]);
  const transitionLayerRef = useRef<HTMLDivElement | null>(null);
  const loadedQualityRef = useRef<Map<number, VideoQuality>>(new Map());
  const seekInFlightRef = useRef<Set<number>>(new Set());
  const seekFrameRef = useRef<number | null>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const jumpAnimationRef = useRef<number | null>(null);
  const latestProgressRef = useRef(0);
  const activeIndexRef = useRef(0);
  const transitionRef = useRef<TimelinePosition["transition"]>(null);
  const targetRef = useRef<number[]>(scenes.map(() => 0));
  const pendingSeekRef = useRef<Array<number | null>>(scenes.map(() => null));
  const seekStartedAtRef = useRef<number[]>(scenes.map(() => 0));
  const seekWatchdogRef = useRef<Array<number | null>>(scenes.map(() => null));
  const diagnosticsRef = useRef<SeekDiagnostics>({
    issued: 0,
    coalesced: 0,
    presented: 0,
    lastLatencyMs: 0,
    maxLatencyMs: 0,
  });
  // The overview intentionally uses one media tier so every desktop session
  // gets the same 1080p/60fps scroll-scrub experience.  Poster images are
  // retained only as the native video loading surface.
  const qualityReadyRef = useRef(true);
  const mediaQuality: VideoQuality = "1080p60";
  const qualityReady = true;
  const [activeIndex, setActiveIndex] = useState(0);
  const progressRef = useRef<HTMLSpanElement | null>(null);
  const ranges = useMemo(() => buildTimelineRanges(scenes), [scenes]);
  const totalScrollVh = useMemo(
    () => Math.round((scenes.reduce((sum, scene) => sum + scene.scrollWeight, 0) + 1) * 100),
    [scenes],
  );

  const clipQuality = useCallback((index: number): VideoQuality => {
    void index;
    return "1080p60";
  }, []);

  const clipSource = useCallback((index: number) => {
    const scene = scenes[index];
    if (!scene) return "";
    return scene.clipDesktop60 ?? "";
  }, [scenes]);

  const unloadClip = useCallback((index: number) => {
    loadedQualityRef.current.delete(index);
    const video = videoRefs.current[index];
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
      delete video.dataset.loadedQuality;
    }
    const watchdog = seekWatchdogRef.current[index];
    if (watchdog !== null) window.clearTimeout(watchdog);
    seekWatchdogRef.current[index] = null;
    seekInFlightRef.current.delete(index);
    pendingSeekRef.current[index] = null;
  }, []);

  const loadClip = useCallback((index: number) => {
    if (!qualityReadyRef.current || index < 0 || index >= scenes.length) return;
    const video = videoRefs.current[index];
    if (!video || video.dataset.failed === "true") return;
    const desiredQuality = clipQuality(index);
    if (loadedQualityRef.current.get(index) === desiredQuality && video.src) return;
    unloadClip(index);
    const source = clipSource(index);
    if (!source) {
      video.dataset.failed = "true";
      video.dataset.mediaError = "missing-1080p60-source";
      console.error(`[StockSense] Missing 1080p60 overview clip for scene ${index}.`);
      return;
    }
    // Keep the URL native so the browser can issue byte-range requests and
    // seek before the complete 1080p60 clip has downloaded.
    loadedQualityRef.current.set(index, desiredQuality);
    video.src = source;
    video.dataset.loadedQuality = desiredQuality;
    video.preload = "auto";
    video.load();
  }, [clipQuality, clipSource, scenes.length, unloadClip]);

  const registerLayer = useCallback((index: number, element: HTMLDivElement | null) => {
    layerRefs.current[index] = element;
  }, []);

  const registerTransitionLayer = useCallback((element: HTMLDivElement | null) => {
    transitionLayerRef.current = element;
  }, []);

  const registerProgress = useCallback((element: HTMLSpanElement | null) => {
    progressRef.current = element;
  }, []);

  const registerVideo = useCallback((index: number, element: HTMLVideoElement | null) => {
    videoRefs.current[index] = element;
    if (!element) return;
    element.muted = true;
    element.playsInline = true;
    element.preload = index < 2 ? "auto" : "metadata";
    element.dataset.sceneIndex = String(index);
  }, []);

  useLayoutEffect(() => {
    let sectionPageTop = 0;
    let sectionHeight = 0;

    const refreshSectionMetrics = () => {
      const section = sectionRef.current;
      if (!section) return;
      const rect = section.getBoundingClientRect();
      sectionPageTop = window.scrollY + rect.top;
      sectionHeight = rect.height;
    };

    function seekVideos() {
      seekFrameRef.current = null;
      const indices = new Set<number>([activeIndexRef.current]);
      const transition = transitionRef.current;
      if (transition) {
        indices.add(transition.index);
        indices.add(transition.index + 1);
      }
      let waitingForMedia = false;
      indices.forEach((index) => {
        const video = videoRefs.current[index];
        const scene = scenes[index];
        if (!video || !scene || video.readyState < 1 || video.dataset.failed === "true") {
          waitingForMedia = true;
          return;
        }
        const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : scene.duration;
        const frameStep = 1 / 60;
        const rawTime = clamp(targetRef.current[index] ?? 0, 0, 0.999) * duration;
        const time = Math.min(Math.max(0, Math.round(rawTime / frameStep) * frameStep), Math.max(0, duration - frameStep));
        if (seekInFlightRef.current.has(index) || video.seeking) {
          if (Math.abs((pendingSeekRef.current[index] ?? time) - time) >= frameStep * 0.5) {
            pendingSeekRef.current[index] = time;
            diagnosticsRef.current.coalesced += 1;
          }
          return;
        }
        if (Math.abs(video.currentTime - time) < frameStep * FRAME_TOLERANCE) return;
        try {
          seekInFlightRef.current.add(index);
          pendingSeekRef.current[index] = time;
          seekStartedAtRef.current[index] = performance.now();
          diagnosticsRef.current.issued += 1;
          video.currentTime = time;
          // A seek can occasionally stall indefinitely while a blob is still
          // being decoded. Clear the in-flight lock so the next scroll frame
          // can retry instead of freezing the whole scrubber.
          const previousWatchdog = seekWatchdogRef.current[index];
          if (previousWatchdog !== null) window.clearTimeout(previousWatchdog);
          seekWatchdogRef.current[index] = window.setTimeout(() => {
            if (!seekInFlightRef.current.has(index)) return;
            seekInFlightRef.current.delete(index);
            seekWatchdogRef.current[index] = null;
            diagnosticsRef.current.coalesced += 1;
            scheduleSeek();
          }, 1500);
        } catch {
          seekInFlightRef.current.delete(index);
          const watchdog = seekWatchdogRef.current[index];
          if (watchdog !== null) window.clearTimeout(watchdog);
          seekWatchdogRef.current[index] = null;
        }
      });
      if (waitingForMedia && qualityReadyRef.current) return;
    }

    function scheduleSeek() {
      if (seekFrameRef.current === null) seekFrameRef.current = window.requestAnimationFrame(seekVideos);
    }

    const updateFromScroll = () => {
      scrollFrameRef.current = null;
      const section = sectionRef.current;
      if (!section) return;
      if (sectionHeight <= 0) refreshSectionMetrics();
      const scrollableDistance = Math.max(1, sectionHeight - window.innerHeight);
      const nextProgress = clamp((window.scrollY - sectionPageTop) / scrollableDistance);
      latestProgressRef.current = nextProgress;
      const seamTransitions = scenes.slice(0, -1).map((scene) => scene.transitionToNext ?? "veil");
      const position = locateTimelinePosition(nextProgress, ranges, {
        crossfadeWidth: 0.014,
        veilWidth: 0.035,
        seamTransitions,
      });
      transitionRef.current = position.transition;
      if (transitionLayerRef.current) {
        transitionLayerRef.current.style.opacity = String(position.transition?.opacity ?? 0);
        transitionLayerRef.current.dataset.transitionKind = position.transition?.kind ?? "none";
      }
      position.opacities.forEach((opacity, index) => {
        const layer = layerRefs.current[index];
        if (!layer) return;
        layer.style.opacity = opacity.toFixed(4);
        layer.style.zIndex = opacity > 0 ? String(20 + Math.round(opacity * 10)) : "1";
        layer.style.pointerEvents = opacity > 0.75 ? "auto" : "none";
      });
      position.localProgress.forEach((local, index) => {
        targetRef.current[index] = applyLinger(local, scenes[index].linger);
      });
      activeIndexRef.current = position.activeIndex;
      if (qualityReadyRef.current) {
        const needed = new Set<number>([position.activeIndex]);
        if (position.transition) {
          needed.add(position.transition.index);
          needed.add(position.transition.index + 1);
        } else if (position.activeIndex + 1 < scenes.length) {
          needed.add(position.activeIndex + 1);
        }
        needed.forEach((index) => loadClip(index));
        for (let index = 0; index < scenes.length; index += 1) {
          if (!needed.has(index)) unloadClip(index);
        }
      }
      setActiveIndex((previous) => previous === position.activeIndex ? previous : position.activeIndex);
      if (progressRef.current) progressRef.current.style.transform = `scaleX(${nextProgress})`;
      scheduleSeek();
    };

    const requestScrollUpdate = () => {
      if (scrollFrameRef.current === null) scrollFrameRef.current = window.requestAnimationFrame(updateFromScroll);
    };

    const seekedHandlers = videoRefs.current.map((video, index) => {
      if (!video) return null;
      const handleSeeked = () => {
        const watchdog = seekWatchdogRef.current[index];
        if (watchdog !== null) window.clearTimeout(watchdog);
        seekWatchdogRef.current[index] = null;
        seekInFlightRef.current.delete(index);
        const startedAt = seekStartedAtRef.current[index];
        if (startedAt) {
          const latency = performance.now() - startedAt;
          diagnosticsRef.current.lastLatencyMs = latency;
          diagnosticsRef.current.maxLatencyMs = Math.max(diagnosticsRef.current.maxLatencyMs, latency);
          if (import.meta.env.DEV && window.__stocksensePerformance) {
            window.__stocksensePerformance.overview = { ...diagnosticsRef.current };
          }
        }
        if (typeof video.requestVideoFrameCallback === "function") {
          video.requestVideoFrameCallback(() => {
            diagnosticsRef.current.presented += 1;
            video.parentElement?.classList.add("has-video");
            scheduleSeek();
          });
        } else {
          video.parentElement?.classList.add("has-video");
          scheduleSeek();
        }
      };
      const handleMediaReady = () => scheduleSeek();
      const handleMediaError = () => {
        video.dataset.failed = "true";
        video.dataset.mediaError = `1080p60-${video.error?.code ?? "unknown"}`;
        console.error(`[StockSense] 1080p60 overview clip failed for scene ${index}.`, scenes[index]?.clipDesktop60, video.error);
      };
      video.addEventListener("seeked", handleSeeked);
      video.addEventListener("loadedmetadata", handleMediaReady);
      video.addEventListener("loadeddata", handleMediaReady);
      video.addEventListener("error", handleMediaError);
      return () => {
        video.removeEventListener("seeked", handleSeeked);
        video.removeEventListener("loadedmetadata", handleMediaReady);
        video.removeEventListener("loadeddata", handleMediaReady);
        video.removeEventListener("error", handleMediaError);
      };
    });

    window.addEventListener("scroll", requestScrollUpdate, { passive: true });
    window.addEventListener("resize", requestScrollUpdate);
    const resizeObserver = typeof ResizeObserver !== "undefined" && sectionRef.current
      ? new ResizeObserver(() => {
        refreshSectionMetrics();
        requestScrollUpdate();
      })
      : null;
    if (resizeObserver && sectionRef.current) resizeObserver.observe(sectionRef.current);
    refreshSectionMetrics();
    updateFromScroll();
    scheduleSeek();

    return () => {
      window.removeEventListener("scroll", requestScrollUpdate);
      window.removeEventListener("resize", requestScrollUpdate);
      resizeObserver?.disconnect();
      seekedHandlers.forEach((cleanup) => cleanup?.());
      // React StrictMode deliberately mounts, cleans up, and mounts effects
      // again in development. Cancelling an animation frame does not clear
      // the stored request id, so the next effect mount would incorrectly
      // believe that scrolling/seeking was already scheduled. Reset every
      // scheduler ref after cancellation so the video scrubber can restart.
      if (scrollFrameRef.current !== null) {
        window.cancelAnimationFrame(scrollFrameRef.current);
        scrollFrameRef.current = null;
      }
      if (seekFrameRef.current !== null) {
        window.cancelAnimationFrame(seekFrameRef.current);
        seekFrameRef.current = null;
      }
      if (jumpAnimationRef.current !== null) {
        window.cancelAnimationFrame(jumpAnimationRef.current);
        jumpAnimationRef.current = null;
      }
      loadedQualityRef.current.clear();
      seekInFlightRef.current.clear();
      seekWatchdogRef.current.forEach((watchdog) => {
        if (watchdog !== null) window.clearTimeout(watchdog);
      });
      seekWatchdogRef.current = scenes.map(() => null);
    };
  }, [clipQuality, loadClip, ranges, scenes, unloadClip, mediaQuality, qualityReady]);

  const jumpToScene = useCallback((index: number) => {
    const section = sectionRef.current;
    const range = ranges[index];
    if (!section || !range) return;
    const sectionTop = window.scrollY + section.getBoundingClientRect().top;
    const maxScroll = Math.max(0, section.offsetHeight - window.innerHeight);
    const targetProgress = (range.start + range.end) / 2;
    const targetTop = sectionTop + maxScroll * targetProgress;
    if (jumpAnimationRef.current !== null) window.cancelAnimationFrame(jumpAnimationRef.current);
    const startTop = window.scrollY;
    const distance = Math.abs(targetTop - startTop);
    const duration = Math.min(3200, Math.max(1250, 900 + distance * 0.34));
    const startedAt = performance.now();
    const easeInOut = (value: number) => value < 0.5
      ? 4 * value * value * value
      : 1 - Math.pow(-2 * value + 2, 3) / 2;
    const animate = (now: number) => {
      const elapsed = Math.min(1, (now - startedAt) / duration);
      window.scrollTo({ top: startTop + (targetTop - startTop) * easeInOut(elapsed), behavior: "auto" });
      if (elapsed < 1) {
        jumpAnimationRef.current = window.requestAnimationFrame(animate);
      } else {
        jumpAnimationRef.current = null;
      }
    };
    jumpAnimationRef.current = window.requestAnimationFrame(animate);
  }, [ranges]);

  return {
    sectionRef,
    activeIndex,
    mediaQuality,
    diagnosticsRef,
    totalScrollVh,
    registerLayer,
    registerTransitionLayer,
    registerVideo,
    registerProgress,
    jumpToScene,
    latestProgressRef,
  };
}
