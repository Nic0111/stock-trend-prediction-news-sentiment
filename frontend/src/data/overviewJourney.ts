export type OverviewSceneId =
  | "exchange"
  | "observatory"
  | "forecast"
  | "newsroom"
  | "compare"
  | "learn";

export interface OverviewCameraStop {
  x: number;
  y: number;
  scale: number;
}

export interface OverviewScene {
  id: OverviewSceneId;
  label: string;
  eyebrow: string;
  title: string;
  body: string;
  tags: string[];
  accent: string;
  art: OverviewSceneId;
  worldPosition: { x: number; y: number };
  camera: OverviewCameraStop;
  route?: string;
  ctaLabel?: string;
}

/**
 * Coordinates live in the virtual Market Town, not in pixels on the screen.
 * Keeping them in data makes the camera journey easy to tune without touching
 * the scroll engine or the artwork components.
 */
export const OVERVIEW_WORLD = {
  width: 5200,
  height: 3100,
  scenes: [
    {
      id: "exchange",
      label: "Welcome Exchange",
      eyebrow: "A calmer starting point",
      title: "Make sense of market movement.",
      body: "StockSense turns prices, predictions and news into a small journey you can follow at your own pace.",
      tags: ["Beginner-friendly", "Bursa Malaysia"],
      accent: "#2F6BFF",
      art: "exchange",
      worldPosition: { x: 520, y: 1500 },
      camera: { x: 910, y: 1720, scale: 0.82 },
      route: "/stock-outlook",
      ctaLabel: "Explore a company",
    },
    {
      id: "observatory",
      label: "Market Observatory",
      eyebrow: "Start with the evidence",
      title: "See what moved.",
      body: "Inspect the latest price, change and market history before interpreting a model signal.",
      tags: ["Price context", "Daily history"],
      accent: "#2EB872",
      art: "observatory",
      worldPosition: { x: 1620, y: 690 },
      camera: { x: 2010, y: 980, scale: 0.88 },
      route: "/market-snapshot",
      ctaLabel: "Open market snapshot",
    },
    {
      id: "forecast",
      label: "Prediction Workshop",
      eyebrow: "Read the model clearly",
      title: "One signal, three horizons.",
      body: "Compare 1, 5 and 10 trading-day outlooks with probability, evidence status and a plain-language explanation.",
      tags: ["1 / 5 / 10 days", "Evidence-aware"],
      accent: "#6B63FF",
      art: "forecast",
      worldPosition: { x: 2870, y: 1470 },
      camera: { x: 3260, y: 1680, scale: 0.92 },
      route: "/stock-outlook",
      ctaLabel: "View stock outlook",
    },
    {
      id: "newsroom",
      label: "Newsroom Garden",
      eyebrow: "Check the surrounding story",
      title: "Signals need context.",
      body: "Read the retrieved articles, source coverage and confidence before deciding how much weight to give the tone.",
      tags: ["NewsAPI", "Marketaux fallback"],
      accent: "#F2AE32",
      art: "newsroom",
      worldPosition: { x: 3910, y: 580 },
      camera: { x: 4310, y: 910, scale: 0.88 },
      route: "/news",
      ctaLabel: "Read news evidence",
    },
    {
      id: "compare",
      label: "Comparison Bridge",
      eyebrow: "Look side by side",
      title: "Compare on equal terms.",
      body: "Place up to five Bursa companies on the same horizon so differences are easy to see and explain.",
      tags: ["Up to 5 tickers", "Shared horizon"],
      accent: "#D46B4C",
      art: "compare",
      worldPosition: { x: 4350, y: 2180 },
      camera: { x: 4350, y: 2180, scale: 0.9 },
      route: "/compare",
      ctaLabel: "Compare tickers",
    },
    {
      id: "learn",
      label: "Learning Pavilion",
      eyebrow: "Keep asking why",
      title: "Turn signals into understanding.",
      body: "Use Ask & Learn and the Learning Center to unpack probability, sentiment and model limitations in plain language.",
      tags: ["Conversational", "Educational boundary"],
      accent: "#E15C9B",
      art: "learn",
      worldPosition: { x: 2920, y: 2580 },
      camera: { x: 2920, y: 2580, scale: 0.94 },
      route: "/ask-learn",
      ctaLabel: "Ask StockSense",
    },
  ] satisfies OverviewScene[],
} as const;

export const OVERVIEW_SCENES = OVERVIEW_WORLD.scenes as readonly OverviewScene[];

export interface CinematicOverviewScene {
  id: OverviewSceneId;
  label: string;
  eyebrow: string;
  title: string;
  body: string;
  tags: readonly string[];
  accent: string;
  posterDesktop: string;
  /** Optional 4K poster used only on very wide/high-DPI displays. */
  posterDesktop4k?: string;
  /** Required motion-interpolated 1080p/60 master used by the scroll scrubber. */
  clipDesktop60: string;
  /** Source frame rate used to document the master. */
  sourceFps: number;
  duration: number;
  scrollWeight: number;
  linger: number;
  /** Transition used at this scene's outgoing seam. */
  transitionToNext?: "crossfade" | "veil";
  route?: string;
  ctaLabel?: string;
}

const MEDIA_ROOT = "/media/overview-world/v2";

/**
 * Versioned media manifest for the video-led Overview. Keeping the paths and
 * durations in data means the 1080p/60 masters can be replaced without
 * changing the scroll engine or page composition.
 */
export const CINEMATIC_OVERVIEW_SCENES: readonly CinematicOverviewScene[] = OVERVIEW_SCENES.map((scene, index) => {
  const mediaName: Record<OverviewSceneId, string> = {
    exchange: "welcome-exchange",
    observatory: "market-observatory",
    forecast: "prediction-workshop",
    newsroom: "newsroom-garden",
    compare: "comparison-bridge",
    learn: "learning-pavilion",
  };
  const name = mediaName[scene.id];
  const pacing = [
    { scrollWeight: 1.2, linger: 0.18 },
    { scrollWeight: 1.08, linger: 0.1 },
    { scrollWeight: 1.16, linger: 0.16 },
    { scrollWeight: 1.12, linger: 0.12 },
    { scrollWeight: 1.04, linger: 0.08 },
    { scrollWeight: 1.3, linger: 0.24 },
  ][index];
  return {
    id: scene.id,
    label: scene.label,
    eyebrow: scene.eyebrow,
    title: scene.title,
    body: scene.body,
    tags: scene.tags,
    accent: scene.accent,
    posterDesktop: `${MEDIA_ROOT}/poster/${name}.webp`,
    posterDesktop4k: `${MEDIA_ROOT}/poster-4k/${name}.webp`,
    clipDesktop60: `${MEDIA_ROOT}/video-1080-60/${name}.mp4`,
    sourceFps: 24,
    duration: 10.005,
    scrollWeight: pacing.scrollWeight,
    linger: pacing.linger,
    transitionToNext: index === 2 ? "crossfade" : "veil",
    route: scene.route,
    ctaLabel: scene.ctaLabel,
  };
});
