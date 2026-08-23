import {
  ArrowDown,
  ArrowRight,
  BarChart3,
  BookOpen,
  GitCompareArrows,
  LineChart,
  Newspaper,
  Sparkles,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import {
  CINEMATIC_OVERVIEW_SCENES,
  type CinematicOverviewScene,
  type OverviewSceneId,
} from "../../data/overviewJourney";
import { useVideoScrollWorld } from "../../hooks/useVideoScrollWorld";

interface ScrollWorldProps {
  companyCount?: number;
  companyLoadFailed?: boolean;
}

const sceneIcons: Record<OverviewSceneId, LucideIcon> = {
  exchange: Sparkles,
  observatory: LineChart,
  forecast: BarChart3,
  newsroom: Newspaper,
  compare: GitCompareArrows,
  learn: BookOpen,
};

function SceneCopy({
  scene,
  index,
  total,
  companyCount,
}: {
  scene: CinematicOverviewScene;
  index: number;
  total: number;
  companyCount?: number;
}) {
  const Icon = sceneIcons[scene.id];
  const isOpeningScene = index === 0;
  return (
    <div className="cinematic-copy" style={{ "--cinematic-accent": scene.accent } as React.CSSProperties}>
      <div className="cinematic-copy-topline">
        <span className="cinematic-copy-kicker"><Icon size={16} /> {scene.eyebrow}</span>
        <span className="cinematic-copy-counter">{String(index + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}</span>
      </div>
      <h1>{scene.title}</h1>
      <p>{scene.body}</p>
      {isOpeningScene && (
        <div className="cinematic-system-facts cinematic-hero-facts">
          <span><strong>{companyCount ?? 10}</strong> companies</span>
          <span><strong>3</strong> horizons</span>
          <span><strong>2</strong> evidence streams</span>
        </div>
      )}
      {!isOpeningScene && (
        <>
          <div className="cinematic-copy-tags">
            {scene.tags.map((tag) => <span key={tag}>{tag}</span>)}
          </div>
          <div className="cinematic-copy-actions">
            {scene.route && scene.ctaLabel && (
              <Link className="cinematic-copy-cta primary" to={scene.route}>
                {scene.ctaLabel}
                <ArrowRight size={17} />
              </Link>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function DesktopScrollWorld({
  companyCount,
}: {
  companyCount?: number;
}) {
  const {
    sectionRef,
    activeIndex,
    totalScrollVh,
    mediaQuality,
    registerLayer,
    registerTransitionLayer,
    registerVideo,
    registerProgress,
    jumpToScene,
  } = useVideoScrollWorld({ scenes: CINEMATIC_OVERVIEW_SCENES });
  const activeScene = CINEMATIC_OVERVIEW_SCENES[activeIndex] ?? CINEMATIC_OVERVIEW_SCENES[0];

  return (
    <section
      ref={sectionRef}
      className="cinematic-world-shell"
      data-video-quality={mediaQuality}
      style={{ "--cinematic-scroll-height": `${totalScrollVh}vh` } as React.CSSProperties}
      aria-label="StockSense interactive overview"
    >
      <div className="cinematic-world-sticky">
        <div className="cinematic-video-stage" aria-hidden="true">
          {CINEMATIC_OVERVIEW_SCENES.map((scene, index) => (
            <div
              key={scene.id}
              ref={(element) => registerLayer(index, element)}
              className={`cinematic-video-layer ${index === 0 ? "is-first" : ""}`}
              style={{ opacity: index === 0 ? 1 : 0 }}
            >
              <picture className="cinematic-video-poster-picture">
                {scene.posterDesktop4k && (
                  <source media="(min-width: 2560px)" srcSet={scene.posterDesktop4k} />
                )}
                <img src={scene.posterDesktop} alt="" className="cinematic-video-poster" />
              </picture>
              <video
                ref={(element) => registerVideo(index, element)}
                className="cinematic-video"
                muted
                playsInline
                preload={index < 2 ? "auto" : "metadata"}
                poster={scene.posterDesktop}
                onLoadedData={(event) => event.currentTarget.parentElement?.classList.add("has-video")}
                onSeeked={(event) => event.currentTarget.parentElement?.classList.add("has-video")}
              />
            </div>
          ))}
          <div className="cinematic-readable-scrim" />
          <div ref={registerTransitionLayer} className="cinematic-transition-veil" aria-hidden="true" />
        </div>

        <div className="cinematic-copy-wrap" key={activeScene.id}>
          <SceneCopy scene={activeScene} index={activeIndex} total={CINEMATIC_OVERVIEW_SCENES.length} companyCount={companyCount} />
        </div>

        <nav className="cinematic-route" aria-label="Overview journey">
          {CINEMATIC_OVERVIEW_SCENES.map((scene, index) => (
            <button
              key={scene.id}
              type="button"
              className={`cinematic-route-stop ${activeIndex === index ? "is-active" : ""}`}
              onClick={() => jumpToScene(index)}
              aria-label={`Go to ${scene.label}`}
              aria-current={activeIndex === index ? "step" : undefined}
              style={{ "--route-accent": scene.accent } as React.CSSProperties}
            >
              <span>{index + 1}</span>
              <em>{scene.label}</em>
            </button>
          ))}
        </nav>

        <div className="cinematic-progress" aria-hidden="true">
          <span ref={registerProgress} />
        </div>
        <div className="cinematic-scroll-hint cinematic-scroll-hint-right" aria-hidden="true">
          <ArrowDown size={15} />
          Scroll to travel
        </div>
      </div>
    </section>
  );
}

export function ScrollWorld({ companyCount }: ScrollWorldProps) {
  // The Overview is intentionally a single video-led experience.  Poster
  // images remain as the native video loading surface, but there is no
  // reduced-motion/static or legacy camera branch to silently change the
  // interaction model between browser instances.
  return <DesktopScrollWorld companyCount={companyCount} />;
}
