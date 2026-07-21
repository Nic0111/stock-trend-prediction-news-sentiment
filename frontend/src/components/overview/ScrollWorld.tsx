import { ArrowDown, ArrowRight, BarChart3, BookOpen, ChevronRight, CircleHelp, GitCompareArrows, LineChart, MessageCircle, Newspaper, Sparkles } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { OVERVIEW_SCENES, OVERVIEW_WORLD, type OverviewScene, type OverviewSceneId } from "../../data/overviewJourney";
import { useScrollCamera } from "../../hooks/useScrollCamera";
import { MiniatureArt } from "./MiniatureArt";

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

function ScenePin({ scene, index, active }: { scene: OverviewScene; index: number; active: boolean }) {
  const style = {
    "--scene-x": `${scene.worldPosition.x}px`,
    "--scene-y": `${scene.worldPosition.y}px`,
    "--scene-accent": scene.accent,
  } as React.CSSProperties;
  const Icon = sceneIcons[scene.id];
  return (
    <article className={`world-scene ${active ? "is-active" : ""}`} style={style} data-scene={scene.id}>
      <div className="world-scene-glow" />
      <div className="world-scene-index">0{index + 1}</div>
      <MiniatureArt kind={scene.art} accent={scene.accent} />
      <div className="world-scene-caption">
        <span className="world-scene-caption-icon"><Icon size={15} strokeWidth={2.3} /></span>
        <span>{scene.label}</span>
      </div>
    </article>
  );
}

function ConnectorPath() {
  return (
    <svg className="market-path" viewBox={`0 0 ${OVERVIEW_WORLD.width} ${OVERVIEW_WORLD.height}`} aria-hidden="true" preserveAspectRatio="none">
      <defs>
        <linearGradient id="market-path-gradient" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#2F6BFF" />
          <stop offset="0.46" stopColor="#6B63FF" />
          <stop offset="1" stopColor="#2EB872" />
        </linearGradient>
      </defs>
      <path d="M910 1690 C1290 1460 1510 1250 2010 930 C2510 650 2790 1160 3260 1670 C3740 2180 3880 1160 4310 830 C4600 600 4620 1510 4350 2180 C4130 2680 3650 2760 2920 2580" fill="none" stroke="#ffffff" strokeWidth="33" strokeLinecap="round" opacity="0.82" />
      <path d="M910 1690 C1290 1460 1510 1250 2010 930 C2510 650 2790 1160 3260 1670 C3740 2180 3880 1160 4310 830 C4600 600 4620 1510 4350 2180 C4130 2680 3650 2760 2920 2580" fill="none" stroke="url(#market-path-gradient)" strokeWidth="13" strokeLinecap="round" strokeDasharray="2 34" />
      <path d="M910 1690 C1290 1460 1510 1250 2010 930 C2510 650 2790 1160 3260 1670 C3740 2180 3880 1160 4310 830 C4600 600 4620 1510 4350 2180 C4130 2680 3650 2760 2920 2580" fill="none" stroke="#2F6BFF" strokeWidth="2" strokeDasharray="13 16" opacity="0.46" />
      <circle cx="910" cy="1690" r="22" fill="#2F6BFF" opacity="0.15" />
      <circle cx="910" cy="1690" r="9" fill="#2F6BFF" />
      <circle cx="2010" cy="930" r="22" fill="#2EB872" opacity="0.15" />
      <circle cx="2010" cy="930" r="9" fill="#2EB872" />
      <circle cx="3260" cy="1670" r="22" fill="#6B63FF" opacity="0.15" />
      <circle cx="3260" cy="1670" r="9" fill="#6B63FF" />
      <circle cx="4310" cy="830" r="22" fill="#F2AE32" opacity="0.15" />
      <circle cx="4310" cy="830" r="9" fill="#F2AE32" />
      <circle cx="2920" cy="2580" r="22" fill="#D46B4C" opacity="0.15" />
      <circle cx="2920" cy="2580" r="9" fill="#D46B4C" />
      <circle cx="4350" cy="2180" r="22" fill="#E15C9B" opacity="0.15" />
      <circle cx="4350" cy="2180" r="9" fill="#E15C9B" />
    </svg>
  );
}

function SceneCopy({ scene, index, total }: { scene: OverviewScene; index: number; total: number }) {
  const Icon = sceneIcons[scene.id];
  return (
    <div className="world-copy" style={{ "--copy-accent": scene.accent } as React.CSSProperties}>
      <div className="world-copy-topline">
        <span className="world-copy-kicker"><Icon size={16} /> {scene.eyebrow}</span>
        <span className="world-copy-counter">0{index + 1} / 0{total}</span>
      </div>
      <h1>{scene.title}</h1>
      <p>{scene.body}</p>
      <div className="world-copy-tags">
        {scene.tags.map((tag) => <span key={tag}>{tag}</span>)}
      </div>
      {scene.route && scene.ctaLabel && (
        <Link className="world-copy-cta" to={scene.route}>
          {scene.ctaLabel}
          <ArrowRight size={17} />
        </Link>
      )}
    </div>
  );
}

export function ScrollWorld({ companyCount, companyLoadFailed = false }: ScrollWorldProps) {
  const cameraStops = useMemo(() => OVERVIEW_SCENES.map((scene) => scene.camera), []);
  const { sectionRef, viewportRef, worldRef, activeIndex, progress } = useScrollCamera({
    stops: cameraStops,
    worldWidth: OVERVIEW_WORLD.width,
    worldHeight: OVERVIEW_WORLD.height,
  });
  const activeScene = OVERVIEW_SCENES[activeIndex] || OVERVIEW_SCENES[0];

  const jumpToScene = (index: number) => {
    const section = sectionRef.current;
    if (!section) return;
    const maxScroll = Math.max(0, section.offsetHeight - window.innerHeight);
    const sectionTop = window.scrollY + section.getBoundingClientRect().top;
    const target = sectionTop + (maxScroll * index) / Math.max(1, OVERVIEW_SCENES.length - 1);
    window.scrollTo({ top: target, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  };

  return (
    <section ref={sectionRef} className="scroll-world-shell" aria-label="StockSense interactive overview">
      <div ref={viewportRef} className="scroll-world-sticky">
        <div className="scroll-world-atmosphere" aria-hidden="true">
          <span className="atmosphere-orb atmosphere-orb-one" />
          <span className="atmosphere-orb atmosphere-orb-two" />
          <span className="atmosphere-line atmosphere-line-one" />
          <span className="atmosphere-line atmosphere-line-two" />
        </div>

        <header className="world-header">
          <Link to="/overview" className="world-brand" aria-label="StockSense overview">
            <img src="/stocksense-logo.png" alt="" />
            <span>StockSense</span>
          </Link>
          <div className="world-header-facts" aria-label="System facts">
            <span><strong>{companyCount ?? "—"}</strong> supported companies</span>
            <span><strong>3</strong> forecast horizons</span>
            <span><strong>2</strong> evidence streams</span>
          </div>
          <div className="world-header-status"><span className="status-dot" /> Educational decision support</div>
        </header>

        <div ref={worldRef} className="scroll-world-canvas" style={{ width: OVERVIEW_WORLD.width, height: OVERVIEW_WORLD.height }}>
          <ConnectorPath />
          {OVERVIEW_SCENES.map((scene, index) => <ScenePin key={scene.id} scene={scene} index={index} active={activeIndex === index} />)}
        </div>

        <div className="world-copy-wrap">
          <SceneCopy scene={activeScene} index={activeIndex} total={OVERVIEW_SCENES.length} />
        </div>

        <nav className="world-rail" aria-label="Overview journey">
          <div className="world-rail-track" aria-hidden="true"><span style={{ transform: `scaleX(${progress})` }} /></div>
          {OVERVIEW_SCENES.map((scene, index) => (
            <button key={scene.id} className={`world-rail-stop ${activeIndex === index ? "is-active" : ""}`} onClick={() => jumpToScene(index)} aria-label={`Go to ${scene.label}`} aria-current={activeIndex === index ? "step" : undefined}>
              <span className="world-rail-dot" style={{ "--rail-accent": scene.accent } as React.CSSProperties}>{index + 1}</span>
              <span>{scene.label}</span>
            </button>
          ))}
        </nav>

        <div className="world-scroll-hint" aria-hidden="true"><ArrowDown size={15} /> Scroll to travel through StockSense</div>
        <div className="world-boundary-note"><CircleHelp size={15} /><span><strong>Educational view:</strong> signals are evidence to explore, not personalised buy or sell advice.</span></div>

        {companyLoadFailed && <div className="world-data-note">Supported company details will appear when the data service is available.</div>}
      </div>
    </section>
  );
}
