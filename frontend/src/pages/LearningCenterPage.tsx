import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BarChart3,
  BookOpen,
  CalendarDays,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Clock3,
  ExternalLink,
  GitBranch,
  Info,
  LineChart,
  ListChecks,
  MessageCircle,
  RotateCcw,
  Scale,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingUp,
  Volume2,
} from "lucide-react";
import { PageHeader } from "../components/PageHeader";
import { useAppState } from "../state/AppState";
import { GLOSSARY_TERMS, LEARNING_MODULES, type LearningModule, type LearningModuleId } from "../data/learningCenterContent";

const PROGRESS_KEY = "stocksense:learning-progress:v1";
const ICONS = { evidence: CalendarDays, market: Activity, model: Target, news: Volume2, signals: GitBranch, compare: Scale };

interface ProgressState {
  current: LearningModuleId;
  completed: LearningModuleId[];
  answers: Record<string, number>;
}

const DEFAULT_PROGRESS: ProgressState = { current: "evidence", completed: [], answers: {} };

function readProgress(): ProgressState {
  try {
    const parsed = JSON.parse(localStorage.getItem(PROGRESS_KEY) || "null") as Partial<ProgressState> | null;
    const current = LEARNING_MODULES.some((module) => module.id === parsed?.current) ? parsed?.current as LearningModuleId : DEFAULT_PROGRESS.current;
    const completed = Array.isArray(parsed?.completed) ? parsed.completed.filter((id): id is LearningModuleId => LEARNING_MODULES.some((module) => module.id === id)) : [];
    const answers = parsed?.answers && typeof parsed.answers === "object" ? parsed.answers as Record<string, number> : {};
    return { current, completed, answers };
  } catch {
    return DEFAULT_PROGRESS;
  }
}

function IconFor({ module, size = 18 }: { module: LearningModule; size?: number }) {
  const Icon = ICONS[module.id];
  return <Icon size={size} aria-hidden="true" />;
}

function DemoVisual({ module, thresholdProbability, setThresholdProbability }: { module: LearningModule; thresholdProbability: number; setThresholdProbability: (value: number) => void }) {
  if (module.visual === "calendar") {
    const sessions = [
      { count: "1", day: "Wed", date: "22" },
      { count: "2", day: "Thu", date: "23" },
      { count: "3", day: "Fri", date: "24" },
      { count: "4", day: "Mon", date: "27" },
      { count: "5", day: "Tue", date: "28" },
    ];
    return <div className="lesson-visual calendar-lab">
      <div className="visual-label"><CalendarDays size={16} /> Count sessions, not calendar days</div>
      <div className="worked-example-heading"><span>Example</span><strong>5-trading-day horizon</strong></div>
      <div className="calendar-journey">
        <div className="calendar-start"><small>Observation</small><strong>Tue 21 Jul</strong><span>Latest complete session</span></div>
        <ArrowRight size={16} aria-hidden="true" />
        <div className="calendar-session-list">
          {sessions.slice(0, 3).map((session) => <span key={session.date}><b>{session.count}</b><small>{session.day}</small><strong>{session.date}</strong></span>)}
          <span className="calendar-weekend"><b>×</b><small>Sat-Sun</small><strong>Skipped</strong></span>
          {sessions.slice(3).map((session) => <span className={session.count === "5" ? "forecast-session" : ""} key={session.date}><b>{session.count}</b><small>{session.day}</small><strong>{session.date}</strong></span>)}
        </div>
      </div>
      <div className="visual-conclusion"><Clock3 size={14} /><span>The forecast asks whether the price is higher at the close of <strong>Tue 28 Jul</strong>.</span></div>
    </div>;
  }
  if (module.visual === "market") {
    return <div className="lesson-visual market-lab">
      <div className="visual-label"><LineChart size={16} /> Read price, trend and activity together</div>
      <div className="worked-example-heading"><span>Illustrative 10 sessions</span><strong>RM10.20 → RM10.90</strong></div>
      <div className="market-worked-chart" role="img" aria-label="Illustrative adjusted close rises from RM10.20 to RM10.90, above its 20-day moving average, while volume increases">
        <svg viewBox="0 0 270 112" preserveAspectRatio="none" aria-hidden="true">
          <g className="chart-grid"><line x1="0" y1="18" x2="270" y2="18" /><line x1="0" y1="52" x2="270" y2="52" /><line x1="0" y1="86" x2="270" y2="86" /></g>
          <g className="market-volume-bars"><rect x="5" y="85" width="15" height="20" /><rect x="32" y="78" width="15" height="27" /><rect x="59" y="82" width="15" height="23" /><rect x="86" y="72" width="15" height="33" /><rect x="113" y="75" width="15" height="30" /><rect x="140" y="66" width="15" height="39" /><rect x="167" y="70" width="15" height="35" /><rect x="194" y="57" width="15" height="48" /><rect x="221" y="61" width="15" height="44" /><rect x="248" y="48" width="15" height="57" /></g>
          <polyline className="market-average-path" points="6,61 34,59 62,57 90,55 118,52 146,49 174,46 202,43 230,40 262,37" />
          <polyline className="market-price-path" points="6,70 34,61 62,66 90,51 118,57 146,40 174,46 202,29 230,35 262,18" />
          <circle className="market-end-dot" cx="262" cy="18" r="4" />
        </svg>
        <span className="market-chart-start">RM10.20</span><span className="market-chart-end">RM10.90</span>
      </div>
      <div className="market-evidence-strip">
        <span><small>10-session return</small><strong className="positive-text">+6.9%</strong></span>
        <span><small>vs MA20</small><strong>Above</strong></span>
        <span><small>Volume ratio</small><strong>1.6×</strong></span>
      </div>
      <div className="visual-conclusion"><Info size={14} /><span>Rising price with above-average activity describes the recent move; it does not guarantee the next one.</span></div>
    </div>;
  }
  if (module.visual === "threshold") {
    const threshold = 0.51;
    const label = thresholdProbability >= threshold ? "Up" : "Down";
    const probabilityPercent = Math.round(thresholdProbability * 100);
    const thresholdPercent = Math.round(threshold * 100);
    const marginPoints = Math.abs(probabilityPercent - thresholdPercent);
    const relation = thresholdProbability >= threshold ? "above" : "below";
    return <div className="lesson-visual threshold-lab">
      <div className="visual-label"><Target size={16} /> See how probability becomes a label</div>
      <div className="worked-example-heading"><span>Illustrative locked rule</span><strong>Up at 51% or higher</strong></div>
      <div className={`threshold-decision ${label.toLowerCase()}`}>
        <span className="threshold-direction-icon"><TrendingUp size={18} /></span>
        <div><small>Directional result</small><strong>{label}</strong></div>
        <div className="threshold-equation"><b>{probabilityPercent}%</b><span>{label === "Up" ? "≥" : "<"}</span><b>{thresholdPercent}%</b></div>
      </div>
      <div className="threshold-scale" aria-label={`${probabilityPercent}% model probability compared with a ${thresholdPercent}% decision threshold`}>
        <div className="threshold-zone down-zone" style={{ width: `${thresholdPercent}%` }}><span>Down zone</span></div>
        <div className="threshold-zone up-zone" style={{ left: `${thresholdPercent}%`, width: `${100 - thresholdPercent}%` }}><span>Up zone</span></div>
        <i className="threshold-cutoff" style={{ left: `${thresholdPercent}%` }}><small>51% cutoff</small></i>
        <i className={`threshold-probability-marker ${label.toLowerCase()}`} style={{ left: `${probabilityPercent}%` }}><small>{probabilityPercent}% estimate</small></i>
      </div>
      <div className="threshold-evidence-strip">
        <span><small>Model estimate</small><strong>{probabilityPercent}% Up</strong></span>
        <span><small>Locked cutoff</small><strong>{thresholdPercent}%</strong></span>
        <span><small>Distance</small><strong>{marginPoints} point{marginPoints === 1 ? "" : "s"} {relation}</strong></span>
      </div>
      <label className="threshold-slider-control" htmlFor="learning-probability">
        <span>Try another probability</span>
        <output htmlFor="learning-probability">{probabilityPercent}%</output>
      </label>
      <input id="learning-probability" className="threshold-slider" type="range" min="0.35" max="0.7" step="0.01" value={thresholdProbability} onChange={(event) => setThresholdProbability(Number(event.target.value))} />
      <div className="visual-conclusion"><Info size={14} /><span><strong>{label}</strong> appears because the estimate is {marginPoints} percentage point{marginPoints === 1 ? "" : "s"} {relation} the cutoff. A small margin is fragile, not a promise.</span></div>
    </div>;
  }
  if (module.visual === "news") {
    const items = [
      { age: "1 day", recency: "0.95", confidence: "0.90", contribution: "0.86", tone: "Positive", className: "positive" },
      { age: "12 days", recency: "0.59", confidence: "0.88", contribution: "0.52", tone: "Negative", className: "negative" },
      { age: "3 days", recency: "0.76", confidence: "0.45", contribution: "0.34", tone: "Positive", className: "uncertain" },
    ];
    return <div className="lesson-visual news-lab">
      <div className="visual-label"><Volume2 size={16} /> See why one article matters more</div>
      <div className="news-formula"><span>Recency weight</span><b>×</b><span>FinBERT confidence</span><b>=</b><strong>Contribution</strong></div>
      <div className="news-weight-table">
        <div className="news-weight-head"><span>Article</span><span>Recency</span><span>Confidence</span><span>Weight</span></div>
        {items.map((item) => <div className="news-weight-example" key={`${item.age}-${item.tone}`}>
          <span><i className={item.className}>{item.tone}</i><small>{item.age} old</small></span>
          <code>{item.recency}</code><code>{item.confidence}</code><strong>{item.contribution}</strong>
        </div>)}
      </div>
      <div className="visual-conclusion"><ShieldCheck size={14} /><span>Weight controls influence. The tone label supplies direction; neither value is a stock-rise probability.</span></div>
    </div>;
  }
  if (module.visual === "signals") {
    return <div className="lesson-visual signals-lab">
      <div className="visual-label"><GitBranch size={16} /> Follow both inputs to the conclusion</div>
      <div className="signal-worked-inputs">
        <div><span className="signal-icon down"><TrendingUp size={15} /></span><small>Stock model</small><strong>58% Down</strong><em>Accepted</em></div>
        <div><span className="signal-icon negative"><Volume2 size={15} /></span><small>News tone</small><strong>-0.42 Negative</strong><em>Medium coverage</em></div>
      </div>
      <div className="signal-relation"><span /><strong>Same direction → reinforcing</strong><span /></div>
      <div className="signal-worked-output"><Sparkles size={16} /><div><small>Aggregate result</small><strong>Down · reinforcing evidence</strong></div></div>
      <div className="visual-conclusion"><Info size={14} /><span>A numerical adjustment is used only when the fusion model and both evidence gates are validated.</span></div>
    </div>;
  }
  return <div className="lesson-visual compare-lab">
    <div className="visual-label"><Scale size={16} /> Normalize first, then compare</div>
    <div className="worked-example-heading"><span>Illustrative common start</span><strong>Base = 100</strong></div>
    <div className="compare-worked-chart" role="img" aria-label="Normalized comparison where Maybank ends at 112, KLCI at 108 and Tenaga Nasional at 105">
      <svg viewBox="0 0 270 118" preserveAspectRatio="none" aria-hidden="true">
        <g className="chart-grid"><line x1="0" y1="20" x2="270" y2="20" /><line x1="0" y1="56" x2="270" y2="56" /><line x1="0" y1="94" x2="270" y2="94" /></g>
        <polyline className="compare-path maybank" points="6,94 45,80 84,86 123,63 162,68 201,42 264,26" />
        <polyline className="compare-path tnb" points="6,94 45,89 84,76 123,83 162,70 201,66 264,59" />
        <polyline className="compare-path klci" points="6,94 45,84 84,82 123,71 162,75 201,56 264,45" />
        <circle className="compare-end maybank" cx="264" cy="26" r="4" /><circle className="compare-end tnb" cx="264" cy="59" r="4" /><circle className="compare-end klci" cx="264" cy="45" r="4" />
      </svg>
      <span className="compare-base-label">100</span>
    </div>
    <div className="compare-result-list">
      <span><i className="maybank" /><b>Maybank</b><strong>112 · +12%</strong></span>
      <span><i className="klci" /><b>KLCI</b><strong>108 · +8%</strong></span>
      <span><i className="tnb" /><b>Tenaga</b><strong>105 · +5%</strong></span>
    </div>
    <div className="visual-conclusion"><Info size={14} /><span>Maybank led this example, but return alone says nothing about risk or future performance.</span></div>
  </div>;
}

function KnowledgeChecks({ module, answers, onAnswer }: { module: LearningModule; answers: Record<string, number>; onAnswer: (id: string, answer: number) => void }) {
  return <section className="learning-checks" aria-labelledby="knowledge-check-heading"><div className="learning-section-title"><div><span className="panel-kicker">Check your understanding</span><h2 id="knowledge-check-heading">Can you explain it back?</h2></div><ListChecks size={21} /></div><div className="knowledge-check-grid">{module.checks.map((check, index) => { const selected = answers[`${module.id}:${check.id}`]; const correct = selected === check.answer; return <article className={`knowledge-check ${selected !== undefined ? correct ? "correct" : "incorrect" : ""}`} key={check.id}><span className="check-number">0{index + 1}</span><h3>{check.question}</h3><div className="check-options">{check.options.map((option, optionIndex) => <button type="button" className={selected === optionIndex ? "selected" : ""} aria-pressed={selected === optionIndex} onClick={() => onAnswer(check.id, optionIndex)} key={option}>{option}</button>)}</div>{selected !== undefined && <p className="check-feedback"><span>{correct ? <Check size={14} /> : <CircleHelp size={14} />}</span>{correct ? "That is right. " : "Not quite. "}{check.explanation}</p>}</article>; })}</div></section>;
}

export function LearningCenterPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { selectedTicker, setSnapshotTicker, setOutlookTicker, setNewsTicker, setAskTicker, comparisonTickers, setComparisonTickers } = useAppState();
  const [progress, setProgress] = useState<ProgressState>(readProgress);
  const [thresholdProbability, setThresholdProbability] = useState(0.54);
  const [glossaryQuery, setGlossaryQuery] = useState("");
  const currentId = searchParams.get("lesson") as LearningModuleId | null;
  const activeId = LEARNING_MODULES.some((module) => module.id === currentId) ? currentId as LearningModuleId : progress.current;
  const activeIndex = LEARNING_MODULES.findIndex((module) => module.id === activeId);
  const module = LEARNING_MODULES[activeIndex] || LEARNING_MODULES[0];
  const filteredGlossary = useMemo(() => GLOSSARY_TERMS.filter((item) => `${item.term} ${item.definition}`.toLowerCase().includes(glossaryQuery.toLowerCase().trim())), [glossaryQuery]);
  const completion = Math.round((progress.completed.length / LEARNING_MODULES.length) * 100);

  useEffect(() => {
    try { localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress)); } catch { /* Progress is an enhancement. */ }
  }, [progress]);

  useEffect(() => {
    if (!currentId || currentId !== activeId) setSearchParams({ lesson: activeId }, { replace: true });
  }, [activeId, currentId, setSearchParams]);

  const selectModule = (id: LearningModuleId) => {
    setProgress((current) => ({ ...current, current: id }));
    setSearchParams({ lesson: id });
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const answerCheck = (checkId: string, answer: number) => {
    setProgress((current) => {
      const answers = { ...current.answers, [`${module.id}:${checkId}`]: answer };
      const complete = module.checks.every((check) => answers[`${module.id}:${check.id}`] === check.answer);
      return { ...current, answers, completed: complete && !current.completed.includes(module.id) ? [...current.completed, module.id] : current.completed };
    });
  };

  const handoff = () => {
    if (module.handoff.path === "/market-snapshot") setSnapshotTicker(selectedTicker);
    if (module.handoff.path === "/stock-outlook") setOutlookTicker(selectedTicker);
    if (module.handoff.path === "/news") setNewsTicker(selectedTicker);
    if (module.handoff.path === "/ask-learn") setAskTicker(selectedTicker);
    if (module.handoff.path === "/compare") {
      const retained = comparisonTickers.filter((ticker) => ticker !== selectedTicker);
      setComparisonTickers([selectedTicker, ...retained].slice(0, 5));
    }
    const query = module.handoff.prompt ? `?prompt=${encodeURIComponent(module.handoff.prompt)}` : "";
    navigate(`${module.handoff.path}${query}`);
  };

  const resetProgress = () => {
    if (window.confirm("Reset your Learning Center progress?")) setProgress(DEFAULT_PROGRESS);
  };

  return <div className="learning-center-page">
    <PageHeader eyebrow="Learn" title="The StockSense field guide" description="A short, visual route from market evidence to a careful interpretation. Learn a concept, try a small example, then apply it in StockSense." />

    <div className="learning-center-layout">
      <aside className="learning-map" aria-label="Learning modules"><div className="learning-map-heading"><strong>Learning route</strong></div><div className="learning-route-progress" aria-label={`${completion}% complete`}><div className="learning-route-progress-ring" style={{ "--progress": `${completion * 3.6}deg` } as CSSProperties}><strong>{completion}%</strong></div><div><span className="panel-kicker">Your progress</span><strong>{progress.completed.length} of {LEARNING_MODULES.length} complete</strong></div></div><button className="learning-reset-progress" type="button" onClick={resetProgress}><RotateCcw size={14} /> Reset progress</button><div className="learning-map-line" />{LEARNING_MODULES.map((item) => { const ItemIcon = ICONS[item.id]; const done = progress.completed.includes(item.id); return <button type="button" className={`learning-map-item ${item.id === module.id ? "active" : ""} ${done ? "done" : ""}`} onClick={() => selectModule(item.id)} key={item.id}><span className="learning-map-dot" style={{ "--module-accent": item.accent } as CSSProperties}>{done ? <Check size={16} /> : <ItemIcon size={18} />}</span><span className="learning-map-label"><small className="learning-map-number">{item.number}.</small><span className="learning-map-name">{item.eyebrow}</span></span></button>; })}</aside>

      <main className="learning-lesson" key={module.id}>
        <header className="learning-lesson-header"><div><span className="lesson-number" style={{ "--module-accent": module.accent } as CSSProperties}>{module.number}</span><span className="panel-kicker">{module.eyebrow}</span><h2>{module.title}</h2><p>{module.summary}</p></div><div className="lesson-icon" style={{ "--module-accent": module.accent } as CSSProperties}><IconFor module={module} size={24} /></div></header>
        <div className="learning-lesson-body"><div className="learning-lesson-copy">{module.sections.map((section) => <section className="lesson-copy-section" key={section.heading}><h3>{section.heading}</h3><p>{section.body}</p></section>)}<div className="lesson-takeaways"><span className="panel-kicker">Keep this in your notebook</span>{module.takeaways.map((takeaway) => <span key={takeaway}><Check size={14} />{takeaway}</span>)}</div></div><DemoVisual module={module} thresholdProbability={thresholdProbability} setThresholdProbability={setThresholdProbability} /></div>
        <KnowledgeChecks module={module} answers={progress.answers} onAnswer={answerCheck} />
        <div className="learning-lesson-footer"><button type="button" className="secondary-button" onClick={() => selectModule(LEARNING_MODULES[Math.max(0, activeIndex - 1)].id)} disabled={activeIndex === 0}><ChevronLeft size={16} /> Previous</button><button type="button" className="learning-handoff" onClick={handoff}><ExternalLink size={15} /> {module.handoff.label}</button><button type="button" className="secondary-button" onClick={() => selectModule(LEARNING_MODULES[Math.min(LEARNING_MODULES.length - 1, activeIndex + 1)].id)} disabled={activeIndex === LEARNING_MODULES.length - 1}>Next <ChevronRight size={16} /></button></div>
      </main>

      <aside className="learning-glossary"><div className="learning-glossary-heading"><BookOpen size={18} /><div><span className="panel-kicker">Vocabulary</span><strong>Words you will see</strong><small>{filteredGlossary.length} of {GLOSSARY_TERMS.length} terms</small></div></div><label className="glossary-search"><Search size={15} /><input value={glossaryQuery} onChange={(event) => setGlossaryQuery(event.target.value)} placeholder="Find a term" aria-label="Search glossary" /></label><div className="glossary-list">{filteredGlossary.map((item) => <details key={item.term}><summary>{item.term}</summary><p>{item.definition}</p></details>)}{!filteredGlossary.length && <p className="glossary-empty">No matching term yet. Ask &amp; Learn can explain it in plain language.</p>}</div><button className="glossary-ask" type="button" onClick={() => navigate(`/ask-learn?prompt=${encodeURIComponent(`Explain ${glossaryQuery || "an unfamiliar StockSense term"} in simple language.`)}`)}><MessageCircle size={15} /> Ask the tutor</button></aside>
    </div>

    <section className="learning-performance-preview"><div><span className="panel-kicker">Model evidence lab</span><h2>See how StockSense was tested.</h2><p>Open the dedicated dashboard to compare champions, baselines, chronological folds, class errors and calibration without confusing historical evaluation with a live forecast.</p></div><button className="learning-handoff" type="button" onClick={() => navigate("/model-performance")}><BarChart3 size={17} /> Open Model Performance <ArrowRight size={16} /></button></section>

    <section className="learning-center-footer-card"><div><span className="panel-kicker">Before you interpret</span><h2>Use the same six questions every time.</h2></div><ol><li><strong>When?</strong><span>Observation and freshness</span></li><li><strong>What?</strong><span>Market movement and risk</span></li><li><strong>How sure?</strong><span>Model status and threshold</span></li><li><strong>What evidence?</strong><span>News relevance and coverage</span></li><li><strong>Do they agree?</strong><span>Signal relationship</span></li><li><strong>What next?</strong><span>Ask, compare and investigate</span></li></ol></section>
  </div>;
}
