import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { ArrowUp, BookOpen, CalendarDays, CheckCircle2, CircleHelp, Clock3, FileText, GraduationCap, LineChart, RefreshCw, Sparkles, Target, TrendingDown, TrendingUp } from "lucide-react";
import { askStockSense, getCompanies, prepareLearningContext } from "../api/client";
import { LoadingState, ErrorState } from "../components/AsyncState";
import { CompanyLogo } from "../components/CompanyLogo";
import { CompanySelector } from "../components/CompanySelector";
import { HorizonSelector } from "../components/HorizonSelector";
import { PageHeader } from "../components/PageHeader";
import { TutorAnswer, tutorMessageFromResponse } from "../components/TutorAnswer";
import { useAppState } from "../state/AppState";
import type { ChatMessage, ChatResponse, Company, LearningContext, NewsArticle } from "../types";

const LEARNING_PATHS = [
  { icon: Target, label: "Understand this outlook", prompt: "Explain the current stock outlook in plain language. What should a beginner notice first?" },
  { icon: LineChart, label: "Read the market", prompt: "What do the current price, recent performance and activity tell me, and what do they not tell me?" },
  { icon: FileText, label: "Read the news", prompt: "Which news evidence matters for this horizon, and how did confidence and recency affect the tone?" },
  { icon: Sparkles, label: "Connect the signals", prompt: "How do the stock trend and news sentiment complement or conflict with each other?" },
  { icon: BookOpen, label: "Learn a term", prompt: "Teach me probability, decision threshold, volatility and sentiment confidence with a simple example." },
];

const FAQ_PATHS = [
  { icon: CircleHelp, label: "What does probability mean?", prompt: "What does the probability in this result mean in simple language?" },
  { icon: CircleHelp, label: "Why is the model experimental?", prompt: "Why is this model labelled experimental or accepted?" },
  { icon: CircleHelp, label: "Why can news be unavailable?", prompt: "Why can news sentiment be unavailable or have low coverage?" },
  { icon: CircleHelp, label: "How are articles weighted?", prompt: "How do recency and sentiment confidence affect the news result?" },
];

function formatDate(value?: string | null, includeTime = false): string {
  if (!value) return "Unavailable";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en-MY", includeTime ? { dateStyle: "medium", timeStyle: "short" } : { dateStyle: "medium" }).format(parsed);
}

function formatPercent(value?: number | null): string {
  return value == null || !Number.isFinite(value) ? "Unavailable" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function statusLabel(context: LearningContext | null, matches: boolean): string {
  if (!context || !matches) return "Prepare evidence to begin";
  if (context.status === "ready") return "Evidence ready";
  if (context.status === "partial") return "Evidence partially available";
  return "Concept mode";
}

function articleRows(context: LearningContext | null): NewsArticle[] {
  const articles = context?.analysis?.news?.articles || [];
  return articles.filter((article) => article.effective_included || article.automatic_included).slice(0, 5);
}

function EvidenceNotebook({ context, matches, onRefresh }: { context: LearningContext | null; matches: boolean; onRefresh: () => void }) {
  const snapshot = context?.market_snapshot;
  const analysis = context?.analysis;
  const stock = analysis?.stock;
  const news = analysis?.news?.contexts?.[String(context?.horizon_days)] || null;
  const combined = analysis?.combined_outlook;
  const articles = articleRows(context);
  return (
    <aside className="learning-notebook" aria-label="Prepared evidence">
      <div className="notebook-heading">
        <div><span className="panel-kicker">Evidence notebook</span><h2>{context && matches ? context.company_name : "No evidence prepared"}</h2></div>
        <button className="icon-button" type="button" title="Refresh evidence" onClick={onRefresh} disabled={!context}><RefreshCw size={16} /></button>
      </div>
      {!context || !matches ? (
        <div className="notebook-empty"><BookOpen size={22} /><p>Prepare a company and horizon to keep the tutor grounded in your actual market, model and news evidence.</p></div>
      ) : <>
        <div className={`notebook-status ${context.status}`}><CheckCircle2 size={15} /><span>{context.status === "ready" ? "Prepared evidence" : "Partial evidence"}</span><small>until {formatDate(context.expires_at, true)}</small></div>
        <section className="notebook-section">
          <div className="notebook-section-title"><LineChart size={15} /><span>Market now</span></div>
          <div className="notebook-metric"><strong>{snapshot?.price == null ? "Unavailable" : `RM ${snapshot.price.toFixed(2)}`}</strong><span className={snapshot?.change_pct != null && snapshot.change_pct >= 0 ? "positive-text" : "negative-text"}>{formatPercent(snapshot?.change_pct)} today</span></div>
          <p className="notebook-meta"><CalendarDays size={13} /> Observed {formatDate(snapshot?.as_of_date)} · {snapshot?.freshness_status || "unknown"}</p>
        </section>
        <section className="notebook-section">
          <div className="notebook-section-title"><Target size={15} /><span>Stock trend</span></div>
          <div className="notebook-signal-row"><span className={`signal-mini ${stock?.prediction_label === "Up" ? "up" : "down"}`}>{stock?.prediction_label === "Up" ? <TrendingUp size={15} /> : <TrendingDown size={15} />}</span><strong>{stock?.prediction_label || "Unavailable"}</strong><span>{stock ? `${(stock.probability_up * 100).toFixed(1)}% Up` : "No model value"}</span></div>
          <p className="notebook-meta"><Clock3 size={13} /> {stock?.evidence_status || "Unavailable"} · {formatDate(stock?.as_of_date)}</p>
        </section>
        <section className="notebook-section">
          <div className="notebook-section-title"><FileText size={15} /><span>News sentiment</span></div>
          <div className="notebook-metric"><strong>{news?.weighted_sentiment_score == null ? "Unavailable" : `${news.weighted_sentiment_score >= 0 ? "+" : ""}${Number(news.weighted_sentiment_score).toFixed(2)}`}</strong><span>{news?.coverage_confidence || "Unavailable"} coverage</span></div>
          <p className="notebook-meta"><FileText size={13} /> {news?.usable_article_count ?? 0} usable article(s) · {analysis?.news?.provider_mode || "No provider"}</p>
        </section>
        <section className="notebook-section notebook-synthesis">
          <div className="notebook-section-title"><Sparkles size={15} /><span>How signals relate</span></div>
          <strong>{combined?.relationship || "Unavailable"}</strong>
          <p>{combined?.explanation || "The aggregate relationship will appear after evidence is prepared."}</p>
        </section>
        <section className="notebook-section">
          <div className="notebook-section-title"><FileText size={15} /><span>Articles in this view</span></div>
          {articles.length ? <div className="notebook-articles">{articles.map((article, index) => <a href={article.url || undefined} target="_blank" rel="noreferrer" key={article.article_id || `${article.title}-${index}`}><span>{article.title || "Untitled article"}</span><small>{article.source_domain || article.source_name || article.provider || "Source unavailable"} · {formatDate(article.published_at)}</small></a>)}</div> : <p className="notebook-meta">No contributing articles are available for this horizon.</p>}
        </section>
        {context.warnings.map((warning) => <div className="notice warning" key={warning}>{warning}</div>)}
      </>}
    </aside>
  );
}

export function AskLearnPage() {
  const companies = useQuery({ queryKey: ["companies"], queryFn: ({ signal }) => getCompanies(signal), staleTime: 86400000 });
  const { askTicker, setAskTicker, askHorizon, setAskHorizon, analysis, setAnalysis, learningContext, setLearningContext, chatMessages, setChatMessages, clearChat } = useAppState();
  const [searchParams, setSearchParams] = useSearchParams();
  const [input, setInput] = useState("");
  const [promptMode, setPromptMode] = useState<"faq" | "learning">("faq");
  const inputRef = useRef<HTMLInputElement>(null);
  const lessonPrompt = searchParams.get("prompt");

  const contextMatches = Boolean(learningContext && learningContext.ticker === askTicker && learningContext.horizon_days === askHorizon);
  const company = useMemo<Company | undefined>(() => companies.data?.find((item) => item.ticker === askTicker), [companies.data, askTicker]);

  useEffect(() => {
    // Remove the former introductory tutor paragraph from sessions created by
    // the previous layout without disturbing any real conversation.
    if (chatMessages.length === 1 && chatMessages[0].id === "welcome") setChatMessages([]);
  }, [chatMessages, setChatMessages]);

  useEffect(() => {
    if (!lessonPrompt) return;
    setInput(lessonPrompt);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete("prompt");
    setSearchParams(nextParams, { replace: true });
    const focusTimer = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(focusTimer);
  }, [lessonPrompt, searchParams, setSearchParams]);

  const prepareMutation = useMutation({
    mutationFn: () => prepareLearningContext(askTicker, askHorizon, analysis?.ticker === askTicker && analysis.horizon_days === askHorizon ? analysis.analysis_id : null),
    onSuccess: (context) => { setLearningContext(context); if (context.analysis) setAnalysis(context.analysis); clearChat(); },
  });

  const chatMutation = useMutation({
    mutationFn: (messages: ChatMessage[]) => askStockSense(askTicker, askHorizon, contextMatches ? learningContext?.context_id || null : null, contextMatches ? learningContext?.analysis?.analysis_id || null : null, messages),
    onSuccess: (result: ChatResponse, messages) => {
      setChatMessages([...messages, tutorMessageFromResponse(result)]);
    },
  });

  if (companies.isLoading) return <><PageHeader eyebrow="Learn" title="Ask & Learn" description="A guided conversation for understanding StockSense evidence in plain language." /><LoadingState /></>;
  if (companies.isError || !companies.data) return <><PageHeader eyebrow="Learn" title="Ask & Learn" description="A guided conversation for understanding StockSense evidence in plain language." /><ErrorState message={companies.error?.message || "Companies could not be loaded."} /></>;

  const send = (event: FormEvent) => {
    event.preventDefault();
    const text = input.trim();
    if (!text || chatMutation.isPending) return;
    const messages = [...chatMessages, { role: "user" as const, id: `user-${Date.now()}`, content: text }];
    setChatMessages(messages);
    setInput("");
    chatMutation.mutate(messages);
  };
  const chooseFollowUp = (question: string) => { setInput(question); inputRef.current?.focus(); };
  const prepare = () => prepareMutation.mutate();

  return (
    <div className="ask-learn-page">
      <PageHeader eyebrow="Learn" title="Ask & Learn" description="A guided conversation for understanding StockSense evidence in plain language." />
      <section className="learning-context-panel">
        <div className="learning-context-main">
          <div className="learning-context-logo"><CompanyLogo company={company} className="selector-logo" /></div>
          <CompanySelector label="Company to discuss" companies={companies.data} value={askTicker} onChange={setAskTicker} />
          <button className="primary-button learning-prepare-button" type="button" onClick={prepare} disabled={prepareMutation.isPending}>{prepareMutation.isPending ? "Preparing…" : contextMatches ? "Refresh evidence" : "Prepare evidence"}</button>
        </div>
        <HorizonSelector value={askHorizon} onChange={setAskHorizon} legend="Learning horizon" />
        <div className={`learning-context-status ${contextMatches ? learningContext?.status || "ready" : "idle"}`}><span className="status-dot" />{statusLabel(learningContext, contextMatches)}{contextMatches && learningContext ? <small>Prepared {formatDate(learningContext.generated_at, true)}</small> : <small>General questions are always available</small>}</div>
      </section>
      {(prepareMutation.isError || chatMutation.isError) && <ErrorState message={(prepareMutation.error || chatMutation.error)?.message || "The tutor could not complete this request."} />}
      <div className="learning-workspace">
        <main className="learning-conversation">
          <div className="learning-conversation-heading"><div><span className="panel-kicker">Your learning trail</span><h2>Ask a question, then follow the evidence.</h2></div><button className="secondary-button" type="button" onClick={clearChat}>Clear conversation</button></div>
          {!chatMessages.some((message) => message.role === "user") && <section className="learning-prompts" aria-label="Question starters">
            <div className="learning-prompt-tabs" role="tablist" aria-label="Question starter type">
              <button type="button" role="tab" aria-selected={promptMode === "faq"} className={promptMode === "faq" ? "active" : ""} onClick={() => setPromptMode("faq")}><CircleHelp size={15} /> FAQ</button>
              <button type="button" role="tab" aria-selected={promptMode === "learning"} className={promptMode === "learning" ? "active" : ""} onClick={() => setPromptMode("learning")}><GraduationCap size={16} /> Learning paths</button>
            </div>
            <div className="learning-path-grid">{(promptMode === "faq" ? FAQ_PATHS : LEARNING_PATHS).map(({ icon: Icon, label, prompt }) => <button type="button" key={label} onClick={() => { setInput(prompt); inputRef.current?.focus(); }}><Icon size={17} /><span>{label}</span><small>Start here</small></button>)}</div>
          </section>}
          <div className="learning-thread" aria-live="polite">
            {chatMessages.map((message, index) => <article className={`learning-message ${message.role}`} key={message.id || `${message.role}-${index}`}><div className="learning-message-mark">{message.role === "user" ? "You" : <><Sparkles size={15} /><span>StockSense tutor</span></>}</div><TutorAnswer message={message} onFollowUp={chooseFollowUp} /></article>)}
            {chatMutation.isPending && <article className="learning-message assistant pending"><div className="learning-message-mark"><Sparkles size={15} /><span>StockSense tutor</span></div><div className="tutor-thinking"><span className="spinner" /> Reading the prepared evidence…</div></article>}
          </div>
          <form className="learning-composer" onSubmit={send}><label htmlFor="ask-learn-input">Your question</label><div className="learning-input-shell"><input ref={inputRef} id="ask-learn-input" value={input} onChange={(event) => setInput(event.target.value)} placeholder={contextMatches ? `Ask about ${company?.company_name || "this company"}…` : "Ask a general investing question…"} /><button className="learning-send-button" aria-label="Send question" title="Send question" type="submit" disabled={!input.trim() || chatMutation.isPending}><ArrowUp size={18} /></button></div><small>{contextMatches ? "Answers can refer to the prepared market, model and news evidence." : "Prepare evidence when you want a company-specific explanation."}</small></form>
        </main>
        <EvidenceNotebook context={learningContext} matches={contextMatches} onRefresh={prepare} />
      </div>
    </div>
  );
}
