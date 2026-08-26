import { type FormEvent, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowUp, RefreshCw, Sparkles } from "lucide-react";
import { ApiClientError, askAggregateExplanation } from "../api/client";
import { TutorAnswer, tutorMessageFromResponse } from "./TutorAnswer";
import { useAppState } from "../state/AppState";
import type { ChatMessage, DashboardAnalysis } from "../types";

// Keep the user-facing request short. The detailed response contract belongs
// in the API system instruction; putting it here encourages weaker models to
// echo the prompt into the explanation panel.
const AGGREGATE_EXPLANATION_PROMPT = "Explain this StockSense result in plain language using only the displayed stock and news evidence.";

const STARTER_QUESTIONS = [
  "What does this probability mean in plain language?",
  "How did the news evidence affect this result?",
  "What should make me cautious about this result?",
];

const MAX_VISIBLE_THREAD_MESSAGES = 24;

function aggregateEvidenceKey(analysis: DashboardAnalysis): string {
  const outlook = analysis.combined_outlook;
  const context = analysis.news?.contexts[String(analysis.horizon_days)];
  const articleState = (analysis.news?.articles || [])
    .map((article, index) => [
      article.article_id || article.url || `article-${index}`,
      article.effective_included ? "included" : "excluded",
      article.user_override || "automatic",
      (article.contributes_to_horizons || []).join(","),
    ].join("~"))
    .sort()
    .join("|");
  return [
    analysis.analysis_id,
    outlook?.mode,
    outlook?.direction,
    outlook?.probability_up,
    outlook?.probability_kind,
    outlook?.validation_status,
    outlook?.relationship,
    outlook?.news_score,
    outlook?.usable_article_count,
    context?.weighted_sentiment_score,
    context?.coverage_confidence,
    context?.usable_article_count,
    articleState,
  ].map((value) => String(value ?? "")).join("::");
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function analysisExpired(error: unknown): boolean {
  return error instanceof ApiClientError && error.code === "ANALYSIS_EXPIRED";
}

export function AggregateExplanationPanel({ analysis }: { analysis: DashboardAnalysis }) {
  const {
    aggregateTutorThread,
    setAggregateTutorThread,
    clearAggregateTutorThread,
  } = useAppState();
  const [input, setInput] = useState("");
  const [lastFailedPrompt, setLastFailedPrompt] = useState<string | null>(null);
  const evidenceKey = useMemo(() => aggregateEvidenceKey(analysis), [analysis]);
  const aggregateLabel = analysis.combined_outlook?.mode === "validated_fusion" ? "Fusion" : "Stock-Led";

  useEffect(() => {
    // A correction changes the evidence revision but keeps the same analysis
    // identifier. Never display a continuation that refers to the old article
    // choices beside the recalculated result.
    if (
      aggregateTutorThread
      && aggregateTutorThread.analysisId === analysis.analysis_id
      && aggregateTutorThread.evidenceKey !== evidenceKey
    ) {
      clearAggregateTutorThread();
      setLastFailedPrompt(null);
    }
  }, [aggregateTutorThread, analysis.analysis_id, clearAggregateTutorThread, evidenceKey]);

  const initialExplanation = useQuery({
    queryKey: ["aggregate-explanation", analysis.analysis_id, evidenceKey],
    queryFn: ({ signal }) => askAggregateExplanation(
      analysis.ticker,
      analysis.horizon_days as 1 | 5 | 10,
      analysis.analysis_id,
      [{ role: "user", content: AGGREGATE_EXPLANATION_PROMPT }],
      signal,
    ),
    staleTime: Infinity,
    retry: false,
    refetchOnWindowFocus: false,
  });

  const initialMessage = useMemo(
    () => initialExplanation.data ? tutorMessageFromResponse(initialExplanation.data) : null,
    [initialExplanation.data],
  );
  const visibleMessages = (
    aggregateTutorThread?.analysisId === analysis.analysis_id
    && aggregateTutorThread.evidenceKey === evidenceKey
  ) ? aggregateTutorThread.messages : [];

  const followUpMutation = useMutation({
    mutationFn: (messages: ChatMessage[]) => askAggregateExplanation(
      analysis.ticker,
      analysis.horizon_days as 1 | 5 | 10,
      analysis.analysis_id,
      messages,
    ),
    onSuccess: (result) => {
      const assistantMessage = tutorMessageFromResponse(result);
      setAggregateTutorThread((current) => {
        if (
          !current
          || current.analysisId !== analysis.analysis_id
          || current.evidenceKey !== evidenceKey
        ) return current;
        return {
          ...current,
          messages: [...current.messages, assistantMessage].slice(-MAX_VISIBLE_THREAD_MESSAGES),
        };
      });
      setLastFailedPrompt(null);
    },
    onError: (_error, messages) => {
      const lastUserMessage = [...messages].reverse().find((message) => message.role === "user" && message.content !== AGGREGATE_EXPLANATION_PROMPT);
      setLastFailedPrompt(lastUserMessage?.content || null);
    },
  });

  const askFollowUp = (rawQuestion: string) => {
    const question = rawQuestion.trim();
    if (!question || followUpMutation.isPending || !initialMessage) return;
    const userMessage: ChatMessage = { role: "user", id: `aggregate-user-${Date.now()}`, content: question };
    const nextVisibleMessages = [...visibleMessages, userMessage].slice(-MAX_VISIBLE_THREAD_MESSAGES);
    const requestMessages: ChatMessage[] = [
      { role: "user", content: AGGREGATE_EXPLANATION_PROMPT },
      initialMessage,
      ...nextVisibleMessages.slice(-(MAX_VISIBLE_THREAD_MESSAGES - 2)),
    ];
    setAggregateTutorThread({
      analysisId: analysis.analysis_id,
      evidenceKey,
      messages: nextVisibleMessages,
    });
    setInput("");
    followUpMutation.mutate(requestMessages);
  };

  const send = (event: FormEvent) => {
    event.preventDefault();
    askFollowUp(input);
  };
  const error = initialExplanation.error || followUpMutation.error;
  const canRetryInitial = initialExplanation.isError && !analysisExpired(initialExplanation.error);

  return (
    <section className="aggregate-explanation-panel" aria-labelledby="aggregate-explanation-title">
      <header className="aggregate-explanation-header">
        <div className="aggregate-explanation-title-wrap">
          <span className="aggregate-explanation-icon" aria-hidden="true"><Sparkles size={18} /></span>
          <div>
            <span className="panel-kicker">Evidence walkthrough</span>
            <h3 id="aggregate-explanation-title">Why this result?</h3>
            <p>{aggregateLabel === "Fusion"
              ? "The stock trend and news sentiment are combined into the displayed probability."
              : "News is shown alongside the stock trend, even when it is not eligible to change the displayed probability."}</p>
          </div>
        </div>
        <span className="aggregate-explanation-status">LLM explanation · {aggregateLabel}</span>
      </header>

      <div className="aggregate-explanation-sources" aria-label="Inputs to this explanation">
        <span>Stock perspective</span><i aria-hidden="true" /><span>News perspective</span>
      </div>

      <div className="aggregate-explanation-thread" aria-live="polite">
        {initialExplanation.isLoading && (
          <div className="aggregate-thinking"><span className="spinner" /> Connecting the stock and news evidence…</div>
        )}
        {initialExplanation.isError && (
          <div className="aggregate-explanation-error" role="alert">
            <p>{analysisExpired(initialExplanation.error)
              ? "This analysis is no longer available. Run Analyze again to create a fresh explanation."
              : errorMessage(initialExplanation.error, "The explanation could not be generated right now.")}</p>
            {canRetryInitial && <button className="secondary-button" type="button" onClick={() => initialExplanation.refetch()}><RefreshCw size={15} /> Try again</button>}
          </div>
        )}
        {initialMessage && (
          <article className="aggregate-message assistant">
            <div className="aggregate-message-mark"><Sparkles size={15} /><span>StockSense explanation</span></div>
            <TutorAnswer message={initialMessage} onFollowUp={askFollowUp} paragraphOnly />
          </article>
        )}
        {visibleMessages.map((message, index) => (
          <article className={`aggregate-message ${message.role}`} key={message.id || `${message.role}-${index}`}>
            <div className="aggregate-message-mark">{message.role === "user" ? "You" : <><Sparkles size={15} /><span>StockSense explanation</span></>}</div>
            {message.role === "assistant" ? <TutorAnswer message={message} onFollowUp={askFollowUp} paragraphOnly /> : <p>{message.content}</p>}
          </article>
        ))}
        {followUpMutation.isPending && <div className="aggregate-thinking"><span className="spinner" /> Checking the same analysis evidence…</div>}
        {followUpMutation.isError && !analysisExpired(followUpMutation.error) && (
          <div className="aggregate-explanation-error compact" role="alert">
            <p>{errorMessage(followUpMutation.error, "The follow-up could not be completed.")}</p>
            {lastFailedPrompt && <button className="secondary-button" type="button" onClick={() => askFollowUp(lastFailedPrompt)}><RefreshCw size={15} /> Retry question</button>}
          </div>
        )}
      </div>

      {initialMessage && !analysisExpired(error) && (
        <>
          {visibleMessages.length === 0 && (
            <div className="aggregate-starter-questions" aria-label="Suggested questions">
              {STARTER_QUESTIONS.map((question) => <button type="button" key={question} onClick={() => askFollowUp(question)}>{question}</button>)}
            </div>
          )}
          <form className="aggregate-composer" onSubmit={send}>
            <label htmlFor="aggregate-question">Ask about this result</label>
            <div className="aggregate-input-shell">
              <input
                id="aggregate-question"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Ask what the signals mean together…"
                disabled={followUpMutation.isPending}
              />
              <button type="submit" aria-label="Send question" title="Send question" disabled={!input.trim() || followUpMutation.isPending}><ArrowUp size={18} /></button>
            </div>
            <small>Questions stay connected to this company, horizon and current article choices.</small>
          </form>
        </>
      )}
    </section>
  );
}
