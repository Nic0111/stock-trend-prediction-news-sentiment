import { FormEvent, useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { askStockSense, getCompanies } from "../api/client";
import { LoadingState, ErrorState } from "../components/AsyncState";
import { CompanyLogo } from "../components/CompanyLogo";
import { CompanySelector } from "../components/CompanySelector";
import { PageHeader } from "../components/PageHeader";
import { useAppState } from "../state/AppState";
import type { ChatMessage, Horizon } from "../types";

const suggestions = ["What does probability of Up mean?", "Why can news sentiment be unavailable?", "What does experimental model mean?"];

export function AskLearnPage() {
  const companies = useQuery({ queryKey: ["companies"], queryFn: ({ signal }) => getCompanies(signal), staleTime: 86400000 });
  const { askTicker, setAskTicker, askHorizon, setAskHorizon, analysis, chatMessages, setChatMessages, clearChat } = useAppState();
  const [input, setInput] = useState("");
  useEffect(() => { if (!chatMessages.length) setChatMessages([{ role: "assistant", content: "Hi — I’m StockSense’s tutor. I can explain probabilities, sentiment, data freshness and model limitations. What would you like to learn?" }]); }, [chatMessages.length, setChatMessages]);
  const mutation = useMutation({ mutationFn: (messages: ChatMessage[]) => askStockSense(askTicker, askHorizon, analysis?.ticker === askTicker && analysis.horizon_days === askHorizon ? analysis.analysis_id : null, messages), onSuccess: (result, messages) => setChatMessages([...messages, { role: "assistant", content: result.fallback_used ? `_Answered by fallback model ${result.model}._\n\n${result.content}` : result.content }]) });
  if (companies.isLoading) return <><PageHeader eyebrow="Learn" title="Ask & Learn" description="A conversational space for understanding the model, the data and investing vocabulary." /><LoadingState /></>;
  if (companies.isError || !companies.data) return <><PageHeader eyebrow="Learn" title="Ask & Learn" description="A conversational space for understanding the model, the data and investing vocabulary." /><ErrorState message={companies.error?.message || "Companies could not be loaded."} /></>;
  const company = companies.data.find((item) => item.ticker === askTicker);
  const send = (event: FormEvent) => { event.preventDefault(); const text = input.trim(); if (!text || mutation.isPending) return; const messages = [...chatMessages, { role: "user" as const, content: text }]; setChatMessages(messages); setInput(""); mutation.mutate(messages); };
  return <><PageHeader eyebrow="Learn" title="Ask & Learn" description="A conversational space for understanding the model, the data and investing vocabulary." /><div className="chat-intro"><strong>Ask StockSense.</strong><br />Gemma 4 answers in a beginner-friendly conversational style using the selected ticker, horizon and any loaded dashboard evidence.</div><section className="selector-panel compact"><CompanyLogo company={company} className="selector-logo" /><CompanySelector label="Conversation ticker" companies={companies.data} value={askTicker} onChange={setAskTicker} /><label className="field-label"><span>Conversation horizon</span><select value={askHorizon} onChange={(event) => setAskHorizon(Number(event.target.value) as Horizon)}><option value={1}>1 trading day(s)</option><option value={5}>5 trading day(s)</option><option value={10}>10 trading day(s)</option></select></label><button className="secondary-button" onClick={clearChat}>Clear chat</button></section><div className="chat-thread">{chatMessages.map((message, index) => <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}><span className="chat-avatar">{message.role === "user" ? "◎" : "✦"}</span><div><strong>{message.role === "user" ? "You" : "StockSense"}</strong><p>{message.content}</p></div></div>)}</div><div className="suggestion-row">{suggestions.map((suggestion) => <button key={suggestion} className="suggestion-button" onClick={() => setInput(suggestion)}>{suggestion}</button>)}</div><form className="chat-form" onSubmit={send}><input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about the prediction, news or investing terms…" /><button className="primary-button" type="submit" disabled={!input.trim() || mutation.isPending}>{mutation.isPending ? "Thinking…" : "Send"}</button></form>{mutation.isError && <ErrorState message={mutation.error.message} />}</>;
}
