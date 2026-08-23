import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { AggregateTutorThread, ChatMessage, DashboardAnalysis, Horizon, LearningContext } from "../types";

interface AppStateValue {
  selectedTicker: string;
  setSelectedTicker: (ticker: string) => void;
  outlookTicker: string;
  setOutlookTicker: (ticker: string) => void;
  outlookHorizon: Horizon;
  setOutlookHorizon: (horizon: Horizon) => void;
  snapshotTicker: string;
  setSnapshotTicker: (ticker: string) => void;
  newsTicker: string;
  setNewsTicker: (ticker: string) => void;
  newsHorizon: Horizon;
  setNewsHorizon: (horizon: Horizon) => void;
  askTicker: string;
  setAskTicker: (ticker: string) => void;
  askHorizon: Horizon;
  setAskHorizon: (horizon: Horizon) => void;
  comparisonTickers: string[];
  setComparisonTickers: (tickers: string[]) => void;
  comparisonHorizon: Horizon;
  setComparisonHorizon: (horizon: Horizon) => void;
  analysis: DashboardAnalysis | null;
  setAnalysis: (analysis: DashboardAnalysis | null) => void;
  learningContext: LearningContext | null;
  setLearningContext: (context: LearningContext | null) => void;
  chatMessages: ChatMessage[];
  setChatMessages: (messages: ChatMessage[]) => void;
  clearChat: () => void;
  aggregateTutorThread: AggregateTutorThread | null;
  setAggregateTutorThread: React.Dispatch<React.SetStateAction<AggregateTutorThread | null>>;
  clearAggregateTutorThread: () => void;
}

const DEFAULT_TICKER = "1155.KL";
const StateContext = createContext<AppStateValue | null>(null);

function sessionValue<T>(key: string, fallback: T, validate?: (value: unknown) => value is T): T {
  try {
    const raw = sessionStorage.getItem(key);
    if (!raw) return fallback;
    const parsed: unknown = JSON.parse(raw);
    if (validate) return validate(parsed) ? parsed : fallback;
    return parsed as T;
  } catch {
    return fallback;
  }
}

function useSessionState<T>(key: string, fallback: T, validate?: (value: unknown) => value is T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => sessionValue(key, fallback, validate));
  useEffect(() => {
    try {
      sessionStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Session state is an enhancement; the page remains usable if storage is blocked.
    }
  }, [key, value]);
  return [value, setValue];
}

const isHorizon = (value: unknown): value is Horizon => value === 1 || value === 5 || value === 10;
const isTicker = (value: unknown): value is string =>
  typeof value === "string" && /^[0-9A-Z.^-]{1,16}$/i.test(value);
const isTickerList = (value: unknown): value is string[] =>
  Array.isArray(value) &&
  value.length >= 2 &&
  value.length <= 5 &&
  value.every(isTicker) &&
  new Set(value.map((ticker) => ticker.toUpperCase())).size === value.length;
const isChatMessages = (value: unknown): value is ChatMessage[] =>
  Array.isArray(value) &&
  value.length <= 50 &&
  value.every((item) =>
    typeof item === "object" &&
    item !== null &&
    ((item as ChatMessage).role === "user" || (item as ChatMessage).role === "assistant") &&
    typeof (item as ChatMessage).content === "string"
  );

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [selectedTicker, setSelectedTicker] = useSessionState("stocksense:selectedTicker", DEFAULT_TICKER, isTicker);
  const [outlookTicker, setOutlookTicker] = useSessionState("stocksense:outlookTicker", DEFAULT_TICKER, isTicker);
  const [outlookHorizon, setOutlookHorizon] = useSessionState<Horizon>("stocksense:outlookHorizon", 1, isHorizon);
  const [snapshotTicker, setSnapshotTicker] = useSessionState("stocksense:snapshotTicker", DEFAULT_TICKER, isTicker);
  const [newsTicker, setNewsTicker] = useSessionState("stocksense:newsTicker", DEFAULT_TICKER, isTicker);
  const [newsHorizon, setNewsHorizon] = useSessionState<Horizon>("stocksense:newsHorizon", 1, isHorizon);
  const [askTicker, setAskTicker] = useSessionState("stocksense:askTicker", DEFAULT_TICKER, isTicker);
  const [askHorizon, setAskHorizon] = useSessionState<Horizon>("stocksense:askHorizon", 1, isHorizon);
  const [comparisonTickers, setComparisonTickers] = useSessionState<string[]>("stocksense:comparisonTickers", [DEFAULT_TICKER, "1295.KL"], isTickerList);
  const [comparisonHorizon, setComparisonHorizon] = useSessionState<Horizon>("stocksense:comparisonHorizon", 1, isHorizon);
  const [analysis, setAnalysis] = useState<DashboardAnalysis | null>(null);
  const [learningContext, setLearningContext] = useState<LearningContext | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(() =>
    sessionValue("stocksense:chatMessages", [], isChatMessages)
  );
  // Analysis results expire server-side after one hour, so keep this thread in
  // memory only. It persists while the learner navigates the app but cannot be
  // mistaken for current evidence after a browser reload.
  const [aggregateTutorThread, setAggregateTutorThread] = useState<AggregateTutorThread | null>(null);

  useEffect(() => {
    try {
      sessionStorage.setItem("stocksense:chatMessages", JSON.stringify(chatMessages));
    } catch {
      // Optional session persistence.
    }
  }, [chatMessages]);

  const value = useMemo<AppStateValue>(() => ({
    selectedTicker,
    setSelectedTicker,
    outlookTicker,
    setOutlookTicker: (ticker) => { setOutlookTicker(ticker); setSelectedTicker(ticker); },
    outlookHorizon,
    setOutlookHorizon,
    snapshotTicker,
    setSnapshotTicker: (ticker) => { setSnapshotTicker(ticker); setSelectedTicker(ticker); },
    newsTicker,
    setNewsTicker: (ticker) => { setNewsTicker(ticker); setSelectedTicker(ticker); },
    newsHorizon,
    setNewsHorizon,
    askTicker,
    setAskTicker: (ticker) => { setAskTicker(ticker); setSelectedTicker(ticker); },
    askHorizon,
    setAskHorizon,
    comparisonTickers,
    setComparisonTickers,
    comparisonHorizon,
    setComparisonHorizon,
    analysis,
    setAnalysis,
    learningContext,
    setLearningContext,
    chatMessages,
    setChatMessages,
    clearChat: () => setChatMessages([]),
    aggregateTutorThread,
    setAggregateTutorThread,
    clearAggregateTutorThread: () => setAggregateTutorThread(null),
  }), [selectedTicker, setSelectedTicker, outlookTicker, setOutlookTicker, outlookHorizon, setOutlookHorizon, snapshotTicker, setSnapshotTicker, newsTicker, setNewsTicker, newsHorizon, setNewsHorizon, askTicker, setAskTicker, askHorizon, setAskHorizon, comparisonTickers, setComparisonTickers, comparisonHorizon, setComparisonHorizon, analysis, learningContext, chatMessages, setChatMessages, aggregateTutorThread]);

  return <StateContext.Provider value={value}>{children}</StateContext.Provider>;
}

export function useAppState(): AppStateValue {
  const context = useContext(StateContext);
  if (!context) throw new Error("useAppState must be used inside AppStateProvider");
  return context;
}
