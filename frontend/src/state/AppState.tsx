import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ChatMessage, DashboardAnalysis, Horizon } from "../types";

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
  chatMessages: ChatMessage[];
  setChatMessages: (messages: ChatMessage[]) => void;
  clearChat: () => void;
}

const DEFAULT_TICKER = "1155.KL";
const StateContext = createContext<AppStateValue | null>(null);

function sessionValue<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function useSessionState<T>(key: string, fallback: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => sessionValue(key, fallback));
  useEffect(() => {
    try {
      sessionStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Session state is an enhancement; the page remains usable if storage is blocked.
    }
  }, [key, value]);
  return [value, setValue];
}

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [selectedTicker, setSelectedTicker] = useSessionState("stocksense:selectedTicker", DEFAULT_TICKER);
  const [outlookTicker, setOutlookTicker] = useSessionState("stocksense:outlookTicker", DEFAULT_TICKER);
  const [outlookHorizon, setOutlookHorizon] = useSessionState<Horizon>("stocksense:outlookHorizon", 1);
  const [snapshotTicker, setSnapshotTicker] = useSessionState("stocksense:snapshotTicker", DEFAULT_TICKER);
  const [newsTicker, setNewsTicker] = useSessionState("stocksense:newsTicker", DEFAULT_TICKER);
  const [newsHorizon, setNewsHorizon] = useSessionState<Horizon>("stocksense:newsHorizon", 1);
  const [askTicker, setAskTicker] = useSessionState("stocksense:askTicker", DEFAULT_TICKER);
  const [askHorizon, setAskHorizon] = useSessionState<Horizon>("stocksense:askHorizon", 1);
  const [comparisonTickers, setComparisonTickers] = useSessionState<string[]>("stocksense:comparisonTickers", [DEFAULT_TICKER, "1295.KL"]);
  const [comparisonHorizon, setComparisonHorizon] = useSessionState<Horizon>("stocksense:comparisonHorizon", 1);
  const [analysis, setAnalysis] = useState<DashboardAnalysis | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(() => sessionValue("stocksense:chatMessages", []));

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
    chatMessages,
    setChatMessages,
    clearChat: () => setChatMessages([]),
  }), [selectedTicker, setSelectedTicker, outlookTicker, setOutlookTicker, outlookHorizon, setOutlookHorizon, snapshotTicker, setSnapshotTicker, newsTicker, setNewsTicker, newsHorizon, setNewsHorizon, askTicker, setAskTicker, askHorizon, setAskHorizon, comparisonTickers, setComparisonTickers, comparisonHorizon, setComparisonHorizon, analysis, chatMessages, setChatMessages]);

  return <StateContext.Provider value={value}>{children}</StateContext.Provider>;
}

export function useAppState(): AppStateValue {
  const context = useContext(StateContext);
  if (!context) throw new Error("useAppState must be used inside AppStateProvider");
  return context;
}
