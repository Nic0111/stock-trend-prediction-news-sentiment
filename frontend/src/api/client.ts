import type {
  AnalysisRequest,
  ChatMessage,
  ChatResponse,
  Company,
  ComparisonResult,
  DashboardAnalysis,
  Horizon,
  LearningContext,
  MarketSnapshot,
  ModelDiagnostics,
  ModelPerformance,
  NewsContext,
  Prediction,
  SystemStatus,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";

export class ApiClientError extends Error {
  code?: string;
  retryable: boolean;
  status: number;

  constructor(message: string, status: number, code?: string, retryable = false) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
  });
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const envelope = body as { detail?: { code?: string; message?: string; retryable?: boolean }; error?: { code?: string; message?: string; retryable?: boolean } } | null;
    const detail = envelope?.error || envelope?.detail;
    throw new ApiClientError(
      detail?.message || `Request failed with status ${response.status}.`,
      response.status,
      detail?.code,
      detail?.retryable,
    );
  }
  return body as T;
}

export function getCompanies(signal?: AbortSignal): Promise<Company[]> {
  return request<Company[]>("/companies", { signal });
}

export function getSystemStatus(signal?: AbortSignal): Promise<SystemStatus> {
  return request<SystemStatus>("/system/status", { signal });
}

export function getModelDiagnostics(signal?: AbortSignal): Promise<ModelDiagnostics> {
  return request<ModelDiagnostics>("/system/model-diagnostics", { signal });
}

export function getModelPerformance(signal?: AbortSignal, runId?: string): Promise<ModelPerformance> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return request<ModelPerformance>(`/system/model-performance${query}`, { signal });
}

export function getMarketSnapshot(ticker: string, signal?: AbortSignal): Promise<MarketSnapshot> {
  return request<MarketSnapshot>(`/market-snapshots/${encodeURIComponent(ticker)}`, { signal });
}

export function getPrediction(ticker: string, horizon: Horizon, signal?: AbortSignal): Promise<Prediction> {
  return request<Prediction>(`/predictions/${encodeURIComponent(ticker)}?horizon=${horizon}`, { signal });
}

export function createAnalysis(requestBody: AnalysisRequest, signal?: AbortSignal): Promise<DashboardAnalysis> {
  return request<DashboardAnalysis>("/analyses", {
    method: "POST",
    body: JSON.stringify(requestBody),
    signal,
  });
}

export function createComparison(tickers: string[], horizon: Horizon, signal?: AbortSignal): Promise<ComparisonResult> {
  return request<ComparisonResult>("/comparisons", {
    method: "POST",
    body: JSON.stringify({ tickers, horizon }),
    signal,
  });
}

export function retrieveNews(ticker: string, horizon: Horizon, signal?: AbortSignal): Promise<NewsContext> {
  return request<NewsContext>("/news", {
    method: "POST",
    body: JSON.stringify({ ticker, horizon }),
    signal,
  });
}

export function updateNewsOverride(
  analysisId: string,
  articleId: string,
  action: "include" | "exclude" | "reset",
  signal?: AbortSignal,
): Promise<DashboardAnalysis> {
  return request<DashboardAnalysis>(`/analyses/${encodeURIComponent(analysisId)}/news-overrides`, {
    method: "PATCH",
    body: JSON.stringify({ article_id: articleId, action }),
    signal,
  });
}

export function resetNewsOverrides(analysisId: string, signal?: AbortSignal): Promise<DashboardAnalysis> {
  return request<DashboardAnalysis>(`/analyses/${encodeURIComponent(analysisId)}/news-overrides`, {
    method: "DELETE",
    signal,
  });
}

export function askStockSense(
  ticker: string,
  horizon: Horizon,
  contextId: string | null,
  analysisId: string | null,
  messages: ChatMessage[],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({ ticker, horizon, context_id: contextId, analysis_id: analysisId, messages }),
    signal,
  });
}

/**
 * Uses the same guarded chat endpoint as Ask & Learn, but binds the prompt to
 * the exact Stock Outlook analysis currently displayed. Keeping this helper
 * separate prevents the two UI conversations from sharing state accidentally.
 */
export function askAggregateExplanation(
  ticker: string,
  horizon: Horizon,
  analysisId: string,
  messages: ChatMessage[],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({
      ticker,
      horizon,
      analysis_id: analysisId,
      intent: "aggregate_explanation",
      messages,
    }),
    signal,
  });
}

export function prepareLearningContext(
  ticker: string,
  horizon: Horizon,
  analysisId: string | null,
  signal?: AbortSignal,
): Promise<LearningContext> {
  return request<LearningContext>("/learning-contexts", {
    method: "POST",
    body: JSON.stringify({ ticker, horizon, analysis_id: analysisId }),
    signal,
  });
}
