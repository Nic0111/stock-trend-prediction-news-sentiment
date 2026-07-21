export type Horizon = 1 | 5 | 10;

export interface AnalysisRequest {
  ticker: string;
  horizon: Horizon;
}

export interface Company {
  ticker: string;
  company_name: string;
  domain: string;
  primary_logo_url: string;
  fallback_logo_url: string;
  initials: string;
}

export interface MarketSnapshot {
  ticker: string;
  company_name: string;
  price: number;
  previous_close: number | null;
  change: number | null;
  change_pct: number | null;
  day_high: number | null;
  day_low: number | null;
  volume: number | null;
  as_of_date: string;
  data_source: string;
  stale_data_warning: boolean;
  history_dates: string[];
  history_prices: number[];
}

export interface PredictionFactor {
  feature: string;
  label: string;
  impact_probability_points: number;
  relationship: string;
}

export interface Prediction {
  ticker: string;
  company_name: string;
  horizon_days: number;
  as_of_date: string;
  prediction_label: string;
  probability_up: number;
  decision_threshold: number;
  model_name: string;
  model_version: string;
  refinement_round: number;
  benchmark_passed: boolean;
  status: string;
  stale_data_warning: boolean;
  latest_data_age_days: number;
  threshold_constraints_passed: boolean;
  evidence_strength: string;
  supporting_factors: PredictionFactor[];
  opposing_factors: PredictionFactor[];
  market_reference_date?: string | null;
  observation_lag_sessions?: number | null;
  freshness_status?: "fresh" | "delayed" | "stale" | "unknown" | string;
  freshness_reason?: string | null;
  feature_schema_version?: string;
  prediction_available: boolean;
  directional_signal_available: boolean;
  evidence_status: string;
}

export interface NewsArticle {
  title?: string;
  description?: string;
  content_snippet?: string;
  url?: string;
  source_name?: string;
  source_domain?: string;
  provider?: string;
  provider_sources?: string[];
  published_at?: string;
  sentiment_label?: string;
  sentiment_probability?: number;
  sentiment_uncertain?: boolean;
  sentiment_score?: number;
  [key: string]: unknown;
}

export interface NewsContext {
  ticker: string;
  company_name: string;
  query_text: string;
  as_of_date: string;
  retrieved_at: string;
  raw_article_count: number;
  relevant_article_count: number;
  results_truncated: boolean;
  contexts: Record<string, Record<string, unknown>>;
  articles: NewsArticle[];
  output_paths: Record<string, string>;
  provider_mode: string;
  provider_article_counts: Record<string, number>;
  provider_status: Record<string, string>;
  provider_metadata: Record<string, unknown>;
  historical_articles: NewsArticle[];
}

export interface DashboardAnalysis {
  ticker: string;
  company_name: string;
  horizon_days: number;
  stock: Prediction | null;
  stock_source: string;
  stock_captured_at: string | null;
  stock_warning: string | null;
  news: NewsContext | null;
  news_source: string;
  news_captured_at: string | null;
  news_warning: string | null;
  generated_at: string;
  analysis_id: string;
}

export interface ComparisonItem {
  ticker: string;
  snapshot: MarketSnapshot | null;
  prediction: Prediction | null;
  error: string | null;
}

export interface ComparisonResult {
  tickers: string[];
  horizon_days: number;
  items: ComparisonItem[];
}

export interface SystemStatus {
  supported_horizons: number[];
  news_retrieval_lookback_days: number;
  news_context_windows: Record<string, number>;
  stock_models_available: Record<string, boolean>;
  finbert_runtime_available: boolean;
  newsapi_configured: boolean;
  marketaux_configured: boolean;
  openrouter_configured: boolean;
  openrouter_model: string;
  openrouter_fallback_model: string;
  feature_schema_version: string;
  model_registry_signature: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  content: string;
  model: string;
  fallback_used: boolean;
}
