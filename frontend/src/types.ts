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
  global_evidence_status?: "accepted" | "limited" | "experimental" | string;
  ticker_evidence_status?: "accepted" | "limited" | "experimental" | string;
  shadow_evidence_status?: "accepted" | "limited" | "experimental" | null | string;
  acceptance_policy_version?: string;
  failed_acceptance_checks?: string[];
  prediction_available: boolean;
  directional_signal_available: boolean;
  evidence_status: string;
}

export interface NewsArticle {
  article_id?: string;
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
  automatic_relevance?: "included" | "review" | "unrelated" | string;
  relevance_score?: number;
  relevance_reasons?: string[];
  matched_entity_terms?: string[];
  collision_signals?: string[];
  automatic_included?: boolean;
  effective_included?: boolean;
  user_override?: "include" | "exclude" | null | string;
  contributes_to_horizons?: number[];
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
  review_articles: NewsArticle[];
  automatic_articles: NewsArticle[];
  automatic_review_articles: NewsArticle[];
  automatic_contexts: Record<string, Record<string, unknown>>;
  user_overrides: Record<string, string>;
}

export type SynthesisRelationship = "reinforcing" | "conflicting" | "balanced" | "news_unavailable";

export interface CombinedOutlook {
  ticker: string;
  company_name: string;
  horizon_days: number;
  mode: "stock_led_synthesis" | "validated_fusion" | "unavailable";
  direction: string | null;
  probability_up: number | null;
  decision_threshold: number | null;
  probability_kind: "stock_model" | "combined" | null;
  validation_status: string;
  relationship: SynthesisRelationship;
  news_tone: string;
  news_coverage: string;
  as_of_date: string | null;
  news_cutoff_at: string | null;
  explanation: string;
  stock_probability_up: number | null;
  stock_direction: string | null;
  news_score: number | null;
  usable_article_count: number;
  source_count: number;
  component_contributions: Record<string, number>;
  fusion_model_version: string | null;
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
  combined_outlook: CombinedOutlook | null;
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
  fusion_data_ready?: boolean;
  fusion_readiness_reason?: string;
  fusion_models_available?: Record<string, boolean>;
  fusion_validation_status?: Record<string, string>;
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
