export type Horizon = 1 | 5 | 10;

export interface EvaluationMetrics {
  accuracy: number | null;
  balanced_accuracy: number | null;
  macro_f1: number | null;
  mcc: number | null;
  roc_auc?: number | null;
  ece?: number | null;
  brier_score?: number | null;
  log_loss?: number | null;
  class_f1?: Record<string, number | null>;
}

export interface StockHorizonDiagnostic {
  horizon_days: Horizon;
  model_name: string;
  status: string;
  observed_status: string;
  shadow_status: string;
  decision_threshold: number | null;
  strongest_baseline: string;
  metrics: EvaluationMetrics;
  baseline_metrics: EvaluationMetrics;
  balanced_accuracy_gain: number | null;
  observed_bootstrap_gain_ci?: {
    lower: number | null;
    median: number | null;
    upper: number | null;
    repetitions?: number | null;
    block_size?: number | null;
  } | null;
  cv_fold_balanced_accuracy: number[];
  observed_failed_checks: string[];
  shadow_failed_checks: string[];
  observed_unique_dates: number | null;
  shadow_rows: number | null;
}

export interface SentimentModelDiagnostic extends EvaluationMetrics {
  key: string;
  label: string;
  class_f1: Record<string, number | null>;
}

export interface ModelDiagnostics {
  generated_at: string;
  disclosure: string;
  stock: {
    available: boolean;
    run_id: string | null;
    search_profile: string | null;
    feature_schema_version: string | null;
    development_period: { start: string | null; end: string | null };
    retrospective_test_period: { start: string | null; end: string | null; disclosure: string | null };
    shadow_period: Record<string, unknown>;
    acceptance_policy_version: string | null;
    horizons: StockHorizonDiagnostic[];
  };
  sentiment: {
    available: boolean;
    active_variant: string;
    backend: string | null;
    promotion_status: string;
    manual_override_active: boolean;
    weights: { bursa: number | null; phrasebank: number | null };
    ensemble_temperature: number | null;
    run_id: string | null;
    selection_scope: string | null;
    evaluation_scope: string | null;
    evaluation_year: number;
    evaluation_rows: number | null;
    models: SentimentModelDiagnostic[];
    dataset: {
      eligible_rows: number | null;
      year_counts: Record<string, number>;
      class_counts: Record<string, number>;
    };
    acceptance: {
      eligible_for_promotion: boolean;
      passed_checks: string[];
      failed_checks: string[];
      thresholds: Record<string, unknown>;
    };
  };
}

export interface LearningCurvePoint {
  fraction: number;
  mean_training_rows: number;
  training_mean: number;
  training_std: number;
  validation_mean: number;
  validation_std: number;
  folds: number;
}

export interface CandidateFamilyDiagnostic {
  model_name: string;
  balanced_accuracy: number | null;
  macro_f1: number | null;
  mcc: number | null;
  refinement_round: number;
}

export interface CompanyModelDiagnostic {
  ticker: string;
  company_name?: string;
  rows: number;
  status?: string;
  balanced_accuracy: number | null;
  macro_f1: number | null;
  mcc: number | null;
  roc_auc?: number | null;
  ece?: number | null;
}

export interface FusionCompanyDiagnostic {
  ticker?: string;
  company_name?: string;
  rows: number;
  stock_balanced_accuracy: number;
  fusion_balanced_accuracy: number;
  gain: number;
}

export interface FusionSubgroupArmMetrics {
  balanced_accuracy?: number | null;
  macro_f1?: number | null;
  mcc?: number | null;
  brier_score?: number | null;
  log_loss?: number | null;
  ece?: number | null;
}

export interface FusionSubgroupDiagnostic {
  rows?: number;
  stock_balanced_accuracy?: number | null;
  fusion_balanced_accuracy?: number | null;
  gain?: number | null;
  stock_only?: FusionSubgroupArmMetrics;
  stock_only_meta?: FusionSubgroupArmMetrics;
  stock_meta?: FusionSubgroupArmMetrics;
  compact_full?: FusionSubgroupArmMetrics;
  full_fusion?: FusionSubgroupArmMetrics;
  stock_plus_news?: FusionSubgroupArmMetrics;
}

export interface StockPerformanceHorizon extends StockHorizonDiagnostic {
  learning_curve: LearningCurvePoint[];
  confusion_matrix: { labels: string[]; values: number[][]; normalized: number[][] } | null;
  calibration_curve: { predicted: number; observed: number }[];
  roc_curve: { false_positive_rate: number; true_positive_rate: number }[];
  precision_recall_curve: { recall: number; precision: number }[];
  threshold_curve: { threshold: number; balanced_accuracy: number; macro_f1: number; up_recall: number; down_recall: number; selected: boolean }[];
  probability_distribution: { bin: string; midpoint: number; actual_down: number; actual_up: number }[];
  permutation_importance: { feature: string; importance_mean: number; importance_std: number }[];
  hyperparameter_trials: { model_name: string; candidate_index: number; selected: boolean; balanced_accuracy: number | null; balanced_accuracy_std: number | null; macro_f1: number | null; mcc: number | null; parameters: string; search_strategy: string; search_budget: number }[];
  seed_stability: { seed: number; balanced_accuracy: number | null; macro_f1: number | null; mcc: number | null }[];
  candidate_families: CandidateFamilyDiagnostic[];
  per_company: CompanyModelDiagnostic[];
}

export interface ModelPerformance {
  generated_at: string;
  disclosure: string;
  available_runs: { run_id: string; label: string; search_profile?: string; active: boolean; artifact_available: boolean }[];
  fusion_ablation?: {
    schema_version?: string;
    created_at?: string;
    claim_scope?: string;
    policy_sha256?: string;
    overall_all_horizon_accepted?: boolean;
    accepted_horizons?: number[];
    overall_status?: string;
    runtime_action?: string;
    runtime_enabled?: boolean;
    scientific_accepted_horizons?: number[];
    scientific_overall_status?: string;
    provenance?: {
      strict_identity_passed?: boolean;
      active_stock_run_id?: string | null;
      recorded_stock_run_ids?: string[];
      checks?: Record<string, boolean>;
    };
    periods: Record<string, string | number[]>;
    arms: string[];
    horizons: Record<string, {
      matched_benchmark_rows?: number;
      benchmark_rows?: number;
      news_present_rows?: number;
      ticker_count: number;
      arms: Record<string, {
        metrics?: { balanced_accuracy: number; macro_f1: number; mcc: number; ece: number; brier_score?: number; log_loss?: number };
        summary?: { iterations: number; actual_percentile: number; mean: number | null; p95: number | null };
      }>;
      stock_plus_news_gain?: number;
      balanced_accuracy_gain?: number;
      brier_improvement?: number;
      actual_minus_placebo_gain?: number;
      block_bootstrap_gain_ci?: { lower: number; median: number; upper: number; iterations: number; block: string };
      bootstrap?: {
        block_trading_sessions: number;
        iterations: number;
        brier_improvement: { lower: number; median: number; upper: number };
        balanced_accuracy_gain: { lower: number; median: number; upper: number };
      };
      clark_west?: { statistic: number; one_sided_p_value: number };
      holm_adjusted_clark_west_p_value?: number;
      placebo?: {
        iterations?: number;
        actual_percentile?: number;
        mean?: number | null;
        p95?: number | null;
        distribution?: number[];
        content_only?: { iterations: number; actual_percentile: number; mean: number | null; p95: number | null; distribution: number[] };
        complete_event?: { iterations: number; actual_percentile: number; mean: number | null; p95: number | null; distribution: number[] };
      };
      calibration?: Record<string, { intercept: number; slope: number; curve: { predicted: number; observed: number; rows: number }[] }>;
      rolling_origins?: { evaluation_year: number; rows: number; ticker_count: number; brier_improvement: number; balanced_accuracy_gain: number }[];
      per_company?: Record<string, FusionCompanyDiagnostic>;
      error_transitions?: Record<string, number>;
      subgroups?: Record<string, Record<string, FusionSubgroupDiagnostic>>;
      confidence_accuracy?: Record<string, { bucket: number; rows: number; minimum_margin: number; maximum_margin: number; accuracy: number }[]>;
      coefficient_stability?: {
        folds: number;
        news_present_intercept?: { mean: number; standard_deviation: number };
        coefficients?: Record<string, { mean: number; standard_deviation: number; sign_consistency: number }>;
      };
      readiness?: { passed: boolean; checks: Record<string, boolean> };
      runtime_label?: string;
      runtime_enabled?: boolean;
      scientific_accepted?: boolean;
      mcnemar_exact_p_value?: number;
      acceptance_checks?: Record<string, boolean>;
      accepted: boolean;
      teaching_conclusion?: string;
    }>;
  };
  stock: Omit<ModelDiagnostics["stock"], "horizons"> & {
    artifact_generated_at: string | null;
    artifact_run_id: string | null;
    artifact_status: "current" | "stale" | "unavailable" | string;
    artifact_warning: string | null;
    detailed_artifact_available: boolean;
    methodology: Record<string, string>;
    horizons: StockPerformanceHorizon[];
  };
  sentiment: ModelDiagnostics["sentiment"] & {
    per_company: CompanyModelDiagnostic[];
    confusion_matrix: { labels: string[]; values: number[][]; normalized?: number[][] } | null;
    calibration_curve: { predicted: number; observed: number }[];
    artifact_notes: Record<string, string>;
  };
}

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
  sector?: string | null;
  price: number;
  previous_close: number | null;
  change: number | null;
  change_pct: number | null;
  open?: number | null;
  adjusted_close?: number | null;
  day_high: number | null;
  day_low: number | null;
  volume: number | null;
  as_of_date: string;
  data_source: string;
  stale_data_warning: boolean;
  freshness_status?: "current" | "delayed" | "stale" | "unknown" | string;
  freshness_reason?: string;
  observation_lag_sessions?: number | null;
  history_dates: string[];
  history_prices: number[];
  history?: MarketHistoryPoint[];
  performance?: MarketPerformance;
  range_52_week?: MarketRange;
  activity?: MarketActivity;
  risk?: MarketRisk;
  market_context?: MarketContext;
}

export interface MarketHistoryPoint {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  adjusted_close: number | null;
  volume: number | null;
  dividend: number | null;
  split: number | null;
  ma20: number | null;
  ma60: number | null;
  rsi14: number | null;
  macd: number | null;
  macd_signal: number | null;
}

export interface MarketPerformance {
  one_day: number | null;
  five_day: number | null;
  one_month: number | null;
  three_month: number | null;
  ytd: number | null;
  one_year: number | null;
}

export interface MarketRange {
  low: number | null;
  high: number | null;
  position_pct: number | null;
}

export interface MarketActivity {
  average_volume_20d: number | null;
  volume_ratio: number | null;
}

export interface MarketRisk {
  annualized_volatility_20d: number | null;
  atr14_pct: number | null;
  daily_range_pct: number | null;
}

export interface BenchmarkPoint {
  date: string;
  close: number | null;
  adjusted_close: number | null;
}

export interface MarketContext {
  history: BenchmarkPoint[];
  excess_return_one_month: number | null;
  excess_return_three_month: number | null;
  beta20: number | null;
  beta60: number | null;
  correlation20: number | null;
  correlation60: number | null;
  klci_return_one_month: number | null;
  klci_trend: string;
  klci_volatility_20d: number | null;
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
  forecast_issued_at?: string | null;
  target_window_start_date?: string | null;
  target_window_end_date?: string | null;
  probability_interval?: Record<string, number> | null;
  directional_margin?: number;
  abstention_reason?: string | null;
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

export interface NewsArticleContribution {
  article_id: string;
  event_id?: string;
  age_days: number;
  recency_weight: number;
  sentiment_confidence: number;
  raw_weight: number;
  event_weight?: number;
  effective_weight?: number;
  normalized_weight: number;
  sentiment_score: number;
  sentiment_label: string;
  signed_contribution: number;
}

export interface NewsHorizonContext {
  horizon_days?: number;
  window_days?: number;
  half_life_days?: number;
  candidate_article_count?: number;
  usable_article_count?: number;
  uncertain_article_count?: number;
  unique_source_count?: number;
  independent_event_count?: number;
  weighted_sentiment_score?: number | null;
  positive_count?: number;
  neutral_count?: number;
  negative_count?: number;
  mean_sentiment_confidence?: number | null;
  max_sentiment_confidence?: number | null;
  mean_article_age_days?: number | null;
  newest_article_age_days?: number | null;
  article_contributions?: NewsArticleContribution[];
  coverage_confidence?: string;
  low_confidence_flag?: boolean;
  weighting_policy_version?: string;
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
  contexts: Record<string, NewsHorizonContext>;
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
  automatic_contexts: Record<string, NewsHorizonContext>;
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
  fusion_method: "transparent_log_odds" | "late_fusion_logistic" | "anchored_incremental_news" | "v4_anchored_incremental_news" | null;
  fallback_reason: string | null;
  acceptance_override_active?: boolean;
  acceptance_override_reason?: string | null;
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

export interface TutorEvidenceReference {
  id: string;
  type: "market" | "model" | "news" | "synthesis";
  label: string;
  value: string;
  as_of?: string | null;
  url?: string | null;
}

export interface TutorKeyTerm {
  term: string;
  definition: string;
}

export interface LearningContext {
  context_id: string;
  ticker: string;
  company_name: string;
  horizon_days: Horizon;
  generated_at: string;
  expires_at: string;
  status: "ready" | "partial" | "concept_only";
  warnings: string[];
  market_snapshot: MarketSnapshot | null;
  analysis: DashboardAnalysis | null;
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
  stock_registry_profile?: string;
  stock_registry_publication_status?: string;
  stock_registry_officially_validated?: boolean;
  stock_registry_manual_override_active?: boolean;
  stock_registry_manual_override_reason?: string | null;
  fusion_data_ready?: boolean;
  fusion_readiness_reason?: string;
  fusion_models_available?: Record<string, boolean>;
  fusion_validation_status?: Record<string, string>;
  news_evidence_ready?: boolean;
  news_evidence_reason?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  id?: string;
  tutor?: TutorResponse;
}

/** A short-lived, analysis-specific thread shown below Aggregate Result. */
export interface AggregateTutorThread {
  analysisId: string;
  evidenceKey: string;
  messages: ChatMessage[];
}

export interface ChatResponse {
  content: string;
  message_id: string;
  mode: "concept" | "evidence" | "news" | "market" | "uncertainty";
  direct_answer: string;
  explanation_steps: string[];
  evidence_refs: TutorEvidenceReference[];
  key_terms: TutorKeyTerm[];
  caution: string | null;
  follow_up_questions: string[];
  model: string;
  fallback_used: boolean;
  context_id: string | null;
}

export interface TutorResponse {
  message_id: string;
  mode: ChatResponse["mode"];
  direct_answer: string;
  explanation_steps: string[];
  evidence_refs: TutorEvidenceReference[];
  key_terms: TutorKeyTerm[];
  caution: string | null;
  follow_up_questions: string[];
  model: string;
  fallback_used: boolean;
  context_id: string | null;
}
