from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Horizon = Literal[1, 5, 10]


class ApiError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ApiError


class CompanyResponse(BaseModel):
    ticker: str
    company_name: str
    domain: str
    primary_logo_url: str
    fallback_logo_url: str
    initials: str


class MarketSnapshotResponse(BaseModel):
    ticker: str
    company_name: str
    sector: Optional[str] = None
    price: float
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    open: Optional[float] = None
    adjusted_close: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[int] = None
    as_of_date: str
    data_source: str
    stale_data_warning: bool
    freshness_status: str = "unknown"
    freshness_reason: str = ""
    observation_lag_sessions: Optional[int] = None
    history_dates: List[str]
    history_prices: List[float]
    history: List[Dict[str, Any]] = Field(default_factory=list)
    performance: Dict[str, Optional[float]] = Field(default_factory=dict)
    range_52_week: Dict[str, Optional[float]] = Field(default_factory=dict)
    activity: Dict[str, Optional[float]] = Field(default_factory=dict)
    risk: Dict[str, Optional[float]] = Field(default_factory=dict)
    market_context: Dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    ticker: str
    company_name: str
    horizon_days: int
    as_of_date: str
    prediction_label: str
    probability_up: float
    decision_threshold: float
    model_name: str
    model_version: str
    refinement_round: int
    benchmark_passed: bool
    status: str
    stale_data_warning: bool
    latest_data_age_days: int
    threshold_constraints_passed: bool = True
    evidence_strength: str = "Unavailable"
    supporting_factors: List[Dict[str, Any]] = Field(default_factory=list)
    opposing_factors: List[Dict[str, Any]] = Field(default_factory=list)
    market_reference_date: Optional[str] = None
    observation_lag_sessions: Optional[int] = None
    freshness_status: str = "unknown"
    freshness_reason: Optional[str] = None
    feature_schema_version: str = "legacy-unversioned"
    global_evidence_status: str = "experimental"
    ticker_evidence_status: str = "experimental"
    shadow_evidence_status: Optional[str] = None
    acceptance_policy_version: str = "legacy"
    failed_acceptance_checks: List[str] = Field(default_factory=list)
    prediction_available: bool = True
    directional_signal_available: bool = False
    evidence_status: str = "experimental"
    forecast_issued_at: Optional[str] = None
    target_window_start_date: Optional[str] = None
    target_window_end_date: Optional[str] = None
    probability_interval: Optional[Dict[str, float]] = None
    directional_margin: float = 0.0
    abstention_reason: Optional[str] = None


class NewsContextResponse(BaseModel):
    ticker: str
    company_name: str
    query_text: str
    as_of_date: str
    retrieved_at: str
    raw_article_count: int
    relevant_article_count: int
    results_truncated: bool
    contexts: Dict[str, Dict[str, Any]]
    articles: List[Dict[str, Any]]
    output_paths: Dict[str, str]
    provider_mode: str = "newsapi"
    provider_article_counts: Dict[str, int] = Field(default_factory=dict)
    provider_status: Dict[str, str] = Field(default_factory=dict)
    provider_metadata: Dict[str, Any] = Field(default_factory=dict)
    historical_articles: List[Dict[str, Any]] = Field(default_factory=list)
    review_articles: List[Dict[str, Any]] = Field(default_factory=list)
    automatic_articles: List[Dict[str, Any]] = Field(default_factory=list)
    automatic_review_articles: List[Dict[str, Any]] = Field(default_factory=list)
    automatic_contexts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    user_overrides: Dict[str, str] = Field(default_factory=dict)


class CombinedOutlookResponse(BaseModel):
    """Top-level Stock Outlook synthesis or validated fusion result."""

    ticker: str
    company_name: str
    horizon_days: int
    mode: Literal["stock_led_synthesis", "validated_fusion", "unavailable"]
    direction: Optional[str] = None
    probability_up: Optional[float] = None
    decision_threshold: Optional[float] = None
    probability_kind: Optional[Literal["stock_model", "combined"]] = None
    validation_status: str = "unavailable"
    relationship: Literal["reinforcing", "conflicting", "balanced", "news_unavailable"]
    news_tone: str = "unavailable"
    news_coverage: str = "Unavailable"
    as_of_date: Optional[str] = None
    news_cutoff_at: Optional[str] = None
    explanation: str
    stock_probability_up: Optional[float] = None
    stock_direction: Optional[str] = None
    news_score: Optional[float] = None
    usable_article_count: int = 0
    source_count: int = 0
    component_contributions: Dict[str, float] = Field(default_factory=dict)
    fusion_model_version: Optional[str] = None
    fusion_method: Optional[Literal["transparent_log_odds", "late_fusion_logistic", "anchored_incremental_news", "v4_anchored_incremental_news"]] = None
    fallback_reason: Optional[str] = None
    acceptance_override_active: bool = False
    acceptance_override_reason: Optional[str] = None


class DashboardAnalysisResponse(BaseModel):
    ticker: str
    company_name: str
    horizon_days: int
    stock: Optional[PredictionResponse] = None
    stock_source: str
    stock_captured_at: Optional[str] = None
    stock_warning: Optional[str] = None
    news: Optional[NewsContextResponse] = None
    news_source: str
    news_captured_at: Optional[str] = None
    news_warning: Optional[str] = None
    generated_at: str
    combined_outlook: Optional[CombinedOutlookResponse] = None


class AnalysisResponse(DashboardAnalysisResponse):
    analysis_id: str


class LearningContextRequest(BaseModel):
    ticker: str = Field(min_length=1)
    horizon: Horizon
    analysis_id: Optional[str] = None


class TutorEvidenceReference(BaseModel):
    id: str
    type: Literal["market", "model", "news", "synthesis"]
    label: str
    value: str = ""
    as_of: Optional[str] = None
    url: Optional[str] = None


class TutorKeyTerm(BaseModel):
    term: str
    definition: str


class LearningContextResponse(BaseModel):
    context_id: str
    ticker: str
    company_name: str
    horizon_days: int
    generated_at: str
    expires_at: str
    status: Literal["ready", "partial", "concept_only"]
    warnings: List[str] = Field(default_factory=list)
    market_snapshot: Optional[MarketSnapshotResponse] = None
    analysis: Optional[AnalysisResponse] = None


class AnalysisRequest(BaseModel):
    ticker: str = Field(min_length=1)
    horizon: Horizon


class NewsOverrideRequest(BaseModel):
    article_id: str = Field(min_length=1)
    action: Literal["include", "exclude", "reset"]


class ComparisonRequest(BaseModel):
    tickers: List[str] = Field(min_length=2, max_length=5)
    horizon: Horizon


class ComparisonItemResponse(BaseModel):
    ticker: str
    snapshot: Optional[MarketSnapshotResponse] = None
    prediction: Optional[PredictionResponse] = None
    error: Optional[str] = None


class ComparisonResponse(BaseModel):
    tickers: List[str]
    horizon_days: int
    items: List[ComparisonItemResponse]


class NewsRequest(BaseModel):
    ticker: str = Field(min_length=1)
    horizon: Horizon


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    ticker: str = Field(min_length=1)
    horizon: Horizon
    context_id: Optional[str] = None
    analysis_id: Optional[str] = None
    # ``aggregate_explanation`` is deliberately separate from the general
    # tutor: it is allowed only when the exact live analysis is still in the
    # cache, so it cannot silently turn into an ungrounded stock explanation.
    intent: Literal["general_tutor", "aggregate_explanation"] = "general_tutor"
    messages: List[ChatMessage] = Field(default_factory=list, max_length=50)


class ChatResponse(BaseModel):
    # ``content`` remains for clients deployed before the structured tutor
    # envelope. New clients should render the fields below.
    content: str = ""
    message_id: str
    mode: Literal["concept", "evidence", "news", "market", "uncertainty"] = "concept"
    direct_answer: str
    explanation_steps: List[str] = Field(default_factory=list)
    evidence_refs: List[TutorEvidenceReference] = Field(default_factory=list)
    key_terms: List[TutorKeyTerm] = Field(default_factory=list)
    caution: Optional[str] = None
    follow_up_questions: List[str] = Field(default_factory=list)
    model: str
    fallback_used: bool = False
    context_id: Optional[str] = None


class SystemStatusResponse(BaseModel):
    supported_horizons: List[int]
    news_retrieval_lookback_days: int
    news_context_windows: Dict[str, int]
    stock_models_available: Dict[str, bool]
    finbert_runtime_available: bool
    newsapi_configured: bool
    marketaux_configured: bool
    openrouter_configured: bool
    openrouter_model: str
    openrouter_fallback_model: str
    feature_schema_version: str
    model_registry_signature: str
    stock_registry_profile: str = "unknown"
    stock_registry_publication_status: str = "unknown"
    stock_registry_officially_validated: bool = False
    stock_registry_manual_override_active: bool = False
    stock_registry_manual_override_reason: Optional[str] = None
    fusion_data_ready: bool = False
    fusion_readiness_reason: str = "Historical point-in-time fusion data is not ready."
    fusion_models_available: Dict[str, bool] = Field(default_factory=dict)
    fusion_validation_status: Dict[str, str] = Field(default_factory=dict)
    news_evidence_ready: bool = False
    news_evidence_reason: str = "A Malaysian labelled news benchmark is not ready."
