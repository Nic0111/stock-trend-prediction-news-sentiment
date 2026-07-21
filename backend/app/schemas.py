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
    price: float
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[int] = None
    as_of_date: str
    data_source: str
    stale_data_warning: bool
    history_dates: List[str]
    history_prices: List[float]


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
    prediction_available: bool = True
    directional_signal_available: bool = False
    evidence_status: str = "experimental"


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


class AnalysisResponse(DashboardAnalysisResponse):
    analysis_id: str


class AnalysisRequest(BaseModel):
    ticker: str = Field(min_length=1)
    horizon: Horizon


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
    analysis_id: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list, max_length=50)


class ChatResponse(BaseModel):
    content: str
    model: str
    fallback_used: bool = False


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
