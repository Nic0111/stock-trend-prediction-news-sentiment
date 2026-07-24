from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.app.cache import MemoryTTLCache
from backend.app.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    ComparisonItemResponse,
    ComparisonRequest,
    ComparisonResponse,
    CompanyResponse,
    MarketSnapshotResponse,
    NewsContextResponse,
    NewsOverrideRequest,
    NewsRequest,
    PredictionResponse,
    SystemStatusResponse,
)
from pipeline.company_logos import get_company_logo
from pipeline.dashboard_service import DashboardAnalysisResult, run_dashboard_analysis
from pipeline.download_financial_news_data import (
    CONTEXT_WINDOWS,
    NEWS_RETRIEVAL_LOOKBACK_DAYS,
    load_stock_metadata,
)
from pipeline.finbert_sentiment_model import FINBERT_MODEL_DIR
from pipeline.fusion_model import DEFAULT_FUSION_DATASET, DEFAULT_FUSION_REGISTRY, check_fusion_dataset_readiness
from pipeline.market_snapshot import MarketSnapshot, fetch_market_snapshot
from pipeline.marketaux_news_provider import MARKETAUX_API_KEY_FILE
from pipeline.news_corrections import NewsCorrectionError, apply_news_override, reset_news_overrides
from pipeline.openrouter_client import (
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_MODEL,
    OpenRouterConfigurationError,
    OpenRouterError,
    complete_chat,
    get_openrouter_api_key,
)
from pipeline.predict_stock_trend import (
    FEATURE_SCHEMA_VERSION,
    REGISTRY_FILE,
    SUPPORTED_HORIZONS,
    PredictionResult,
    predict_stock_trend,
)
from pipeline.runtime_provenance import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("stocksense.api")

market_cache: MemoryTTLCache[MarketSnapshot] = MemoryTTLCache()
prediction_cache: MemoryTTLCache[PredictionResult] = MemoryTTLCache()
analysis_cache: MemoryTTLCache[DashboardAnalysisResult] = MemoryTTLCache()
analysis_signature_cache: MemoryTTLCache[str] = MemoryTTLCache()


def _normalise_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _metadata() -> dict[str, str]:
    return load_stock_metadata()


def _validate_ticker(ticker: str) -> str:
    normalized = _normalise_ticker(ticker)
    if normalized not in _metadata():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "UNSUPPORTED_TICKER", "message": f"Unsupported ticker: {normalized}"},
        )
    return normalized


def _prediction_key(ticker: str, horizon: int) -> str:
    return f"{_normalise_ticker(ticker)}:{horizon}:{_model_registry_signature()}"


def _model_registry_signature() -> str:
    """Return a short signature so a registry update cannot reuse old memory-cache entries."""
    try:
        digest = hashlib.sha256(REGISTRY_FILE.read_bytes()).hexdigest()
    except OSError:
        return "missing"
    return digest[:16]


def _prediction_payload(result: PredictionResult) -> dict[str, Any]:
    payload = result.to_dict()
    # A valid model inference always has a directional class.  Validation
    # status is intentionally kept separate from prediction availability.
    payload["prediction_available"] = result.prediction_label in {"Up", "Down"}
    payload["directional_signal_available"] = result.directional_signal_available
    payload["evidence_status"] = result.evidence_status
    return payload


def _prediction_response(result: PredictionResult) -> PredictionResponse:
    return PredictionResponse.model_validate(_prediction_payload(result))


def _prediction(ticker: str, horizon: int) -> PredictionResult:
    normalized = _validate_ticker(ticker)
    key = _prediction_key(normalized, horizon)
    cached = prediction_cache.get(key)
    if cached is not None:
        return cached
    result = predict_stock_trend(normalized, horizon)
    prediction_cache.set(key, result, 900)
    return result


def _snapshot(ticker: str) -> MarketSnapshot:
    normalized = _validate_ticker(ticker)
    cached = market_cache.get(normalized)
    if cached is not None:
        return cached
    result = fetch_market_snapshot(normalized)
    market_cache.set(normalized, result, 300)
    return result


def _analysis_response(result: DashboardAnalysisResult, analysis_id: str | None = None) -> AnalysisResponse:
    payload = result.to_dict()
    if result.stock is not None:
        payload["stock"] = _prediction_payload(result.stock)
    if analysis_id is None:
        analysis_id = str(uuid.uuid4())
    payload["analysis_id"] = analysis_id
    return AnalysisResponse.model_validate(payload)


def _configuration_present(path: Path, env_name: str) -> bool:
    if os.getenv(env_name, "").strip():
        return True
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError):
        return False


def _stock_model_availability() -> dict[str, bool]:
    try:
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {str(horizon): False for horizon in SUPPORTED_HORIZONS}
    result: dict[str, bool] = {}
    for horizon in SUPPORTED_HORIZONS:
        entry = registry.get("horizons", {}).get(str(horizon), {})
        artifact = PROJECT_ROOT / str(entry.get("artifact_path", ""))
        expected_hash = entry.get("artifact_sha256")
        hash_ok = not expected_hash or (artifact.exists() and sha256_file(artifact) == expected_hash)
        result[str(horizon)] = bool(entry and artifact.exists() and hash_ok)
    return result


def _finbert_available() -> bool:
    return bool(
        FINBERT_MODEL_DIR.exists()
        and importlib.util.find_spec("torch")
        and importlib.util.find_spec("transformers")
    )


def _runtime_artifact_integrity() -> dict[str, Any]:
    """Report hash integrity and identify registries awaiting provenance migration."""
    try:
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ok": False, "state": "registry_unreadable"}
    missing_provenance: list[str] = []
    mismatches: list[str] = []
    for horizon in SUPPORTED_HORIZONS:
        entry = registry.get("horizons", {}).get(str(horizon), {})
        path = PROJECT_ROOT / str(entry.get("artifact_path", ""))
        if not entry.get("artifact_sha256"):
            missing_provenance.append(f"stock-{horizon}d")
        elif not path.exists() or sha256_file(path) != entry["artifact_sha256"]:
            mismatches.append(f"stock-{horizon}d")
    if not FINBERT_MODEL_DIR.exists() or not (FINBERT_MODEL_DIR / "model.safetensors").exists():
        mismatches.append("finbert")
    state = "verified" if not missing_provenance and not mismatches else "migration_required"
    return {
        "ok": not mismatches,
        "state": state,
        "missing_provenance": missing_provenance,
        "mismatches": mismatches,
    }


def _fusion_availability() -> tuple[dict[str, Any], dict[str, bool], dict[str, str]]:
    """Expose fusion readiness without implying that an experimental model is live."""
    try:
        readiness_payload = check_fusion_dataset_readiness(DEFAULT_FUSION_DATASET).to_dict()
    except Exception as exc:  # status must remain available during setup
        readiness_payload = {
            "ready": False,
            "dataset_path": str(DEFAULT_FUSION_DATASET),
            "reason": f"Fusion readiness check failed: {exc}",
            "errors": [str(exc)],
        }
    model_available = {str(horizon): False for horizon in SUPPORTED_HORIZONS}
    validation_status = {str(horizon): "unavailable" for horizon in SUPPORTED_HORIZONS}
    try:
        registry_payload = json.loads(DEFAULT_FUSION_REGISTRY.read_text(encoding="utf-8"))
        for horizon in SUPPORTED_HORIZONS:
            entry = registry_payload.get("horizons", {}).get(str(horizon), {})
            status = str(entry.get("validation_status", "unavailable"))
            validation_status[str(horizon)] = status
            artifact_reference = str(entry.get("artifact_path", "")).strip()
            if artifact_reference:
                artifact = Path(artifact_reference)
                if not artifact.is_absolute():
                    artifact = PROJECT_ROOT / artifact
                model_available[str(horizon)] = status == "accepted" and artifact.is_file()
    except (OSError, ValueError, TypeError):
        pass
    return readiness_payload, model_available, validation_status


def _chat_context(request: ChatRequest) -> str:
    lines = [
        f"Selected company: {request.ticker}",
        f"Selected prediction horizon: {request.horizon} trading day(s)",
    ]
    analysis = analysis_cache.get(request.analysis_id or "")
    analysis_signature = analysis_signature_cache.get(request.analysis_id or "")
    if (
        analysis is None
        or analysis_signature != _model_registry_signature()
        or analysis.ticker != _normalise_ticker(request.ticker)
        or analysis.horizon_days != request.horizon
    ):
        lines.append("No matching Stock Outlook analysis is loaded in this session. Do not invent stock or news values.")
        return "\n".join(lines)

    lines.append(f"Analysis generated at: {analysis.generated_at}")
    if analysis.stock is not None:
        stock = analysis.stock
        validation = stock.evidence_status
        stock_context = (
            f"direction={stock.prediction_label}; probability_up={stock.probability_up:.3f}; "
            f"decision_threshold={stock.decision_threshold:.3f}"
        )
        lines.append(
            f"Stock signal: {stock_context}; as_of={stock.as_of_date}; "
            f"validation={validation}; source={analysis.stock_source}."
        )
        if stock.failed_acceptance_checks:
            lines.append(f"Acceptance checks not passed: {', '.join(stock.failed_acceptance_checks)}.")
        supporting = ", ".join(factor["label"] for factor in stock.supporting_factors) or "none identified"
        opposing = ", ".join(factor["label"] for factor in stock.opposing_factors) or "none identified"
        lines.append(f"Local model reasons: supports={supporting}; opposes={opposing}; threshold_margin_strength={stock.evidence_strength}.")
    else:
        lines.append("Stock signal: unavailable.")

    if analysis.news is not None:
        context = analysis.news.contexts.get(str(request.horizon), {})
        score = context.get("weighted_sentiment_score")
        score_text = "unavailable" if score is None else f"{float(score):+.2f}"
        metadata = analysis.news.provider_metadata or {}
        lines.append(
            f"News context: weighted_score={score_text}; coverage={context.get('coverage_confidence', 'Unavailable')}; "
            f"usable_articles={context.get('usable_article_count', 0)}; context_window_days={context.get('window_days', 'unknown')}; "
            f"retrieval_lookback_days={metadata.get('retrieval_lookback_days', NEWS_RETRIEVAL_LOOKBACK_DAYS)}; "
            f"historical_background_articles={metadata.get('historical_article_count', 0)} (excluded from sentiment); source={analysis.news_source}."
        )
        if analysis.news.user_overrides:
            lines.append(
                f"News evidence was adjusted by the user for this analysis ({len(analysis.news.user_overrides)} article correction(s)); "
                "treat it as a subjective what-if view, not an automatic system result."
            )
    else:
        lines.append("News context: unavailable.")
    combined = analysis.combined_outlook
    if combined is not None:
        combined_probability = (
            "unavailable"
            if combined.probability_up is None
            else f"{float(combined.probability_up):.3f}"
        )
        lines.append(
            f"Evidence synthesis: mode={combined.mode}; direction={combined.direction or 'unavailable'}; "
            f"probability_up={combined_probability}; probability_kind={combined.probability_kind or 'unavailable'}; "
            f"relationship={combined.relationship}; validation={combined.validation_status}."
        )
        lines.append(f"Synthesis explanation: {combined.explanation}")
    return "\n".join(lines)


app = FastAPI(
    title="StockSense API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

dev_origins = [origin.strip() for origin in os.getenv("STOCKSENSE_CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=dev_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Keep client errors in the same envelope as unexpected API failures."""
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    payload = {
        "error": {
            "code": str(detail.get("code", "HTTP_ERROR")),
            "message": str(detail.get("message", "The request could not be completed.")),
            "retryable": bool(detail.get("retryable", exc.status_code >= 500)),
            "request_id": getattr(request.state, "request_id", request.headers.get("X-Request-ID")),
        }
    }
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Check the request fields and try again.",
                "retryable": False,
                "request_id": getattr(request.state, "request_id", request.headers.get("X-Request-ID")),
                "fields": exc.errors(),
            }
        },
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled API error request_id=%s path=%s", request_id, request.url.path)
        response = JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": "The request could not be completed.", "retryable": False, "request_id": request_id}})
    response.headers["X-Request-ID"] = request_id
    if request.url.path.startswith(("/api/v1/predictions", "/api/v1/analyses", "/api/v1/comparisons", "/api/v1/system/status")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/v1/health/live")
def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/health/ready")
def health_ready() -> dict[str, Any]:
    metadata_ok = (PROJECT_ROOT / "data" / "stock-metadata.csv").exists()
    registry_ok = REGISTRY_FILE.exists()
    models = _stock_model_availability()
    artifact_integrity = _runtime_artifact_integrity()
    schema_ok = _registry_schema_version() == FEATURE_SCHEMA_VERSION
    ready = metadata_ok and registry_ok and schema_ok and all(models.values()) and artifact_integrity["ok"]
    return {"status": "ready" if ready else "not_ready", "metadata": metadata_ok, "registry": registry_ok, "feature_schema_version": _registry_schema_version(), "schema_expected": FEATURE_SCHEMA_VERSION, "schema_ok": schema_ok, "models": models, "finbert_runtime": _finbert_available(), "artifact_integrity": artifact_integrity}


def _registry_schema_version() -> str:
    try:
        registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "missing"
    return str(registry.get("feature_schema_version") or "missing")


@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
def system_status() -> SystemStatusResponse:
    fusion_readiness, fusion_models, fusion_validation = _fusion_availability()
    return SystemStatusResponse(
        supported_horizons=list(SUPPORTED_HORIZONS),
        news_retrieval_lookback_days=NEWS_RETRIEVAL_LOOKBACK_DAYS,
        news_context_windows={str(key): value for key, value in CONTEXT_WINDOWS.items()},
        stock_models_available=_stock_model_availability(),
        finbert_runtime_available=_finbert_available(),
        newsapi_configured=_configuration_present(Path(__import__("pipeline.download_financial_news_data", fromlist=["NEWSAPI_KEY_FILE"]).NEWSAPI_KEY_FILE), "NEWSAPI_KEY"),
        marketaux_configured=_configuration_present(MARKETAUX_API_KEY_FILE, "MARKETAUX_API_TOKEN"),
        openrouter_configured=bool(get_openrouter_api_key()),
        openrouter_model=OPENROUTER_MODEL,
        openrouter_fallback_model=OPENROUTER_FALLBACK_MODEL,
        feature_schema_version=_registry_schema_version(),
        model_registry_signature=_model_registry_signature(),
        fusion_data_ready=bool(fusion_readiness.get("ready", False)),
        fusion_readiness_reason=str(fusion_readiness.get("reason", "Fusion data is not ready.")),
        fusion_models_available=fusion_models,
        fusion_validation_status=fusion_validation,
    )


@app.get("/api/v1/companies", response_model=list[CompanyResponse])
def companies() -> list[CompanyResponse]:
    metadata = _metadata()
    result: list[CompanyResponse] = []
    for ticker, company_name in metadata.items():
        logo = get_company_logo(ticker, company_name)
        result.append(CompanyResponse(ticker=ticker, company_name=company_name, domain=logo.domain, primary_logo_url=logo.primary_url, fallback_logo_url=logo.fallback_url, initials=logo.initials))
    return result


@app.get("/api/v1/market-snapshots/{ticker}", response_model=MarketSnapshotResponse)
def market_snapshot(ticker: str) -> MarketSnapshotResponse:
    return MarketSnapshotResponse.model_validate(asdict(_snapshot(ticker)))


@app.get("/api/v1/predictions/{ticker}", response_model=PredictionResponse)
def prediction(ticker: str, horizon: int) -> PredictionResponse:
    if horizon not in SUPPORTED_HORIZONS:
        raise HTTPException(status_code=422, detail={"code": "INVALID_HORIZON", "message": "Horizon must be 1, 5 or 10 trading days."})
    return _prediction_response(_prediction(ticker, horizon))


@app.post("/api/v1/analyses", response_model=AnalysisResponse)
def analysis(request: AnalysisRequest) -> AnalysisResponse:
    normalized = _validate_ticker(request.ticker)
    result = run_dashboard_analysis(normalized, request.horizon)
    analysis_id = str(uuid.uuid4())
    analysis_cache.set(analysis_id, result, 3600)
    analysis_signature_cache.set(analysis_id, _model_registry_signature(), 3600)
    return _analysis_response(result, analysis_id)


@app.patch("/api/v1/analyses/{analysis_id}/news-overrides", response_model=AnalysisResponse)
def news_override(analysis_id: str, request: NewsOverrideRequest) -> AnalysisResponse:
    result = analysis_cache.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"code": "ANALYSIS_EXPIRED", "message": "This analysis has expired. Run Analyze again."})
    if analysis_signature_cache.get(analysis_id) != _model_registry_signature():
        raise HTTPException(status_code=409, detail={"code": "STALE_ANALYSIS", "message": "The deployed model changed. Run Analyze again before correcting articles."})
    try:
        updated = apply_news_override(result, request.article_id, request.action)
    except NewsCorrectionError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_NEWS_OVERRIDE", "message": str(exc)}) from exc
    analysis_cache.set(analysis_id, updated, 3600)
    return _analysis_response(updated, analysis_id)


@app.delete("/api/v1/analyses/{analysis_id}/news-overrides", response_model=AnalysisResponse)
def reset_news_override(analysis_id: str) -> AnalysisResponse:
    result = analysis_cache.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"code": "ANALYSIS_EXPIRED", "message": "This analysis has expired. Run Analyze again."})
    if analysis_signature_cache.get(analysis_id) != _model_registry_signature():
        raise HTTPException(status_code=409, detail={"code": "STALE_ANALYSIS", "message": "The deployed model changed. Run Analyze again before resetting articles."})
    updated = reset_news_overrides(result)
    analysis_cache.set(analysis_id, updated, 3600)
    return _analysis_response(updated, analysis_id)


@app.post("/api/v1/comparisons", response_model=ComparisonResponse)
def comparison(request: ComparisonRequest) -> ComparisonResponse:
    tickers = [_normalise_ticker(ticker) for ticker in request.tickers]
    if len(set(tickers)) != len(tickers):
        raise HTTPException(status_code=422, detail={"code": "DUPLICATE_TICKER", "message": "Comparison tickers must be unique."})
    if len(tickers) < 2 or len(tickers) > 5:
        raise HTTPException(status_code=422, detail={"code": "INVALID_COMPARISON_SIZE", "message": "Choose between two and five tickers."})
    items: list[ComparisonItemResponse] = []
    for ticker in tickers:
        try:
            snapshot = _snapshot(ticker)
            model = _prediction(ticker, request.horizon)
            items.append(ComparisonItemResponse(ticker=ticker, snapshot=MarketSnapshotResponse.model_validate(asdict(snapshot)), prediction=_prediction_response(model)))
        except Exception as exc:
            items.append(ComparisonItemResponse(ticker=ticker, error=str(exc)))
    return ComparisonResponse(tickers=tickers, horizon_days=request.horizon, items=items)


@app.post("/api/v1/news", response_model=NewsContextResponse)
def news(request: NewsRequest) -> NewsContextResponse:
    from pipeline.marketaux_news_provider import retrieve_news_with_marketaux_fallback

    normalized = _validate_ticker(request.ticker)
    result = retrieve_news_with_marketaux_fallback(normalized, request.horizon)
    return NewsContextResponse.model_validate(result.to_dict())


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    prompt = [
        {"role": "system", "content": (
            "You are StockSense's educational Ask & Learn tutor. Answer beginner questions in clear, friendly language. "
            "Use the supplied StockSense context as evidence, distinguish model outputs from facts, and mention timestamps or missing data when relevant. "
            "Never give personalized financial advice and never tell the user to buy, sell, or hold a security. "
            "If context is missing, say so. Explain one concept at a time, use short sections or bullets, and stay under 350 words.\n\n"
            f"StockSense context:\n{_chat_context(request)}"
        )}
    ]
    prompt.extend({"role": message.role, "content": message.content} for message in request.messages)
    try:
        answer = complete_chat(prompt)
    except OpenRouterConfigurationError as exc:
        raise HTTPException(status_code=503, detail={"code": "OPENROUTER_NOT_CONFIGURED", "message": str(exc), "retryable": False}) from exc
    except OpenRouterError as exc:
        raise HTTPException(status_code=503, detail={"code": "OPENROUTER_UNAVAILABLE", "message": str(exc), "retryable": True}) from exc
    return ChatResponse(content=answer.content, model=answer.model, fallback_used=answer.model != OPENROUTER_MODEL)
