"""Resumable dual-LLM pseudo-labelling for recovered Bursa news.

This module is offline-only. It never runs from FastAPI and it never exposes
FinBERT predictions to the labellers. Two fixed OpenRouter models label each
article independently; only strict consensus becomes training data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from pipeline.bursa_news_dataset import (
    SENTIMENT_LABELS,
    build_annotation_template,
    read_table,
)
from pipeline.openrouter_client import (
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_MODEL,
    OpenRouterAnswer,
    OpenRouterError,
    OpenRouterRateLimitError,
    complete_chat_with_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABEL_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "news-evidence-benchmark"
    / "bursa-news-llm-pseudolabelled.jsonl"
)
DEFAULT_LABEL_REPORT = (
    PROJECT_ROOT
    / "results"
    / "bursa-news-dataset"
    / "llm-consensus-report.json"
)
DEFAULT_LABEL_CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "llm-label-checkpoint.jsonl"
)
DEFAULT_HEARTBEAT_PATH = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "llm-label-heartbeat.json"
)
DEFAULT_APPROVAL_QUEUE = (
    PROJECT_ROOT
    / "data"
    / "historical-news-fusion"
    / "gdelt"
    / "llm-label-approval-queue.jsonl"
)
LABEL_MODEL_A = OPENROUTER_MODEL
LABEL_MODEL_B = OPENROUTER_FALLBACK_MODEL
PROMPT_VERSION = "bursa-news-consensus-v1"
MINIMUM_LABEL_CONFIDENCE = 0.70
MAX_MODEL_ATTEMPTS = 3
SENTIMENT_VALUES = set(SENTIMENT_LABELS)
DAILY_FREE_RATE_LIMIT_MARKERS = (
    "free-models-per-day",
    "daily free-model quota",
    "daily free model quota",
)
EVENT_CATEGORIES = {
    "earnings",
    "guidance",
    "contract",
    "corporate_action",
    "financing",
    "operations",
    "regulatory_legal",
    "management",
    "product_service",
    "macro_sector_exposure",
    "esg",
    "other",
}


def _label_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "bursa_news_weak_label",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "entity_relevant",
                    "sentiment",
                    "event_category",
                    "language",
                    "confidence",
                    "evidence",
                    "rationale",
                ],
                "properties": {
                    "entity_relevant": {"type": "boolean"},
                    "sentiment": {
                        "type": ["string", "null"],
                        "enum": ["negative", "neutral", "positive", None],
                    },
                    "event_category": {
                        "type": "string",
                        "enum": sorted(EVENT_CATEGORIES),
                    },
                    "language": {"type": "string", "enum": ["en", "other"]},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "evidence": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {"type": "string", "maxLength": 240},
                    },
                    "rationale": {"type": "string", "maxLength": 800},
                },
            },
        },
    }


@dataclass(frozen=True)
class LLMArticleLabel:
    entity_relevant: bool
    sentiment: str | None
    event_category: str
    language: str
    confidence: float
    evidence: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically refresh a small status file despite transient Windows locks.

    Antivirus, indexers, or Explorer can briefly hold the previous heartbeat
    file open.  A unique sibling temporary file avoids competing writers, and
    the short retry budget prevents a status update from losing an otherwise
    valid, resumable labelling pass.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str)
    for attempt in range(6):
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        try:
            temporary.replace(path)
            return
        except PermissionError:
            temporary.unlink(missing_ok=True)
            if attempt >= 5:
                raise
            time.sleep(0.05 * (attempt + 1))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _is_daily_free_model_limit(error: BaseException) -> bool:
    """Identify an account-wide daily quota without treating it as a bad row."""

    if not isinstance(error, OpenRouterRateLimitError):
        return False
    message = str(error).casefold()
    return any(marker in message for marker in DAILY_FREE_RATE_LIMIT_MARKERS)


def _prompt_hash() -> str:
    return hashlib.sha256(
        f"{PROMPT_VERSION}|{_system_prompt()}".encode("utf-8")
    ).hexdigest()


def _system_prompt() -> str:
    return (
        "You label historical financial news for an academic Bursa Malaysia "
        "sentiment dataset. Judge entity relevance and the article's financial "
        "or operating implication for the named company, never future share-price "
        "direction. Exclude passing mentions, similarly named foreign entities, "
        "software/package pages, directories and generic sector news without a "
        "defensible company connection. Return exactly one JSON object with keys "
        "entity_relevant (boolean), sentiment (negative|neutral|positive|null), "
        "event_category, language (en|other), confidence (0 to 1), evidence "
        "(array of at most two short excerpts), and rationale. If entity_relevant "
        "is false, sentiment must be null. If the company connection is real but "
        "the text does not support a company-specific financial sentiment, use "
        "null; the pipeline will abstain instead of inventing a label. Do not use "
        "markdown."
    )


def build_label_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    company = str(row.get("company_name", "")).strip()
    ticker = str(row.get("ticker", "")).strip().upper()
    text = str(row.get("model_text", "")).strip()[:12000]
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": (
                f"Company: {company}\nTicker: {ticker}\n"
                f"Publication timestamp: {row.get('published_at', '')}\n"
                f"Article:\n{text}"
            ),
        },
    ]


def _json_object(raw: str) -> dict[str, Any]:
    value = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.S | re.I)
    if fenced:
        value = fenced.group(1)
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(value)
    except json.JSONDecodeError:
        match = re.search(r"\{", value)
        if not match:
            raise ValueError("LLM response does not contain a JSON object")
        payload, _ = decoder.raw_decode(value[match.start() :])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def parse_llm_label(raw: str) -> LLMArticleLabel:
    payload = _json_object(raw)
    entity = payload.get("entity_relevant")
    if not isinstance(entity, bool):
        raise ValueError("entity_relevant must be boolean")
    sentiment_value = payload.get("sentiment")
    sentiment = (
        None
        if sentiment_value is None
        else str(sentiment_value).strip().casefold()
    )
    # Some free structured-output providers populate an enum value even when
    # they correctly mark the target entity irrelevant. Sentiment for an
    # irrelevant article is never used, so discard it deterministically rather
    # than halting the resumable dataset run or accepting a contradictory label.
    if not entity:
        sentiment = None
    if entity and sentiment is not None and sentiment not in SENTIMENT_VALUES:
        raise ValueError("relevant article sentiment must be valid or null")
    category = str(payload.get("event_category", "")).strip().casefold()
    if category not in EVENT_CATEGORIES:
        category = "other"
    language = str(payload.get("language", "")).strip().casefold()
    if language in {"en", "eng", "english"}:
        language = "en"
    elif language not in {"other"}:
        language = "other"
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    evidence_raw = payload.get("evidence", [])
    if not isinstance(evidence_raw, list):
        raise ValueError("evidence must be an array")
    evidence = tuple(
        str(item).strip()[:240]
        for item in evidence_raw[:2]
        if str(item).strip()
    )
    rationale = str(payload.get("rationale", "")).strip()[:800]
    if not rationale:
        raise ValueError("rationale is required")
    return LLMArticleLabel(
        entity_relevant=entity,
        sentiment=sentiment,
        event_category=category,
        language=language,
        confidence=confidence,
        evidence=evidence,
        rationale=rationale,
    )


def _article_id(row: Mapping[str, Any]) -> str:
    existing = str(row.get("article_id", "")).strip()
    if existing:
        return existing
    return hashlib.sha256(
        (
            f"{str(row.get('ticker', '')).upper()}|"
            f"{str(row.get('url', '')).strip()}|"
            f"{str(row.get('published_at', '')).strip()}"
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_records(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        key = (
            str(item.get("article_id", "")),
            str(item.get("model", "")),
            str(item.get("prompt_sha256", "")),
        )
        records[key] = item
    return records


def _checkpoint_failures(
    path: Path,
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], str]]:
    attempts: dict[tuple[str, str, str], int] = {}
    terminal: dict[tuple[str, str, str], str] = {}
    if not path.exists():
        return attempts, terminal
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("article_id", "")),
            str(item.get("model", "")),
            str(item.get("prompt_sha256", "")),
        )
        status = str(item.get("status", ""))
        if status not in {"retryable_error", "terminal_error"}:
            continue
        attempts[key] = max(
            attempts.get(key, 0),
            int(item.get("attempt", attempts.get(key, 0) + 1)),
        )
        if status == "terminal_error":
            terminal[key] = str(item.get("error", "free model unavailable"))
    return attempts, terminal


def _append_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")


def _call_content(
    completion: Callable[..., Any],
    messages: Sequence[Mapping[str, str]],
    *,
    model: str,
) -> str:
    result = completion(
        messages,
        model=model,
        temperature=0.0,
        max_tokens=900,
        timeout=60.0,
        response_format=_label_response_format(),
        reasoning={"enabled": False, "exclude": True},
    )
    if isinstance(result, OpenRouterAnswer):
        if result.model != model:
            raise ValueError("fixed label model was silently substituted")
        return result.content
    if isinstance(result, str):
        return result
    raise ValueError("completion callable returned an unsupported value")


def _role_columns(role: str, model: str, label: LLMArticleLabel) -> dict[str, Any]:
    return {
        f"{role}_model": model,
        f"{role}_entity_relevant": label.entity_relevant,
        f"{role}_sentiment": label.sentiment or "",
        f"{role}_event_category": label.event_category,
        f"{role}_language": label.language,
        f"{role}_confidence": label.confidence,
        f"{role}_evidence": json.dumps(label.evidence, ensure_ascii=False),
        f"{role}_rationale": label.rationale,
    }


def _consensus(
    label_a: LLMArticleLabel,
    label_b: LLMArticleLabel,
    *,
    minimum_confidence: float,
) -> dict[str, Any]:
    base = {
        "label_source": "llm_consensus",
        "entity_relevant": False,
        "sentiment_label": "",
        "event_category": "",
        "language": "",
        "consensus_status": "abstained",
        "consensus_reason": "",
    }
    if label_a.confidence < minimum_confidence or label_b.confidence < minimum_confidence:
        return {**base, "consensus_reason": "one or both labels below confidence floor"}
    if label_a.entity_relevant != label_b.entity_relevant:
        return {**base, "consensus_reason": "entity relevance disagreement"}
    if label_a.language != label_b.language:
        return {**base, "consensus_reason": "language disagreement"}
    if not label_a.entity_relevant:
        return {
            **base,
            "consensus_status": "accepted",
            "consensus_reason": "both models agree article is irrelevant",
            "language": label_a.language,
        }
    if label_a.language != "en":
        return {**base, "consensus_reason": "non-English article"}
    if label_a.sentiment is None or label_b.sentiment is None:
        return {
            **base,
            "consensus_reason": (
                "one or both models found no defensible company-specific sentiment"
            ),
        }
    if label_a.sentiment != label_b.sentiment:
        return {**base, "consensus_reason": "sentiment disagreement"}
    category = (
        label_a.event_category
        if label_a.event_category == label_b.event_category
        else "other"
    )
    return {
        **base,
        "entity_relevant": True,
        "sentiment_label": label_a.sentiment or "",
        "event_category": category,
        "language": "en",
        "consensus_status": "accepted",
        "consensus_reason": "independent model agreement",
    }


def consensus_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "consensus_coverage": 0.0,
            "entity_kappa": None,
            "sentiment_kappa": None,
            "parse_success_rate": 0.0,
        }
    completed = frame.get(
        "labelling_completed", pd.Series(False, index=frame.index)
    ).astype(bool)
    accepted_status = frame.get(
        "consensus_status", pd.Series("", index=frame.index)
    ).eq("accepted")
    label_source = frame.get(
        "label_source", pd.Series("", index=frame.index)
    ).astype(str).str.casefold()
    accepted = accepted_status & label_source.eq("llm_consensus")
    parse_success = float(completed.mean())
    coverage = float(accepted[completed].mean()) if completed.any() else 0.0
    entity_kappa: float | None = None
    sentiment_kappa: float | None = None
    if completed.sum() >= 2:
        entity_value = float(
            cohen_kappa_score(
                frame.loc[completed, "labeler_a_entity_relevant"].astype(bool),
                frame.loc[completed, "labeler_b_entity_relevant"].astype(bool),
            )
        )
        entity_kappa = entity_value if math.isfinite(entity_value) else None
        relevant = completed & frame["labeler_a_entity_relevant"].astype(bool) & frame[
            "labeler_b_entity_relevant"
        ].astype(bool)
        if relevant.sum() >= 2:
            sentiment_value = float(
                cohen_kappa_score(
                    frame.loc[relevant, "labeler_a_sentiment"],
                    frame.loc[relevant, "labeler_b_sentiment"],
                    labels=list(SENTIMENT_LABELS),
                )
            )
            sentiment_kappa = (
                sentiment_value if math.isfinite(sentiment_value) else None
            )
    return {
        "rows": int(len(frame)),
        "consensus_rows": int(accepted.sum()),
        "abstained_rows": int((completed & ~accepted).sum()),
        "consensus_coverage": coverage,
        "entity_kappa": entity_kappa,
        "sentiment_kappa": sentiment_kappa,
        "parse_success_rate": parse_success,
    }


def build_llm_approval_queue(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose only completed abstentions for optional user adjudication."""

    if frame.empty:
        return pd.DataFrame()
    status = frame.get(
        "consensus_status", pd.Series("", index=frame.index)
    ).astype(str).str.casefold()
    queue = frame.loc[status.eq("abstained")].copy()
    if queue.empty:
        return queue
    queue["approval_status"] = "pending"
    queue["approved_entity_relevant"] = ""
    queue["approved_sentiment_label"] = ""
    queue["approved_event_category"] = ""
    queue["approval_notes"] = ""
    keep = [
        "article_id",
        "ticker",
        "company_name",
        "published_at",
        "source_domain",
        "url",
        "title",
        "model_text",
        "consensus_reason",
        "labeler_a_entity_relevant",
        "labeler_a_sentiment",
        "labeler_a_confidence",
        "labeler_a_error",
        "labeler_b_entity_relevant",
        "labeler_b_sentiment",
        "labeler_b_confidence",
        "labeler_b_error",
        "approval_status",
        "approved_entity_relevant",
        "approved_sentiment_label",
        "approved_event_category",
        "approval_notes",
    ]
    return queue.reindex(columns=keep)


def apply_llm_approvals(
    frame: pd.DataFrame, approvals: pd.DataFrame | None
) -> pd.DataFrame:
    """Apply explicit decisions only to rows that originally abstained."""

    if approvals is None or approvals.empty or frame.empty:
        return frame.copy()
    output = frame.copy()
    indexed = approvals.copy()
    indexed["article_id"] = indexed["article_id"].astype(str)
    indexed = indexed.drop_duplicates("article_id", keep="last").set_index("article_id")
    for index, row in output.iterrows():
        article_id = str(row.get("article_id", ""))
        if article_id not in indexed.index:
            continue
        if str(row.get("consensus_status", "")).casefold() != "abstained":
            continue
        decision = indexed.loc[article_id]
        if str(decision.get("approval_status", "")).strip().casefold() != "approved":
            continue
        entity = str(decision.get("approved_entity_relevant", "")).strip().casefold()
        if entity not in {"true", "false", "1", "0", "yes", "no"}:
            continue
        is_relevant = entity in {"true", "1", "yes"}
        sentiment = str(
            decision.get("approved_sentiment_label", "")
        ).strip().casefold()
        if is_relevant and sentiment not in SENTIMENT_VALUES:
            continue
        output.at[index, "entity_relevant"] = is_relevant
        output.at[index, "sentiment_label"] = sentiment if is_relevant else ""
        output.at[index, "event_category"] = str(
            decision.get("approved_event_category", "other")
        ).strip().casefold() or "other"
        output.at[index, "consensus_status"] = "accepted"
        output.at[index, "consensus_reason"] = "user adjudicated LLM abstention"
        output.at[index, "label_source"] = "user_adjudicated_after_llm_abstention"
    return output


def _label_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarise usable weak labels without declaring them training-ready."""

    if frame.empty or "consensus_status" not in frame:
        return {
            "accepted_consensus_rows": 0,
            "accepted_relevant_sentiment_rows": 0,
            "accepted_irrelevant_rows": 0,
            "accepted_relevant_by_ticker": {},
            "accepted_relevant_by_year": {},
            "accepted_sentiment_distribution": {},
        }
    accepted = frame.loc[
        frame["consensus_status"].astype(str).str.casefold().eq("accepted")
    ].copy()
    relevant_mask = accepted.get(
        "entity_relevant", pd.Series(False, index=accepted.index)
    ).map(
        lambda value: value is True
        or str(value).strip().casefold() in {"true", "1", "yes"}
    )
    relevant = accepted.loc[relevant_mask].copy()
    irrelevant = accepted.loc[~relevant_mask].copy()
    published = pd.to_datetime(
        relevant.get("published_at", pd.Series(dtype=object)),
        utc=True,
        errors="coerce",
    )

    def counts(series: pd.Series) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in series.dropna().astype(str).value_counts().sort_index().items()
            if str(key).strip()
        }

    return {
        "accepted_consensus_rows": int(len(accepted)),
        "accepted_relevant_sentiment_rows": int(len(relevant)),
        "accepted_irrelevant_rows": int(len(irrelevant)),
        "accepted_relevant_by_ticker": counts(
            relevant.get("ticker", pd.Series(dtype=str))
        ),
        "accepted_relevant_by_year": counts(published.dt.year.dropna().astype(int)),
        "accepted_sentiment_distribution": counts(
            relevant.get("sentiment_label", pd.Series(dtype=str))
        ),
    }


def label_bursa_news_consensus(
    frame: pd.DataFrame,
    *,
    checkpoint_path: Path = DEFAULT_LABEL_CHECKPOINT,
    heartbeat_path: Path = DEFAULT_HEARTBEAT_PATH,
    completion: Callable[..., Any] = complete_chat_with_model,
    model_a: str = LABEL_MODEL_A,
    model_b: str = LABEL_MODEL_B,
    minimum_confidence: float = MINIMUM_LABEL_CONFIDENCE,
    max_runtime_minutes: float = 45.0,
    max_requests: int | None = None,
    request_interval_seconds: float = 1.0,
    recommended_start_year: int = 2019,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if model_a == model_b:
        raise ValueError("consensus label roles require two different model IDs")
    if int(recommended_start_year) < 2015 or int(recommended_start_year) > 2025:
        raise ValueError("recommended_start_year must be between 2015 and 2025")
    prepared = (
        frame.copy()
        if {"article_id", "model_text", "split"}.issubset(frame.columns)
        else build_annotation_template(
            frame,
            recommended_start_year=int(recommended_start_year),
        )
    )
    cache = _checkpoint_records(Path(checkpoint_path))
    failure_attempts, terminal_failures = _checkpoint_failures(
        Path(checkpoint_path)
    )
    prompt_sha256 = _prompt_hash()
    started = time.monotonic()
    requests = 0
    output_rows: list[dict[str, Any]] = []
    stop_reason: str | None = None
    for source_row in prepared.to_dict(orient="records"):
        if (time.monotonic() - started) / 60.0 >= max_runtime_minutes:
            stop_reason = "maximum runtime reached"
            break
        article_id = _article_id(source_row)
        labels: dict[str, LLMArticleLabel] = {}
        terminal_roles: list[str] = []
        row = {**source_row, "article_id": article_id}
        for role, model in (("labeler_a", model_a), ("labeler_b", model_b)):
            key = (article_id, model, prompt_sha256)
            record = cache.get(key)
            if record is None and key in terminal_failures:
                row[f"{role}_model"] = model
                row[f"{role}_error"] = terminal_failures[key]
                terminal_roles.append(role)
                continue
            if record is None:
                if max_requests is not None and requests >= max_requests:
                    stop_reason = "maximum request count reached"
                    break
                try:
                    content = _call_content(
                        completion,
                        build_label_messages(source_row),
                        model=model,
                    )
                    label = parse_llm_label(content)
                except (OpenRouterError, ValueError, TypeError) as exc:
                    if _is_daily_free_model_limit(exc):
                        _append_checkpoint(
                            Path(checkpoint_path),
                            {
                                "article_id": article_id,
                                "model": model,
                                "prompt_sha256": prompt_sha256,
                                "status": "quota_paused",
                                "attempt": failure_attempts.get(key, 0),
                                "error": str(exc),
                                "created_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        stop_reason = (
                            f"{model} daily free-model quota reached; labelling is "
                            "paused until the provider quota resets without consuming "
                            "a retry."
                        )
                        break
                    attempt = failure_attempts.get(key, 0) + 1
                    terminal_error = attempt >= MAX_MODEL_ATTEMPTS
                    _append_checkpoint(
                        Path(checkpoint_path),
                        {
                            "article_id": article_id,
                            "model": model,
                            "prompt_sha256": prompt_sha256,
                            "status": (
                                "terminal_error"
                                if terminal_error
                                else "retryable_error"
                            ),
                            "attempt": attempt,
                            "error": str(exc),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    failure_attempts[key] = attempt
                    if terminal_error:
                        terminal_failures[key] = str(exc)
                        row[f"{role}_model"] = model
                        row[f"{role}_error"] = str(exc)
                        terminal_roles.append(role)
                        continue
                    stop_reason = (
                        f"{model} labelling stopped after retryable attempt "
                        f"{attempt}/{MAX_MODEL_ATTEMPTS}: {exc}"
                    )
                    break
                requests += 1
                record = {
                    "article_id": article_id,
                    "model": model,
                    "prompt_sha256": prompt_sha256,
                    "status": "completed",
                    "label": label.to_dict(),
                    "raw_response_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                _append_checkpoint(Path(checkpoint_path), record)
                cache[key] = record
                if request_interval_seconds > 0:
                    time.sleep(request_interval_seconds)
            try:
                label = LLMArticleLabel(**record["label"])
            except (KeyError, TypeError, ValueError) as exc:
                stop_reason = f"invalid cached label for {model}: {exc}"
                break
            labels[role] = label
            row.update(_role_columns(role, model, label))
        if len(labels) != 2:
            if stop_reason is not None:
                break
            missing = ", ".join(terminal_roles) or "unknown label role"
            for role in ("labeler_a", "labeler_b"):
                row.setdefault(f"{role}_model", model_a if role == "labeler_a" else model_b)
                row.setdefault(f"{role}_entity_relevant", False)
                row.setdefault(f"{role}_sentiment", "")
                row.setdefault(f"{role}_confidence", 0.0)
            row.update(
                {
                    "entity_relevant": "",
                    "sentiment_label": "",
                    "event_category": "",
                    "label_source": "",
                    "consensus_status": "abstained",
                    "consensus_reason": (
                        "Free automated labelling remained unavailable after "
                        f"{MAX_MODEL_ATTEMPTS} attempts for: {missing}."
                    ),
                    "automatic_entity_relevant": str(
                        source_row.get(
                            "automatic_entity_relevant",
                            source_row.get("automatic_relevance", ""),
                        )
                    ).strip().casefold()
                    in {"included", "true", "1", "yes"},
                    "labelling_completed": False,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": prompt_sha256,
                    "labelled_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            output_rows.append(row)
            _atomic_json(
                Path(heartbeat_path),
                {
                    "status": "running",
                    "completed_articles": len(output_rows),
                    "requests_this_run": requests,
                    "last_article_id": article_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue
        row.update(
            _consensus(
                labels["labeler_a"],
                labels["labeler_b"],
                minimum_confidence=minimum_confidence,
            )
        )
        row.update(
            {
                "automatic_entity_relevant": str(
                    source_row.get(
                        "automatic_entity_relevant",
                        source_row.get("automatic_relevance", ""),
                    )
                ).strip().casefold()
                in {"included", "true", "1", "yes"},
                "labelling_completed": True,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_sha256,
                "labelled_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        output_rows.append(row)
        _atomic_json(
            Path(heartbeat_path),
            {
                "status": "running",
                "completed_articles": len(output_rows),
                "requests_this_run": requests,
                "last_article_id": article_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    result = pd.DataFrame(output_rows)
    metrics = consensus_metrics(result)
    report = {
        "schema_version": "bursa-news-llm-consensus-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_source": "llm_consensus",
        "model_a": model_a,
        "model_b": model_b,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "minimum_confidence": float(minimum_confidence),
        "recommended_start_year": int(recommended_start_year),
        "input_rows": int(len(prepared)),
        "completed_rows": int(len(result)),
        "requests_this_run": requests,
        "stop_reason": stop_reason,
        **metrics,
    }
    _atomic_json(
        Path(heartbeat_path),
        {
            "status": "completed" if stop_reason is None else "paused",
            **report,
        },
    )
    return result, report


def run_llm_consensus_labelling(
    source_path: Path | str,
    *,
    output_path: Path | str = DEFAULT_LABEL_OUTPUT,
    report_path: Path | str = DEFAULT_LABEL_REPORT,
    checkpoint_path: Path | str = DEFAULT_LABEL_CHECKPOINT,
    heartbeat_path: Path | str = DEFAULT_HEARTBEAT_PATH,
    approval_queue_path: Path | str = DEFAULT_APPROVAL_QUEUE,
    approvals_path: Path | str | None = None,
    max_runtime_minutes: float = 45.0,
    max_requests: int | None = None,
    request_interval_seconds: float = 1.0,
    recommended_start_year: int = 2019,
) -> dict[str, Any]:
    non_free_models = [
        model
        for model in (LABEL_MODEL_A, LABEL_MODEL_B)
        if not str(model).casefold().endswith(":free")
    ]
    if non_free_models:
        raise ValueError(
            "Historical Bursa-news automation is free-only; refusing non-free "
            f"OpenRouter models: {non_free_models}"
        )
    labelled, report = label_bursa_news_consensus(
        read_table(source_path),
        checkpoint_path=Path(checkpoint_path),
        heartbeat_path=Path(heartbeat_path),
        max_runtime_minutes=max_runtime_minutes,
        max_requests=max_requests,
        request_interval_seconds=request_interval_seconds,
        recommended_start_year=recommended_start_year,
    )
    approvals = read_table(approvals_path) if approvals_path else None
    labelled = apply_llm_approvals(labelled, approvals)
    approval_queue = build_llm_approval_queue(labelled)
    queue_destination = Path(approval_queue_path)
    queue_destination.parent.mkdir(parents=True, exist_ok=True)
    approval_queue.to_json(
        queue_destination, orient="records", lines=True, date_format="iso"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.casefold() == ".csv":
        labelled.to_csv(destination, index=False)
    elif destination.suffix.casefold() in {".parquet", ".pq"}:
        labelled.to_parquet(destination, index=False)
    else:
        labelled.to_json(
            destination, orient="records", lines=True, date_format="iso"
        )
    report_destination = Path(report_path)
    final_report = {
        **report,
        "free_only": True,
        "output": str(destination),
        "approval_queue": str(queue_destination),
        "pending_approval_rows": int(len(approval_queue)),
        "approvals_applied": str(approvals_path) if approvals_path else None,
        "weak_label_distribution": _label_distribution(labelled),
    }
    _atomic_json(report_destination, final_report)
    return {**final_report, "report": str(report_destination)}
