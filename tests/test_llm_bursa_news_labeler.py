import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.llm_bursa_news_labeler import (
    LABEL_MODEL_A,
    LABEL_MODEL_B,
    _label_distribution,
    apply_llm_approvals,
    build_llm_approval_queue,
    label_bursa_news_consensus,
    parse_llm_label,
)
from pipeline.openrouter_client import OpenRouterAnswer, OpenRouterRateLimitError


def _response(*, entity=True, sentiment="positive", confidence=0.91):
    return json.dumps(
        {
            "entity_relevant": entity,
            "sentiment": sentiment if entity else None,
            "event_category": "earnings" if entity else "other",
            "language": "en",
            "confidence": confidence,
            "evidence": ["named company reported results"],
            "rationale": "The article directly discusses the named company.",
        }
    )


def _frame():
    return pd.DataFrame(
        [
            {
                "article_id": "article-1",
                "ticker": "1155.KL",
                "company_name": "Malayan Banking Berhad",
                "published_at": "2023-06-01T00:00:00Z",
                "model_text": "Maybank reported higher quarterly profit.",
                "split": "training",
                "automatic_entity_relevant": True,
            }
        ]
    )


def test_parse_llm_label_accepts_fenced_json():
    label = parse_llm_label(f"```json\n{_response()}\n```")
    assert label.entity_relevant is True
    assert label.sentiment == "positive"
    assert label.confidence == pytest.approx(0.91)


def test_parse_llm_label_uses_first_complete_object_when_provider_duplicates_it():
    response = _response()

    label = parse_llm_label(f"{response}\n{response}")

    assert label.entity_relevant is True
    assert label.sentiment == "positive"


def test_irrelevant_article_discards_spurious_sentiment():
    payload = json.loads(_response(entity=False))
    payload["sentiment"] = "neutral"

    label = parse_llm_label(json.dumps(payload))

    assert label.entity_relevant is False
    assert label.sentiment is None


def test_unknown_descriptive_metadata_is_normalized_to_other():
    payload = json.loads(_response())
    payload["event_category"] = "market_update"
    payload["language"] = "English"

    label = parse_llm_label(json.dumps(payload))

    assert label.event_category == "other"
    assert label.language == "en"


def test_dual_model_consensus_and_checkpoint_resume(tmp_path: Path):
    calls = []

    def complete(messages, *, model, **kwargs):
        calls.append(model)
        assert "FinBERT" not in json.dumps(messages)
        return OpenRouterAnswer(_response(), model)

    checkpoint = tmp_path / "labels.jsonl"
    heartbeat = tmp_path / "heartbeat.json"
    first, report = label_bursa_news_consensus(
        _frame(),
        checkpoint_path=checkpoint,
        heartbeat_path=heartbeat,
        completion=complete,
        request_interval_seconds=0,
    )
    assert calls == [LABEL_MODEL_A, LABEL_MODEL_B]
    assert first.loc[0, "consensus_status"] == "accepted"
    assert first.loc[0, "sentiment_label"] == "positive"
    assert report["consensus_coverage"] == pytest.approx(1.0)

    def should_not_call(*args, **kwargs):
        raise AssertionError("checkpoint was not reused")

    second, _ = label_bursa_news_consensus(
        _frame(),
        checkpoint_path=checkpoint,
        heartbeat_path=heartbeat,
        completion=should_not_call,
        request_interval_seconds=0,
    )
    assert second.loc[0, "consensus_status"] == "accepted"


def test_disagreement_abstains(tmp_path: Path):
    def complete(messages, *, model, **kwargs):
        sentiment = "positive" if model == LABEL_MODEL_A else "negative"
        return OpenRouterAnswer(_response(sentiment=sentiment), model)

    labelled, report = label_bursa_news_consensus(
        _frame(),
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        heartbeat_path=tmp_path / "heartbeat.json",
        completion=complete,
        request_interval_seconds=0,
    )
    assert labelled.loc[0, "consensus_status"] == "abstained"
    assert labelled.loc[0, "sentiment_label"] == ""
    assert report["consensus_coverage"] == pytest.approx(0.0)
    queue = build_llm_approval_queue(labelled)
    assert queue["article_id"].tolist() == ["article-1"]

    approvals = queue.copy()
    approvals["approval_status"] = "approved"
    approvals["approved_entity_relevant"] = "true"
    approvals["approved_sentiment_label"] = "neutral"
    approvals["approved_event_category"] = "other"
    adjudicated = apply_llm_approvals(labelled, approvals)
    assert adjudicated.loc[0, "consensus_status"] == "accepted"
    assert adjudicated.loc[0, "label_source"] == (
        "user_adjudicated_after_llm_abstention"
    )
    assert adjudicated.loc[0, "sentiment_label"] == "neutral"


def test_relevant_but_sentiment_unavailable_abstains(tmp_path: Path):
    def complete(messages, *, model, **kwargs):
        return OpenRouterAnswer(
            _response(entity=True, sentiment=None, confidence=0.95),
            model,
        )

    labelled, report = label_bursa_news_consensus(
        _frame(),
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        heartbeat_path=tmp_path / "heartbeat.json",
        completion=complete,
        request_interval_seconds=0,
    )

    assert labelled.loc[0, "consensus_status"] == "abstained"
    assert labelled.loc[0, "sentiment_label"] == ""
    assert "no defensible" in labelled.loc[0, "consensus_reason"]
    assert report["consensus_coverage"] == pytest.approx(0.0)


def test_same_model_cannot_fill_both_roles(tmp_path: Path):
    with pytest.raises(ValueError, match="two different model"):
        label_bursa_news_consensus(
            _frame(),
            checkpoint_path=tmp_path / "checkpoint.jsonl",
            heartbeat_path=tmp_path / "heartbeat.json",
            model_a=LABEL_MODEL_A,
            model_b=LABEL_MODEL_A,
        )


def test_explicit_2015_start_includes_approved_historical_backfill(
    tmp_path: Path,
):
    frame = pd.DataFrame(
        [
            {
                "ticker": "1155.KL",
                "company_name": "Malayan Banking Berhad",
                "published_at": "2015-06-01T00:00:00Z",
                "title": "Maybank reports higher quarterly profit",
                "description": "Malayan Banking Berhad reported higher profit.",
                "url": "https://example.test/maybank-profit",
            }
        ]
    )

    labelled, report = label_bursa_news_consensus(
        frame,
        checkpoint_path=tmp_path / "checkpoint.jsonl",
        heartbeat_path=tmp_path / "heartbeat.json",
        recommended_start_year=2015,
        max_runtime_minutes=0,
        request_interval_seconds=0,
    )

    assert labelled.empty
    assert report["input_rows"] == 1
    assert report["recommended_start_year"] == 2015


def test_label_distribution_keeps_entity_and_sentiment_counts_separate():
    frame = pd.DataFrame(
        [
            {
                "consensus_status": "accepted",
                "entity_relevant": True,
                "ticker": "1155.KL",
                "published_at": "2023-01-01T00:00:00Z",
                "sentiment_label": "positive",
            },
            {
                "consensus_status": "accepted",
                "entity_relevant": False,
                "ticker": "1155.KL",
                "published_at": "2023-01-02T00:00:00Z",
                "sentiment_label": "",
            },
            {
                "consensus_status": "abstained",
                "entity_relevant": True,
                "ticker": "1295.KL",
                "published_at": "2023-01-03T00:00:00Z",
                "sentiment_label": "negative",
            },
        ]
    )

    report = _label_distribution(frame)

    assert report["accepted_consensus_rows"] == 2
    assert report["accepted_relevant_sentiment_rows"] == 1
    assert report["accepted_irrelevant_rows"] == 1
    assert report["accepted_relevant_by_ticker"] == {"1155.KL": 1}
    assert report["accepted_relevant_by_year"] == {"2023": 1}
    assert report["accepted_sentiment_distribution"] == {"positive": 1}


def test_repeated_provider_failure_becomes_abstention_and_does_not_block(
    tmp_path: Path,
):
    def unavailable(*args, **kwargs):
        raise ValueError("provider returned no answer")

    checkpoint = tmp_path / "checkpoint.jsonl"
    heartbeat = tmp_path / "heartbeat.json"
    labelled = pd.DataFrame()
    report = {}
    # Each of the two independent label roles receives three bounded attempts.
    for _ in range(6):
        labelled, report = label_bursa_news_consensus(
            _frame(),
            checkpoint_path=checkpoint,
            heartbeat_path=heartbeat,
            completion=unavailable,
            request_interval_seconds=0,
        )

    assert report["stop_reason"] is None
    assert labelled.loc[0, "consensus_status"] == "abstained"
    assert bool(labelled.loc[0, "labelling_completed"]) is False
    assert "3 attempts" in labelled.loc[0, "consensus_reason"]


def test_daily_free_model_limit_pauses_without_consuming_a_retry(tmp_path: Path):
    calls = []

    def daily_limit(*args, **kwargs):
        calls.append(kwargs["model"])
        raise OpenRouterRateLimitError(
            "OpenRouter returned HTTP 429: Rate limit exceeded: "
            "free-models-per-day-high-balance."
        )

    checkpoint = tmp_path / "checkpoint.jsonl"
    labelled, report = label_bursa_news_consensus(
        _frame(),
        checkpoint_path=checkpoint,
        heartbeat_path=tmp_path / "heartbeat.json",
        completion=daily_limit,
        request_interval_seconds=0,
    )

    assert labelled.empty
    assert calls == [LABEL_MODEL_A]
    assert "daily free-model quota reached" in str(report["stop_reason"])
    checkpoint_rows = [
        json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()
    ]
    assert checkpoint_rows[-1]["status"] == "quota_paused"
    assert checkpoint_rows[-1]["attempt"] == 0
