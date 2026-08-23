"""Guarded promotion and rollback for versioned FinBERT ensemble variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "models" / "sentiment-runtime-config.json"
DEFAULT_BENCHMARK_REPORT = PROJECT_ROOT / "results" / "finbert-ensemble-benchmark" / "latest.json"
DEFAULT_POLICY = PROJECT_ROOT / "config" / "bursa_news_dataset_policy.json"


class SentimentEnsemblePromotionError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SentimentEnsemblePromotionError(f"Unable to read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SentimentEnsemblePromotionError(f"JSON object required: {path}")
    return payload


def _component_hashes(component: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return hashes for one model or an explicitly averaged seed component."""

    seed_models = component.get("seed_models")
    if isinstance(seed_models, list):
        resolved: dict[str, Any] = {}
        for index, seed_model in enumerate(seed_models):
            if not isinstance(seed_model, Mapping):
                return {}
            key = str(seed_model.get("seed", index))
            hashes = seed_model.get("artifact_hashes")
            if not isinstance(hashes, Mapping) or not hashes or key in resolved:
                return {}
            resolved[key] = dict(hashes)
        return resolved
    hashes = component.get("artifact_hashes")
    return hashes if isinstance(hashes, Mapping) else {}


def _hashes_match(config: Mapping[str, Any], report: Mapping[str, Any], variant: str) -> bool:
    challenger = (config.get("challengers") or {}).get(variant)
    report_hashes = report.get("artifact_hashes")
    if not isinstance(challenger, Mapping) or not isinstance(report_hashes, Mapping):
        return False
    components = challenger.get("components")
    if not isinstance(components, Mapping):
        return False
    for name in ("bursa", "phrasebank"):
        component = components.get(name)
        expected = _component_hashes(component) if isinstance(component, Mapping) else {}
        observed = report_hashes.get(name)
        if not expected or expected != observed:
            return False
    return True


def _candidate_from_report(report: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Validate the immutable runtime recipe emitted by the benchmark."""

    raw = report.get("runtime_candidate")
    if not isinstance(raw, Mapping):
        raise SentimentEnsemblePromotionError(
            "Benchmark report has no runtime candidate recipe; rerun the v2 benchmark"
        )
    candidate = dict(raw)
    if str(report.get("variant_id", "")) != str(variant):
        raise SentimentEnsemblePromotionError("Runtime candidate variant does not match report")
    if str(candidate.get("backend", "")).casefold() != "weighted_finbert_ensemble":
        raise SentimentEnsemblePromotionError("Benchmark runtime candidate is not a FinBERT ensemble")
    if not isinstance(candidate.get("weights"), Mapping) or not isinstance(
        candidate.get("components"), Mapping
    ):
        raise SentimentEnsemblePromotionError("Benchmark runtime candidate is incomplete")
    candidate["promotion_status"] = "promoted"
    return candidate


def _validate_promotion_protocol(
    report: Mapping[str, Any], *, policy: Mapping[str, Any]
) -> None:
    """Require the promotion protocol declared by the active policy.

    A report from the inspected 2025 retrospective evaluation is still useful
    development evidence, but it cannot impersonate an untouched 2026 final
    holdout.  When the policy requires the prospective protocol, only a sealed
    one-time 2026 evaluation of the exact runtime recipe may be approved.
    """

    ensemble_policy = (policy.get("sentiment_ensemble") or {}).get("promotion_gates") or {}
    protocol = report.get("promotion_protocol")
    if protocol is None:
        if bool(ensemble_policy.get("require_prospective_holdout", False)):
            raise SentimentEnsemblePromotionError(
                "Current policy requires a sealed 2026 prospective evaluation before promotion"
            )
        return
    if not isinstance(protocol, Mapping):
        raise SentimentEnsemblePromotionError("Benchmark promotion protocol is invalid")
    kind = str(protocol.get("kind", ""))
    if kind == "prospective_2026_sealed_holdout":
        required = {
            "independent_holdout": True,
            "labels_sealed": True,
            "evaluation_only_once": True,
        }
        missing = [name for name, expected in required.items() if protocol.get(name) is not expected]
        if missing:
            raise SentimentEnsemblePromotionError(
                "Prospective benchmark protocol is incomplete: " + ", ".join(missing)
            )
        if int(protocol.get("final_holdout_year", 0)) != 2026:
            raise SentimentEnsemblePromotionError("Prospective benchmark must use the sealed 2026 holdout")
        if str(protocol.get("exact_runtime_recipe", "")) != (
            "three_bursa_seed_average_then_phrasebank_probability_blend"
        ):
            raise SentimentEnsemblePromotionError(
                "Prospective benchmark does not represent the required three-seed runtime recipe"
            )
        candidate = report.get("runtime_candidate") or {}
        components = candidate.get("components") if isinstance(candidate, Mapping) else None
        bursa_component = components.get("bursa") if isinstance(components, Mapping) else None
        if not isinstance(bursa_component, Mapping) or len(bursa_component.get("seed_models") or []) != 3:
            raise SentimentEnsemblePromotionError(
                "Prospective benchmark candidate does not contain the required three Bursa seeds"
            )
        return
    if bool(ensemble_policy.get("require_prospective_holdout", False)):
        raise SentimentEnsemblePromotionError(
            "Current policy requires a sealed 2026 prospective evaluation before promotion"
        )
    if kind != "retrospective_locked_2025":
        raise SentimentEnsemblePromotionError("Unknown benchmark promotion protocol")
    if not bool(ensemble_policy.get("allow_retrospective_manual_promotion", False)):
        raise SentimentEnsemblePromotionError(
            "Current policy does not permit manual promotion from the retrospective 2025 test"
        )
    if not bool(protocol.get("manual_promotion_only")):
        raise SentimentEnsemblePromotionError(
            "Retrospective benchmark must require manual promotion"
        )
    if not bool(protocol.get("retrospective_test_disclosure")):
        raise SentimentEnsemblePromotionError(
            "Retrospective benchmark disclosure is missing"
        )


def promote_sentiment_ensemble(
    *,
    variant: str,
    benchmark_report: Path | str = DEFAULT_BENCHMARK_REPORT,
    runtime_config: Path | str = DEFAULT_RUNTIME_CONFIG,
    policy_path: Path | str = DEFAULT_POLICY,
    approve: bool = False,
) -> dict[str, Any]:
    """Activate only a benchmarked, independently-held-out ensemble variant."""

    if not approve:
        raise SentimentEnsemblePromotionError("Promotion requires --approve")
    config_path = Path(runtime_config)
    report_path = Path(benchmark_report)
    policy_file = Path(policy_path)
    config = _read_json(config_path)
    report = _read_json(report_path)
    policy = _read_json(policy_file)
    if str(report.get("variant_id", "")) != str(variant):
        raise SentimentEnsemblePromotionError(
            "Benchmark variant does not match the requested runtime variant"
        )
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or not bool(gates.get("eligible_for_promotion")):
        raise SentimentEnsemblePromotionError(
            "Benchmark gates did not pass; the challenger must remain opt-in"
        )
    _validate_promotion_protocol(report, policy=policy)
    prior = str(config.get("active_variant") or "finbert_phrasebank_v1")
    challengers = config.setdefault("challengers", {})
    if not isinstance(challengers, dict):
        raise SentimentEnsemblePromotionError("Runtime challengers must be a JSON object")
    if variant not in challengers:
        # Adding a candidate is safe only during this explicitly approved
        # promotion.  The recipe is copied from the immutable benchmark report
        # and then checked against the report hashes below.
        challengers[variant] = _candidate_from_report(report, variant)
    if not _hashes_match(config, report, variant):
        raise SentimentEnsemblePromotionError(
            "Configured component hashes do not match the benchmarked artifacts"
        )
    history = list(config.get("promotion_history") or [])
    history.append(
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "from": prior,
            "to": variant,
            "benchmark_report": str(report_path),
            "benchmark_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    )
    config["previous_active_variant"] = prior
    config["active_variant"] = variant
    config["promotion_history"] = history
    _atomic_json(config_path, config)
    return {
        "promoted": True,
        "previous_active_variant": prior,
        "active_variant": variant,
        "runtime_config": str(config_path),
    }


def manually_activate_sentiment_ensemble(
    *,
    variant: str,
    benchmark_report: Path | str = DEFAULT_BENCHMARK_REPORT,
    runtime_config: Path | str = DEFAULT_RUNTIME_CONFIG,
    approve: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """Activate a candidate with an explicit, auditable experimental override.

    This path intentionally does not alter the benchmark report or claim that
    the acceptance gates passed.  It is for a user-approved demo/field trial
    only; the active runtime entry is marked ``manual_override_active`` and the
    failed-gate snapshot is retained in both the candidate and promotion history.
    The normal ``promote`` action remains gate-protected.
    """

    if not approve:
        raise SentimentEnsemblePromotionError("Manual activation requires --approve")
    normalized_reason = str(reason or "").strip()
    if len(normalized_reason) < 12:
        raise SentimentEnsemblePromotionError(
            "Manual activation requires a meaningful --reason (at least 12 characters)"
        )

    config_path = Path(runtime_config)
    report_path = Path(benchmark_report)
    config = _read_json(config_path)
    report = _read_json(report_path)
    if str(report.get("variant_id", "")) != str(variant):
        raise SentimentEnsemblePromotionError(
            "Benchmark variant does not match the requested runtime variant"
        )

    candidate = _candidate_from_report(report, variant)
    gates = report.get("gates") if isinstance(report.get("gates"), Mapping) else {}
    artifact_hashes = report.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise SentimentEnsemblePromotionError(
            "Manual activation requires an immutable benchmark artifact manifest"
        )

    challengers = config.setdefault("challengers", {})
    if not isinstance(challengers, dict):
        raise SentimentEnsemblePromotionError("Runtime challengers must be a JSON object")
    candidate["promotion_status"] = "manual_override_active"
    candidate["manual_override"] = {
        "active": True,
        "reason": normalized_reason,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_report": str(report_path),
        "benchmark_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "gates_eligible_for_promotion": bool(gates.get("eligible_for_promotion", False)),
        "failed_gates": [
            str(name)
            for name, passed in (gates.get("checks") or {}).items()
            if passed is False
        ],
    }
    challengers[variant] = candidate
    if not _hashes_match(config, report, variant):
        raise SentimentEnsemblePromotionError(
            "Configured component hashes do not match the benchmarked artifacts"
        )

    prior = str(config.get("active_variant") or "finbert_phrasebank_v1")
    history = list(config.get("promotion_history") or [])
    history.append(
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "from": prior,
            "to": variant,
            "benchmark_report": str(report_path),
            "benchmark_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "manual_override": True,
            "reason": normalized_reason,
            "gates_eligible_for_promotion": bool(gates.get("eligible_for_promotion", False)),
        }
    )
    config["previous_active_variant"] = prior
    config["active_variant"] = variant
    config["promotion_history"] = history
    _atomic_json(config_path, config)
    return {
        "manually_activated": True,
        "manual_override": True,
        "previous_active_variant": prior,
        "active_variant": variant,
        "runtime_config": str(config_path),
        "gates_eligible_for_promotion": bool(gates.get("eligible_for_promotion", False)),
        "failed_gates": [
            str(name)
            for name, passed in (gates.get("checks") or {}).items()
            if passed is False
        ],
    }


def rollback_sentiment_ensemble(
    *,
    runtime_config: Path | str = DEFAULT_RUNTIME_CONFIG,
    variant: str | None = None,
    approve: bool = False,
) -> dict[str, Any]:
    """Restore a known runtime variant without altering model artifacts."""

    if not approve:
        raise SentimentEnsemblePromotionError("Rollback requires --approve")
    path = Path(runtime_config)
    config = _read_json(path)
    current = str(config.get("active_variant") or "finbert_phrasebank_v1")
    target = str(variant or config.get("previous_active_variant") or "finbert_phrasebank_v1")
    valid = {"finbert_phrasebank_v1", *(config.get("challengers") or {}).keys()}
    if target not in valid:
        raise SentimentEnsemblePromotionError(f"Unknown rollback variant: {target}")
    config["previous_active_variant"] = current
    config["active_variant"] = target
    history = list(config.get("promotion_history") or [])
    history.append(
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "from": current,
            "to": target,
            "reason": "manual rollback",
        }
    )
    config["promotion_history"] = history
    _atomic_json(path, config)
    return {
        "rolled_back": True,
        "previous_active_variant": current,
        "active_variant": target,
        "runtime_config": str(path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guarded FinBERT ensemble promotion")
    actions = parser.add_subparsers(dest="action", required=True)
    promote = actions.add_parser("promote")
    promote.add_argument("--variant", required=True)
    promote.add_argument("--benchmark-report", default=str(DEFAULT_BENCHMARK_REPORT))
    promote.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG))
    promote.add_argument("--policy", default=str(DEFAULT_POLICY))
    promote.add_argument("--approve", action="store_true")
    manual = actions.add_parser(
        "manual-activate",
        help="Activate a candidate as an explicitly marked experimental override.",
    )
    manual.add_argument("--variant", required=True)
    manual.add_argument("--benchmark-report", default=str(DEFAULT_BENCHMARK_REPORT))
    manual.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG))
    manual.add_argument("--reason", required=True)
    manual.add_argument("--approve", action="store_true")
    rollback = actions.add_parser("rollback")
    rollback.add_argument("--variant", default=None)
    rollback.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG))
    rollback.add_argument("--approve", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "promote":
            result = promote_sentiment_ensemble(
                variant=args.variant,
                benchmark_report=args.benchmark_report,
                runtime_config=args.runtime_config,
                policy_path=args.policy,
                approve=args.approve,
            )
        elif args.action == "manual-activate":
            result = manually_activate_sentiment_ensemble(
                variant=args.variant,
                benchmark_report=args.benchmark_report,
                runtime_config=args.runtime_config,
                reason=args.reason,
                approve=args.approve,
            )
        else:
            result = rollback_sentiment_ensemble(
                variant=args.variant,
                runtime_config=args.runtime_config,
                approve=args.approve,
            )
    except SentimentEnsemblePromotionError as exc:
        print(json.dumps({"completed": False, "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
