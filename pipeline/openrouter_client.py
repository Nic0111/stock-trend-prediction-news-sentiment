from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_FALLBACK_MODEL = "google/gemma-4-26b-a4b-it:free"
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
OPENROUTER_API_KEY_FILE = Path(__file__).resolve().parents[1] / "config" / "openrouter_api_key.txt"


class OpenRouterError(RuntimeError):
    """A safe, user-facing error from the OpenRouter adapter."""


class OpenRouterConfigurationError(OpenRouterError):
    """Raised when the application has no OpenRouter API key configured."""


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when a provider rejects a request because it is rate-limited."""


@dataclass(frozen=True)
class OpenRouterAnswer:
    content: str
    model: str


def _config_file_key() -> str | None:
    """Read the optional project-local key file, without exposing its contents."""
    try:
        value = OPENROUTER_API_KEY_FILE.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def get_openrouter_api_key() -> str | None:
    """Read the key from the process environment or ignored config file."""
    value = os.getenv(OPENROUTER_API_KEY_ENV_VAR)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _config_file_key()


def _extract_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return ""


def _error_message(raw_body: bytes) -> str:
    try:
        payload = json.loads(raw_body.decode("utf-8", errors="replace"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "The provider returned an unreadable error."
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        metadata = error.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("raw"), str):
            raw_message = metadata["raw"].strip()
            if raw_message:
                return raw_message[:500]
        if isinstance(error.get("message"), str):
            return error["message"].strip() or "The provider rejected the request."
    return "The provider rejected the request."


def _complete_chat_once(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str,
    temperature: float = 0.35,
    max_tokens: int = 650,
    timeout: float = 45.0,
    response_format: Mapping[str, Any] | None = None,
    reasoning: Mapping[str, Any] | None = None,
) -> str:
    """Send one non-streaming chat completion request to OpenRouter."""
    api_key = get_openrouter_api_key()
    if not api_key:
        raise OpenRouterConfigurationError(
            "OpenRouter is not configured. Add a key to config/openrouter_api_key.txt or set OPENROUTER_API_KEY."
        )

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        request_payload["response_format"] = dict(response_format)
    if reasoning is not None:
        request_payload["reasoning"] = dict(reasoning)
    payload = json.dumps(request_payload).encode("utf-8")
    request = Request(
        OPENROUTER_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://stocksense.local",
            "X-Title": "StockSense Ask & Learn",
        },
    )

    for attempt in range(2):
        try:
            with urlopen(request, timeout=timeout) as response:
                raw_body = response.read()
            break
        except HTTPError as exc:
            error_detail = _error_message(exc.read())
            if (exc.code == 429 or exc.code >= 500) and attempt == 0:
                time.sleep(1.5)
                continue
            error_type = OpenRouterRateLimitError if exc.code == 429 else OpenRouterError
            raise error_type(f"OpenRouter returned HTTP {exc.code}: {error_detail}") from exc
        except URLError as exc:
            raise OpenRouterError("OpenRouter could not be reached. Check the network connection.") from exc
        except TimeoutError as exc:
            raise OpenRouterError("OpenRouter took too long to respond. Please try again.") from exc

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenRouterError("OpenRouter returned an invalid response.") from exc

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("OpenRouter returned no answer.")
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = _extract_content(message.get("content") if isinstance(message, dict) else None)
    if not content and isinstance(first_choice, dict):
        content = _extract_content(first_choice.get("text"))
    if not content:
        raise OpenRouterError("OpenRouter returned an empty answer.")
    return content


def complete_chat(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str = OPENROUTER_MODEL,
    fallback_model: str | None = OPENROUTER_FALLBACK_MODEL,
    temperature: float = 0.35,
    max_tokens: int = 650,
    timeout: float = 45.0,
) -> OpenRouterAnswer:
    """Use the primary model and fall back only when it is provider-rate-limited."""
    try:
        return OpenRouterAnswer(
            content=_complete_chat_once(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            ),
            model=model,
        )
    except OpenRouterConfigurationError:
        raise
    except OpenRouterRateLimitError as primary_error:
        if not fallback_model or fallback_model == model:
            raise
        try:
            return OpenRouterAnswer(
                content=_complete_chat_once(
                    messages,
                    model=fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                ),
                model=fallback_model,
            )
        except OpenRouterConfigurationError:
            raise
        except OpenRouterRateLimitError as fallback_error:
            raise OpenRouterRateLimitError(
                f"{primary_error} Fallback model {fallback_model} also failed: {fallback_error}"
            ) from fallback_error
        except OpenRouterError as fallback_error:
            raise OpenRouterError(
                f"{primary_error} Fallback model {fallback_model} also failed: {fallback_error}"
            ) from fallback_error


def complete_chat_with_model(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    timeout: float = 60.0,
    response_format: Mapping[str, Any] | None = None,
    reasoning: Mapping[str, Any] | None = None,
) -> OpenRouterAnswer:
    """Call one fixed model without substituting a fallback.

    Offline consensus labelling assigns independent model roles. Substituting a
    fallback into either role would make agreement statistics misleading, so
    this entry point deliberately fails closed when the requested model is
    unavailable.
    """

    return OpenRouterAnswer(
        content=_complete_chat_once(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            reasoning=reasoning,
        ),
        model=model,
    )
