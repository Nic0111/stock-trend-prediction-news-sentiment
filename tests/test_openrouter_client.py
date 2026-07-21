import json
import os
import unittest
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import patch

from pipeline.openrouter_client import (
    OPENROUTER_ENDPOINT,
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_MODEL,
    OpenRouterConfigurationError,
    OpenRouterRateLimitError,
    complete_chat,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self._body


class OpenRouterClientTests(unittest.TestCase):
    def test_complete_chat_uses_bearer_auth_and_extracts_response(self):
        response = _FakeResponse(
            {"choices": [{"message": {"content": "A probability is an estimate."}}]}
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch("pipeline.openrouter_client.urlopen", return_value=response) as mocked_urlopen:
                answer = complete_chat([{"role": "user", "content": "What is probability?"}])

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, OPENROUTER_ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(answer.content, "A probability is an estimate.")
        self.assertEqual(answer.model, OPENROUTER_MODEL)

    def test_missing_key_is_reported_without_a_network_call(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pipeline.openrouter_client._config_file_key", return_value=None):
                with patch("pipeline.openrouter_client.urlopen") as mocked_urlopen:
                    with self.assertRaises(OpenRouterConfigurationError):
                        complete_chat([{"role": "user", "content": "Hello"}])
        mocked_urlopen.assert_not_called()

    def test_rate_limit_surfaces_provider_detail_after_one_retry(self):
        error_body = json.dumps(
            {
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"raw": "Gemma is temporarily rate-limited upstream."},
                }
            }
        ).encode("utf-8")
        first_error = HTTPError(OPENROUTER_ENDPOINT, 429, "Too Many Requests", {}, BytesIO(error_body))
        second_error = HTTPError(OPENROUTER_ENDPOINT, 429, "Too Many Requests", {}, BytesIO(error_body))
        third_error = HTTPError(OPENROUTER_ENDPOINT, 429, "Too Many Requests", {}, BytesIO(error_body))
        fourth_error = HTTPError(OPENROUTER_ENDPOINT, 429, "Too Many Requests", {}, BytesIO(error_body))

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch(
                "pipeline.openrouter_client.urlopen",
                side_effect=[first_error, second_error, third_error, fourth_error],
            ):
                with patch("pipeline.openrouter_client.time.sleep") as mocked_sleep:
                    with self.assertRaisesRegex(OpenRouterRateLimitError, "temporarily rate-limited upstream"):
                        complete_chat([{"role": "user", "content": "Hello"}])
        self.assertEqual(mocked_sleep.call_count, 2)

    def test_rate_limit_falls_back_to_gemma_26b(self):
        error_body = json.dumps(
            {"error": {"metadata": {"raw": "Gemma 31B is temporarily rate-limited."}}}
        ).encode("utf-8")
        first_error = HTTPError(OPENROUTER_ENDPOINT, 429, "Too Many Requests", {}, BytesIO(error_body))
        second_error = HTTPError(OPENROUTER_ENDPOINT, 429, "Too Many Requests", {}, BytesIO(error_body))
        fallback_response = _FakeResponse(
            {"choices": [{"message": {"content": "Fallback answer."}}]}
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch(
                "pipeline.openrouter_client.urlopen",
                side_effect=[first_error, second_error, fallback_response],
            ) as mocked_urlopen:
                with patch("pipeline.openrouter_client.time.sleep") as mocked_sleep:
                    answer = complete_chat([{"role": "user", "content": "Hello"}])

        models = [json.loads(call.args[0].data.decode("utf-8"))["model"] for call in mocked_urlopen.call_args_list]
        self.assertEqual(models, [OPENROUTER_MODEL, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODEL])
        self.assertEqual(answer.model, OPENROUTER_FALLBACK_MODEL)
        self.assertEqual(answer.content, "Fallback answer.")
        mocked_sleep.assert_called_once_with(1.5)


if __name__ == "__main__":
    unittest.main()
