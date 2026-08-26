from __future__ import annotations

import unittest

from backend.app.main import _aggregate_response_is_unusable, _clean_tutor_prose, _extract_grounded_answer


class TutorGroundingTests(unittest.TestCase):
    def test_valid_markers_become_references_and_are_hidden(self) -> None:
        text, references = _extract_grounded_answer(
            "The model estimates 54% upward probability [MODEL-1]. "
            "Do not trust this invented source [MODEL-99].",
            {
                "model-1": {
                    "id": "model-1",
                    "type": "model",
                    "label": "Stock trend model",
                    "value": "Up, probability_up=0.540",
                    "as_of": "2026-07-21",
                }
            },
        )
        self.assertNotIn("[MODEL-1]", text)
        self.assertNotIn("[MODEL-99]", text)
        self.assertEqual([item.id for item in references], ["model-1"])

    def test_prompt_echo_is_reduced_to_connected_perspective_paragraphs(self) -> None:
        echoed = (
            "We need to produce answer two short paragraphs for beginner. Actually instruction: "
            "start with Stock perspective then News perspective. Stock perspective: The stock model points Down "
            "with a 55.9% probability for this horizon. News perspective: Two usable articles were retrieved, "
            "but low coverage means the news is context only. This satisfies two paragraphs."
        )
        cleaned = _clean_tutor_prose(echoed)
        self.assertEqual(
            cleaned,
            "The stock model points Down with a 55.9% probability for this horizon.\n\n"
            "Two usable articles were retrieved, but low coverage means the news is context only.",
        )
        self.assertNotIn("Actually instruction", cleaned)
        self.assertNotIn("Stock perspective:", cleaned)

    def test_prompt_only_response_uses_fallback(self) -> None:
        fallback = "The stock model provides the displayed probability.\n\nNews is shown as context only."
        cleaned = _clean_tutor_prose(
            "For the first request, answer in two short paragraphs. Do not output JSON.",
            fallback=fallback,
        )
        self.assertEqual(cleaned, fallback)

    def test_aggregate_prompt_echo_is_rejected_before_display(self) -> None:
        echoed = (
            "We need to answer: explain the current Aggregate Result. We need to use the supplied context. "
            "Thus displayed aggregate outcome is Up with 54%."
        )
        normal = "The stock trend points Up, while the available news is balanced."
        self.assertTrue(_aggregate_response_is_unusable(echoed))
        self.assertFalse(_aggregate_response_is_unusable(normal))


if __name__ == "__main__":
    unittest.main()
