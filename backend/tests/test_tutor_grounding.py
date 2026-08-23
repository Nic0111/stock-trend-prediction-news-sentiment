from __future__ import annotations

import unittest

from backend.app.main import _extract_grounded_answer


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


if __name__ == "__main__":
    unittest.main()
