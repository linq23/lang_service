"""Detector unit tests. The fastText wheel is never imported here: the model is
126 MB and a stub proves the parts we actually wrote (label stripping, the
empty-input short circuit, clamping) without it.

    python -m unittest discover -s test
"""

from __future__ import annotations

import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from detector import UNKNOWN, Detector, sanitize  # noqa: E402


class FakeModel:
    """Stands in for fasttext.load_model()'s return value."""

    def __init__(self, labels, probabilities):
        self.labels = labels
        self.probabilities = probabilities
        self.seen = []

    def predict(self, text, k=1):
        self.seen.append(text)
        return self.labels, self.probabilities


def detector_with(model) -> Detector:
    detector = Detector("/nonexistent/lid.176.bin", max_text_chars=20)
    detector._model = model  # bypass load(); the wheel is not installed in CI
    return detector


class SanitizeTest(unittest.TestCase):
    def test_newlines_tabs_and_nulls_become_spaces(self):
        self.assertEqual(sanitize("a\nb\r\nc\td\x00e", 100), "a b  c d e")

    def test_trims_surrounding_whitespace(self):
        self.assertEqual(sanitize("  привіт  ", 100), "привіт")

    def test_truncates_to_max_chars(self):
        self.assertEqual(sanitize("abcdefghij", 4), "abcd")

    def test_whitespace_only_collapses_to_empty(self):
        self.assertEqual(sanitize("\n\t  \r", 100), "")

    def test_non_string_is_empty(self):
        self.assertEqual(sanitize(None, 100), "")


class DetectManyTest(unittest.TestCase):
    def test_strips_label_prefix_and_rounds_confidence(self):
        detector = detector_with(FakeModel(["__label__uk"], [0.993142]))
        self.assertEqual(detector.detect_many(["привіт"]), [{"language": "uk", "confidence": 0.9931}])

    def test_lowercases_label(self):
        detector = detector_with(FakeModel(["__label__EN"], [0.5]))
        self.assertEqual(detector.detect_many(["hello"])[0]["language"], "en")

    def test_clamps_confidence_into_zero_one(self):
        detector = detector_with(FakeModel(["__label__en"], [1.0000004]))
        self.assertEqual(detector.detect_many(["hello"])[0]["confidence"], 1.0)
        detector = detector_with(FakeModel(["__label__en"], [-0.2]))
        self.assertEqual(detector.detect_many(["hello"])[0]["confidence"], 0.0)

    def test_empty_input_never_reaches_the_model(self):
        model = FakeModel(["__label__en"], [0.99])
        detector = detector_with(model)
        self.assertEqual(detector.detect_many(["", "   ", "\n"]), [
            {"language": UNKNOWN, "confidence": 0.0},
            {"language": UNKNOWN, "confidence": 0.0},
            {"language": UNKNOWN, "confidence": 0.0},
        ])
        self.assertEqual(model.seen, [])

    def test_one_result_per_input_in_order(self):
        detector = detector_with(FakeModel(["__label__ru"], [0.8]))
        results = detector.detect_many(["a", "", "b"])
        self.assertEqual([r["language"] for r in results], ["ru", UNKNOWN, "ru"])

    def test_text_is_sanitized_and_capped_before_predict(self):
        model = FakeModel(["__label__en"], [0.9])
        detector = detector_with(model)  # max_text_chars=20
        detector.detect_many(["line one\nline two padded out well past the cap"])
        self.assertEqual(model.seen, ["line one line two pa"])

    def test_empty_label_falls_back_to_unknown(self):
        detector = detector_with(FakeModel(["__label__"], [0.9]))
        self.assertEqual(detector.detect_many(["hello"]), [{"language": UNKNOWN, "confidence": 0.0}])

    def test_unusable_prediction_falls_back_to_unknown(self):
        detector = detector_with(FakeModel([], []))
        self.assertEqual(detector.detect_many(["hello"]), [{"language": UNKNOWN, "confidence": 0.0}])

    def test_raises_when_model_is_not_loaded(self):
        detector = Detector("/nonexistent/lid.176.bin")
        self.assertFalse(detector.ready)
        with self.assertRaises(RuntimeError):
            detector.detect_many(["hello"])


class LoadTest(unittest.TestCase):
    def test_failure_is_recorded_not_raised(self):
        detector = Detector("/nonexistent/lid.176.bin")
        detector.load()
        self.assertFalse(detector.ready)
        self.assertIsNotNone(detector.load_error)


if __name__ == "__main__":
    unittest.main()
