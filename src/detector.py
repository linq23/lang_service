"""fastText language identification, wrapped so the HTTP layer stays dumb.

The model (lid.176.bin, 176 languages) is loaded once into the process and
lives for its lifetime — that is the whole reason this service exists instead
of PHP shelling out to the fasttext binary per post.
"""

from __future__ import annotations

import threading

# ISO 639-2 "undetermined". Mirrors App\Language\Model\Language::UNKNOWN in the
# backend; both sides must agree or the confidence gate has nothing to compare.
UNKNOWN = "und"

_LABEL_PREFIX = "__label__"


def sanitize(text: str, max_chars: int) -> str:
    """fastText predicts one line at a time and raises on embedded newlines."""
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("\x00", " ")
    cleaned = cleaned.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


class Detector:
    """Thread-safe front for one loaded fastText model."""

    def __init__(self, model_path: str, max_text_chars: int = 1000) -> None:
        self._model_path = model_path
        self._max_text_chars = max_text_chars
        self._model = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        """Blocking model load. Call once, off the request path."""
        try:
            import fasttext  # provided by the fasttext-predict wheel

            model = fasttext.load_model(self._model_path)
        except Exception as exc:  # noqa: BLE001 — surfaced through /ready, not raised
            self._load_error = f"{type(exc).__name__}: {exc}"
            return

        self._model = model
        self._load_error = None

    def detect_many(self, texts: list[str]) -> list[dict[str, object]]:
        """One {language, confidence} per input, in the same order.

        Empty (or emoji/punctuation-only) input never reaches the model: it has
        no signal, and fastText would answer with a confident-looking label
        anyway.
        """
        model = self._model
        if model is None:
            raise RuntimeError("model not loaded")

        results: list[dict[str, object]] = []
        for text in texts:
            cleaned = sanitize(text, self._max_text_chars)
            if cleaned == "":
                results.append({"language": UNKNOWN, "confidence": 0.0})
                continue
            results.append(self._predict_one(model, cleaned))

        return results

    def _predict_one(self, model, text: str) -> dict[str, object]:
        # The binding is not documented as thread-safe; predictions are
        # microseconds, so serializing them costs nothing at our volume.
        with self._lock:
            labels, probabilities = model.predict(text, k=1)

        try:
            label = str(labels[0])
            confidence = float(probabilities[0])
        except (IndexError, TypeError, ValueError):
            return {"language": UNKNOWN, "confidence": 0.0}

        if label.startswith(_LABEL_PREFIX):
            label = label[len(_LABEL_PREFIX):]
        label = label.strip().lower()
        if label == "":
            return {"language": UNKNOWN, "confidence": 0.0}

        return {
            "language": label,
            "confidence": round(min(1.0, max(0.0, confidence)), 4),
        }
