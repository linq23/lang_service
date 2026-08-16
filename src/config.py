"""Runtime configuration, read once from the environment.

Every knob has a working default so the container starts with only
LANG_API_KEY set. Thresholding is deliberately absent: the confidence gate
lives in the PHP caller (App\\Language\\Detector\\ConfidenceThresholdLanguageDetector),
so this service always reports what the model actually said.
"""

from __future__ import annotations

import os


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


class Config:
    def __init__(self) -> None:
        self.host: str = os.environ.get("HOST", "0.0.0.0").strip() or "0.0.0.0"
        self.port: int = _int("PORT", 8090)
        # Shared secret with the backend (X-API-Key). Empty means the service
        # refuses to start — an unauthenticated detector on the compose network
        # is one SSRF away from being a free translation oracle.
        self.api_key: str = os.environ.get("LANG_API_KEY", "").strip()
        self.model_path: str = (
            os.environ.get("LANG_MODEL_PATH", "").strip() or "/app/models/lid.176.bin"
        )
        # Language ID saturates long before this; the cap keeps one 10k-char
        # post from costing 10x a normal one.
        self.max_text_chars: int = _int("LANG_MAX_TEXT_CHARS", 1000)
        self.max_batch: int = _int("LANG_MAX_BATCH", 64)
        # Enough for max_batch * 10k chars of UTF-8 plus JSON overhead.
        self.max_body_bytes: int = _int("LANG_MAX_BODY_BYTES", 2_000_000)
        self.log_level: str = (os.environ.get("LOG_LEVEL", "info").strip() or "info").lower()

    def redacted(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "model_path": self.model_path,
            "max_text_chars": self.max_text_chars,
            "max_batch": self.max_batch,
            "api_key_set": self.api_key != "",
        }
