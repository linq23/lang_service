"""Config unit tests — the defaults are what the container runs on."""

from __future__ import annotations

import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import Config  # noqa: E402

KEYS = [
    "HOST", "PORT", "LANG_API_KEY", "LANG_MODEL_PATH", "LANG_MAX_TEXT_CHARS",
    "LANG_MAX_BATCH", "LANG_MAX_BODY_BYTES", "LOG_LEVEL",
]


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.saved = {key: os.environ.pop(key, None) for key in KEYS}

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_defaults(self):
        config = Config()
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 8090)
        self.assertEqual(config.api_key, "")
        self.assertEqual(config.model_path, "/app/models/lid.176.bin")
        self.assertEqual(config.max_text_chars, 1000)
        self.assertEqual(config.max_batch, 64)
        self.assertEqual(config.max_body_bytes, 2_000_000)
        self.assertEqual(config.log_level, "info")

    def test_environment_overrides(self):
        os.environ.update({
            "PORT": "9001",
            "LANG_API_KEY": "  s3cret  ",
            "LANG_MODEL_PATH": "/models/lid.176.ftz",
            "LANG_MAX_BATCH": "8",
            "LOG_LEVEL": "DEBUG",
        })
        config = Config()
        self.assertEqual(config.port, 9001)
        self.assertEqual(config.api_key, "s3cret")
        self.assertEqual(config.model_path, "/models/lid.176.ftz")
        self.assertEqual(config.max_batch, 8)
        self.assertEqual(config.log_level, "debug")

    def test_unparseable_int_falls_back_to_default(self):
        os.environ["PORT"] = "not-a-port"
        self.assertEqual(Config().port, 8090)

    def test_blank_string_falls_back_to_default(self):
        os.environ["LANG_MODEL_PATH"] = "   "
        self.assertEqual(Config().model_path, "/app/models/lid.176.bin")

    def test_int_floor_is_one(self):
        os.environ["LANG_MAX_BATCH"] = "0"
        self.assertEqual(Config().max_batch, 1)

    def test_redacted_reports_whether_the_key_is_set_never_the_key(self):
        os.environ["LANG_API_KEY"] = "s3cret"
        redacted = Config().redacted()
        self.assertTrue(redacted["api_key_set"])
        self.assertNotIn("s3cret", repr(redacted))

    def test_redacted_reports_missing_key(self):
        self.assertFalse(Config().redacted()["api_key_set"])


if __name__ == "__main__":
    unittest.main()
