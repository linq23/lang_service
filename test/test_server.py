"""HTTP contract tests. A real server on an ephemeral port, a stub detector —
these are the statuses and shapes App\\Language\\Detector\\FastTextLanguageDetector
is written against, so they are the ones worth pinning.

    python -m unittest discover -s test
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import unittest
import urllib.error
import urllib.request

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import server as server_mod  # noqa: E402
from config import Config  # noqa: E402
from detector import UNKNOWN  # noqa: E402

API_KEY = "test-key"


class StubDetector:
    def __init__(self, ready=True, load_error=None):
        self._ready = ready
        self.load_error = load_error
        self.calls = []

    @property
    def ready(self):
        return self._ready

    def detect_many(self, texts):
        self.calls.append(list(texts))
        return [
            {"language": UNKNOWN, "confidence": 0.0} if text.strip() == ""
            else {"language": "uk", "confidence": 0.9931}
            for text in texts
        ]


class ServerTestCase(unittest.TestCase):
    """Boots the real handler; each test gets its own port and detector."""

    ENV = {
        "LANG_API_KEY": API_KEY,
        "LANG_MAX_BATCH": "3",
        "LANG_MAX_BODY_BYTES": "300",
        "LANG_MODEL_PATH": "/app/models/lid.176.bin",
    }

    def setUp(self):
        self.saved = {key: os.environ.get(key) for key in self.ENV}
        os.environ.update(self.ENV)
        self.addCleanup(self._restore_env)

        # Silence the JSON log lines; failures still report through unittest.
        saved_threshold = server_mod._LOG_THRESHOLD
        server_mod._LOG_THRESHOLD = 100
        self.addCleanup(lambda: setattr(server_mod, "_LOG_THRESHOLD", saved_threshold))

        self.detector = StubDetector()
        self.httpd = server_mod.LangServer(("127.0.0.1", 0), server_mod.Handler, Config(), self.detector)
        self.port = self.httpd.server_address[1]
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop, thread)

    def _restore_env(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _stop(self, thread):
        self.httpd.shutdown()
        self.httpd.server_close()
        thread.join(timeout=5)

    # ── request helpers ──────────────────────────────────────────────────────

    def call(self, method, path, payload=None, key=API_KEY, raw=None):
        """-> (status, decoded body)."""
        url = f"http://127.0.0.1:{self.port}{path}"
        data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode("utf-8"))
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if key is not None:
            request.add_header("X-API-Key", key)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def detect(self, payload, key=API_KEY):
        return self.call("POST", "/v1/detect", payload, key=key)


class HealthTest(ServerTestCase):
    def test_health_answers_ok_without_a_key(self):
        self.assertEqual(self.call("GET", "/health", key=None), (200, {"status": "ok"}))

    def test_ready_reports_the_loaded_model(self):
        status, body = self.call("GET", "/ready")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ready", "model": "lid.176.bin"})

    def test_ready_is_503_while_loading(self):
        self.detector._ready = False
        status, body = self.call("GET", "/ready")
        self.assertEqual(status, 503)
        self.assertEqual(body["error"], "MODEL_NOT_READY")

    def test_ready_surfaces_a_load_failure(self):
        self.detector._ready = False
        self.detector.load_error = "ValueError: bad file"
        _, body = self.call("GET", "/ready")
        self.assertEqual(body["message"], "ValueError: bad file")

    def test_unknown_path_is_404(self):
        status, body = self.call("GET", "/v1/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "NOT_FOUND")

    def test_trailing_slash_is_the_same_route(self):
        self.assertEqual(self.call("GET", "/health/", key=None)[0], 200)


class DetectTest(ServerTestCase):
    def test_single_text(self):
        status, body = self.detect({"text": "привіт як справи"})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], [{"language": "uk", "confidence": 0.9931}])
        self.assertEqual(body["model"], "lid.176.bin")

    def test_batch_keeps_order_and_arity(self):
        status, body = self.detect({"texts": ["привіт", "", "ok"]})
        self.assertEqual(status, 200)
        self.assertEqual([r["language"] for r in body["results"]], ["uk", UNKNOWN, "uk"])
        self.assertEqual(self.detector.calls, [["привіт", "", "ok"]])

    def test_empty_text_is_answered_not_rejected(self):
        status, body = self.detect({"text": ""})
        self.assertEqual(status, 200)
        self.assertEqual(body["results"], [{"language": UNKNOWN, "confidence": 0.0}])

    def test_wrong_key_is_401(self):
        status, body = self.detect({"text": "hi"}, key="nope")
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "UNAUTHORIZED")

    def test_missing_key_is_401(self):
        self.assertEqual(self.detect({"text": "hi"}, key=None)[0], 401)

    def test_auth_is_checked_before_the_payload(self):
        status, _ = self.call("POST", "/v1/detect", None, key="nope", raw=b"{not json")
        self.assertEqual(status, 401)

    def test_model_not_ready_is_503(self):
        self.detector._ready = False
        status, body = self.detect({"text": "hi"})
        self.assertEqual(status, 503)
        self.assertEqual(body["error"], "MODEL_NOT_READY")

    def test_broken_json_is_400(self):
        status, body = self.call("POST", "/v1/detect", None, raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "INVALID_JSON")

    def test_json_array_is_400(self):
        status, body = self.call("POST", "/v1/detect", ["hi"])
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "INVALID_JSON")

    def test_missing_text_field_is_422(self):
        status, body = self.detect({"lang": "uk"})
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "MISSING_TEXT")

    def test_non_string_text_is_422(self):
        self.assertEqual(self.detect({"text": 42})[1]["error"], "INVALID_TEXT")

    def test_non_list_texts_is_422(self):
        self.assertEqual(self.detect({"texts": "hi"})[1]["error"], "INVALID_TEXTS")

    def test_empty_batch_is_422(self):
        self.assertEqual(self.detect({"texts": []})[1]["error"], "INVALID_TEXTS")

    def test_non_string_batch_item_is_422(self):
        self.assertEqual(self.detect({"texts": ["hi", 7]})[1]["error"], "INVALID_TEXTS")

    def test_batch_over_the_cap_is_422(self):
        status, body = self.detect({"texts": ["a", "b", "c", "d"]})  # LANG_MAX_BATCH=3
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "BATCH_TOO_LARGE")
        self.assertEqual(self.detector.calls, [])

    def test_oversized_body_is_413_without_reading_it(self):
        status, body = self.call("POST", "/v1/detect", {"text": "x" * 400})  # cap is 300 bytes
        self.assertEqual(status, 413)
        self.assertEqual(body["error"], "PAYLOAD_TOO_LARGE")

    def test_empty_body_is_400(self):
        status, body = self.call("POST", "/v1/detect", None)  # http.client sends Content-Length: 0
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "INVALID_JSON")

    def test_body_without_content_length_is_411(self):
        # Raw socket, because http.client always supplies Content-Length on POST.
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(
                b"POST /v1/detect HTTP/1.1\r\nHost: localhost\r\n"
                b"X-API-Key: " + API_KEY.encode() + b"\r\n\r\n"
            )
            raw = sock.recv(4096)
        self.assertIn(b" 411 ", raw.split(b"\r\n")[0])
        self.assertIn(b"LENGTH_REQUIRED", raw)

    def test_unknown_post_path_is_404(self):
        self.assertEqual(self.call("POST", "/v1/translate", {"text": "hi"})[0], 404)

    def test_keep_alive_survives_a_rejected_request(self):
        """A 422 must not desync the connection: the body is read before the
        error is written, so the next request on the socket still parses."""
        self.assertEqual(self.detect({"texts": []})[0], 422)
        self.assertEqual(self.detect({"text": "привіт"})[0], 200)


if __name__ == "__main__":
    unittest.main()
