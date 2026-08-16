"""HTTP front for fastText language identification.

    POST /v1/detect   {"text": "..."} | {"texts": ["...", ...]}
                   -> {"results": [{"language": "uk", "confidence": 0.99}], "model": "lid.176.bin"}
    GET  /health      liveness — answers before the model is loaded
    GET  /ready       503 until the model is in memory (what compose polls)

Stdlib-only on purpose: the one dependency is the fastText wheel, so the image
builds without a compiler, a package index resolution step, or a web framework
whose CVE feed we would have to follow. Traffic is one detect per post/comment
write plus small chat batches.
"""

from __future__ import annotations

import hmac
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config  # noqa: E402
from detector import Detector  # noqa: E402

_LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_LOG_THRESHOLD = 20


def log(level: str, message: str, **fields: object) -> None:
    if _LEVELS.get(level, 20) < _LOG_THRESHOLD:
        return
    record = {"level": level, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "msg": message}
    record.update(fields)
    print(json.dumps(record, ensure_ascii=False), flush=True)


class LangServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, config: Config, detector: Detector) -> None:
        self.config = config
        self.detector = detector
        super().__init__(address, handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kolo-lang/1.0"
    sys_version = ""
    # A client that opens a connection and stops talking must not hold a thread.
    timeout = 30

    # ── routing ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802 — stdlib naming
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/ready":
            detector = self.server.detector
            if detector.ready:
                self._send_json(200, {"status": "ready", "model": os.path.basename(self.server.config.model_path)})
            else:
                self._send_json(503, {
                    "error": "MODEL_NOT_READY",
                    "message": detector.load_error or "model is still loading",
                })
            return
        self._error(404, "NOT_FOUND", "Unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802 — stdlib naming
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/detect":
            self._error(404, "NOT_FOUND", "Unknown endpoint")
            return

        config = self.server.config
        length = self._content_length()
        if length is None:
            self._error(411, "LENGTH_REQUIRED", "Content-Length is required", close=True)
            return
        if length > config.max_body_bytes:
            self._error(413, "PAYLOAD_TOO_LARGE", "Request body is too large", close=True)
            return

        raw = self.rfile.read(length) if length > 0 else b""

        if not self._authorized():
            self._error(401, "UNAUTHORIZED", "Invalid or missing X-API-Key")
            return

        detector = self.server.detector
        if not detector.ready:
            self._error(503, "MODEL_NOT_READY", detector.load_error or "model is still loading")
            return

        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "INVALID_JSON", "Request body is not valid UTF-8 JSON")
            return

        if not isinstance(payload, dict):
            self._error(400, "INVALID_JSON", "Request body must be a JSON object")
            return

        texts, error = self._extract_texts(payload, config.max_batch)
        if error is not None:
            self._error(422, error[0], error[1])
            return

        started = time.perf_counter()
        try:
            results = detector.detect_many(texts)
        except RuntimeError as exc:
            self._error(503, "MODEL_NOT_READY", str(exc))
            return
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        log(
            "info",
            "detected",
            count=len(results),
            ms=elapsed_ms,
            languages=[r["language"] for r in results[:8]],
        )
        self._send_json(200, {
            "results": results,
            "model": os.path.basename(config.model_path),
        })

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_texts(payload: dict, max_batch: int):
        """-> (texts, None) or (None, (error_code, message))."""
        if "texts" in payload:
            texts = payload["texts"]
            if not isinstance(texts, list):
                return None, ("INVALID_TEXTS", "`texts` must be an array of strings")
            if len(texts) == 0:
                return None, ("INVALID_TEXTS", "`texts` must not be empty")
            if len(texts) > max_batch:
                return None, ("BATCH_TOO_LARGE", f"`texts` must hold at most {max_batch} items")
            if any(not isinstance(item, str) for item in texts):
                return None, ("INVALID_TEXTS", "`texts` must hold strings only")
            return texts, None

        if "text" in payload:
            text = payload["text"]
            if not isinstance(text, str):
                return None, ("INVALID_TEXT", "`text` must be a string")
            return [text], None

        return None, ("MISSING_TEXT", "Provide either `text` or `texts`")

    def _content_length(self) -> int | None:
        raw = self.headers.get("Content-Length")
        if raw is None:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return max(0, value)

    def _authorized(self) -> bool:
        expected = self.server.config.api_key
        provided = self.headers.get("X-API-Key") or ""
        return hmac.compare_digest(expected, provided)

    def _send_json(self, status: int, payload: dict, close: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str, close: bool = False) -> None:
        log("warn" if status >= 500 else "debug", "request rejected", status=status, error=code, path=self.path)
        self._send_json(status, {"error": code, "message": message}, close=close)

    def log_message(self, fmt: str, *args) -> None:
        # Default implementation writes an access line to stderr per request.
        log("debug", "access", line=fmt % args)


def main() -> int:
    global _LOG_THRESHOLD

    config = Config()
    _LOG_THRESHOLD = _LEVELS.get(config.log_level, 20)

    if config.api_key == "":
        log("error", "LANG_API_KEY is empty — refusing to start an unauthenticated detector")
        return 1
    if not os.path.isfile(config.model_path):
        log("error", "model file is missing", path=config.model_path)
        return 1

    detector = Detector(config.model_path, config.max_text_chars)

    def load() -> None:
        started = time.perf_counter()
        detector.load()
        if detector.ready:
            log("info", "model loaded", ms=round((time.perf_counter() - started) * 1000), path=config.model_path)
        else:
            log("error", "model failed to load", error=detector.load_error, path=config.model_path)

    threading.Thread(target=load, name="model-loader", daemon=True).start()

    httpd = LangServer((config.host, config.port), Handler, config, detector)
    log("info", "listening", **config.redacted())

    def shutdown(signum, _frame) -> None:
        log("info", "shutting down", signal=signum)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
