# lang_service — fastText language identification

One HTTP endpoint over Facebook Research's `lid.176` model (176 languages). The
backend calls it once per post/comment write and for chat messages on demand;
the model stays resident in the process, which is the whole reason this runs as
a service instead of PHP shelling out to a binary per text.

The service reports **what the model actually said**. It does not decide whether
a verdict is good enough — the 0.85 confidence gate lives in PHP
(`App\Language\Detector\ConfidenceThresholdLanguageDetector`), next to the code
that has to store or ignore the result.

## Contract

```
POST /v1/detect          X-API-Key: <LANG_API_KEY>
  {"text": "привіт, як справи?"}
  {"texts": ["hello", "привіт"]}          # ≤ LANG_MAX_BATCH items
→ 200 {"results": [{"language": "uk", "confidence": 0.9931}], "model": "lid.176.bin"}

GET /health   → 200 {"status":"ok"}                      liveness, no key needed
GET /ready    → 200 {"status":"ready","model":"…"}       model is in memory
              → 503 {"error":"MODEL_NOT_READY", …}       still loading, or load failed
```

`results` always holds exactly one entry per input, in input order. Text with no
signal (empty, whitespace) is answered `{"language": "und", "confidence": 0.0}`
rather than rejected — `und` is ISO 639-2 "undetermined" and is the same
sentinel `App\Language\Model\Language::UNKNOWN` uses.

Errors are `{"error": "CODE", "message": "…"}`:

| Status | Code | Cause |
|---|---|---|
| 400 | `INVALID_JSON` | body is not a UTF-8 JSON object |
| 401 | `UNAUTHORIZED` | missing or wrong `X-API-Key` |
| 404 | `NOT_FOUND` | unknown path |
| 411 | `LENGTH_REQUIRED` | no `Content-Length` |
| 413 | `PAYLOAD_TOO_LARGE` | body over `LANG_MAX_BODY_BYTES` |
| 422 | `MISSING_TEXT` / `INVALID_TEXT` / `INVALID_TEXTS` / `BATCH_TOO_LARGE` | bad payload shape |
| 503 | `MODEL_NOT_READY` | model not loaded yet, or failed to load |

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `HOST` / `PORT` | `0.0.0.0` / `8090` | |
| `LANG_API_KEY` | — | **required**; the process exits 1 when empty |
| `LANG_MODEL_PATH` | `/app/models/lid.176.bin` | |
| `LANG_MAX_TEXT_CHARS` | `1000` | language id saturates long before this |
| `LANG_MAX_BATCH` | `64` | items per `texts` request |
| `LANG_MAX_BODY_BYTES` | `2000000` | |
| `LOG_LEVEL` | `info` | `debug` adds an access line per request |

An empty `LANG_API_KEY` is fatal on purpose: an unauthenticated detector on the
compose network is one SSRF away from being a free language oracle.

## Running it

```bash
python scripts/fetch_model.py                    # ~126 MB into ./models
pip install -r requirements.txt
LANG_API_KEY=dev LANG_MODEL_PATH=./models/lid.176.bin python src/server.py
```

```bash
python -m unittest discover -s test              # 48 tests, no model needed
```

The suite stubs the model, so it runs without the wheel and without the 126 MB
download. In the image the same command works: `docker compose exec lang python
-m unittest discover -s test`.

## Docker

```bash
docker compose up -d --build lang
docker compose exec lang curl -s localhost:8090/ready
```

The model is downloaded in a separate build stage and verified before it is
given the model's name (minimum size, no HTML body, fastText magic; the sha256
is printed and can be pinned with `--build-arg LANG_MODEL_SHA256=…`). Editing
`src/` does not re-download it.

## Model

`lid.176.bin` from <https://fasttext.cc/docs/en/language-identification.html>,
**CC-BY-SA 3.0**. The quantized `lid.176.ftz` (917 kB) is available via
`python scripts/fetch_model.py ftz` if the 126 MB resident set ever matters more
than the accuracy difference; set `LANG_MODEL_PATH` accordingly.
