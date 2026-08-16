# fastText language identification, behind a stdlib HTTP server.
#
# Two stages so the 126 MB model lands in its own cached layer: editing src/
# does not re-download it. The runtime stage installs exactly one wheel
# (fasttext-predict) — prebuilt for manylinux x86_64/aarch64, so there is no
# compiler, no pybind11 and no numpy in this image.
FROM python:3.12-slim-bookworm AS model

WORKDIR /build

# ca-certificates for the model download; urllib verifies TLS itself, so no curl.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pulled from dl.fbaipublicfiles.com and checked (minimum size, no HTML body,
# fastText magic) before it is given the model's name. Pass LANG_MODEL_SHA256 as
# a build arg to additionally pin the exact bytes.
COPY scripts/fetch_model.py ./scripts/
ARG LANG_MODEL_VARIANT=bin
ARG LANG_MODEL_SHA256=
RUN python scripts/fetch_model.py "${LANG_MODEL_VARIANT}" --target /build/models


FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8090 \
    LANG_MODEL_PATH=/app/models/lid.176.bin

# One dependency, installed from a wheel. --only-binary makes a missing wheel a
# build failure instead of a silent fallback to the sdist, which would need g++.
COPY requirements.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && python -c "import fasttext; print('fasttext binding OK')"

RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin lang

COPY --from=model --chown=lang:lang /build/models ./models
COPY --chown=lang:lang src ./src
COPY --chown=lang:lang scripts ./scripts
# test/ ships too: it is a stdlib unittest suite with a stubbed model, so
# `docker compose exec lang python -m unittest discover -s test` is a valid
# post-deploy check. A few KB, no runtime cost.
COPY --chown=lang:lang test ./test

USER lang

EXPOSE 8090

# Liveness only — readiness (model in memory) is /ready, which compose polls.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,sys,urllib.request; \
        u='http://127.0.0.1:'+os.environ.get('PORT','8090')+'/health'; \
        sys.exit(0 if urllib.request.urlopen(u, timeout=3).status == 200 else 1)"

CMD ["python", "src/server.py"]
