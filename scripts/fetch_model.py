#!/usr/bin/env python3
"""Downloads the fastText language-identification model from its upstream home.

`lid.176.bin` (~126 MB, 176 languages) is published by Facebook Research at a
stable path under dl.fbaipublicfiles.com. The quantized `lid.176.ftz` (~917 kB)
is the same model at a small accuracy cost — useful for a laptop, not what the
container ships.

Nothing is vendored into git: a 126 MB binary in a repo this size is a tax on
every clone. Nothing is trusted on arrival either — the bytes are checked before
they are given the model's name, because a truncated download or an error page
would otherwise fail much later, inside fasttext.load_model(), on a container
that was already declared built.

Upstream publishes no digest for these files, so the checks are: a minimum size,
a rejection of HTML bodies, and the fastText file-format magic. The sha256 of
what arrived is always printed; pass it back via --sha256 (or LANG_MODEL_SHA256)
to pin the exact bytes on later builds.

Usage:
    python scripts/fetch_model.py                 # lid.176.bin
    python scripts/fetch_model.py ftz             # the quantized one
    python scripts/fetch_model.py all
    python scripts/fetch_model.py --sha256 <hex>  # verify against a known hash
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import urllib.error
import urllib.request

BASE = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/"

# fastText writes this int32 at the head of every model it saves
# (FASTTEXT_FILEFORMAT_MAGIC_INT32 in src/fasttext.cc).
MAGIC = 793712314

MODELS = {
    "bin": {"file": "lid.176.bin", "url": BASE + "lid.176.bin", "min_bytes": 120_000_000},
    "ftz": {"file": "lid.176.ftz", "url": BASE + "lid.176.ftz", "min_bytes": 800_000},
}
DEFAULT = "bin"

ATTEMPTS = 3
TIMEOUT_SECONDS = 120


def target_dir() -> str:
    from_env = os.environ.get("LANG_MODEL_DIR", "").strip()
    if from_env:
        return os.path.abspath(from_env)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def mib(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MB"


def looks_like_html(payload: bytes, content_type: str) -> bool:
    head = payload[:512].lstrip().lower()
    return "text/html" in content_type.lower() or head.startswith(b"<!doctype") or head.startswith(b"<html")


def verify(model: dict, payload: bytes, content_type: str, expected_sha: str | None) -> str | None:
    """-> what is wrong with these bytes, or None."""
    if looks_like_html(payload, content_type):
        return f"got a {len(payload)}-byte HTML page instead of a model"
    if len(payload) < model["min_bytes"]:
        return f"got {len(payload)} bytes, expected at least {model['min_bytes']}"
    if expected_sha is not None:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_sha.lower():
            return f"sha256 {actual} != {expected_sha.lower()}"
    return None


def warn_on_magic(payload: bytes) -> None:
    """Advisory: a wrong magic means load_model() will fail, but the format has
    changed before, so this must not be the thing that breaks an image build."""
    if len(payload) < 4:
        return
    (found,) = struct.unpack("<i", payload[:4])
    if found != MAGIC:
        print(f"  warning: file does not start with the fastText magic ({found} != {MAGIC})")


def attempt_once(model: dict, destination: str, expected_sha: str | None) -> str | None:
    request = urllib.request.Request(model["url"], headers={"User-Agent": "kolo-lang-service/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read()
    except (urllib.error.URLError, OSError) as exc:
        return f"request failed: {exc}"

    problem = verify(model, payload, content_type, expected_sha)
    if problem is not None:
        return problem

    warn_on_magic(payload)
    print(f"  sha256 {hashlib.sha256(payload).hexdigest()}")

    # Write beside the target and rename: rename is atomic within a directory,
    # so an interrupted build cannot leave a partial file that the next run
    # would find, accept by size, and hand to load_model().
    staging = destination + ".partial"
    with open(staging, "wb") as handle:
        handle.write(payload)
    os.replace(staging, destination)
    return None


def fetch(name: str, directory: str, expected_sha: str | None, force: bool) -> None:
    model = MODELS.get(name)
    if model is None:
        known = ", ".join(MODELS) + ", all"
        raise SystemExit(f"unknown model {name!r} — known: {known}")

    os.makedirs(directory, exist_ok=True)
    destination = os.path.join(directory, model["file"])

    if not force and os.path.isfile(destination) and os.path.getsize(destination) >= model["min_bytes"]:
        print(f"  {model['file']} already present ({mib(os.path.getsize(destination))})")
        return

    print(f"  {model['file']} … ", flush=True)
    failures = []
    for attempt in range(1, ATTEMPTS + 1):
        problem = attempt_once(model, destination, expected_sha)
        if problem is None:
            print(f"  {model['file']} {mib(os.path.getsize(destination))} ok")
            return
        failures.append(problem)
        retrying = f", retrying ({attempt}/{ATTEMPTS})" if attempt < ATTEMPTS else ""
        print(f"  {problem}{retrying}")

    raise SystemExit(f"refusing to write {model['file']} — {'; '.join(failures)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the fastText language-id model.")
    parser.add_argument("models", nargs="*", default=[], help="bin (default), ftz, or all")
    parser.add_argument("--target", default=None, help="directory to write into")
    parser.add_argument("--sha256", default=os.environ.get("LANG_MODEL_SHA256", "").strip() or None)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    requested = args.models or [DEFAULT]
    if "all" in requested:
        requested = list(MODELS)

    directory = os.path.abspath(args.target) if args.target else target_dir()
    print(f"fetching into {directory}")
    for name in requested:
        fetch(name, directory, args.sha256, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
