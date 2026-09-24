"""Bounded HTTPS downloads: byte caps, content-type and format checks, atomic writes.

Every response is streamed and abandoned as soon as it passes its byte cap (or declares a
larger Content-Length), so a misbehaving server cannot fill the disk or memory. Files are
written to a temporary name beside the destination, validated, then renamed into place;
a failed or truncated transfer never leaves a file at the destination, so "the file
exists" keeps meaning "the file is complete".
"""

from __future__ import annotations

import gzip
import io
import json
import os
import threading
import zlib
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image

JSON_MAX_BYTES = 32 * 1024 * 1024  # an API page of 175 cards is well under 2 MB
BULK_MAX_BYTES = 2 * 1024**3  # all_cards.jsonl.gz is ~400 MB
IMAGE_MAX_BYTES = 8 * 1024 * 1024  # art crops and normal scans are 50-300 KB
IMAGE_MAX_SIDE = 4096
IMAGE_MIN_SIDE = 16
CARD_ASPECT_TOLERANCE = 0.05  # normal scans are 488x680 portrait (63x88 mm)

JSON_TYPES = ("application/json",)
GZIP_TYPES = ("application/gzip", "application/x-gzip", "application/octet-stream")
IMAGE_TYPES = ("image/jpeg", "image/png")
CHUNK = 1 << 16


class DownloadError(Exception):
    """A response that was refused: wrong scheme/type, over its cap, truncated or malformed."""


def _check_response(response: httpx.Response, content_types: Iterable[str], max_bytes: int) -> int | None:
    """Raise unless the (final, post-redirect) response is HTTPS 200 with an allowed type and
    no declared length over the cap; returns the declared length when it describes the body."""
    if response.url.scheme != "https":
        raise DownloadError(f"refusing non-HTTPS URL {response.url}")
    response.raise_for_status()
    ctype = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if ctype not in content_types:
        raise DownloadError(f"{response.url}: unexpected content-type {ctype or '(none)'}")
    declared = response.headers.get("content-length")
    if declared is None:
        return None
    try:
        length = int(declared)
    except ValueError:
        raise DownloadError(f"{response.url}: invalid content-length {declared!r}") from None
    if length > max_bytes:
        raise DownloadError(f"{response.url}: {length} bytes exceeds the {max_bytes}-byte cap")
    # httpx decodes Content-Encoding, after which the declared length no longer counts the body.
    return None if response.headers.get("content-encoding") else length


def _chunks(response: httpx.Response, max_bytes: int, declared: int | None) -> Iterable[bytes]:
    received = 0
    for chunk in response.iter_bytes(CHUNK):
        received += len(chunk)
        if received > max_bytes:
            raise DownloadError(f"{response.url}: body exceeds the {max_bytes}-byte cap")
        yield chunk
    if declared is not None and received != declared:
        raise DownloadError(f"{response.url}: truncated ({received} of {declared} bytes)")


def fetch_bytes(client: httpx.Client, url: str, *, max_bytes: int, content_types: Iterable[str], params=None, timeout: float = 30) -> bytes:
    """The whole body in memory, for small responses (API pages, images)."""
    _require_https(url)
    with client.stream("GET", url, params=params, timeout=timeout) as response:
        declared = _check_response(response, tuple(content_types), max_bytes)
        return b"".join(_chunks(response, max_bytes, declared))


def fetch_json(client: httpx.Client, url: str, *, params=None, max_bytes: int = JSON_MAX_BYTES, timeout: float = 30):
    body = fetch_bytes(client, url, max_bytes=max_bytes, content_types=JSON_TYPES, params=params, timeout=timeout)
    try:
        return json.loads(body)
    except ValueError:
        raise DownloadError(f"{url}: response is not JSON") from None


def download_file(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    max_bytes: int,
    content_types: Iterable[str],
    validate: Callable[[Path], None],
    progress: Callable[[int, int | None], None] | None = None,
    timeout: float = 60,
) -> Path:
    """Stream `url` to `dest` atomically: capped, validated by `validate(tmp)` (raise to
    reject), then renamed. Partial files are always removed."""
    _require_https(url)
    tmp = temp_path(dest)
    try:
        with client.stream("GET", url, timeout=timeout) as response:
            declared = _check_response(response, tuple(content_types), max_bytes)
            with tmp.open("wb") as f:
                for chunk in _chunks(response, max_bytes, declared):
                    f.write(chunk)
                    if progress:
                        progress(len(chunk), declared)
        validate(tmp)
        os.replace(tmp, dest)
        return dest
    finally:
        tmp.unlink(missing_ok=True)


def write_atomic(dest: Path, data: bytes) -> None:
    tmp = temp_path(dest)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def temp_path(dest: Path) -> Path:
    """Per process and thread, beside `dest` so the rename stays on one filesystem."""
    return dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.part")


def validate_gzip_jsonl(path: Path) -> None:
    """A complete gzip stream (a truncated one raises EOFError) whose first line is a JSON object."""
    try:
        with gzip.open(path, "rb") as f:
            first = f.readline()
            while f.read(1 << 20):
                pass
        if not isinstance(json.loads(first), dict):
            raise ValueError("first line is not a JSON object")
    except (OSError, EOFError, zlib.error, ValueError) as error:
        raise DownloadError(f"{path.name}: not a complete gzip JSON-lines file ({error})") from None


def decode_image(data: bytes, *, card: bool = False) -> np.ndarray:
    """BGR pixels of a JPEG/PNG within the size limits, checked from the header before any
    pixels are decoded. `card` also requires a portrait 63x88 full-card scan."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except (OSError, Image.DecompressionBombError) as error:
        raise DownloadError(f"not an image ({error})") from None
    if fmt not in ("JPEG", "PNG"):
        raise DownloadError(f"unexpected image format {fmt}")
    if not (IMAGE_MIN_SIDE <= min(w, h) and max(w, h) <= IMAGE_MAX_SIDE):
        raise DownloadError(f"unexpected image size {w}x{h}")
    if card and abs((h / w) / (88 / 63) - 1) > CARD_ASPECT_TOLERANCE:
        raise DownloadError(f"{w}x{h} is not a portrait card scan")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (h, w):
        raise DownloadError("image does not decode")
    return image


def _require_https(url: str) -> None:
    if httpx.URL(url).scheme != "https":
        raise DownloadError(f"refusing non-HTTPS URL {url}")
