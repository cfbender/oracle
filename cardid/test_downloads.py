"""Bounded Scryfall downloads: byte caps, type/format checks, atomic writes, no partial files."""

from __future__ import annotations

import gzip
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from PIL import Image

from . import downloads, scryfall
from .downloads import DownloadError


def png(w: int, h: int) -> bytes:
    return cv2.imencode(".png", np.full((h, w, 3), 90, np.uint8))[1].tobytes()


def gif(w: int, h: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h)).save(out, "GIF")
    return out.getvalue()


def gzip_jsonl(lines: int = 3) -> bytes:
    return gzip.compress(b"".join(json.dumps({"n": i}).encode() + b"\n" for i in range(lines)))


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


class DownloadHelpersTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def assert_empty(self):
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [])

    def test_json_type_and_size_caps(self):
        with client(lambda r: httpx.Response(200, json={"ok": True})) as c:
            self.assertEqual(downloads.fetch_json(c, "https://api.test/x"), {"ok": True})
            with self.assertRaisesRegex(DownloadError, "cap"):
                downloads.fetch_json(c, "https://api.test/x", max_bytes=5)
            with self.assertRaisesRegex(DownloadError, "non-HTTPS"):
                downloads.fetch_json(c, "http://api.test/x")
        with client(lambda r: httpx.Response(200, text="<html>", headers={"content-type": "text/html"})) as c:
            with self.assertRaisesRegex(DownloadError, "content-type text/html"):
                downloads.fetch_json(c, "https://api.test/x")
        # No Content-Length (chunked): the stream itself is cut off at the cap.
        streamed = lambda r: httpx.Response(200, headers={"content-type": "application/json"}, content=iter([b"[" + b"1," * 50, b"1]"]))  # noqa: E731
        with client(streamed) as c:
            with self.assertRaisesRegex(DownloadError, "exceeds"):
                downloads.fetch_json(c, "https://api.test/x", max_bytes=20)
            self.assertEqual(len(downloads.fetch_json(c, "https://api.test/x")), 51)

    def test_redirect_to_plain_http_is_refused(self):
        def handler(request):
            if request.url.scheme == "https":
                return httpx.Response(302, headers={"location": "http://api.test/plain"})
            return httpx.Response(200, json={})

        with client(handler) as c, self.assertRaisesRegex(DownloadError, "non-HTTPS"):
            downloads.fetch_json(c, "https://api.test/x")

    def test_download_file_is_atomic_and_leaves_no_partial_files(self):
        body = gzip_jsonl()
        dest = self.root / "bulk.gz"

        def gz(content, **headers):
            return lambda r: httpx.Response(200, headers={"content-type": "application/gzip", **headers}, content=content)

        def fetch(c, **kwargs):
            opts = {"max_bytes": 10_000, "content_types": downloads.GZIP_TYPES, "validate": downloads.validate_gzip_jsonl, **kwargs}
            return downloads.download_file(c, "https://data.test/bulk.gz", dest, **opts)

        cases = {
            "declared over cap": (gz(body), {"max_bytes": len(body) - 1}, "exceeds"),
            "streamed over cap": (gz(iter([body[:10], body[10:]])), {"max_bytes": 12}, "exceeds"),
            "truncated vs content-length": (gz(body[:-8], **{"content-length": str(len(body))}), {}, "truncated"),
            "truncated gzip": (gz(body[:-8]), {}, "not a complete gzip"),
            "not gzip": (gz(b"<html>not gzip</html>"), {}, "not a complete gzip"),
            "wrong type": (lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=body), {}, "content-type"),
        }
        for name, (handler, kwargs, message) in cases.items():
            with self.subTest(name), client(handler) as c:
                with self.assertRaisesRegex(DownloadError, message):
                    fetch(c, **kwargs)
                self.assert_empty()
        with client(lambda r: httpx.Response(503)) as c, self.assertRaises(httpx.HTTPStatusError):
            fetch(c)
        self.assert_empty()
        with client(gz(body)) as c:
            self.assertEqual(fetch(c), dest)
        self.assertEqual(dest.read_bytes(), body)
        self.assertEqual([p.name for p in self.root.iterdir()], ["bulk.gz"])

    def test_decode_image_checks_format_size_and_card_aspect(self):
        self.assertEqual(downloads.decode_image(png(626, 457)).shape, (457, 626, 3))
        self.assertEqual(downloads.decode_image(png(488, 680), card=True).shape, (680, 488, 3))
        for data, card, message in (
            (png(626, 457), True, "portrait card"),
            (png(8, 8), False, "size"),
            (png(5000, 20), False, "size"),
            (gif(64, 64), False, "format GIF"),
            (b"<html>", False, "not an image"),
            (png(64, 64)[:60], False, "decode|not an image"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(DownloadError, message):
                downloads.decode_image(data, card=card)


class ScryfallDownloadTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        gap = patch.object(scryfall, "REQUEST_GAP_S", 0)
        gap.start()
        self.addCleanup(gap.stop)

    def test_fetch_image_rejects_bad_responses_without_leaving_files(self):
        entry = {"id": "art", "url": "https://img.test/art.jpg", "card_url": "https://img.test/card.jpg"}
        responses = {
            "html error page": httpx.Response(200, text="<html>", headers={"content-type": "text/html"}),
            "oversized body": httpx.Response(200, content=png(64, 64), headers={"content-type": "image/png", "content-length": str(1 << 30)}),
            "undecodable": httpx.Response(200, content=b"\x89PNG garbage", headers={"content-type": "image/png"}),
            "tiny": httpx.Response(200, content=png(4, 4), headers={"content-type": "image/png"}),
        }
        for name, response in responses.items():
            with self.subTest(name), client(lambda r, response=response: response) as c:
                self.assertEqual(scryfall.fetch_image(c, entry, self.root), ("art", False))
                self.assertEqual(list(self.root.iterdir()), [])
        landscape = httpx.Response(200, content=png(626, 457), headers={"content-type": "image/png"})
        with client(lambda r: landscape) as c:
            self.assertEqual(scryfall.fetch_image(c, entry, self.root, "card_url"), ("art", False))  # not a card scan
            self.assertEqual(list(self.root.iterdir()), [])
            self.assertEqual(scryfall.fetch_image(c, entry, self.root), ("art", True))  # a fine art crop
        self.assertEqual([p.name for p in self.root.iterdir()], ["art.jpg"])
        self.assertEqual((self.root / "art.jpg").read_bytes(), png(626, 457))  # stored as served

    def test_existing_truncated_bulk_is_replaced_and_valid_one_reused(self):
        body = gzip_jsonl(20)
        dest = self.root / "all-cards.jsonl.gz"
        dest.write_bytes(body[: len(body) // 2])  # what an interrupted old download left behind
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.url.host == "api.scryfall.com":
                entry = {"type": "all_cards", "jsonl_download_uri": "https://data.test/all.jsonl.gz", "compressed_size": len(body)}
                return httpx.Response(200, json={"data": [entry]})
            return httpx.Response(200, content=body, headers={"content-type": "application/gzip"})

        with client(handler) as c, redirect_stdout(io.StringIO()) as out:
            scryfall.download_bulk(c, dest)
            self.assertIn("discarding", out.getvalue())
            self.assertEqual(dest.read_bytes(), body)
            scryfall.download_bulk(c, dest)
        self.assertEqual(calls, ["/bulk-data", "/all.jsonl.gz"])  # the complete file was reused

    def test_failed_update_keeps_the_previous_bulk(self):
        (self.root / "arts.json").write_text("[]")
        bulk = self.root / "all-cards.jsonl.gz"
        bulk.write_bytes(gzip_jsonl())
        original = bulk.read_bytes()

        def handler(request):
            if request.url.host == "api.scryfall.com":
                return httpx.Response(200, json={"data": [{"type": "all_cards", "jsonl_download_uri": "https://data.test/x.gz"}]})
            return httpx.Response(200, content=b"truncated", headers={"content-type": "application/gzip"})

        real_client = httpx.Client
        with (
            patch.object(scryfall, "DATA_DIR", self.root),
            patch.object(scryfall, "ART_DIR", self.root / "art"),
            patch.object(scryfall.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)),
            patch("sys.argv", ["scryfall", "--update"]),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(SystemExit, "bulk download failed"):
                scryfall.main()
        self.assertEqual(bulk.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.name.startswith(".")), [])


if __name__ == "__main__":
    unittest.main()
