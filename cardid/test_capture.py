"""capture.py's local HTTP server: origin/host checks and request/image limits, without a model."""

from __future__ import annotations

import base64
import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import cv2
import numpy as np

from . import capture


class FakeSession:
    def __init__(self):
        self.identified = []
        self.labels = []

    def identify(self, crop, click, quad):
        self.identified.append((crop.shape, click, quad))
        return {"capture_id": "abc", "quad": quad}

    def label(self, capture_id, label, method):
        self.labels.append((capture_id, label, method))
        return {"capture_id": capture_id}

    def stats(self):
        return {"labeled": len(self.labels)}

    def search(self, q):
        return [{"q": q}]


def jpeg_b64(w: int, h: int) -> str:
    _, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class CaptureServerTest(unittest.TestCase):
    def start(self, allow_remote: bool = False):
        self.session = FakeSession()
        server = ThreadingHTTPServer(("127.0.0.1", 0), capture.make_handler(self.session, allow_remote=allow_remote))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.port = server.server_address[1]
        self.origin = f"http://127.0.0.1:{self.port}"

    def setUp(self):
        self.start()

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for key, value in (headers or {}).items():
            conn.putheader(key, value)
        conn.endheaders(body)
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"{}")

    def post(self, path: str, payload, **overrides) -> tuple[int, dict]:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        headers = {
            "Host": f"127.0.0.1:{self.port}",
            "Origin": self.origin,
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        headers.update(overrides)
        return self.request("POST", path, body, {k: v for k, v in headers.items() if v is not None})

    def test_same_origin_json_identify_and_label_succeed(self):
        status, reply = self.post("/identify", {"image": jpeg_b64(64, 48), "click": [10, 20], "quad": [[0, 0], [5, 0], [5, 5], [0, 5]]})
        self.assertEqual(status, 200, reply)
        self.assertEqual(self.session.identified, [((48, 64, 3), (10.0, 20.0), [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])])
        status, reply = self.post("/label", {"capture_id": "abc", "label": None, "method": "skip"}, **{"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(status, 200, reply)
        self.assertEqual(self.session.labels, [("abc", None, "skip")])
        for host in (f"localhost:{self.port}", f"[::1]:{self.port}"):
            status, _ = self.post("/label", {"capture_id": "abc", "label": "x"}, Host=host, Origin=f"http://{host}")
            self.assertEqual(status, 200, host)

    def test_cross_origin_and_foreign_host_requests_are_refused(self):
        cases = {
            "foreign origin": {"Origin": "https://evil.example"},
            "missing origin on POST": {"Origin": None},
            "origin for another port": {"Origin": f"http://127.0.0.1:{self.port + 1}"},
            "null origin": {"Origin": "null"},
            "rebinding host": {"Host": f"evil.example:{self.port}", "Origin": f"http://evil.example:{self.port}"},
            "missing host": {"Host": None, "Origin": None},
        }
        for name, headers in cases.items():
            with self.subTest(name):
                status, reply = self.post("/label", {"capture_id": "abc", "label": "x"}, **headers)
                self.assertEqual(status, 403, reply)
        status, _ = self.request("GET", "/stats", headers={"Host": f"evil.example:{self.port}"})
        self.assertEqual(status, 403)
        status, _ = self.request("GET", "/stats", headers={"Host": f"127.0.0.1:{self.port}", "Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, reply = self.request("GET", "/stats", headers={"Host": f"127.0.0.1:{self.port}"})
        self.assertEqual((status, reply), (200, {"labeled": 0}))
        self.assertEqual(self.session.labels, [])

    def test_simple_request_content_types_are_refused(self):
        # text/plain and form bodies are what a cross-site page can POST without a preflight.
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", None):
            with self.subTest(ctype):
                status, _ = self.post("/label", {"capture_id": "abc", "label": "x"}, **{"Content-Type": ctype})
                self.assertEqual(status, 415)
        self.assertEqual(self.session.labels, [])

    def test_body_size_limits(self):
        with patch.object(capture, "MAX_BODY_BYTES", 1000):
            # The declared length is checked before anything is read.
            status, _ = self.post("/identify", b"{}", **{"Content-Length": "1001"})
            self.assertEqual(status, 413)
            status, reply = self.post("/identify", {"image": jpeg_b64(8, 8), "click": [1, 1]})
            self.assertEqual(status, 200, reply)
        for length in (None, "abc", "-1"):
            with self.subTest(length=length):
                status, _ = self.post("/label", b"{}", **{"Content-Length": length})
                self.assertIn(status, (400, 411))
        status, _ = self.post("/label", b"{}", **{"Transfer-Encoding": "chunked"})
        self.assertEqual(status, 411)
        status, _ = self.post("/label", b"[1, 2]")
        self.assertEqual(status, 400)
        status, _ = self.post("/label", b"not json")
        self.assertEqual(status, 400)

    def test_image_dimension_and_format_limits(self):
        with patch.object(capture, "MAX_IMAGE_SIDE", 32), patch.object(capture, "MAX_IMAGE_PIXELS", 32 * 32):
            status, reply = self.post("/identify", {"image": jpeg_b64(33, 8), "click": [1, 1]})
            self.assertEqual(status, 413, reply)
            status, _ = self.post("/identify", {"image": jpeg_b64(32, 32), "click": [1, 1]})
            self.assertEqual(status, 200)
        gif = base64.b64encode(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;").decode()
        for image in (gif, "!!!not base64!!!", base64.b64encode(b"garbage").decode(), 12):
            with self.subTest(image=str(image)[:12]):
                status, _ = self.post("/identify", {"image": image, "click": [1, 1]})
                self.assertIn(status, (400, 415))
        for click in ([1], ["x", 1], [float("nan"), 1], None):
            with self.subTest(click=click):
                status, _ = self.post("/identify", {"image": jpeg_b64(8, 8), "click": click})  # NaN serialises as a JSON NaN literal
                self.assertEqual(status, 400)
        self.assertEqual(len(self.session.identified), 1)

    def test_label_validation(self):
        for payload in (
            {"label": "x"},
            {"capture_id": 3, "label": "x"},
            {"capture_id": "a", "label": 5},
            {"capture_id": "a", "label": "x", "method": "<script>"},
        ):
            with self.subTest(payload=payload):
                status, _ = self.post("/label", payload)
                self.assertEqual(status, 400)
        self.assertEqual(self.session.labels, [])

    def test_remote_mode_drops_host_allowlist_but_keeps_same_origin(self):
        self.start(allow_remote=True)
        lan = f"192.168.1.20:{self.port}"
        status, _ = self.post("/label", {"capture_id": "abc", "label": "x"}, Host=lan, Origin=f"http://{lan}")
        self.assertEqual(status, 200)
        status, _ = self.post("/label", {"capture_id": "abc", "label": "x"}, Host=lan, Origin="https://evil.example")
        self.assertEqual(status, 403)

    def test_loopback_detection(self):
        for host in ("127.0.0.1", "127.8.0.1", "::1", "[::1]", "localhost"):
            self.assertTrue(capture.loopback_host(host), host)
        for host in ("0.0.0.0", "::", "192.168.1.2", "example.com", "localhost.example.com"):
            self.assertFalse(capture.loopback_host(host), host)

    def test_cli_refuses_non_loopback_host_without_flag(self):
        with patch("sys.argv", ["capture", "--checkpoint", "x.pt", "--host", "0.0.0.0"]), patch.object(capture, "Session") as session:
            with self.assertRaises(SystemExit) as raised, patch("sys.stderr"):
                capture.main()
        self.assertEqual(raised.exception.code, 2)
        session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
