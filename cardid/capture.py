"""Local click-to-identify tool for collecting and labeling real webcam captures.

    uv run python -m cardid.capture --checkpoint data/runs/full/best.pt [--detector data/runs/det/best.pt] [--port 8765]

Open http://localhost:8765 in Chrome, allow the camera, and click a card in the live 1080p
feed. The browser sends a full-resolution crop around the click; the server finds the card
quad, warps it, cuts the art box, embeds it, and shows the top-5 candidates. Press 1-5 (or
click a candidate) to confirm, type a name to search when none is right, or press S to skip.
Shift-click the card's four corners (any starting corner, either way round) when the
automatic quad is wrong or missing; this works for tilted and tapped cards, unlike a box.

With `--detector`, the quad comes from the learned `cardid.detector` (always returns one);
without it, from the classical edge finder in `cardid.detect`.

Everything labeled lands in data/real/ (see `cardid.real`) for `train --real` and
`evaluate --real`. The page keeps a running top-1/top-5 over what you have labeled.

Stdlib http.server only, so this adds no dependencies; localhost is a secure context for
getUserMedia in Chrome, so no TLS is needed.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

from . import ART_DIR
from .detect import (
    FRAME_PENALTY,
    art_crops,
    card_orientations,
    find_card_quad,
    warp_card,
)
from .detector import Detector, cyclic_order
from .index import ArtIndex
from .real import load_labels, save_label

PAGE = Path(__file__).with_name("capture.html")


class Session:
    """Server-side state: the index plus captures that are identified but not yet labeled."""

    def __init__(self, checkpoint: Path, detector: Path | None = None, frame_penalty: float = FRAME_PENALTY):
        self.index = ArtIndex(checkpoint, frame_penalty)
        self.detector = Detector(detector) if detector else None
        self.pending: dict[str, dict] = {}
        self.lock = threading.Lock()

    def identify(self, crop: np.ndarray, click: tuple[float, float], manual_quad: list[list[float]] | None) -> dict:
        t0 = time.perf_counter()
        up_vote = None
        if manual_quad:
            quad = cyclic_order(np.float32(manual_quad))
            source = "manual"
        elif self.detector is not None:
            quad, up_vote = self.detector.locate_up(crop, click)
            source = "detector"
        else:
            quad = find_card_quad(crop, click)
            source = "auto"
        if quad is None:
            return {"quad": None, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        card = warp_card(crop, quad)
        cards = card_orientations(card)
        # every frame's art box from both orientations, embedded in one batch: 2 x F x D
        arts = np.stack([art_crops(c) for c in cards])
        vecs = self.index.embed(arts.reshape(-1, *arts.shape[2:])).reshape(len(cards), arts.shape[1], -1)
        results = [self.index.search(v, 5) for v in vecs]
        if source == "detector":
            # The learned detector orders the quad from the printed top-left, so index 0 is
            # upright by its judgement; the 180-degree turn is only embedded so labelling can
            # still store the card the right way up when the detector was wrong.
            best = 0
        else:
            # The classical finder and hand-drawn boxes do not know which way is up: keep the
            # rotation the recogniser is more confident about.
            best = int(np.argmax([r[0]["similarity"] for r in results]))
        top = results[best]
        capture_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.pending[capture_id] = {
                "crop": crop,
                "quad": quad,
                "cards": cards,
                "vecs": vecs,
                "click": click,
                "source": source,
                "top": top,
                "up_vote": up_vote,
            }
            if len(self.pending) > 50:
                self.pending.pop(next(iter(self.pending)))
        short = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
        return {
            "capture_id": capture_id,
            "quad": quad.tolist(),
            "quad_source": source,
            "orientation": best * 180,
            "up_vote": None if up_vote is None else round(up_vote, 2),
            "card_px": round(float(short)),
            "card_png": png_b64(cards[best]),
            "candidates": candidates(top),
            "margin": round(top[0]["similarity"] - top[1]["similarity"], 3),
            # the 180-degree turn, so the UI can flip when the orientation call was wrong
            "flipped": {"card_png": png_b64(cards[1 - best]), "candidates": candidates(results[1 - best])},
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def label(self, capture_id: str, label: str | None, method: str) -> dict:
        with self.lock:
            p = self.pending.pop(capture_id, None)
        if p is None:
            raise KeyError(capture_id)
        if label is None:
            row = {"label": None, "method": method, "top5": [a["id"] for a in p["top"]]}
            return save_label(capture_id, row, crop_rgb=p["crop"])
        # Pick the orientation whose embedding is closest to the labeled art, so a wrong
        # top-1 on an upside-down card still stores the card the right way up.
        target = self.index.embeddings[self.index.by_id[label]]
        frame = self.index.frames[self.index.by_id[label]]
        best = int(np.argmax(p["vecs"][:, frame] @ target))
        quad = p["quad"]
        short = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
        row = {
            "label": label,
            "method": method,
            "top5": [a["id"] for a in p["top"]],
            "top5_sim": [round(a["similarity"], 4) for a in p["top"]],
            "click": list(p["click"]),
            "quad": quad.tolist(),
            "quad_source": p["source"],
            "orientation": best * 180,
            "card_px": round(float(short)),
            "art_px": round(float(short) * 0.84),
            "checkpoint": str(self.index.checkpoint),
        }
        if p["source"] == "detector":
            # the detector said index 0 was upright; the labeled art says which really was
            row["up_correct"] = best == 0
            row["up_vote"] = round(p["up_vote"], 3)
        return save_label(capture_id, row, crop_rgb=p["crop"], card_rgb=p["cards"][best])

    def stats(self) -> dict:
        rows = load_labels()
        n = len(rows)
        top1 = sum(r["top5"][0] == r["label"] for r in rows)
        top5 = sum(r["label"] in r["top5"] for r in rows)
        judged = [r["up_correct"] for r in rows if "up_correct" in r]
        return {
            "labeled": n,
            "top1": top1,
            "top5": top5,
            "train": sum(r["split"] == "train" for r in rows),
            "eval": sum(r["split"] == "eval" for r in rows),
            "up_correct": sum(judged),
            "up_judged": len(judged),
        }

    def search(self, q: str) -> list[dict]:
        """Name search for labelling. Every word must appear in the name, except that a word
        equal to a set code filters by set instead ("forest fin" -> the Final Fantasy Forests),
        since basics and staples have hundreds of printings, and a number (or "#280") must
        equal the collector number, since one set alone can have dozens of Forests."""
        words = q.strip().lower().split()
        if not words:
            return []
        sets = {a["set"] for a in self.index.arts}
        set_words = [w for w in words if w in sets]
        numbers = [w.lstrip("#") for w in words if w not in sets and (w.startswith("#") or w.isdigit())]
        name_words = [w for w in words if w not in sets and w.lstrip("#") not in numbers]
        # a word that is both a set code and part of the name ("war", "fin") keeps the name meaning too
        hits = [
            a
            for a in self.index.arts
            if (not set_words or a["set"] in set_words or all(w in a["name"].lower() for w in words))
            and all(w in a["name"].lower() for w in name_words)
            and all(str(a.get("collector_number", "")).lower() == n for n in numbers)
        ]
        prefix = " ".join(name_words)
        hits.sort(key=lambda a: (not a["name"].lower().startswith(prefix), a["set"] not in set_words, a["name"], a["set"], collector_key(a)))
        # a set filter is how the user scrolls one set's printings, so show all of them
        limit = 400 if set_words else 60
        return [{"id": a["id"], "name": a["name"], "set": a["set"], "number": a.get("collector_number")} for a in hits[:limit]]


def collector_key(art: dict) -> tuple[int, str]:
    """Sort key putting collector numbers in printed order: 9 before 10, then 10a, 10b."""
    number = str(art.get("collector_number", ""))
    digits = "".join(itertools.takewhile(str.isdigit, number))
    return (int(digits) if digits else 10**9, number[len(digits) :])


def candidates(hits: list[dict]) -> list[dict]:
    return [{"id": a["id"], "name": a["name"], "set": a["set"], "similarity": round(a["similarity"], 3), "frame": a["frame"]} for a in hits]


def png_b64(rgb: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def decode_jpeg_b64(data: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(data.split(",", 1)[-1]), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def send_json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_bytes(self, body: bytes, ctype: str):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path.startswith("/art/"):
                art_id = url.path[len("/art/") :].removesuffix(".jpg")
                p = ART_DIR / f"{art_id}.jpg"
                if art_id in session.index.by_id and p.exists():
                    self.send_bytes(p.read_bytes(), "image/jpeg")
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            elif url.path == "/stats":
                self.send_json(session.stats())
            elif url.path == "/search":
                self.send_json(session.search(parse_qs(url.query).get("q", [""])[0]))
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                if self.path == "/identify":
                    crop = decode_jpeg_b64(body["image"])
                    click = (float(body["click"][0]), float(body["click"][1]))
                    self.send_json(session.identify(crop, click, body.get("quad")))
                elif self.path == "/label":
                    row = session.label(body["capture_id"], body.get("label"), body.get("method", "confirm"))
                    self.send_json({"saved": row, "stats": session.stats()})
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except KeyError as e:
                self.send_json({"error": f"unknown or expired capture {e}"}, HTTPStatus.GONE)
            except Exception as e:  # surface to the page instead of dying
                self.send_json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--detector", help="CornerNet checkpoint from cardid.train_detector; omit to use the classical edge finder")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--frame-penalty",
        type=float,
        default=FRAME_PENALTY,
        help=f"similarity penalty for rare-frame (tall/saga/class) arts, 0 disables the frame prior (default {FRAME_PENALTY})",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    session = Session(Path(args.checkpoint), Path(args.detector) if args.detector else None, args.frame_penalty)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    locator = f"detector {args.detector}" if args.detector else "classical edge finder"
    print(f"gallery: {len(session.index.arts)} arts; quads from {locator}; open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
