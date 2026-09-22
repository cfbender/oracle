"""Run an exported bundle (`cardid.export`) with onnxruntime: the reference for the glue a
browser runtime has to write around the three graphs, and the parity check for the export.

    uv run python -m cardid.bundle data/bundles/2026-09-22-full-3 --image frame.jpg --click 660,350

Per click, mirroring `Detector.locate_up` + `capture.Session.identify`:

1. Resample the SCENE (640) px square around the click to 256x256 (`synth.window_around`, an
   affine scale + translation with edge replication) and run detector.onnx. Map its quad,
   centre and short side back to image pixels through the inverse affine.
2. Second pass on a window of side max(short * 88/63 / 0.6, 64) px around that centre, so the
   card spans ~60% of the input; its quad is the answer, its up vote is |up| / 4.
3. embed.onnx on the image and the quad -> one embedding per frame cut.
4. search.onnx -> top-k gallery indices and scores; `arts.json` names them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import CARD_ASPECT
from .graphs import ROTATIONS
from .synth import DET_INPUT, SCENE

REFINE_FILL = 0.6  # the card's long side as a fraction of the refine window
REFINE_MIN_SIDE = 64.0


def rgba(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> RGBA uint8, the layout a canvas `getImageData` hands a browser runtime."""
    return np.ascontiguousarray(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]))


def window(img: np.ndarray, cx: float, cy: float, side: float) -> tuple[np.ndarray, float]:
    """The `side` px square around (cx, cy) as a DET_INPUT square (see `synth.window_around`)
    and its scale s: window = s * (image - (cx, cy)) + DET_INPUT / 2."""
    s = DET_INPUT / side
    M = np.float32([[s, 0, DET_INPUT / 2 - s * cx], [0, s, DET_INPUT / 2 - s * cy]])
    flags = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (DET_INPUT, DET_INPUT), flags=flags, borderMode=cv2.BORDER_REPLICATE), s


class Bundle:
    def __init__(self, path: Path, providers: list[str] | None = None):
        import onnxruntime as ort

        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.arts = json.loads((self.path / "arts.json").read_text())
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = providers or ["CPUExecutionProvider"]
        self.detector = ort.InferenceSession(str(self.path / "detector.onnx"), opts, providers=providers)
        self.embed = ort.InferenceSession(str(self.path / "embed.onnx"), opts, providers=providers)
        self.search = ort.InferenceSession(str(self.path / "search.onnx"), opts, providers=providers)
        self.timings: dict[str, float] = {}

    def _run(self, name: str, session, feeds: dict) -> list[np.ndarray]:
        t0 = time.perf_counter()
        out = session.run(None, feeds)
        self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return out

    def detect_window(self, img: np.ndarray, cx: float, cy: float, side: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(quad (4, 2) image px in printed order, up (2,), centre (2,) image px, short px)."""
        win, s = window(img, cx, cy, side)
        quad, up, centre, short = self._run("detector", self.detector, {"window": rgba(win)})
        offset = np.float32([cx, cy])
        return (quad - DET_INPUT / 2) / s + offset, up, (centre - DET_INPUT / 2) / s + offset, float(short) / s

    def locate_up(self, img: np.ndarray, click: tuple[float, float]) -> tuple[np.ndarray, float]:
        """`Detector.locate_up` on the exported graph: (printed-order quad, up vote in [0, 1])."""
        _, _, centre, short = self.detect_window(img, click[0], click[1], SCENE)
        side = max(short * CARD_ASPECT / REFINE_FILL, REFINE_MIN_SIDE)
        quad, up, _, _ = self.detect_window(img, float(centre[0]), float(centre[1]), side)
        return quad.astype(np.float32), float(np.linalg.norm(up) / ROTATIONS)

    def embed_card(self, img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """(F, D) embeddings of the frame cuts of the card at `quad` (image px, printed order)."""
        return self._run("embed", self.embed, {"scene": rgba(img), "quad": np.ascontiguousarray(quad, np.float32)})[0]

    def rank(self, embeddings: np.ndarray) -> list[dict]:
        indices, scores = self._run("search", self.search, {"embeddings": np.ascontiguousarray(embeddings, np.float32)})
        return [dict(self.arts[int(i)], index=int(i), score=float(s)) for i, s in zip(indices, scores, strict=True)]

    def identify(self, img: np.ndarray, click: tuple[float, float]) -> dict:
        """Everything for one click on an RGB uint8 image."""
        t0 = time.perf_counter()
        quad, vote = self.locate_up(img, click)
        results = self.rank(self.embed_card(img, quad))
        return {"quad": quad, "up_vote": vote, "results": results, "ms": (time.perf_counter() - t0) * 1000}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("--image", required=True)
    ap.add_argument("--click", action="append", required=True, help="x,y in image pixels; repeatable")
    args = ap.parse_args()

    bundle = Bundle(Path(args.bundle))
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    for spec in args.click:
        x, y = (float(v) for v in spec.split(","))
        out = bundle.identify(img, (x, y))
        print(f"click ({x:.0f},{y:.0f}): {out['ms']:.0f} ms, up vote {out['up_vote']:.2f}, quad {np.round(out['quad']).astype(int).tolist()}")
        for r in out["results"]:
            print(f"    {r['score']:.3f}  {r['name']} [{r['set']}] #{r.get('collector_number', '?')} ({r['frame']})")


if __name__ == "__main__":
    main()
