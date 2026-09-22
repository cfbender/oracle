"""CPU latency of the recognizer's per-click work: embed one crop, then nearest-neighbor
search over a production-sized gallery (~49k unique artworks).

    uv run python -m cardid.bench [--checkpoint data/runs/m0/best.pt]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .degrade import INPUT_SIZE
from .model import EMBED_DIM, Embedder

GALLERY_SIZE = 48_787  # usable unique artworks in Scryfall's bulk file on 2026-09-22


def timeit(fn, n: int, warmup: int = 5) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--threads", type=int, default=1, help="1 mimics a worker handling one click; the server has other work")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = Embedder(pretrained=False).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        med, p95 = timeit(lambda: model(x), 50)
    print(f"embed 1x{INPUT_SIZE}px (torch eager, {args.threads} thread): median {med:.1f} ms, p95 {p95:.1f} ms")

    rng = np.random.default_rng(0)
    gallery = rng.standard_normal((GALLERY_SIZE, EMBED_DIM), dtype=np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    q = gallery[123] + 0.1 * rng.standard_normal(EMBED_DIM, dtype=np.float32)

    def nn_search():
        sims = gallery @ q
        idx = np.argpartition(-sims, 5)[:5]
        return idx[np.argsort(-sims[idx])]

    med, p95 = timeit(nn_search, 200)
    print(f"cosine top-5 over {GALLERY_SIZE} x {EMBED_DIM} float32 ({gallery.nbytes / 1e6:.1f} MB): median {med:.2f} ms, p95 {p95:.2f} ms")

    g8 = np.clip(np.round(gallery * 127), -127, 127).astype(np.int8)
    q8 = np.clip(np.round(q * 127), -127, 127).astype(np.int32)

    def nn_search_int8():
        sims = g8.astype(np.int32) @ q8
        idx = np.argpartition(-sims, 5)[:5]
        return idx[np.argsort(-sims[idx])]

    med, p95 = timeit(nn_search_int8, 200)
    print(f"int8 gallery ({g8.nbytes / 1e6:.1f} MB, browser-shippable): median {med:.2f} ms, p95 {p95:.2f} ms")


if __name__ == "__main__":
    main()
