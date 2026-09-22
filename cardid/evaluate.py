"""Retrieval evaluation: gallery = clean views of every downloaded art, queries = degraded
eval-split arts (never seen in training). Reports top-1/top-5, accuracy by simulated art
width, and the confidence margin (best minus second-best similarity) needed to reject wrong
answers, which is what the UI will use to decide between "show card" and "show top-3".

    uv run python -m cardid.evaluate --method dhash
    uv run python -m cardid.evaluate --method phash --hash-size 16
    uv run python -m cardid.evaluate --method pretrained
    uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .data import cached_eval_queries, gallery_images, load_arts, to_tensor
from .degrade import PROFILES
from .hashing import hamming_topk, hash_images
from .model import Embedder, PretrainedBaseline

WIDTH_BUCKETS = [(56, 80), (80, 110), (110, 141)]  # simulated art width; 56-79 only occurs in "harsh"


@torch.no_grad()
def embed_images(model: torch.nn.Module, images: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        out.append(model(to_tensor(images[i : i + batch])).numpy())
    return np.concatenate(out)


def cosine_topk(q: np.ndarray, g: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims = q @ g.T
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    top = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-top, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(top, order, axis=1)


def report(idx: np.ndarray, sims: np.ndarray, targets: np.ndarray, infos: list[dict], label: str) -> dict:
    top1 = idx[:, 0] == targets
    top5 = (idx[:, :5] == targets[:, None]).any(axis=1)
    widths = np.array([i["width"] for i in infos])
    strong = np.array([i["strong_perspective"] for i in infos])
    margin = sims[:, 0] - sims[:, 1]

    result = {
        "method": label,
        "queries": int(len(targets)),
        "top1": float(top1.mean()),
        "top5": float(top5.mean()),
        "by_width": {},
        "strong_perspective_top1": float(top1[strong].mean()) if strong.any() else None,
        "mild_perspective_top1": float(top1[~strong].mean()),
    }
    for lo, hi in WIDTH_BUCKETS:
        m = (widths >= lo) & (widths < hi)
        if m.any():
            result["by_width"][f"{lo}-{hi - 1}px"] = float(top1[m].mean())

    # Rejection threshold: smallest margin such that >=99% of accepted answers are right.
    # Coverage tells us how often the UI can show a single confident answer.
    order = np.argsort(-margin)
    correct_sorted = top1[order]
    precision = np.cumsum(correct_sorted) / np.arange(1, len(order) + 1)
    ok = np.where(precision >= 0.99)[0]
    if len(ok):
        n = ok.max() + 1
        result["margin_for_99pct_precision"] = float(margin[order][n - 1])
        result["coverage_at_99pct_precision"] = float(n / len(order))
    else:
        result["margin_for_99pct_precision"] = None
        result["coverage_at_99pct_precision"] = 0.0

    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dhash", "phash", "pretrained", "checkpoint"], required=True)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--per-art", type=int, default=3)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="harsh")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 8)

    arts = load_arts()
    t0 = time.time()
    gallery = gallery_images(arts)
    queries, targets, infos = cached_eval_queries(arts, per_art=args.per_art, profile=args.profile)
    print(f"[{args.profile}] gallery {len(gallery)} arts, {len(queries)} queries ({time.time() - t0:.0f}s to load)")

    if args.method in ("dhash", "phash"):
        g = hash_images(gallery, args.method, args.hash_size)
        q = hash_images(queries, args.method, args.hash_size)
        idx, sims = hamming_topk(q, g, 5)
        report(idx, sims, targets, infos, f"{args.method}-{args.hash_size * args.hash_size}bit")
    else:
        if args.method == "pretrained":
            model = PretrainedBaseline()
            label = "mobilenetv3-imagenet-untrained"
        else:
            model = Embedder(pretrained=False)
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
            label = f"checkpoint:{args.checkpoint}"
        t0 = time.time()
        g = embed_images(model, gallery)
        q = embed_images(model, queries)
        print(f"embedded in {time.time() - t0:.1f}s")
        idx, sims = cosine_topk(q, g, 5)
        report(idx, sims, targets, infos, label)


if __name__ == "__main__":
    main()
