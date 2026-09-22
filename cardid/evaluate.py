"""Retrieval evaluation: gallery = clean views of every downloaded art, queries = degraded
eval-split arts (never seen in training). Reports top-1/top-5, accuracy by simulated art
width, and the confidence margin (best minus second-best similarity) needed to reject wrong
answers, which is what the UI will use to decide between "show card" and "show top-3".

    uv run python -m cardid.evaluate --method dhash
    uv run python -m cardid.evaluate --method phash --hash-size 16
    uv run python -m cardid.evaluate --method pretrained
    uv run python -m cardid.evaluate --method checkpoint --checkpoint data/runs/<run>/best.pt
    uv run python -m cardid.evaluate --method checkpoint --checkpoint ... --real   # held-out webcam captures
    uv run python -m cardid.evaluate --method checkpoint --checkpoint ... --real --detector data/runs/det/best.pt   # re-locating the card first
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
from .detector import Detector
from .hashing import hamming_topk, hash_images
from .model import Embedder, PretrainedBaseline, describe_device, pick_device
from .real import load_labels, real_detector_queries, real_eval_queries

WIDTH_BUCKETS = [(56, 80), (80, 110), (110, 141), (141, 10_000)]  # art width in px; 56-79 only occurs in "harsh", 141+ only in real captures


@torch.no_grad()
def embed_images(model: torch.nn.Module, images: np.ndarray, batch: int = 256) -> np.ndarray:
    """Embed on whatever device the model lives on; always returns CPU float32 numpy."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(images), batch):
        out.append(model(to_tensor(images[i : i + batch]).to(device)).cpu().numpy())
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
            result["by_width"][f"{lo}+px" if hi >= 10_000 else f"{lo}-{hi - 1}px"] = float(top1[m].mean())

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


def print_misses(idx: np.ndarray, sims: np.ndarray, targets: np.ndarray, infos: list[dict], arts: list[dict]) -> None:
    """One line per wrong real capture: what it was, what came back, how confident, and (with
    a detector) how far its quad sat from the labeled one. A miss with a quad within a few
    percent is the recogniser's; one with a quad way off is the detector's."""
    misses = np.where(idx[:, 0] != targets)[0]
    if not len(misses):
        return
    print(f"misses ({len(misses)}):")
    for i in misses:
        truth, got = arts[targets[i]], arts[idx[i, 0]]
        rank = np.where(idx[i] == targets[i])[0]
        line = (
            f"  {infos[i].get('capture_id', i)}: {truth['name']} [{truth['set']}] -> {got['name']} [{got['set']}]"
            f" sim {sims[i, 0]:.3f} margin {sims[i, 0] - sims[i, 1]:.3f}, truth {'rank ' + str(rank[0] + 1) if len(rank) else 'not in top 5'}"
        )
        if "quad_err" in infos[i]:
            line += f", detector quad {infos[i]['quad_err'] * 100:.1f}% of short side off the label"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dhash", "phash", "pretrained", "checkpoint"], required=True)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--per-art", type=int, default=3)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="harsh")
    parser.add_argument("--real", action="store_true", help="query with the eval split of labeled real captures instead of synthetic degradation")
    parser.add_argument(
        "--detector",
        help="with --real: re-locate the card in each stored crop with this CornerNet checkpoint (or 'classical' for the edge finder) instead of using the stored quad",
    )
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cpu, or cuda (also AMD/ROCm)")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 8)
    device = pick_device(args.device)

    arts = load_arts()
    t0 = time.time()
    gallery = gallery_images(arts)
    per_query = 1
    if args.real and args.detector:
        profile = f"real+{'classical' if args.detector == 'classical' else 'detector'}"
        locate = classical_locate if args.detector == "classical" else Detector(args.detector).locate
        queries, targets, infos = real_detector_queries(load_labels("eval"), {a["id"]: i for i, a in enumerate(arts)}, locate)
        per_query = 2  # both orientations; the more confident one is kept below
    elif args.real:
        profile = "real"
        queries, targets, infos = real_eval_queries(load_labels("eval"), {a["id"]: i for i, a in enumerate(arts)})
    else:
        profile = args.profile
        queries, targets, infos = cached_eval_queries(arts, per_art=args.per_art, profile=args.profile)
    print(f"[{profile}] gallery {len(gallery)} arts, {len(queries)} queries ({time.time() - t0:.0f}s to load)")

    if args.method in ("dhash", "phash"):
        g = hash_images(gallery, args.method, args.hash_size)
        q = hash_images(queries, args.method, args.hash_size)
        idx, sims = hamming_topk(q, g, 5)
        report(idx, sims, targets, infos, f"{args.method}-{args.hash_size * args.hash_size}bit")
        if args.real:
            print_misses(idx, sims, targets, infos, arts)
    else:
        if args.method == "pretrained":
            model = PretrainedBaseline()
            label = "mobilenetv3-imagenet-untrained"
        else:
            model = Embedder(pretrained=False)
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
            label = f"checkpoint:{args.checkpoint}:{profile}"
        model.to(device)
        t0 = time.time()
        g = embed_images(model, gallery)
        q = embed_images(model, queries)
        print(f"embedded in {time.time() - t0:.1f}s on {describe_device(device)}")
        idx, sims = cosine_topk(q, g, 5)
        if per_query > 1:
            # capture.py keeps the orientation whose best match is most similar
            idx, sims = idx.reshape(-1, per_query, 5), sims.reshape(-1, per_query, 5)
            pick = sims[:, :, 0].argmax(axis=1)
            idx, sims = idx[np.arange(len(idx)), pick], sims[np.arange(len(sims)), pick]
        report(idx, sims, targets, infos, label)
        if args.real:
            print_misses(idx, sims, targets, infos, arts)


def classical_locate(crop: np.ndarray, click: tuple[float, float]) -> np.ndarray:
    """The edge finder as a `locate` function; when it finds nothing, a centred upright card
    of a typical size so the capture still yields a (wrong) guess rather than crashing."""
    from .detect import find_card_quad
    from .detector import card_rect

    quad = find_card_quad(crop, click)
    return quad if quad is not None else card_rect(click[0], click[1], 200.0, 0.0)


if __name__ == "__main__":
    main()
