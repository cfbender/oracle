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

from .data import art_frames, cached_eval_queries, gallery_images, load_arts, to_tensor
from .degrade import PROFILES
from .detect import FRAME_NAMES, FRAME_PENALTY, frame_penalties
from .detector import Detector
from .gallery import printing_index
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


def topk(sims: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(indices, similarities) of the k best gallery entries along the last axis, best first."""
    idx = np.argpartition(-sims, k, axis=-1)[..., :k]
    top = np.take_along_axis(sims, idx, axis=-1)
    order = np.argsort(-top, axis=-1)
    return np.take_along_axis(idx, order, axis=-1), np.take_along_axis(top, order, axis=-1)


def cosine_topk(q: np.ndarray, g: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return topk(q @ g.T, k)


def frame_topk(q: np.ndarray, g: np.ndarray, frames: np.ndarray, k: int, penalty: float = FRAME_PENALTY) -> tuple[np.ndarray, np.ndarray]:
    """`cosine_topk` for per-frame query embeddings (... x F x D, `detect.FRAME_NAMES` order):
    each gallery art is scored against the cut for its own frame (`frames`, G ints), minus the
    frame prior (`detect.frame_penalties`), as `index.ArtIndex.search` does."""
    sims = np.einsum("...fd,gd->...fg", q, g)
    sims = np.take_along_axis(sims, np.broadcast_to(frames, (*sims.shape[:-2], 1, len(frames))), axis=-2)[..., 0, :]
    return topk(sims - frame_penalties(frames, penalty), k)


def report(idx: np.ndarray, sims: np.ndarray, targets: np.ndarray, infos: list[dict], label: str) -> dict:
    top1 = idx[:, 0] == targets
    top5 = (idx[:, :5] == targets[:, None]).any(axis=1)
    widths = np.array([i["width"] for i in infos])
    strong = np.array([i["strong_perspective"] for i in infos])
    margin = sims[:, 0] - sims[:, 1]

    result = {
        "method": label,
        "queries": len(targets),
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


def print_misses(idx: np.ndarray, sims: np.ndarray, targets: np.ndarray, infos: list[dict], arts: list[dict], frames: np.ndarray | None = None) -> None:
    """One line per wrong real capture: what it was, what came back, how confident, and (with
    a detector) how far its quad sat from the labeled one. A miss with a quad within a few
    percent is the recogniser's; one with a quad way off is the detector's. Non-modern art
    frames of the truth are named, since those are cut differently (`detect.FRAMES`)."""
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
        if frames is not None and FRAME_NAMES[frames[targets[i]]] != "modern":
            line += f", truth has {FRAME_NAMES[frames[targets[i]]]} art"
        if "other_orientation" in infos[i]:
            art, sim, top = infos[i]["other_orientation"]
            rank = np.where(top == targets[i])[0]
            line += (
                f"\n      other orientation: {art['name']} [{art['set']}] sim {sim:.3f}, truth {'rank ' + str(rank[0] + 1) if len(rank) else 'not in top 5'}"
            )
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
    parser.add_argument(
        "--frame-penalty",
        type=float,
        default=FRAME_PENALTY,
        help=f"with --real: similarity penalty for rare-frame (tall/saga/class) arts, 0 disables the frame prior (default {FRAME_PENALTY})",
    )
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cpu, or cuda (also AMD/ROCm)")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 8)
    device = pick_device(args.device)

    arts = load_arts()
    t0 = time.time()
    gallery = gallery_images(arts)
    frames = art_frames(arts)
    per_query = 1
    learned_up = False
    if args.real and args.detector:
        profile = f"real+{'classical' if args.detector == 'classical' else 'detector'}"
        locate = classical_locate if args.detector == "classical" else Detector(args.detector).locate
        queries, targets, infos = real_detector_queries(load_labels("eval"), printing_index(arts), locate)
        per_query = 2  # both orientations; which one counts is decided below
        learned_up = args.detector != "classical"  # the learned detector orders the quad upright itself
    elif args.real:
        profile = "real"
        queries, targets, infos = real_eval_queries(load_labels("eval"), printing_index(arts))
    else:
        profile = args.profile
        queries, targets, infos = cached_eval_queries(arts, per_art=args.per_art, profile=args.profile)
    print(f"[{profile}] gallery {len(gallery)} arts, {len(queries)} queries ({time.time() - t0:.0f}s to load)")

    if args.method in ("dhash", "phash"):
        if queries.ndim > 4:
            queries = queries[..., 0, :, :, :].reshape(-1, *queries.shape[-3:])  # hashes only get the modern cut
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
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
            label = f"checkpoint:{args.checkpoint}:{profile}"
        model.to(device)
        t0 = time.time()
        g = embed_images(model, gallery)
        q = embed_images(model, queries.reshape(-1, *queries.shape[-3:]))
        print(f"embedded in {time.time() - t0:.1f}s on {describe_device(device)}")
        if queries.ndim > 4:
            # real captures carry every frame's cut: score each art against the cut for its frame
            idx, sims = frame_topk(q.reshape(*queries.shape[:-3], q.shape[-1]), g, frames, 5, args.frame_penalty)
        else:
            idx, sims = cosine_topk(q, g, 5)
        if per_query > 1:
            idx, sims = idx.reshape(-1, per_query, 5), sims.reshape(-1, per_query, 5)
            rows = np.arange(len(idx))
            if learned_up:
                # capture.py trusts the detector's up output: query 0 is the card as it ordered it
                pick = np.zeros(len(idx), dtype=int)
            else:
                # capture.py keeps the orientation whose best match is most similar
                pick = sims[:, :, 0].argmax(axis=1)
            other = idx[rows, 1 - pick], sims[rows, 1 - pick]
            for i, info in enumerate(infos):
                info["other_orientation"] = (arts[other[0][i, 0]], float(other[1][i, 0]), other[0][i])
            # what a correct up/down decision would recover: the truth is top-1 in either orientation
            hit_each = idx[:, :, 0] == targets[:, None]
            oracle = hit_each.any(axis=1).mean()
            print(f"orientation oracle top1 (truth is top-1 in either orientation): {oracle:.2f}")
            if learned_up:
                # a wrong up call is visible when the truth is top-1 only in the rotation the detector rejected
                wrong = int((hit_each[:, 1] & ~hit_each[:, 0]).sum())
                confident = int(hit_each.sum(axis=1).astype(bool).sum())
                print(f"detector up output wrong on {wrong} of the {confident} captures where the truth is top-1 in one orientation")
            idx, sims = idx[rows, pick], sims[rows, pick]
        report(idx, sims, targets, infos, label)
        if args.real:
            print_misses(idx, sims, targets, infos, arts, frames)


def classical_locate(crop: np.ndarray, click: tuple[float, float]) -> np.ndarray:
    """The edge finder as a `locate` function; when it finds nothing, a centred upright card
    of a typical size so the capture still yields a (wrong) guess rather than crashing."""
    from .detect import find_card_quad
    from .detector import card_rect

    quad = find_card_quad(crop, click)
    return quad if quad is not None else card_rect(click[0], click[1], 200.0, 0.0)


if __name__ == "__main__":
    main()
