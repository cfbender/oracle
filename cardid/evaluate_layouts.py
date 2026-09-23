"""Stratified two-part evaluation with real full scans, never a half pasted into a modern box.

Run after scryfall --update. --prepare-only can fetch the new layout gallery on a smaller
existing sample without changing arts.json or its held-out split. Cached bulk is required.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import torch

from . import ART_DIR, DATA_DIR
from .data import art_frames, art_path, load_arts
from .degrade import PROFILES, clean_view, degraded_view, load_rgb
from .detect import art_crops, warp_card
from .detector import Detector
from .evaluate import cosine_topk, topk
from .gallery import printing_index
from .index import ArtIndex, frame_similarities
from .scryfall import HEADERS, WORKERS, fetch_image, usable_entries
from .synth import SCENE, ArtBank, CardBank, render_scene

GROUPS = ("room", "split", "aftermath", "flip")
ROOT = DATA_DIR / "layout-eval"


def prepare() -> list[dict]:
    ROOT.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    entries = [a for a in usable_entries(DATA_DIR / "all-cards.jsonl.gz") if a["layout"] in {"split", "flip"}]
    for group in GROUPS:
        arts = [a for a in entries if a["layout_group"] == group]
        print(f"{group}: {len(arts)} artworks, {sum(len(a['printings']) for a in arts)} printing faces")
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client, ThreadPoolExecutor(WORKERS) as pool:
        failures = [art_id for art_id, ok in pool.map(lambda art: fetch_image(client, art), entries) if not ok]
        if failures:
            raise SystemExit(f"Failed layout downloads (rerun to retry): {failures}")
    (ROOT / "arts.json").write_text(json.dumps(entries))
    return entries


def accuracy(ranks, targets) -> dict:
    """A full card visibly contains both faces; either face is a correct card identity."""
    return {
        "queries": len(ranks),
        **{f"top{k}": sum(bool(set(row[:k]) & truth) for row, truth in zip(ranks, targets, strict=True)) / len(ranks) for k in (1, 5)},
    }


def evaluate(args, entries: list[dict]) -> dict:
    index = ArtIndex(args.checkpoint)
    known = set(index.by_id)
    extra = [a for a in entries if a["id"] not in known]
    gallery = index.arts + extra
    embeddings = np.concatenate([index.embeddings, index.embed(np.stack([clean_view(load_rgb(art_path(a))) for a in extra]))]) if extra else index.embeddings
    frames = art_frames(gallery)
    labels = printing_index(gallery)
    rng = np.random.default_rng(args.seed)
    detector = Detector(args.detector) if args.detector else None
    backgrounds = ArtBank()
    results = {"checkpoint": str(args.checkpoint), "gallery": len(gallery), "seed": args.seed, "groups": {}}
    scans = ROOT / "cards"
    scans.mkdir(exist_ok=True)
    for group in GROUPS:
        arts = [a for a in entries if a["layout_group"] == group]
        queries, targets = [], []
        for art in arts:
            for _ in range(args.per_card):
                queries.append(degraded_view(load_rgb(art_path(art)), rng, PROFILES["realistic"])[0])
                targets.append({labels[art["id"]]})
        ranks, _ = cosine_topk(index.embed(np.stack(queries)), embeddings, 5)
        metrics = {"isolated_art_realistic": accuracy(ranks, targets)}
        # Sample distinct physical scans, including both face targets when present. The
        # renderer supplies all printed details, perspective, sleeves and occluders.
        cards = list({a["id"].removesuffix("-1"): a for a in arts}.items())
        cards.sort()
        rng.shuffle(cards)
        cards = cards[: args.per_group]
        with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
            for card_id, art in cards:
                if not fetch_image(client, {"id": card_id, "card_url": art["url"]}, scans, "card_url")[1]:
                    raise SystemExit(f"Failed scan {card_id}; rerun")
        bank = CardBank([scans / f"{card_id}.jpg" for card_id, _ in cards])
        modes = {"clean_scan": [], "scene_known_quad": [], "scene_old_crops": []}
        if detector:
            modes["scene_detector"] = []
        truths, clean_truths = [], []
        for i, (card_id, _) in enumerate(cards):
            truth = {labels[p] for p in (card_id, card_id + "-1") if p in labels}
            scan = load_rgb(bank.paths[i])
            quad = np.float32([[0, 0], [scan.shape[1], 0], [scan.shape[1], scan.shape[0]], [0, scan.shape[0]]])
            modes["clean_scan"].append(index.embed(art_crops(warp_card(scan, quad))))
            clean_truths.append(truth)
            for _ in range(args.per_card):
                scene, quad = render_scene(rng, bank, backgrounds, out=SCENE, target_index=i)
                vectors = index.embed(art_crops(warp_card(scene, quad)))
                modes["scene_known_quad"].append(vectors)
                modes["scene_old_crops"].append(vectors)
                if detector:
                    detected, _ = detector.locate_up(scene, tuple(quad.mean(axis=0)))
                    modes["scene_detector"].append(index.embed(art_crops(warp_card(scene, detected))))
                truths.append(truth)
        for mode, vectors in modes.items():
            ranks = []
            for vec in vectors:
                if mode == "scene_old_crops":
                    # Best of the original six cuts is a generous no-new-geometry control.
                    scores = embeddings @ vec[:6].T
                    scores = scores.max(axis=1)
                else:
                    scores = frame_similarities(vec, embeddings, frames)
                ranks.append(topk(scores, 5)[0])
            metrics[mode] = accuracy(ranks, clean_truths if mode == "clean_scan" else truths)
        metrics["cards"] = [card_id for card_id, _ in cards]
        results["groups"][group] = metrics
        print(group, json.dumps(metrics), flush=True)
    # Existing eval split, same degraded pixels/targets but expanded distractor gallery.
    from .data import cached_eval_queries

    images, target, _ = cached_eval_queries(load_arts(), profile="realistic")
    ranks, _ = cosine_topk(index.embed(images), embeddings, 5)
    results["existing_eval_expanded_gallery"] = accuracy(ranks, [{int(i)} for i in target])
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--per-card", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cached = ROOT / "arts.json"
    entries = json.loads(cached.read_text()) if cached.exists() else prepare()
    if not args.prepare_only:
        if not args.checkpoint:
            parser.error("--checkpoint is required for evaluation")
        (ROOT / "report.json").write_text(json.dumps(evaluate(args, entries), indent=2))


if __name__ == "__main__":
    main()
