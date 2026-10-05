"""Export-time hub penalty: a CSLS-style correction for gallery arts that match everything.

Some arts sit near the centre of the cuts every card produces for their frame. Funeral Room's
`room_0` half is the clearest case: the room_0 cut of any card is a sideways strip of art and
text box, those strips look alike, and Funeral Room was in the top 5 for 12% of all queries
(a typical art: 0.007%). The flat frame penalty cannot single such arts out.

For each gallery art, `hub_scores` takes the mean of its `HUB_NEIGHBOURS` highest similarities to
other cards' cuts for its frame (CSLS's r term), using full card scans as stand-ins for queries
(`data/cards`, the detector's scans). An art's own card never counts. `hub_penalties` turns the
excess over the gallery median into a penalty: `HUB_WEIGHT` times the excess, clipped to
[0, HUB_CAP]. It is only ever subtracted, so scores stay at or below the cosine similarity.
`cardid.export` adds it to the frame penalty baked into search.onnx and records the settings
in the manifest (`gallery.hub_penalty`). Without enough card scans it exports without one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .degrade import load_rgb
from .detect import CARD_H, CARD_W, FRAME_NAMES, art_crops
from .gallery import printing_index

HUB_WEIGHT = 0.25  # share of an art's excess r that it pays
HUB_NEIGHBOURS = 10  # r = mean of an art's k best similarities to the background cards
HUB_CAP = 0.1  # keeps a hub's own scan recognisable: Funeral Room pays ~0.09
HUB_MIN_CARDS = 200  # fewer background cards than this give a noisy r; export skips the penalty
HUB_MAX_CARDS = 3000  # a seeded sample of data/cards beyond this


def hub_scores(embeddings: np.ndarray, frames: np.ndarray, cuts: np.ndarray, owners: list[list[int]], k: int = HUB_NEIGHBOURS, chunk: int = 4096) -> np.ndarray:
    """CSLS r per gallery art: the mean of its k highest similarities to the background cards'
    cuts for its own frame. `cuts` is (M, F, D) in FRAME_NAMES order; `owners[m]` lists the
    gallery arts printed on card m, which never count against it."""
    if len(cuts) <= k:
        raise ValueError(f"need more than {k} background cards, got {len(cuts)}")
    pairs = np.array([(art, card) for card, arts in enumerate(owners) for art in arts], np.int64).reshape(-1, 2)
    r = np.empty(len(embeddings), np.float32)
    for frame in np.unique(frames):
        rows = np.flatnonzero(frames == frame)
        queries = cuts[:, frame].T  # (D, M)
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            sims = embeddings[part] @ queries
            own = pairs[np.isin(pairs[:, 0], part)]
            sims[np.searchsorted(part, own[:, 0]), own[:, 1]] = -np.inf
            r[part] = -np.partition(-sims, k - 1, axis=1)[:, :k].mean(axis=1)
    return r


def hub_penalties(r: np.ndarray, weight: float = HUB_WEIGHT, cap: float = HUB_CAP) -> np.ndarray:
    """`weight` times each art's r above the gallery median, clipped to [0, cap]."""
    return np.clip(weight * (r - np.median(r)), 0.0, cap).astype(np.float32)


def background_cards(card_dir: Path, limit: int = HUB_MAX_CARDS, seed: int = 0) -> list[Path]:
    """The card scans to measure hubs against: all of `card_dir`, or a seeded sample of `limit`."""
    paths = sorted(card_dir.glob("*.jpg"))
    if len(paths) > limit:
        pick = np.sort(np.random.default_rng(seed).choice(len(paths), limit, replace=False))
        paths = [paths[int(i)] for i in pick]
    return paths


def card_owners(paths: list[Path], arts: list[dict]) -> list[list[int]]:
    """Gallery arts on each scan. Scans are named by printing ID; a two-part printing owns its
    face-1 half (`<id>-1`) as well."""
    by_id = printing_index(arts)
    return [sorted({by_id[i] for i in (p.stem, f"{p.stem}-1", p.stem.removesuffix("-1")) if i in by_id}) for p in paths]


def background_cuts(embed, paths: list[Path], batch: int = 32) -> np.ndarray:
    """(M, F, D) embeddings of every frame cut of each scan, cut like a query (`art_crops` of
    the card resized to the canonical 250x350). `embed` maps (N, H, W, 3) uint8 to (N, D)."""
    out = []
    for start in tqdm(range(0, len(paths), batch), desc="hub background cards", leave=False):
        cards = [cv2.resize(load_rgb(p), (CARD_W, CARD_H), interpolation=cv2.INTER_AREA) for p in paths[start : start + batch]]
        crops = np.concatenate([art_crops(card) for card in cards])
        out.append(embed(crops).reshape(len(cards), len(FRAME_NAMES), -1))
    return np.concatenate(out)


def fingerprint(paths: list[Path]) -> str:
    return hashlib.sha256("\n".join(p.name for p in paths).encode()).hexdigest()[:16]


def export_hub_penalty(
    index, card_dir: Path, weight: float = HUB_WEIGHT, k: int = HUB_NEIGHBOURS, cap: float = HUB_CAP, min_cards: int = HUB_MIN_CARDS, limit: int = HUB_MAX_CARDS
) -> tuple[np.ndarray, dict | None]:
    """(per-art penalty, manifest record) for an `ArtIndex`. All zeros and None when disabled
    (`weight` 0) or when `card_dir` has fewer than `min_cards` scans."""
    zeros = np.zeros(len(index.embeddings), np.float32)
    if weight <= 0:
        return zeros, None
    paths = background_cards(card_dir, limit)
    if len(paths) < min_cards:
        print(
            f"WARNING: no hub penalty: {len(paths)} card scans in {card_dir}, need {min_cards} "
            "(`python -m cardid.scryfall --cards 3000`); hub arts such as Funeral Room keep matching everything"
        )
        return zeros, None
    cuts = background_cuts(index.embed, paths)
    r = hub_scores(index.embeddings, index.frames, cuts, card_owners(paths, index.arts), k)
    penalty = hub_penalties(r, weight, cap)
    worst = np.argsort(-penalty)[:5]
    print(f"hub penalty from {len(paths)} card scans: {int((penalty > 0).sum())} arts pay > 0, {int((penalty >= cap).sum())} the cap {cap}")
    for i in worst:
        print(f"  {penalty[i]:.3f}  {index.arts[i]['name']} ({FRAME_NAMES[index.frames[i]]}), r {r[i]:.3f}")
    info = {
        "weight": weight,
        "neighbours": k,
        "cap": cap,
        "cards": len(paths),
        "cards_fingerprint": fingerprint(paths),
        "median_r": round(float(np.median(r)), 4),
        "penalised": int((penalty > 0).sum()),
        "max": round(float(penalty.max()), 4),
    }
    return penalty, info
