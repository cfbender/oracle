"""Embedding index over the downloaded arts for a given checkpoint, cached next to it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import art_frames, gallery_images, load_arts
from .detect import FRAME_NAMES, FRAME_PENALTY, frame_penalties
from .evaluate import embed_images
from .model import Embedder


def frame_similarities(vecs: np.ndarray, embeddings: np.ndarray, frames: np.ndarray, penalty: float = FRAME_PENALTY) -> np.ndarray:
    """Cosine similarity of every gallery art to the query cut for that art's frame, minus the
    frame prior (`detect.frame_penalties`). `vecs` is F x D (one query embedding per frame,
    FRAME_NAMES order) or D for a single cut used against every art; `frames` indexes
    FRAME_NAMES per gallery art."""
    if vecs.ndim == 1:
        return embeddings @ vecs
    return (embeddings @ vecs.T)[np.arange(len(embeddings)), frames] - frame_penalties(frames, penalty)


class ArtIndex:
    def __init__(self, checkpoint: Path, frame_penalty: float = FRAME_PENALTY):
        self.checkpoint = Path(checkpoint)
        self.frame_penalty = frame_penalty
        self.model = Embedder(pretrained=False).eval()
        self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
        self.arts = load_arts()
        self.by_id = {a["id"]: i for i, a in enumerate(self.arts)}
        self.frames = art_frames(self.arts)
        cache = self.checkpoint.with_name(f"{self.checkpoint.stem}-gallery-{len(self.arts)}-{int(self.checkpoint.stat().st_mtime)}.npy")
        if cache.exists():
            self.embeddings = np.load(cache)
        else:
            print(f"embedding {len(self.arts)} gallery arts with {self.checkpoint} ...")
            self.embeddings = embed_images(self.model, gallery_images(self.arts))
            np.save(cache, self.embeddings)

    @torch.no_grad()
    def embed(self, images: np.ndarray) -> np.ndarray:
        return embed_images(self.model, images)

    def frame_name(self, art_id: str) -> str:
        return FRAME_NAMES[self.frames[self.by_id[art_id]]]

    def search(self, vecs: np.ndarray, k: int = 5) -> list[dict]:
        """Top-k arts for the per-frame query embeddings (F x D, see `detect.art_crops`), each
        with its score (similarity minus the frame prior) and the frame it was matched in."""
        sims = frame_similarities(vecs, self.embeddings, self.frames, self.frame_penalty)
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [dict(self.arts[i], similarity=float(sims[i]), frame=FRAME_NAMES[self.frames[i]]) for i in idx]
