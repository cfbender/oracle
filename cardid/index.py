"""Embedding index over the downloaded arts for a given checkpoint, cached next to it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import gallery_images, load_arts
from .evaluate import embed_images
from .model import Embedder


class ArtIndex:
    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint)
        self.model = Embedder(pretrained=False).eval()
        self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
        self.arts = load_arts()
        self.by_id = {a["id"]: i for i, a in enumerate(self.arts)}
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

    def search(self, vec: np.ndarray, k: int = 5) -> list[dict]:
        sims = self.embeddings @ vec
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [dict(self.arts[i], similarity=float(sims[i])) for i in idx]
