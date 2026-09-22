"""Perceptual-hash baselines (what ManaVault's scanner used) for comparison with the CNN."""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image


def hash_images(images: np.ndarray, method: str, hash_size: int) -> np.ndarray:
    """NHWC uint8 -> (N, hash_size*hash_size) bool bit matrix."""
    fn = {"dhash": imagehash.dhash, "phash": imagehash.phash}[method]
    return np.stack([fn(Image.fromarray(img), hash_size=hash_size).hash.reshape(-1) for img in images])


def hamming_topk(queries: np.ndarray, gallery: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) of the k nearest gallery hashes; similarity = 1 - dist/bits."""
    bits = gallery.shape[1]
    q = np.packbits(queries, axis=1)
    g = np.packbits(gallery, axis=1)
    lut = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    out_idx = np.empty((len(q), k), np.int64)
    out_sim = np.empty((len(q), k), np.float32)
    for i, row in enumerate(q):
        dist = lut[np.bitwise_xor(g, row)].sum(axis=1)
        idx = np.argpartition(dist, k)[:k]
        idx = idx[np.argsort(dist[idx])]
        out_idx[i] = idx
        out_sim[i] = 1.0 - dist[idx] / bits
    return out_idx, out_sim
