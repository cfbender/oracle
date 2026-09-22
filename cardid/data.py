"""Dataset plumbing shared by the baselines, training, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import ART_DIR, DATA_DIR
from .degrade import PROFILES, Degradation, clean_view, degraded_view, load_rgb

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_arts() -> list[dict]:
    arts = json.loads((DATA_DIR / "arts.json").read_text())
    return [a for a in arts if (ART_DIR / f"{a['id']}.jpg").exists()]


def split(arts: list[dict], name: str) -> list[dict]:
    return [a for a in arts if a["split"] == name]


def art_path(art: dict) -> Path:
    return ART_DIR / f"{art['id']}.jpg"


def to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 (or NHWC) -> normalized CHW float tensor."""
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, -3)))


class PairDataset(Dataset):
    """One (clean, degraded) pair per art, with a fresh random degradation every access."""

    def __init__(self, arts: list[dict], cfg: Degradation = Degradation(), seed: int = 0):
        self.arts = arts
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.arts)

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        img = load_rgb(art_path(self.arts[i]))
        # The gallery is built from clean views, so the anchor stays clean apart from a tiny
        # crop jitter that stops the model from keying on exact border pixels.
        clean = clean_view(img)
        degraded, _ = degraded_view(img, rng, self.cfg)
        return to_tensor(clean), to_tensor(degraded), i


def gallery_images(arts: list[dict]) -> np.ndarray:
    """Clean 128px views of every art, cached as one uint8 array (49k arts = 2.4 GB)."""
    cache = DATA_DIR / f"gallery-{len(arts)}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    images = np.stack([clean_view(load_rgb(art_path(a))) for a in arts])
    np.save(cache, images)
    return images


def build_eval_queries(arts: list[dict], gallery_index: dict[str, int], per_art: int, seed: int, cfg: Degradation) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Deterministic degraded queries for the eval split.

    Returns (images NHWC uint8, gallery target indices, per-query difficulty infos).
    """
    rng = np.random.default_rng(seed)
    images, targets, infos = [], [], []
    for a in arts:
        img = load_rgb(art_path(a))
        for _ in range(per_art):
            q, info = degraded_view(img, rng, cfg)
            images.append(q)
            targets.append(gallery_index[a["id"]])
            infos.append(info)
    return np.stack(images), np.array(targets), infos


def cached_eval_queries(arts_all: list[dict], per_art: int = 3, seed: int = 2024, profile: str = "harsh"):
    """Gallery = every downloaded art (clean); queries = degraded eval-split arts. Cached on disk."""
    suffix = "" if profile == "harsh" else f"-{profile}"
    cache = DATA_DIR / f"eval-queries-{per_art}-{seed}{suffix}.npz"
    gallery_index = {a["id"]: i for i, a in enumerate(arts_all)}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["images"], z["targets"], list(z["infos"])
    images, targets, infos = build_eval_queries(split(arts_all, "eval"), gallery_index, per_art, seed, PROFILES[profile])
    np.savez(cache, images=images, targets=targets, infos=np.array(infos, dtype=object))
    return images, targets, infos
