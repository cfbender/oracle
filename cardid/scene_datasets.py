"""Detector training samples: rendered scenes and re-windowed real captures, as tensors."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DET_INPUT, SCENE
from .data import IMAGENET_MEAN, IMAGENET_STD, to_tensor
from .degrade import load_rgb
from .image_bank import ArtBank, CardBank
from .scene_geometry import apply_affine, window_around
from .scene_renderer import photometrics, render_scene


def scene_to_input(scene: np.ndarray) -> torch.Tensor:
    """One HWC uint8 scene -> normalised CHW float tensor (inference path)."""
    if scene.shape[0] != DET_INPUT:
        scene = cv2.resize(scene, (DET_INPUT, DET_INPUT), interpolation=cv2.INTER_AREA)
    return to_tensor(scene)


def batch_to_input(scenes: torch.Tensor) -> torch.Tensor:
    """NHWC uint8 batch (any device) -> normalised NCHW float batch on the same device. The
    datasets ship uint8 so each sample crosses the worker queue, collation and pinned memory
    at a quarter of the float size; the arithmetic is cheaper on the GPU than the transfer."""
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=torch.float32, device=scenes.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=torch.float32, device=scenes.device).view(1, 3, 1, 1)
    x = scenes.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - mean) / std


class SceneDataset(Dataset):
    """`length` fresh scenes per epoch; deterministic in (seed, epoch, index). Yields the HWC
    uint8 scene (normalise batches with `batch_to_input`), the quad in [0, 1] window units
    (or in pixels with `raw`) in printed order, and `up_valid` = True: a rendered scene knows
    which way its card is printed, so the detector's up output can learn from it."""

    def __init__(
        self,
        length: int,
        cards: CardBank | None = None,
        arts: ArtBank | None = None,
        seed: int = 0,
        raw: bool = False,
        target_indices: np.ndarray | None = None,
    ):
        self.length = length
        self.cards = cards or CardBank()
        self.arts = arts if arts is not None else ArtBank()
        self.cards.build()  # in the parent, so workers find the cache instead of each building it
        self.arts.build()
        self.seed = seed
        self.raw = raw
        self.epoch = 0
        self.target_indices = target_indices

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        target = int(self.target_indices[i]) if self.target_indices is not None else None
        scene, quad = render_scene(rng, self.cards, self.arts, target_index=target)
        return torch.from_numpy(scene), torch.from_numpy(quad if self.raw else quad / DET_INPUT), torch.tensor(True)


def trusted_quad(row: dict) -> bool:
    """Whether a labeled capture's quad is tight enough to supervise the detector."""
    top5 = row.get("top5") or []
    return row.get("quad_source") == "manual" or (bool(top5) and top5[0] == row.get("label"))


class RealSceneDataset(Dataset):
    """Labeled real captures (`data/real/<id>/crop.jpg` + the quad the identification used) as
    detector samples, re-windowed around the click like `render_scene`. With `augment`, the
    window is randomly rotated, scaled, and shifted so a few hundred captures go further.

    Only quads worth learning from are kept: ones the user drew by hand, or automatic ones the
    recogniser confirmed with a top-1 hit (a loose quad that still identified the card is
    fine for the embedder but would teach the detector to be loose). Stored quads are only
    cyclically ordered, not printed-ordered (the card may have been identified upside down),
    so samples carry `up_valid` = False and do not train the up output."""

    def __init__(self, rows: list[dict], repeat: int = 1, augment: bool = True, seed: int = 1):
        from .real import REAL_DIR

        self.rows = [r for r in rows if r.get("quad") and trusted_quad(r) and (REAL_DIR / r["capture_id"] / "crop.jpg").exists()]
        self.dir = REAL_DIR
        self.repeat = repeat
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows) * self.repeat

    def sample(self, i: int, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray]:
        """(DET_INPUT x DET_INPUT RGB uint8, quad in those pixels)."""
        r = self.rows[i % len(self.rows)]
        img = load_rgb(self.dir / r["capture_id"] / "crop.jpg")
        quad = np.float32(r["quad"])
        cx, cy = r.get("click") or (img.shape[1] / 2, img.shape[0] / 2)
        side, angle = float(SCENE), 0.0
        if rng is not None:
            side *= rng.uniform(0.8, 1.25)
            angle = rng.uniform(0, 360)
            cx, cy = cx + rng.uniform(-20, 20), cy + rng.uniform(-20, 20)
        win, M = window_around(img, cx, cy, side, DET_INPUT)
        if angle:
            R = cv2.getRotationMatrix2D((DET_INPUT / 2, DET_INPUT / 2), angle, 1.0)
            win = cv2.warpAffine(win, R, (DET_INPUT, DET_INPUT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            M = np.float32(R @ np.vstack([M, [0, 0, 1]]))
        if rng is not None:
            win = photometrics(win, rng, DET_INPUT / side)
        return win, apply_affine(M, quad)

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.augment else None
        scene, quad = self.sample(i, rng)
        return torch.from_numpy(np.ascontiguousarray(scene)), torch.from_numpy(quad / DET_INPUT), torch.tensor(False)
