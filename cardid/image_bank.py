"""Decoded image banks the scene renderer draws cards and table backgrounds from."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import ART_DIR, CACHE_DIR, CARD_DIR
from .degrade import load_rgb


def list_cards() -> list[Path]:
    cards = sorted(CARD_DIR.glob("*.jpg"))
    if not cards:
        raise SystemExit(f"no full-card images in {CARD_DIR}; run `python -m cardid.scryfall --cards 3000`")
    return cards


def list_arts() -> list[Path]:
    return sorted(ART_DIR.glob("*.jpg"))


CARD_SHAPE = (680, 488)  # Scryfall "normal" full-card JPEGs
ART_SHAPE = (457, 626)  # the common Scryfall art-crop size; odd crops are centre-cut to it
BG_ARTS = 1500  # backgrounds sample this many arts (a fixed seeded subset), not the whole gallery
TWO_PART_RATE = 0.10  # enough rotated-content examples to learn the outline, while 90% remain ordinary cards


class ImageBank:
    """The images the renderer draws from, decoded once at half resolution into a memory-mapped
    `.npy` under data/cache. JPEG decoding is entropy-bound (~5 ms per 110 KB card whatever the
    DCT-scaling flag), and a scene decodes three or four images, so it was a quarter of the
    render time; a memmap slice is a page-cache read shared by every DataLoader worker.
    Half resolution (244 px across a card) is enough for every card the renderer draws once
    the 640 px window is downscaled 2.5x for the detector; see `CardBank.load`."""

    def __init__(self, paths: list[Path], shape: tuple[int, int], name: str):
        self.paths = paths
        self.shape = (shape[0] // 2, shape[1] // 2)
        key = hashlib.sha1("\n".join(p.name for p in paths).encode()).hexdigest()[:10]
        self.path = CACHE_DIR / f"{name}-{self.shape[1]}x{self.shape[0]}-{len(paths)}-{key}.npy"
        self._mm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_mm": None}  # workers open the memmap themselves

    def __getitem__(self, i: int) -> np.ndarray:
        """Half-resolution RGB uint8 view (H/2, W/2, 3); do not write to it."""
        if self._mm is None:
            self.build()
            self._mm = np.load(self.path, mmap_mode="r")
        return self._mm[i]

    def build(self, threads: int = 8) -> None:
        """Decode every image into the cache file if it is not there yet."""
        if self.path.exists() or not self.paths:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp.npy")  # per process, in case workers race
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(self.paths), *self.shape, 3))
        h, w = self.shape

        def decode(i: int) -> None:
            img = cv2.imread(str(self.paths[i]), cv2.IMREAD_REDUCED_COLOR_2)
            ih, iw = img.shape[:2]
            if (ih, iw) != (h, w):  # centre-cut to the bank's aspect, then resize
                if iw / ih > w / h:
                    cut = round(ih * w / h)
                    img = img[:, (iw - cut) // 2 : (iw - cut) // 2 + cut]
                else:
                    cut = round(iw * h / w)
                    img = img[(ih - cut) // 2 : (ih - cut) // 2 + cut]
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            mm[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with ThreadPoolExecutor(threads) as pool:  # cv2 releases the GIL while decoding
            list(tqdm(pool.map(decode, range(len(self.paths))), total=len(self.paths), desc=f"caching {self.path.name}", leave=False))
        mm.flush()
        tmp.replace(self.path)


class CardBank(ImageBank):
    def __init__(self, paths: list[Path] | None = None):
        super().__init__(paths if paths is not None else list_cards(), CARD_SHAPE, "cards")
        manifests = {}
        for parent in {p.parent for p in self.paths}:
            path = parent / "two-part.json"
            manifests[parent] = json.loads(path.read_text()) if path.exists() else {}
        # Older --cards downloads can contain a second copy named <printing>-1.jpg.
        self.groups = np.array([manifests[p.parent].get(p.stem.removesuffix("-1"), "ordinary") for p in self.paths])
        self.two_part = {group: np.flatnonzero(self.groups == group) for group in ("room", "split", "aftermath", "flip") if np.any(self.groups == group)}
        self.ordinary = np.flatnonzero(self.groups == "ordinary")

    def sample_index(self, rng: np.random.Generator, two_part_rate: float = TWO_PART_RATE) -> int:
        """Balance layout groups, not printing counts (translations must not swamp flip).
        A bank with no manifest retains the old uniform draw, including its RNG sequence."""
        if not self.two_part:
            return int(rng.integers(len(self)))
        if not len(self.ordinary) or rng.random() < two_part_rate:
            group = list(self.two_part)[int(rng.integers(len(self.two_part)))]
            return int(rng.choice(self.two_part[group]))
        return int(rng.choice(self.ordinary))

    def load(self, i: int, short: float) -> np.ndarray:
        """The card at index `i` with at least `short` pixels across: the half-res bank copy, or
        the full JPEG when the card is drawn larger than that. `short` is measured in output
        pixels, so with the 2.5x scene downscale the bank covers every card size the renderer
        draws and no scene decodes a JPEG."""
        if short > self.shape[1]:
            return load_rgb(self.paths[i])
        return self[i]


class ArtBank(ImageBank):
    def __init__(self, paths: list[Path] | None = None, n: int = BG_ARTS, seed: int = 0):
        arts = paths if paths is not None else list_arts()
        if len(arts) > n:
            pick = np.sort(np.random.default_rng(seed).choice(len(arts), n, replace=False))
            arts = [arts[int(i)] for i in pick]
        super().__init__(arts, ART_SHAPE, "arts")
