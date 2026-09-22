"""Synthetic table scenes for the corner detector, plus the real-capture equivalent.

A scene is what `capture.html` sends on a click: a SCENE x SCENE native-pixel window centred
on the click, with the clicked card somewhere under the centre. The target is the card's four
corners (its own edge, not the sleeve's) in printed order: top-left, top-right, bottom-right,
bottom-left of the card face, in scene pixels.

What the renderer varies, because each is a failure mode of the classical edge finder:
busy playmats (random art crops as background), sleeves (a ring outside the card plus a
glossy tint and glare over it), rigid top-loaders (a larger clear rectangle with a specular
edge), borderless cards (the card image cropped inside its border),
neighbouring and overlapping cards, dice and fingers, any rotation, mild perspective, and
webcam photometrics.

    uv run python -m cardid.synth --n 16 --out /tmp/scenes.png   # sheet with the targets drawn
"""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from . import ART_DIR, CACHE_DIR, CARD_DIR
from .data import IMAGENET_MEAN, IMAGENET_STD, to_tensor
from .degrade import load_rgb

SCENE = 640  # native px around the click (capture.html CROP)
DET_INPUT = 256  # detector input; the scene is downscaled to this
CORNER_RADIUS = 0.045  # card corner radius as a fraction of the short side (3 mm on 63 mm)
PRINTED = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])  # unit square in printed order


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


def rounded_mask(w: int, h: int, radius: float) -> np.ndarray:
    """float32 HxW alpha of a rounded rectangle filling the image."""
    m = np.zeros((h, w), np.float32)
    r = round(radius)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 1.0, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 1.0, -1)
    for cx, cy in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
        cv2.circle(m, (cx, cy), r, 1.0, -1)
    return m


def quad_from_pose(cx: float, cy: float, short: float, angle_deg: float, rng: np.random.Generator | None) -> np.ndarray:
    """Printed-order corners of a 63x88 card with the given centre, short side and rotation,
    optionally perturbed by a mild perspective (camera not exactly overhead)."""
    w, h = short, short * 88 / 63
    box = (PRINTED - 0.5) * np.float32([w, h])
    if rng is not None:
        # trapezoid: one pair of opposite edges shorter than the other (tilt), plus jitter
        tilt = rng.uniform(0.88, 1.0)
        if rng.random() < 0.5:
            box[[0, 1], 0] *= tilt  # top edge shorter
        else:
            box[[0, 3], 1] *= tilt  # left edge shorter
        box += rng.uniform(-0.03, 0.03, size=(4, 2)).astype(np.float32) * short
    a = np.deg2rad(angle_deg)
    rot = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (box @ rot.T + np.float32([cx, cy])).astype(np.float32)


def quad_roi(quad: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """Bounding box of `quad` clipped to the canvas (x0, y0, x1, y1), or None if outside.
    Warps only touch this box; a small card on a big canvas is the common case."""
    x0, y0 = np.floor(quad.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, shape[1]), min(y1, shape[0])
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def warp_alpha(alpha: np.ndarray, quad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Canvas-sized float alpha of `alpha` (an image-shaped mask) warped onto `quad`."""
    out = np.zeros(shape[:2], np.float32)
    roi = quad_roi(quad, shape)
    if roi is None:
        return out
    x0, y0, x1, y1 = roi
    h, w = alpha.shape[:2]
    H = cv2.getPerspectiveTransform(PRINTED * np.float32([w, h]), (quad - np.float32([x0, y0])).astype(np.float32))
    out[y0:y1, x0:x1] = cv2.warpPerspective(alpha, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    return out


def paste(canvas: np.ndarray, img: np.ndarray, alpha: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp `img` (with float alpha) so its corners land on `quad` and blend onto the canvas.
    Returns the warped alpha (canvas-sized) so callers can draw shadows and sleeve effects.
    The blend is `dst += (src - dst) * a` in place over the quad's bounding box: three passes
    over the box instead of the four temporaries of `src * a + dst * (1 - a)`."""
    full_alpha = warp_alpha(alpha, quad, canvas.shape)
    roi = quad_roi(quad, canvas.shape)
    if roi is None:
        return full_alpha
    x0, y0, x1, y1 = roi
    h, w = img.shape[:2]
    H = cv2.getPerspectiveTransform(PRINTED * np.float32([w, h]), (quad - np.float32([x0, y0])).astype(np.float32))
    # converting the (small) source to float before the warp is cheaper than converting the warp
    src = cv2.warpPerspective(np.asarray(img, dtype=np.float32), H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    a = full_alpha[y0:y1, x0:x1, None]
    dst = canvas[y0:y1, x0:x1]
    src -= dst
    src *= a
    dst += src
    return full_alpha


def quad_short(quad: np.ndarray) -> float:
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return float(min(w, h))


def expand(quad: np.ndarray, factor: float) -> np.ndarray:
    c = quad.mean(axis=0)
    return ((quad - c) * factor + c).astype(np.float32)


def background(rng: np.random.Generator, arts: ArtBank, size: int) -> np.ndarray:
    """Float32 `size` x `size` x 3 table surface. Full-canvas passes go through cv2 (SIMD, no
    broadcasting temporaries); numpy's broadcast fills and `[..., None]` multiplies cost more
    here than the warp itself."""
    kind = rng.choice(["art", "flat", "gradient", "tiled"], p=[0.45, 0.35, 0.1, 0.1])
    if kind == "art" and arts:
        img = np.asarray(arts[int(rng.integers(len(arts)))], dtype=np.float32)  # half res; blown up anyway
        # cover the canvas at 1-2.5x so the mat's artwork is at playmat scale, any rotation
        scale = rng.uniform(1.0, 2.5) * size / min(img.shape[:2])
        # a slightly out-of-focus mat: blur the small source by sigma/scale, which is what a
        # blur of sigma on the upscaled canvas looks like, at a fraction of the pixels
        sigma = rng.uniform(0, 2.0) / scale
        if sigma > 0.12:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rng.uniform(0, 360), scale)
        M[:, 2] += np.float32([size / 2, size / 2]) - np.float32([img.shape[1] / 2, img.shape[0] / 2])
        M[:, 2] += rng.uniform(-0.3, 0.3, size=2) * size
        bg = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif kind == "tiled" and arts:
        img = arts[int(rng.integers(len(arts)))]
        tile = int(rng.integers(40, 160))
        t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        reps = size // tile + 2
        bg = np.tile(t, (reps, reps, 1))[:size, :size].astype(np.float32)
    elif kind == "gradient":
        c0, c1 = rng.uniform(0, 255, size=3).astype(np.float32), rng.uniform(0, 255, size=3).astype(np.float32)
        t = np.linspace(0, 1, size, dtype=np.float32)[:, None]
        strip = c0 * (1 - t) + c1 * t  # (size, 3): one colour per row
        horizontal = rng.random() < 0.5
        bg = cv2.resize(strip[None] if horizontal else strip[:, None], (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        # desks and plain mats: white, grey, black, wood, felt green/blue/red
        palette = np.float32([[240, 240, 238], [200, 200, 195], [30, 30, 32], [120, 80, 45], [160, 120, 80], [40, 90, 50], [40, 50, 100], [110, 30, 30]])
        col = palette[int(rng.integers(len(palette)))] * rng.uniform(0.8, 1.1)
        # monochrome grain (paper, felt, wood texture) at half resolution: the scene is
        # downscaled 2.5x for the detector anyway, and colour noise is the sensor's job.
        # Uniform noise with the same std as the old Gaussian; drawing it is 6x cheaper.
        grain = (rng.random(size=(size // 2, size // 2), dtype=np.float32) - 0.5) * np.float32(rng.uniform(1, 8) * np.sqrt(12))
        grain = cv2.resize(grain, (size, size), interpolation=cv2.INTER_LINEAR)
        bg = cv2.merge([grain, grain, grain])
        cv2.add(bg, (*(float(c) for c in col), 0.0), dst=bg)
    # uneven lighting across the table: a linear ramp, so it separates into a row and a column
    # vector (no full-size meshgrid)
    ramp = (np.arange(size, dtype=np.float32) / size - 0.5) * 2
    gx, gy = rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
    light = 1 + gx * ramp[None, :] + gy * ramp[:, None]
    cv2.multiply(bg, cv2.merge([light, light, light]), dst=bg)
    return np.clip(bg, 0, 255, out=bg)


def card_face(rng: np.random.Generator, cards: CardBank, index: int, short: float, detail: float = DET_INPUT / SCENE) -> tuple[np.ndarray, np.ndarray]:
    """A card image and its alpha. Sometimes cut inside the black border so the scene has
    borderless/extended-art cards even when the sample does not. `short` is the drawn size in
    canvas pixels and `detail` the canvas-to-output scale."""
    img = cards.load(index, short * detail)
    h, w = img.shape[:2]
    if rng.random() < 0.2:
        ix, iy = int(w * rng.uniform(0.035, 0.07)), int(h * rng.uniform(0.035, 0.07))
        img = img[iy : h - iy, ix : w - ix]
        h, w = img.shape[:2]
    return img, rounded_mask(w, h, CORNER_RADIUS * w)


def draw_card(
    canvas: np.ndarray, rng: np.random.Generator, cards: CardBank, quad: np.ndarray, shadow: bool = True, detail: float = DET_INPUT / SCENE
) -> np.ndarray:
    """Draw a random card from the bank on `quad`; returns its canvas-sized alpha."""
    if quad_roi(quad, canvas.shape) is None:  # entirely outside the window: nothing to decode
        return np.zeros(canvas.shape[:2], np.float32)
    img, alpha = card_face(rng, cards, int(rng.integers(len(cards))), quad_short(quad), detail)
    if shadow and rng.random() < 0.7:
        # soft drop shadow: darken under a shifted, blurred copy of the card's alpha
        sh_quad = quad + rng.uniform(-6, 6, size=2).astype(np.float32)
        sigma = rng.uniform(2, 6)
        roi = quad_roi(expand(sh_quad, 1 + 4 * sigma / quad_short(sh_quad)), canvas.shape)  # room for the blur tail
        if roi is not None:
            x0, y0, x1, y1 = roi
            sh = warp_alpha(alpha, sh_quad, canvas.shape)[y0:y1, x0:x1]
            sh = cv2.GaussianBlur(sh, (0, 0), sigma)
            sh *= -rng.uniform(0.15, 0.45)
            sh += 1
            canvas[y0:y1, x0:x1] *= sh[..., None]
    return paste(canvas, img, alpha, quad)


def draw_sleeve_ring(canvas: np.ndarray, rng: np.random.Generator, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The part of a sleeve that shows outside the card: a slightly larger rounded card shape.
    Returns (ring quad, ring alpha) so the gloss can be applied over card and ring together."""
    ring = expand(quad, rng.uniform(1.025, 1.07))
    # sleeves are open at the top, so the card often sits low: shift the ring up a little
    ring[[0, 1]] += (ring[0] - ring[3]) / np.linalg.norm(ring[0] - ring[3]) * rng.uniform(0, 0.02) * quad_short(quad)
    dark = rng.random() < 0.6
    color = rng.uniform(5, 40, size=3) if dark else rng.uniform(30, 255, size=3)
    if rng.random() < 0.25:  # clear/frosted sleeve: ring is a translucent haze
        color = np.float32([230, 230, 235])
        opacity = rng.uniform(0.25, 0.6)
    else:
        opacity = 1.0
    unit = 200
    solid = np.empty((int(unit * 88 / 63), unit, 3), np.float32)
    solid[:] = color
    alpha = rounded_mask(solid.shape[1], solid.shape[0], CORNER_RADIUS * unit * 1.3) * opacity
    ring_alpha = paste(canvas, solid, alpha, ring)
    return ring, ring_alpha


def draw_toploader(canvas: np.ndarray, rng: np.random.Generator, quad: np.ndarray) -> np.ndarray:
    """A rigid top-loader (or semi-rigid card saver) around the card: a clear rectangle about
    a fifth larger than the card with square corners, a specular edge line where the plastic
    catches the light, a slight haze, and the card sitting low inside because the loader is
    open at the top. Its corners are a card-shaped rectangle a few percent of the short side
    outside the real ones, exactly where the corner heatmap would otherwise fire.
    Returns the loader alpha (canvas-sized) for the gloss pass."""
    short = quad_short(quad)
    up = (quad[0] - quad[3]) / np.linalg.norm(quad[0] - quad[3])
    sx, sy = rng.uniform(1.12, 1.26), rng.uniform(1.08, 1.2)  # 76x102 mm around 63x88, with some slop
    c = quad.mean(axis=0)
    right = (quad[1] - quad[0]) / np.linalg.norm(quad[1] - quad[0])
    hw, hh = short * sx / 2, short * 88 / 63 * sy / 2
    # the card rests on the loader's bottom edge: shift the loader up by most of the spare height
    c = c + up * rng.uniform(0.3, 1.0) * (hh - short * 88 / 63 / 2)
    loader = np.float32([c - right * hw + up * hh, c + right * hw + up * hh, c + right * hw - up * hh, c - right * hw - up * hh])
    unit = 200
    haze = np.empty((int(unit * hh / hw), unit, 3), np.float32)
    haze[:] = rng.uniform(200, 245, size=3)
    alpha = np.full(haze.shape[:2], rng.uniform(0.04, 0.22), np.float32)
    loader_alpha = paste(canvas, haze, alpha, loader)
    roi = quad_roi(expand(loader, 1.05), canvas.shape)
    if roi is not None:
        # specular edge: a thin bright (or dark, when it shadows the mat) line along the plastic's edge
        bright = rng.random() < 0.7
        color = tuple(float(v) for v in (rng.uniform(190, 255, size=3) if bright else rng.uniform(10, 70, size=3)))
        thickness = max(1, round(rng.uniform(0.008, 0.02) * short))
        x0, y0, x1, y1 = roi
        line = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.polylines(line, [np.round(loader - np.float32([x0, y0])).astype(np.int32)], True, 1.0, thickness, cv2.LINE_AA)
        a = (line * rng.uniform(0.5, 1.0))[..., None]
        dst = canvas[y0:y1, x0:x1]
        dst += (np.float32(color) - dst) * a
        np.maximum(loader_alpha[y0:y1, x0:x1], line, out=loader_alpha[y0:y1, x0:x1])
    return loader_alpha


def gloss(canvas: np.ndarray, rng: np.random.Generator, alpha: np.ndarray, quad: np.ndarray) -> None:
    """Sleeve/foil sheen over the card: a milky tint plus a highlight band or blob."""
    # `alpha` is zero outside the sleeve ring or top-loader, which sit within ~1.3x the card; work in that box
    roi = quad_roi(expand(quad, 1.4), canvas.shape)
    if roi is None:
        return
    x0, y0, x1, y1 = roi
    a = alpha[y0:y1, x0:x1]
    dst = canvas[y0:y1, x0:x1]
    tint = rng.uniform(0.03, 0.15)
    dst += (235 - dst) * (a * tint)[..., None]
    if rng.random() < 0.7:
        # highlights are smooth, so compute them on a coarse grid over the box and upsample
        step = 4
        xs = np.arange(x0, x1, step, dtype=np.float32) + step / 2
        ys = np.arange(y0, y1, step, dtype=np.float32) + step / 2
        c = quad.mean(axis=0)
        s = quad_short(quad)
        if rng.random() < 0.5:  # band across the card, any direction
            ang = rng.uniform(0, np.pi)
            d = (xs - c[0])[None, :] * np.cos(ang) + (ys - c[1])[:, None] * np.sin(ang) + rng.uniform(-0.5, 0.5) * s
            band = np.exp(-((d / (rng.uniform(0.06, 0.25) * s)) ** 2))
        else:
            gx, gy = c + rng.uniform(-0.5, 0.5, size=2) * s
            band = np.exp(-(((xs - gx)[None, :] / (rng.uniform(0.2, 0.6) * s)) ** 2 + ((ys - gy)[:, None] / (rng.uniform(0.2, 0.6) * s)) ** 2))
        band = cv2.resize(band.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        dst += (band * a * rng.uniform(40, 160))[..., None]


def occluders(canvas: np.ndarray, rng: np.random.Generator, quad: np.ndarray, cards: CardBank, detail: float = DET_INPUT / SCENE) -> None:
    c = quad.mean(axis=0)
    s = quad_short(quad)
    if rng.random() < 0.15:  # another card lying partly over this one (aura, equipment, a sloppy stack)
        off = rng.uniform(0.6, 1.1) * s
        a = rng.uniform(0, 2 * np.pi)
        q = quad_from_pose(c[0] + off * np.cos(a), c[1] + off * np.sin(a), s * rng.uniform(0.9, 1.1), rng.uniform(0, 360), rng)
        draw_card(canvas, rng, cards, q, detail=detail)
    if rng.random() < 0.3:  # dice and counters
        for _ in range(int(rng.integers(1, 4))):
            r = int(rng.uniform(0.05, 0.13) * s)
            p = c + rng.uniform(-0.9, 0.9, size=2) * s
            color = tuple(float(v) for v in rng.uniform(0, 255, size=3))
            cv2.circle(canvas, (int(p[0]), int(p[1])), r, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(p[0]), int(p[1])), r, tuple(v * 0.6 for v in color), 2, lineType=cv2.LINE_AA)
    if rng.random() < 0.15:  # a finger reaching in from an edge
        size = canvas.shape[0]
        edge = rng.integers(0, 4)
        length = rng.uniform(0.3, 0.7) * size
        width = rng.uniform(0.06, 0.12) * size
        start = {0: (rng.uniform(0, size), 0), 1: (size, rng.uniform(0, size)), 2: (rng.uniform(0, size), size), 3: (0, rng.uniform(0, size))}[int(edge)]
        toward = c - np.float32(start)
        toward = toward / (np.linalg.norm(toward) + 1e-6)
        end = np.float32(start) + toward * length
        skin = np.float32([rng.uniform(150, 240), rng.uniform(100, 180), rng.uniform(80, 150)])
        cv2.line(canvas, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), tuple(float(v) for v in skin), int(width), lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(end[0]), int(end[1])), int(width / 2), tuple(float(v) for v in skin * 0.95), -1, lineType=cv2.LINE_AA)


def photometrics(img: np.ndarray, rng: np.random.Generator, scale: float = DET_INPUT / SCENE) -> np.ndarray:
    """Webcam look: exposure, white balance, gamma, saturation, defocus, sensor noise, and
    the stream's compression. Applied at detector-input resolution, so blur and noise are
    scaled by `scale` (native px -> input px) from what a 1080p sensor produces. Returns uint8."""
    # exposure/white balance/contrast/brightness/gamma are per-value maps, so apply them to the
    # 256-entry channel LUTs rather than to every pixel
    levels = np.arange(256, dtype=np.float32)[:, None] * rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    levels = (levels - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-30, 30)
    levels = (255 * (np.clip(levels, 0, 255) / 255) ** rng.uniform(0.8, 1.25)).astype(np.float32)
    x = np.stack([levels[:, ch][img[..., ch]] for ch in range(3)], axis=2)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.7, 1.2)
    sigma = rng.uniform(0, 1.6) * scale
    if sigma > 0.15:
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    if rng.random() < 0.1:
        k = int(rng.integers(3, 6))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0), (k, k))
        x = cv2.filter2D(x, -1, kernel / max(kernel.sum(), 1e-6))
    # averaging 1/scale^2 sensor pixels per input pixel shrinks the noise by `scale`
    x += rng.standard_normal(size=x.shape, dtype=np.float32) * np.float32(rng.uniform(1, 8) * max(scale, 0.4))
    x = np.clip(x, 0, 255, out=x).astype(np.uint8)
    quality = int(rng.integers(50, 95))
    _ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def render_scene(rng: np.random.Generator, cards: CardBank, arts: ArtBank, size: int = SCENE, out: int = DET_INPUT) -> tuple[np.ndarray, np.ndarray]:
    """Compose a `size` x `size` native-pixel window and return it downscaled to `out` x `out`
    RGB uint8 with the 4x2 float32 printed-order corners of the clicked card in `out` pixels."""
    canvas = background(rng, arts, size)
    # short side of the clicked card: ~70 px (1080p over a 4-player table) to ~380 (4K, close)
    short = float(np.exp(rng.uniform(np.log(70), np.log(380))))
    angle = rng.uniform(0, 360)
    # neighbours underneath: other cards at a similar scale anywhere in the window
    for _ in range(int(rng.integers(0, 4))):
        q = quad_from_pose(rng.uniform(0, size), rng.uniform(0, size), short * rng.uniform(0.7, 1.3), rng.uniform(0, 360), rng)
        draw_card(canvas, rng, cards, q, detail=out / size)
    # pose the clicked card so a random point on its face sits at the window centre
    quad = quad_from_pose(0, 0, short, angle, rng)
    u, v = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
    click = (1 - v) * ((1 - u) * quad[0] + u * quad[1]) + v * ((1 - u) * quad[3] + u * quad[2])
    quad = quad - click + np.float32([size / 2, size / 2]) + rng.uniform(-12, 12, size=2).astype(np.float32)
    sleeved = rng.random() < 0.55
    loader_alpha = draw_toploader(canvas, rng, quad) if rng.random() < 0.15 else None
    ring_alpha = None
    if sleeved:
        _, ring_alpha = draw_sleeve_ring(canvas, rng, quad)
    card_alpha = draw_card(canvas, rng, cards, quad, shadow=not sleeved, detail=out / size)
    if loader_alpha is not None:
        # the loader's plastic catches the light over card, sleeve and its own margin alike
        alpha = np.maximum(card_alpha, loader_alpha) if ring_alpha is None else np.maximum(np.maximum(card_alpha, ring_alpha), loader_alpha)
        gloss(canvas, rng, alpha, quad)
    elif sleeved:
        gloss(canvas, rng, np.maximum(card_alpha, ring_alpha), quad)
    elif rng.random() < 0.25:  # foil or a glossy unsleeved card
        gloss(canvas, rng, card_alpha, quad)
    occluders(canvas, rng, quad, cards, detail=out / size)
    small = cv2.resize(np.clip(canvas, 0, 255, out=canvas).astype(np.uint8), (out, out), interpolation=cv2.INTER_AREA)
    return photometrics(small, rng, out / size), (quad * (out / size)).astype(np.float32)


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

    def __init__(self, length: int, cards: CardBank | None = None, arts: ArtBank | None = None, seed: int = 0, raw: bool = False):
        self.length = length
        self.cards = cards or CardBank()
        self.arts = arts if arts is not None else ArtBank()
        self.cards.build()  # in the parent, so workers find the cache instead of each building it
        self.arts.build()
        self.seed = seed
        self.raw = raw
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        scene, quad = render_scene(rng, self.cards, self.arts)
        return torch.from_numpy(scene), torch.from_numpy(quad if self.raw else quad / DET_INPUT), torch.tensor(True)


def window_around(img: np.ndarray, cx: float, cy: float, side: float, out: int = SCENE) -> tuple[np.ndarray, np.ndarray]:
    """Resample the `side` x `side` square centred on (cx, cy) to an `out` x `out` image.
    Returns (window, M) where M is the 2x3 affine from image to window coordinates."""
    s = out / side
    M = np.float32([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy]])
    win = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return win, M


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (np.c_[pts, np.ones(len(pts))] @ M.T).astype(np.float32)


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


def sheet(n: int, seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    cards, arts = CardBank(), ArtBank()
    tiles = []
    for _ in range(n):
        scene, quad = render_scene(rng, cards, arts)
        vis = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 1)
        cv2.circle(vis, tuple(quad[0].astype(int)), 3, (0, 0, 255), -1)  # printed top-left
        tiles.append(vis)
    cols = 4
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles) - len(tiles) % cols, cols)]
    cv2.imwrite(str(out), np.vstack(rows))
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/scenes.png"))
    args = parser.parse_args()
    sheet(args.n, args.seed, args.out)
