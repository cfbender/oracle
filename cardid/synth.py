"""Synthetic table scenes for the corner detector, plus the real-capture equivalent.

A scene is what `capture.html` sends on a click: a SCENE x SCENE native-pixel window centred
on the click, with the clicked card somewhere under the centre. The target is the card's four
corners (its own edge, not the sleeve's) in printed order: top-left, top-right, bottom-right,
bottom-left of the card face, in scene pixels.

What the renderer varies, because each is a failure mode of the classical edge finder:
busy playmats (random art crops as background), sleeves (a ring outside the card plus a
glossy tint and glare over it), borderless cards (the card image cropped inside its border),
neighbouring and overlapping cards, dice and fingers, any rotation, mild perspective, and
webcam photometrics.

    uv run python -m cardid.synth --n 16 --out /tmp/scenes.png   # sheet with the targets drawn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import ART_DIR, CARD_DIR
from .data import to_tensor
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


def load_card(path: Path, short: float) -> np.ndarray:
    """Decode a full-card JPEG at half resolution when the rendered card is small enough that
    the extra pixels would only be averaged away (JPEG DCT scaling makes this ~3x faster)."""
    flag = cv2.IMREAD_REDUCED_COLOR_2 if short < 230 else cv2.IMREAD_COLOR
    return cv2.cvtColor(cv2.imread(str(path), flag), cv2.COLOR_BGR2RGB)


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
    Returns the warped alpha (canvas-sized) so callers can draw shadows and sleeve effects."""
    full_alpha = warp_alpha(alpha, quad, canvas.shape)
    roi = quad_roi(quad, canvas.shape)
    if roi is None:
        return full_alpha
    x0, y0, x1, y1 = roi
    h, w = img.shape[:2]
    H = cv2.getPerspectiveTransform(PRINTED * np.float32([w, h]), (quad - np.float32([x0, y0])).astype(np.float32))
    warped = cv2.warpPerspective(img, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    a = full_alpha[y0:y1, x0:x1, None]
    dst = canvas[y0:y1, x0:x1]
    dst[:] = warped.astype(np.float32) * a + dst * (1 - a)
    return full_alpha


def quad_short(quad: np.ndarray) -> float:
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return float(min(w, h))


def expand(quad: np.ndarray, factor: float) -> np.ndarray:
    c = quad.mean(axis=0)
    return ((quad - c) * factor + c).astype(np.float32)


def background(rng: np.random.Generator, arts: list[Path], size: int) -> np.ndarray:
    kind = rng.choice(["art", "flat", "gradient", "tiled"], p=[0.45, 0.35, 0.1, 0.1])
    if kind == "art" and arts:
        img = load_card(arts[int(rng.integers(len(arts)))], 0)  # half-res decode; it is blown up anyway
        # cover the canvas at 1-2.5x so the mat's artwork is at playmat scale, any rotation
        scale = rng.uniform(1.0, 2.5) * size / min(img.shape[:2])
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rng.uniform(0, 360), scale)
        M[:, 2] += np.float32([size / 2, size / 2]) - np.float32([img.shape[1] / 2, img.shape[0] / 2])
        M[:, 2] += rng.uniform(-0.3, 0.3, size=2) * size
        bg = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        sigma = rng.uniform(0, 2.0)
        if sigma > 0.3:
            bg = cv2.GaussianBlur(bg, (0, 0), sigma)
    elif kind == "tiled" and arts:
        img = load_rgb(arts[int(rng.integers(len(arts)))])
        tile = int(rng.integers(40, 160))
        t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        reps = size // tile + 2
        bg = np.tile(t, (reps, reps, 1))[:size, :size].astype(np.float32)
    elif kind == "gradient":
        c0, c1 = rng.uniform(0, 255, size=3), rng.uniform(0, 255, size=3)
        t = np.linspace(0, 1, size, dtype=np.float32)
        if rng.random() < 0.5:
            t = t[None, :, None]
        else:
            t = t[:, None, None]
        bg = c0 * (1 - t) + c1 * t
        bg = np.broadcast_to(bg, (size, size, 3)).copy()
    else:
        # desks and plain mats: white, grey, black, wood, felt green/blue/red
        palette = np.float32([[240, 240, 238], [200, 200, 195], [30, 30, 32], [120, 80, 45], [160, 120, 80], [40, 90, 50], [40, 50, 100], [110, 30, 30]])
        col = palette[int(rng.integers(len(palette)))] * rng.uniform(0.8, 1.1)
        bg = np.broadcast_to(col, (size, size, 3)).copy()
        bg += rng.normal(0, rng.uniform(1, 8), size=bg.shape).astype(np.float32)
    # uneven lighting across the table
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    gx, gy = rng.uniform(-0.25, 0.25, size=2)
    bg *= (1 + gx * (xx - 0.5) * 2 + gy * (yy - 0.5) * 2)[..., None]
    return np.clip(bg, 0, 255)


def card_face(rng: np.random.Generator, path: Path, short: float) -> tuple[np.ndarray, np.ndarray]:
    """A card image and its alpha. Sometimes cut inside the black border so the scene has
    borderless/extended-art cards even when the sample does not."""
    img = load_card(path, short)
    h, w = img.shape[:2]
    if rng.random() < 0.2:
        ix, iy = int(w * rng.uniform(0.035, 0.07)), int(h * rng.uniform(0.035, 0.07))
        img = img[iy : h - iy, ix : w - ix]
        h, w = img.shape[:2]
    return img, rounded_mask(w, h, CORNER_RADIUS * w)


def draw_card(canvas: np.ndarray, rng: np.random.Generator, path: Path, quad: np.ndarray, shadow: bool = True) -> np.ndarray:
    img, alpha = card_face(rng, path, quad_short(quad))
    if shadow and rng.random() < 0.7:
        # soft drop shadow: darken under a shifted, blurred copy of the card's alpha
        sh = warp_alpha(alpha, quad + rng.uniform(-6, 6, size=2).astype(np.float32), canvas.shape)
        sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(2, 6))
        canvas *= 1 - sh[..., None] * rng.uniform(0.15, 0.45)
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


def gloss(canvas: np.ndarray, rng: np.random.Generator, alpha: np.ndarray, quad: np.ndarray) -> None:
    """Sleeve/foil sheen over the card: a milky tint plus a highlight band or blob."""
    size = canvas.shape[0]
    tint = rng.uniform(0.03, 0.15)
    canvas[:] = canvas * (1 - alpha[..., None] * tint) + 235 * alpha[..., None] * tint
    if rng.random() < 0.7:
        # highlights are smooth, so compute them on a coarse grid and upsample
        coarse = size // 4
        yy, xx = np.mgrid[0:coarse, 0:coarse].astype(np.float32) * 4 + 2
        c = quad.mean(axis=0)
        s = quad_short(quad)
        if rng.random() < 0.5:  # band across the card, any direction
            a = rng.uniform(0, np.pi)
            d = (xx - c[0]) * np.cos(a) + (yy - c[1]) * np.sin(a) + rng.uniform(-0.5, 0.5) * s
            band = np.exp(-((d / (rng.uniform(0.06, 0.25) * s)) ** 2))
        else:
            gx, gy = c + rng.uniform(-0.5, 0.5, size=2) * s
            band = np.exp(-(((xx - gx) / (rng.uniform(0.2, 0.6) * s)) ** 2 + ((yy - gy) / (rng.uniform(0.2, 0.6) * s)) ** 2))
        band = cv2.resize(band, (size, size), interpolation=cv2.INTER_LINEAR)
        canvas[:] = canvas + (band * alpha)[..., None] * rng.uniform(40, 160)


def occluders(canvas: np.ndarray, rng: np.random.Generator, quad: np.ndarray, cards: list[Path]) -> None:
    c = quad.mean(axis=0)
    s = quad_short(quad)
    if rng.random() < 0.15:  # another card lying partly over this one (aura, equipment, a sloppy stack)
        off = rng.uniform(0.6, 1.1) * s
        a = rng.uniform(0, 2 * np.pi)
        q = quad_from_pose(c[0] + off * np.cos(a), c[1] + off * np.sin(a), s * rng.uniform(0.9, 1.1), rng.uniform(0, 360), rng)
        draw_card(canvas, rng, cards[int(rng.integers(len(cards)))], q)
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
    x = img.astype(np.float32) * rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    x = (x - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-30, 30)
    x = np.clip(x, 0, 255)
    x = 255 * (x / 255) ** rng.uniform(0.8, 1.25)
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
    x = x + rng.normal(0, rng.uniform(1, 8) * max(scale, 0.4), size=x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    quality = int(rng.integers(50, 95))
    _ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def render_scene(rng: np.random.Generator, cards: list[Path], arts: list[Path], size: int = SCENE, out: int = DET_INPUT) -> tuple[np.ndarray, np.ndarray]:
    """Compose a `size` x `size` native-pixel window and return it downscaled to `out` x `out`
    RGB uint8 with the 4x2 float32 printed-order corners of the clicked card in `out` pixels."""
    canvas = background(rng, arts, size)
    # short side of the clicked card: ~70 px (1080p over a 4-player table) to ~380 (4K, close)
    short = float(np.exp(rng.uniform(np.log(70), np.log(380))))
    angle = rng.uniform(0, 360)
    # neighbours underneath: other cards at a similar scale anywhere in the window
    for _ in range(int(rng.integers(0, 4))):
        q = quad_from_pose(rng.uniform(0, size), rng.uniform(0, size), short * rng.uniform(0.7, 1.3), rng.uniform(0, 360), rng)
        draw_card(canvas, rng, cards[int(rng.integers(len(cards)))], q)
    # pose the clicked card so a random point on its face sits at the window centre
    quad = quad_from_pose(0, 0, short, angle, rng)
    u, v = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
    click = (1 - v) * ((1 - u) * quad[0] + u * quad[1]) + v * ((1 - u) * quad[3] + u * quad[2])
    quad = quad - click + np.float32([size / 2, size / 2]) + rng.uniform(-12, 12, size=2).astype(np.float32)
    sleeved = rng.random() < 0.55
    ring_alpha = None
    if sleeved:
        _, ring_alpha = draw_sleeve_ring(canvas, rng, quad)
    card_alpha = draw_card(canvas, rng, cards[int(rng.integers(len(cards)))], quad, shadow=not sleeved)
    if sleeved:
        gloss(canvas, rng, np.maximum(card_alpha, ring_alpha), quad)
    elif rng.random() < 0.25:  # foil or a glossy unsleeved card
        gloss(canvas, rng, card_alpha, quad)
    occluders(canvas, rng, quad, cards)
    small = cv2.resize(np.clip(canvas, 0, 255).astype(np.uint8), (out, out), interpolation=cv2.INTER_AREA)
    return photometrics(small, rng, out / size), (quad * (out / size)).astype(np.float32)


def scene_to_input(scene: np.ndarray) -> torch.Tensor:
    if scene.shape[0] != DET_INPUT:
        scene = cv2.resize(scene, (DET_INPUT, DET_INPUT), interpolation=cv2.INTER_AREA)
    return to_tensor(scene)


class SceneDataset(Dataset):
    """`length` fresh scenes per epoch; deterministic in (seed, epoch, index). With `raw`,
    yields (uint8 scene, quad in px) instead of the normalised model input/target."""

    def __init__(self, length: int, cards: list[Path] | None = None, arts: list[Path] | None = None, seed: int = 0, raw: bool = False):
        self.length = length
        self.cards = cards or list_cards()
        self.arts = arts if arts is not None else list_arts()
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
        if self.raw:
            return torch.from_numpy(scene), torch.from_numpy(quad)
        return scene_to_input(scene), torch.from_numpy(quad / DET_INPUT)


def window_around(img: np.ndarray, cx: float, cy: float, side: float, out: int = SCENE) -> tuple[np.ndarray, np.ndarray]:
    """Resample the `side` x `side` square centred on (cx, cy) to an `out` x `out` image.
    Returns (window, M) where M is the 2x3 affine from image to window coordinates."""
    s = out / side
    M = np.float32([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy]])
    win = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return win, M


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (np.c_[pts, np.ones(len(pts))] @ M.T).astype(np.float32)


class RealSceneDataset(Dataset):
    """Labeled real captures (`data/real/<id>/crop.jpg` + the quad the identification used) as
    detector samples, re-windowed around the click like `render_scene`. With `augment`, the
    window is randomly rotated, scaled, and shifted so a few hundred captures go further."""

    def __init__(self, rows: list[dict], repeat: int = 1, augment: bool = True, seed: int = 1):
        from .real import REAL_DIR

        self.rows = [r for r in rows if r.get("quad") and (REAL_DIR / r["capture_id"] / "crop.jpg").exists()]
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
        return scene_to_input(scene), torch.from_numpy(quad / DET_INPUT)


def sheet(n: int, seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    cards, arts = list_cards(), list_arts()
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
