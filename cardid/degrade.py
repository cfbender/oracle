"""Synthetic webcam degradation of a clean Scryfall art crop.

The model input is what the recognizer sees *after* the detector has found a card quad,
perspective-warped it to a canonical card, and cut out the fixed art box. So a degraded
sample simulates: detector misalignment, residual perspective, the camera's low pixel
density on the art (60-140 px wide on a 1080p table shot), optics blur, lighting and white
balance, glare off sleeves, sensor noise, video compression, and partial occlusion by
dice/counters/overlapping cards.

    uv run python -m cardid.degrade <art_id>   # writes data/preview-<id>.png with 12 samples
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

INPUT_SIZE = 128  # model input (square); aspect is squashed identically for gallery and query


@dataclass
class Degradation:
    """Knobs so evaluation can bucket by difficulty. Widths are the simulated art width in px."""

    min_width: int = 56
    max_width: int = 140
    strong_perspective_p: float = 0.25
    occlusion_p: float = 0.25
    glare_p: float = 0.4
    motion_blur_p: float = 0.2


# Training uses the harsh defaults above for robustness. Evaluation also reports this profile,
# which assumes the detector returned a quad that was perspective-warped, so only crop jitter
# and mild residual perspective remain, and the art is at least ~70 px wide (a 1080p camera
# over a 4-player table; smaller than that and the card is unreadable to a human too).
PROFILES = {
    "harsh": Degradation(),
    "realistic": Degradation(min_width=70, max_width=140, strong_perspective_p=0.0, occlusion_p=0.15, glare_p=0.4, motion_blur_p=0.1),
}


def clean_view(img: np.ndarray) -> np.ndarray:
    """Gallery view: the art crop resized straight to model input."""
    return cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)


def degraded_view(img: np.ndarray, rng: np.random.Generator, cfg: Degradation = Degradation()) -> tuple[np.ndarray, dict]:
    """Return (128x128 RGB uint8, info) where info records the sampled difficulty knobs."""
    h, w = img.shape[:2]
    width = int(rng.integers(cfg.min_width, cfg.max_width + 1))
    height = max(8, int(round(width * h / w)))

    # 1. Geometry: crop jitter (detector error) + residual perspective + slight rotation,
    #    rendered directly at the low capture resolution so downscaling is part of the warp.
    strong = rng.random() < cfg.strong_perspective_p
    persp = 0.18 if strong else 0.05
    scale = rng.uniform(0.90, 1.12)
    shift = rng.uniform(-0.06, 0.06, size=2)
    angle = rng.uniform(-4, 4)
    # A sensor integrates over its pixel area, so downscale with INTER_AREA to ~1.5x the target
    # first; warpPerspective only offers point-sampling interpolation and would alias.
    pre_w, pre_h = int(width * 1.5), max(8, int(height * 1.5))
    pre = cv2.resize(img, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
    src = np.float32([[0, 0], [pre_w, 0], [pre_w, pre_h], [0, pre_h]])
    # destination box in low-res space, then perturbed
    cx, cy = width / 2 + shift[0] * width, height / 2 + shift[1] * height
    hw, hh = width * scale / 2, height * scale / 2
    dst = np.float32([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
    dst += rng.uniform(-persp, persp, size=(4, 2)).astype(np.float32) * np.float32([width, height])
    rot = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    dst = (rot @ np.c_[dst, np.ones(4)].T).T.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    # border: what leaks in when the crop overshoots the art box is card frame, so a dark
    # or parchment-ish flat colour rather than a mirrored art edge.
    frames = [(20, 20, 20), (30, 28, 25), (200, 190, 165), (60, 60, 70), (235, 230, 220)]
    frame = tuple(float(c) for c in frames[int(rng.integers(0, len(frames)))])
    low = cv2.warpPerspective(pre, H, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=frame)
    low = low.astype(np.float32)

    # 2. Photometrics: exposure, contrast, white balance, gamma, saturation.
    gain = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    low = low * gain
    low = (low - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-35, 35)
    low = np.clip(low, 0, 255)
    low = 255 * (low / 255) ** rng.uniform(0.8, 1.25)
    gray = low.mean(axis=2, keepdims=True)
    low = gray + (low - gray) * rng.uniform(0.65, 1.2)

    # 3. Glare: soft elliptical highlight or a sleeve band.
    if rng.random() < cfg.glare_p:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        gx, gy = rng.uniform(0, width), rng.uniform(0, height)
        sx, sy = rng.uniform(0.15, 0.6) * width, rng.uniform(0.1, 0.6) * height
        blob = np.exp(-(((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2))
        low = low + blob[..., None] * rng.uniform(40, 140)

    # 4. Occlusion: a die/counter (circle) or an overlapping card edge (rectangle from a side).
    if rng.random() < cfg.occlusion_p:
        color = tuple(float(c) for c in rng.integers(0, 255, size=3))
        if rng.random() < 0.5:
            r = int(rng.uniform(0.08, 0.18) * width)
            cv2.circle(low, (int(rng.uniform(0, width)), int(rng.uniform(0, height))), r, color, -1)
        else:
            frac = rng.uniform(0.08, 0.22)
            side = rng.integers(0, 4)
            if side == 0:
                low[:, : int(width * frac)] = color
            elif side == 1:
                low[:, width - int(width * frac) :] = color
            elif side == 2:
                low[: int(height * frac), :] = color
            else:
                low[height - int(height * frac) :, :] = color

    # 5. Optics: defocus and occasionally motion blur.
    sigma = rng.uniform(0, 1.3)
    if sigma > 0.2:
        low = cv2.GaussianBlur(low, (0, 0), sigma)
    if rng.random() < cfg.motion_blur_p:
        k = int(rng.integers(3, 8))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0), (k, k))
        low = cv2.filter2D(low, -1, kernel / max(kernel.sum(), 1e-6))

    # 6. Sensor noise, then video-codec style compression at the low resolution.
    low = low + rng.normal(0, rng.uniform(1.5, 9), size=low.shape).astype(np.float32)
    low = np.clip(low, 0, 255).astype(np.uint8)
    quality = int(rng.integers(35, 90))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(low, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    low = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # 7. Upsample to model input as the recognizer would.
    out = cv2.resize(low, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return out, {"width": width, "strong_perspective": strong, "jpeg_quality": quality, "blur_sigma": round(sigma, 2)}


def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preview(art_id: str, n: int = 12) -> None:
    from . import ART_DIR, DATA_DIR

    img = load_rgb(ART_DIR / f"{art_id}.jpg")
    rng = np.random.default_rng(0)
    tiles = [clean_view(img)] + [degraded_view(img, rng)[0] for _ in range(n - 1)]
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (INPUT_SIZE + 4), cols * (INPUT_SIZE + 4), 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * (INPUT_SIZE + 4) : r * (INPUT_SIZE + 4) + INPUT_SIZE, c * (INPUT_SIZE + 4) : c * (INPUT_SIZE + 4) + INPUT_SIZE] = t
    out = DATA_DIR / f"preview-{art_id}.png"
    Image.fromarray(sheet).save(out)
    print(out)


if __name__ == "__main__":
    preview(sys.argv[1])
