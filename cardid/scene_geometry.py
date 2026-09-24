"""Card-quad geometry shared by the scene renderer, the detector and its datasets.

Quads are 4x2 float32 corners in printed order (top-left, top-right, bottom-right,
bottom-left of the card face) unless a function says otherwise.
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import SCENE

PRINTED = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])  # unit square in printed order


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


def quad_short(quad: np.ndarray) -> float:
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return float(min(w, h))


def expand(quad: np.ndarray, factor: float) -> np.ndarray:
    c = quad.mean(axis=0)
    return ((quad - c) * factor + c).astype(np.float32)


def window_around(img: np.ndarray, cx: float, cy: float, side: float, out: int = SCENE) -> tuple[np.ndarray, np.ndarray]:
    """Resample the `side` x `side` square centred on (cx, cy) to an `out` x `out` image.
    Returns (window, M) where M is the 2x3 affine from image to window coordinates."""
    s = out / side
    M = np.float32([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy]])
    win = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return win, M


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (np.c_[pts, np.ones(len(pts))] @ M.T).astype(np.float32)
