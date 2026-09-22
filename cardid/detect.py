"""Find the card under a click, warp it to a canonical card, and cut out the art box.

This is the classical stand-in for the M2 detector model. It looks for a convex
quadrilateral with a card-like aspect ratio that contains the click, preferring the smallest
one (the card, not the playmat). Real webcam frames have dark card borders on lighter mats,
so Canny + contour approximation gets most face-up cards; the capture UI has a drag-a-box
fallback for the rest, and every labeled click is saved so a learned detector can replace this.
"""

from __future__ import annotations

import cv2
import numpy as np

from .degrade import INPUT_SIZE

CARD_W, CARD_H = 250, 350  # canonical portrait card; 63x88mm is 1.397
CARD_ASPECT = CARD_H / CARD_W

# Scryfall's art_crop is a fixed template per frame, not a per-card cut: template-matching
# the art_crop back into the card image gives the same box (within 1% of the card) for every
# card of a frame. Each frame here is (x0, y0, x1) as fractions of the card; the box's height
# follows the art image's own aspect ratio (`frame_box`), which is also what tells the frames
# apart (`frame_of`): ~1.37 modern, ~1.24 old (1993/1997), ~1.62 extended art, 0.6-1.1
# full-art basics and most tokens (same x range as modern, art runs down to ~0.8), ~0.41 the
# half-width art of sagas (right half) and class/case cards (left half). The query side
# cuts every frame from the warped card (`art_crops`) and each gallery art is compared
# against the cut for its own frame.
FRAMES = {
    "modern": (0.08, 0.115, 0.92),
    "old": (0.121, 0.101, 0.881),
    "extended": (0.0, 0.121, 1.0),
    "tall": (0.08, 0.115, 0.92),
    "right": (0.502, 0.112, 0.922),
    "left": (0.074, 0.112, 0.494),
}
FRAME_NAMES = list(FRAMES)
# representative art aspect (width / height) per frame, for the query-side cut
FRAME_ASPECT = {"modern": 1.37, "old": 1.24, "extended": 1.62, "tall": 0.88, "right": 0.415, "left": 0.415}
HALF_LEFT_LAYOUTS = {"class", "case"}


def frame_of(aspect: float, layout: str | None = None) -> str:
    """Frame name for a gallery art from its image aspect (width / height) and, for the
    half-width frames, the Scryfall layout (sagas put the art on the right, class and case
    cards on the left; without a layout assume saga, they outnumber the others 5:1)."""
    if aspect < 0.6:
        return "left" if layout in HALF_LEFT_LAYOUTS else "right"
    if aspect < 1.1:
        return "tall"
    if aspect < 1.3:
        return "old"
    if aspect < 1.44:
        return "modern"
    return "extended"


def frame_box(frame: str, aspect: float | None = None) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of a frame's art box as fractions of the card, its height from the
    art's aspect (the frame's representative aspect when not given)."""
    x0, y0, x1 = FRAMES[frame]
    aspect = aspect or FRAME_ASPECT[frame]
    return x0, y0, x1, y0 + (x1 - x0) * CARD_W / aspect / CARD_H


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def quad_aspect(quad: np.ndarray) -> float:
    """Long side / short side of an ordered quad."""
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return max(w, h) / max(min(w, h), 1e-6)


def find_card_quad(img: np.ndarray, click: tuple[float, float], min_side: int = 40) -> np.ndarray | None:
    """Return an ordered 4x2 float32 quad containing `click`, or None. `img` is RGB uint8."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    median = float(np.median(gray))
    candidates = []
    # Two edge maps: auto-threshold Canny catches borders on light mats; the adaptive
    # threshold catches cards on dark mats where the border contrast is low.
    edge_maps = [
        cv2.Canny(gray, max(0, 0.66 * median), min(255, 1.33 * median)),
        cv2.Canny(gray, 30, 90),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5),
    ]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for edges in edge_maps:
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < (min_side * min_side * CARD_ASPECT):
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = order_corners(approx)
            if cv2.pointPolygonTest(quad, click, False) < 0:
                continue
            if not 1.15 <= quad_aspect(quad) <= 1.7:
                continue
            candidates.append((cv2.contourArea(quad), quad))
    if not candidates:
        return None
    # The smallest card-shaped quad is usually the card's inner frame line, not its outer
    # edge. Both are card-shaped and nested, so take the outermost quad that is still within
    # 1.35x the smallest area (a black border is ~5% per side, an area ratio of ~1.22); the
    # playmat, a sleeve stack, or an adjacent card is far larger than that.
    smallest = min(a for a, _ in candidates)
    return max((t for t in candidates if t[0] <= 1.35 * smallest), key=lambda t: t[0])[1]


def warp_card(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp an ordered quad to the canonical portrait card. A landscape quad
    (tapped card) is rotated so the long side is vertical; orientation ambiguity (which way
    was it tapped, is it upside down) is resolved by the caller trying rotations."""
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    if w > h:
        quad = np.roll(quad, -1, axis=0)  # tr, br, bl, tl -> treat the right edge as the top
    dst = np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (CARD_W, CARD_H), flags=cv2.INTER_AREA)


def card_orientations(card: np.ndarray) -> list[np.ndarray]:
    """The canonical card and its 180-degree rotation; the recognizer scores both."""
    return [card, cv2.rotate(card, cv2.ROTATE_180)]


def art_crop(card: np.ndarray, frame: str = "modern") -> np.ndarray:
    """The recogniser's input for one frame: that frame's art box cut from the canonical card
    and squashed to the square input the same way the gallery art_crop is."""
    x0, y0, x1, y1 = frame_box(frame)
    art = card[int(y0 * CARD_H) : int(y1 * CARD_H), int(x0 * CARD_W) : int(x1 * CARD_W)]
    return cv2.resize(art, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)


def art_crops(card: np.ndarray) -> np.ndarray:
    """One recogniser input per frame, in FRAME_NAMES order (F x INPUT_SIZE x INPUT_SIZE x 3)."""
    return np.stack([art_crop(card, f) for f in FRAME_NAMES])


def rect_to_quad(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
