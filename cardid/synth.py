"""Synthetic table scenes for the corner detector, plus the real-capture equivalent.

A scene is what `capture.html` sends on a click: a SCENE x SCENE native-pixel window centred
on the click, with the clicked card somewhere under the centre. The target is the card's four
corners (its own edge, not the sleeve's) in printed order: top-left, top-right, bottom-right,
bottom-left of the card face, in scene pixels.

The implementation lives in `image_bank` (decoded card/art banks), `scene_geometry` (quads
and windows), `scene_renderer` (compositing and webcam photometrics) and `scene_datasets`
(torch datasets and input normalisation); this module re-exports them and draws a sheet:

    uv run python -m cardid.synth --n 16 --out /tmp/scenes.png   # sheet with the targets drawn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .constants import DET_INPUT, SCENE
from .image_bank import ART_SHAPE, BG_ARTS, CARD_SHAPE, TWO_PART_RATE, ArtBank, CardBank, ImageBank, list_arts, list_cards
from .scene_datasets import RealSceneDataset, SceneDataset, batch_to_input, scene_to_input, trusted_quad
from .scene_geometry import PRINTED, apply_affine, expand, quad_from_pose, quad_roi, quad_short, window_around
from .scene_renderer import (
    CORNER_RADIUS,
    background,
    card_face,
    draw_card,
    draw_sleeve_ring,
    draw_toploader,
    gloss,
    occluders,
    paste,
    photometrics,
    render_scene,
    rounded_mask,
    warp_alpha,
)

__all__ = [
    "ART_SHAPE",
    "BG_ARTS",
    "CARD_SHAPE",
    "CORNER_RADIUS",
    "DET_INPUT",
    "PRINTED",
    "SCENE",
    "TWO_PART_RATE",
    "ArtBank",
    "CardBank",
    "ImageBank",
    "RealSceneDataset",
    "SceneDataset",
    "apply_affine",
    "background",
    "batch_to_input",
    "card_face",
    "draw_card",
    "draw_sleeve_ring",
    "draw_toploader",
    "expand",
    "gloss",
    "list_arts",
    "list_cards",
    "occluders",
    "paste",
    "photometrics",
    "quad_from_pose",
    "quad_roi",
    "quad_short",
    "render_scene",
    "rounded_mask",
    "scene_to_input",
    "sheet",
    "trusted_quad",
    "warp_alpha",
    "window_around",
]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/scenes.png"))
    args = parser.parse_args()
    sheet(args.n, args.seed, args.out)


if __name__ == "__main__":
    main()
