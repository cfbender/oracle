"""Profile `render_scene` single-threaded: `python -m cardid.profile_synth [--n 60] [--top 20]`.

Prints ms/scene and the cProfile hot spots by own time, which is what decides how many
scenes per second each DataLoader worker can produce."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import cv2
import numpy as np

from .image_bank import ArtBank, CardBank
from .scene_renderer import render_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cv2.setNumThreads(0)
    cards, arts = CardBank(), ArtBank()
    rng = np.random.default_rng(args.seed)
    render_scene(rng, cards, arts)  # warm caches
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    for _ in range(args.n):
        render_scene(rng, cards, arts)
    prof.disable()
    dt = time.perf_counter() - t0
    print(f"{args.n} scenes: {1000 * dt / args.n:.1f} ms/scene")
    pstats.Stats(prof).sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
