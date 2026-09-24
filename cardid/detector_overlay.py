"""Overlay learned and classical card quads on a screenshot (`python -m cardid.detector` CLI)."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .detect import find_card_quad
from .detector import Detector, fit_card_pose


def main() -> None:
    """Overlay the learned (green) and classical (blue) quads for clicks on a screenshot.

    python -m cardid.detector --checkpoint data/runs/det/best.pt --image frame.png \
        --click 660,350 --click 1120,480 --out /tmp/overlay.jpg
    """
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--click", action="append", required=True, help="x,y in image pixels; repeatable")
    ap.add_argument("--out", required=True, help="output image (jpg/png)")
    ap.add_argument("--snap", action="store_true", help="snap the learned quad to an exact 63x88 rectangle")
    args = ap.parse_args()

    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    detector = Detector(args.checkpoint)
    vis = img.copy()
    for spec in args.click:
        x, y = (int(v) for v in spec.split(","))
        classical = find_card_quad(img, (x, y))
        if classical is not None:
            cv2.polylines(vis, [classical.astype(np.int32)], True, (60, 120, 255), 3)
        learned = detector.locate(img, (x, y), snap=args.snap)
        cv2.polylines(vis, [learned.astype(np.int32)], True, (40, 230, 60), 3)
        cv2.circle(vis, tuple(learned[0].astype(int)), 10, (255, 40, 40), -1)  # printed top-left
        cv2.circle(vis, (x, y), 8, (255, 255, 0), -1)
        cx, cy, short, angle = fit_card_pose(learned)
        up = (learned[0] + learned[1]) / 2 - learned.mean(axis=0)
        classical_note = "found" if classical is not None else "none"
        print(
            f"click ({x},{y}): learned centre=({cx:.0f},{cy:.0f}) short={short:.0f}px angle={angle:.0f}deg up={np.degrees(np.arctan2(up[1], up[0])):.0f}deg; classical {classical_note}"
        )
    cv2.imwrite(args.out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
