"""Where does the detector still fail? Error distribution and a sheet of the worst cases.

The trainer's median hides the tail: a median of 4% with a mean of 11% means most samples are
fine and a minority is grossly wrong. This tool breaks the fixed validation set down by
percentile, by failure kind (snap could not reach vs. the pose was on the wrong card or turned
90 degrees) and by card size, and renders the worst samples with the target (green), the raw
pose (blue), the snapped prediction (red) and the corner heatmap (magenta) so the failure
mode can be read off the picture.

    uv run python -m cardid.inspect_detector --checkpoint data/runs/det3/best.pt --out /tmp/worst.png
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch

from .detector import CornerNet, corner_error, fit_card_pose, load_checkpoint
from .model import pick_device
from .synth import DET_INPUT, batch_to_input, quad_short
from .train_detector import HIT, predict_scenes, val_scenes

GROSS = 0.25  # relative corner error above which the pose was on the wrong object or turned 90 degrees


def analyse(snapped: np.ndarray, raw: np.ndarray, quads: np.ndarray) -> dict:
    short = np.array([quad_short(q) for q in quads])
    rel, rel_raw = corner_error(snapped, quads) / short, corner_error(raw, quads) / short
    angle = np.array([fit_card_pose(q)[3] for q in quads])
    angle_raw = np.array([fit_card_pose(q)[3] for q in raw])
    turned = np.abs(((angle - angle_raw) + np.pi / 2) % np.pi - np.pi / 2) > np.pi / 4  # pose rotated ~90 degrees from the target
    small, large = short < np.percentile(short, 25), short > np.percentile(short, 75)
    pct = lambda a: {f"p{p}": round(float(np.percentile(a, p)), 4) for p in (50, 75, 90, 95)}
    gross = rel > GROSS
    return {
        "n": len(rel),
        "snapped": pct(rel),
        "pose": pct(rel_raw),
        "hit": round(float((rel < HIT).mean()), 3),
        "gross": round(float(gross.mean()), 3),
        "gross_turned_90": round(float((gross & turned).sum() / max(gross.sum(), 1)), 3),
        "snap_helped": round(float((rel < rel_raw - 0.005).mean()), 3),
        "snap_hurt": round(float((rel > rel_raw + 0.005).mean()), 3),
        "small_cards": {"short_px_max": round(float(short[small].max()), 1), "median": round(float(np.median(rel[small])), 4), "hit": round(float((rel[small] < HIT).mean()), 3)},
        "large_cards": {"short_px_min": round(float(short[large].min()), 1), "median": round(float(np.median(rel[large])), 4), "hit": round(float((rel[large] < HIT).mean()), 3)},
    }, rel


def draw_case(scene: np.ndarray, target: np.ndarray, raw: np.ndarray, snapped: np.ndarray, heat: np.ndarray, label: str) -> np.ndarray:
    vis = scene.copy()
    h = cv2.resize(heat, (DET_INPUT, DET_INPUT), interpolation=cv2.INTER_LINEAR)
    vis = (vis * (1 - 0.6 * h[..., None]) + np.array([255, 0, 255]) * 0.6 * h[..., None]).astype(np.uint8)
    cv2.polylines(vis, [target.astype(np.int32)], True, (40, 230, 60), 1)
    cv2.polylines(vis, [raw.astype(np.int32)], True, (60, 120, 255), 1)
    cv2.polylines(vis, [snapped.astype(np.int32)], True, (255, 40, 40), 1)
    cv2.putText(vis, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def sheet(tiles: list[np.ndarray], cols: int = 4) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    blank = np.zeros_like(tiles[0])
    tiles = tiles + [blank] * (rows * cols - len(tiles))
    return np.concatenate([np.concatenate(tiles[r * cols : (r + 1) * cols], axis=1) for r in range(rows)], axis=0)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val", type=int, default=1000, help="validation set size (must match a cached det-val-<n>-999.npz or it is rendered)")
    ap.add_argument("--worst", type=int, default=16, help="how many worst cases to render")
    ap.add_argument("--out", default="/tmp/detector-worst.png")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = pick_device(args.device)
    model = CornerNet(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    scenes, quads = val_scenes(args.val, args.workers)
    snapped, raw, _ = predict_scenes(model, scenes, device)
    stats, rel = analyse(snapped, raw, quads)
    print(json.dumps(stats, indent=2))

    worst = np.argsort(-rel)[: args.worst]
    model.eval()
    heat = torch.sigmoid(model(batch_to_input(torch.from_numpy(scenes[worst]).to(device)))[3])[:, 0].cpu().numpy()
    tiles = [draw_case(scenes[i], quads[i], raw[i], snapped[i], heat[k], f"#{i} err {rel[i]:.2f} short {quad_short(quads[i]):.0f}px") for k, i in enumerate(worst)]
    cv2.imwrite(args.out, cv2.cvtColor(sheet(tiles), cv2.COLOR_RGB2BGR))
    print(f"wrote {args.out} (green target, blue pose, red snapped, magenta heatmap)")


if __name__ == "__main__":
    main()
