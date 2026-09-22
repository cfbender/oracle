"""Learned card localiser: MobileNetV3-Small on the 256px click window -> card pose.

A Magic card is always 63x88 mm, so instead of regressing four free corners the head predicts
a *pose* (centre, short side, rotation) and the corners are those of a 63x88 rectangle at
that pose, plus small bounded per-corner residuals for the mild perspective of a camera that
is not exactly overhead. The ratio is therefore built in: the network cannot output a
square or a strip, and the residuals are penalised so they only carry real perspective.

Rotation is predicted as (cos 2t, sin 2t): a rectangle's geometry repeats every 180 degrees
and the corner loss is taken over cyclic orderings, so t in (-90, 90] covers every card.

Inference is two-stage: the pose from the 640px click window, then the same network on a
tight window around that estimate, where 1% of the input is a couple of native pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .synth import (
    DET_INPUT,
    PRINTED,
    SCENE,
    apply_affine,
    scene_to_input,
    window_around,
)

CARD_ASPECT = 88 / 63
RESIDUAL = 0.08  # max per-corner residual as a fraction of the short side
UNIT_CARD = torch.tensor(PRINTED - 0.5) * torch.tensor([1.0, CARD_ASPECT])  # (4, 2), short side 1


def pose_to_quad(pose: torch.Tensor) -> torch.Tensor:
    """(N, 5) [cx, cy, log short, cos2t, sin2t] -> (N, 4, 2) corners in the same units as cx."""
    cx, cy, log_s, c2, s2 = pose.unbind(dim=1)
    short = log_s.exp()
    # halve the doubled angle; atan2 keeps it in (-90, 90] degrees
    t = 0.5 * torch.atan2(s2, c2)
    cos, sin = torch.cos(t), torch.sin(t)
    rot = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)  # (N, 2, 2)
    box = UNIT_CARD.to(pose.device)[None] * short[:, None, None]  # (N, 4, 2)
    return box @ rot.transpose(1, 2) + torch.stack([cx, cy], -1)[:, None, :]


class CornerNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # (N, 576, 8, 8) at 256px input
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(576, 256), nn.Hardswish(), nn.Dropout(0.1), nn.Linear(256, 5 + 8))
        # start as an upright card of short side 0.3 centred in the window
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()
            self.head[-1].bias[:5] = torch.tensor([0.5, 0.5, float(np.log(0.3)), 1.0, 0.0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (corners (N, 4, 2) in [0, 1] window units, residuals (N, 4, 2) in short-side units)."""
        out = self.head(self.pool(self.features(x)).flatten(1))
        pose, res = out[:, :5], out[:, 5:].view(-1, 4, 2)
        res = torch.tanh(res) * RESIDUAL
        quad = pose_to_quad(pose)
        short = pose[:, 2].exp()[:, None, None]
        return quad + res * short, res


def corner_loss(pred: torch.Tensor, target: torch.Tensor, residual: torch.Tensor | None = None, residual_weight: float = 0.5) -> torch.Tensor:
    """Mean L1 corner error under the best of the 4 cyclic corner orderings (a rotated card has
    no privileged first corner), plus an L2 penalty keeping the residuals small."""
    losses = torch.stack([(pred - target.roll(k, dims=1)).abs().mean(dim=(1, 2)) for k in range(4)], dim=1)
    loss = losses.min(dim=1).values.mean()
    if residual is not None:
        loss = loss + residual_weight * residual.pow(2).mean()
    return loss


def corner_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample mean corner distance under the best cyclic ordering; same units as the inputs."""
    errs = np.stack([np.linalg.norm(pred - np.roll(target, k, axis=1), axis=2).mean(axis=1) for k in range(4)], axis=1)
    return errs.min(axis=1)


def fit_card_pose(quad: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares-ish 63x88 rectangle for any 4 cyclically ordered corners:
    (cx, cy, short side, angle in degrees of the short edge). Snaps a classical or manual quad
    to the known ratio."""
    q = np.asarray(quad, np.float32)
    edges = np.roll(q, -1, axis=0) - q  # 0->1, 1->2, 2->3, 3->0
    lengths = np.linalg.norm(edges, axis=1)
    short_pair = 0 if lengths[0] + lengths[2] < lengths[1] + lengths[3] else 1
    short = (lengths[short_pair] + lengths[short_pair + 2]) / 2
    long = (lengths[1 - short_pair] + lengths[3 - short_pair]) / 2
    s = (short + long / CARD_ASPECT) / 2  # both edge pairs vote on the size
    e = edges[short_pair] - edges[short_pair + 2]  # the two short edges point opposite ways
    angle = float(np.degrees(np.arctan2(e[1], e[0])))
    c = q.mean(axis=0)
    return float(c[0]), float(c[1]), float(s), angle


def card_rect(cx: float, cy: float, short: float, angle_deg: float) -> np.ndarray:
    """Cyclically ordered corners of a 63x88 rectangle; inverse of `fit_card_pose`."""
    a = np.deg2rad(angle_deg)
    rot = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    box = (PRINTED - 0.5) * np.float32([short, short * CARD_ASPECT])
    return (box @ rot.T + np.float32([cx, cy])).astype(np.float32)


def cyclic_order(quad: np.ndarray) -> np.ndarray:
    """Clockwise (image coords) order starting at the corner nearest the top-left; robust for
    any rotation, unlike `detect.order_corners`' sum/diff heuristic."""
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0).astype(np.float32)


class Detector:
    """Loads a CornerNet checkpoint and locates the card under a click in a full frame."""

    def __init__(self, checkpoint: Path | None = None, device: torch.device | None = None, model: CornerNet | None = None):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device = device or torch.device("cpu")
        if model is None:
            model = CornerNet(pretrained=False).to(self.device)
            model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.model = model.eval()

    @torch.no_grad()
    def predict_window(self, img: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
        """Corners (4x2, image px) of the card in the `side`-px square centred on (cx, cy)."""
        win, M = window_around(img, cx, cy, side, DET_INPUT)
        x = scene_to_input(win)[None].to(self.device)
        quad, _ = self.model(x)
        quad = quad[0].cpu().numpy() * DET_INPUT
        Minv = cv2_invert(M)
        return apply_affine(Minv, quad)

    def locate(self, img: np.ndarray, click: tuple[float, float], refine: bool = True, snap: bool = False) -> np.ndarray:
        """Ordered 4x2 float32 quad of the card under `click` in `img` (RGB uint8). Always
        returns something: a wrong quad still gets the recogniser a guess, which the UI can
        show alongside its alternatives."""
        quad = self.predict_window(img, click[0], click[1], SCENE)
        if refine:
            # second pass on a window where the card spans ~60% of the input
            cx, cy, short, _ = fit_card_pose(quad)
            side = max(short * CARD_ASPECT / 0.6, 64.0)
            quad = self.predict_window(img, cx, cy, side)
        if snap:
            quad = card_rect(*fit_card_pose(quad))
        return cyclic_order(quad)


def cv2_invert(M: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine."""
    A, t = M[:, :2], M[:, 2]
    Ainv = np.linalg.inv(A)
    return np.hstack([Ainv, (-Ainv @ t)[:, None]]).astype(np.float32)


def main() -> None:
    """Overlay the learned (green) and classical (blue) quads for clicks on a screenshot.

    python -m cardid.detector --checkpoint data/runs/det/best.pt --image frame.png \
        --click 660,350 --click 1120,480 --out /tmp/overlay.jpg
    """
    import argparse

    import cv2

    from .detect import find_card_quad

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
        cv2.circle(vis, (x, y), 8, (255, 255, 0), -1)
        cx, cy, short, angle = fit_card_pose(learned)
        classical_note = "found" if classical is not None else "none"
        print(f"click ({x},{y}): learned centre=({cx:.0f},{cy:.0f}) short={short:.0f}px angle={angle:.0f}deg; classical {classical_note}")
    cv2.imwrite(args.out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
