"""Learned card localiser: MobileNetV3-Small on the 256px click window -> card pose.

A Magic card is always 63x88 mm, so instead of regressing four free corners the head predicts
a *pose* (centre, short side, rotation) and the corners are those of a 63x88 rectangle at
that pose, plus small bounded per-corner residuals for the mild perspective of a camera that
is not exactly overhead. The ratio is therefore built in: the network cannot output a
square or a strip, and the residuals are penalised so they only carry real perspective.

Rotation is predicted as (cos 2t, sin 2t): a rectangle's geometry repeats every 180 degrees
and the corner loss is taken over cyclic orderings, so t in (-90, 90] covers every card.

Regressing coordinates through a flattened fully connected head has a precision floor (about
a tenth of the short side here, however long it trains), so the network also predicts a
class-agnostic *corner heatmap* at stride 4. Each pose corner snaps to the nearest heatmap
peak within a radius scaled by the card size, with a soft-argmax over the peak's 3x3
neighbourhood for sub-pixel position; a corner with no peak nearby keeps the pose estimate.
The pose gives the ordering, the 90-degree disambiguation and a guaranteed answer; the
heatmap gives the precision.

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
    """MobileNetV3-Small trunk with a head that keeps the feature map's geometry.

    Global average pooling (the first version of this head) keeps only *how much* card is in
    the window, so it learned scale well but not where the card is or which way its edges
    point. Here the stride-32 and stride-16 maps are fused at 16x16, squeezed to a few channels
    and flattened, so every position keeps its own weights into the pose regression."""

    FUSE = 64
    SQUEEZE = 32
    DECODE = 32

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        # the stem is split where the decoder takes laterals; state-dict keys stay `stem.<i>`
        self.stem = backbone.features[:9]  # (N, 48, 16, 16) at 256px input
        self.top = backbone.features[9:]  # (N, 576, 8, 8)
        self.lateral = nn.Conv2d(48, self.FUSE, 1)
        self.reduce = nn.Conv2d(576, self.FUSE, 1)
        self.squeeze = nn.Sequential(nn.Conv2d(self.FUSE, self.SQUEEZE, 3, padding=1), nn.Hardswish())
        self.head = nn.Sequential(nn.Linear(self.SQUEEZE * 16 * 16, 256), nn.Hardswish(), nn.Dropout(0.1), nn.Linear(256, 5 + 8))
        # corner heatmap decoder: fused stride 16 -> 8 (lateral from features[3], 24 ch) -> 4
        # (lateral from features[1], 16 ch) -> one logit per position
        self.lat8 = nn.Conv2d(24, self.DECODE, 1)
        self.lat4 = nn.Conv2d(16, self.DECODE, 1)
        self.up16 = nn.Conv2d(self.FUSE, self.DECODE, 1)
        self.dec8 = nn.Sequential(nn.Conv2d(self.DECODE, self.DECODE, 3, padding=1), nn.Hardswish())
        self.dec4 = nn.Sequential(nn.Conv2d(self.DECODE, self.DECODE, 3, padding=1), nn.Hardswish(), nn.Conv2d(self.DECODE, 1, 1))
        # start as an upright card of short side 0.3 centred in the window
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()
            self.head[-1].bias[:5] = torch.tensor([0.5, 0.5, float(np.log(0.3)), 1.0, 0.0])
            self.dec4[-1].bias.fill_(-2.19)  # sigmoid 0.1: corners are rare (focal-loss prior)

    def backbone_parameters(self):
        yield from self.stem.parameters()
        yield from self.top.parameters()

    def head_parameters(self):
        for m in (self.lateral, self.reduce, self.squeeze, self.head, self.lat8, self.lat4, self.up16, self.dec8, self.dec4):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (corners (N, 4, 2) in [0, 1] window units, residuals (N, 4, 2) in short-side
        units, raw pose (N, 5) [cx, cy, log short, cos2t, sin2t], corner heatmap logits
        (N, 1, 64, 64) at stride HEAT_STRIDE)."""
        s4 = self.stem[:2](x)  # (N, 16, 64, 64)
        s8 = self.stem[2:4](s4)  # (N, 24, 32, 32)
        mid = self.stem[4:](s8)  # (N, 48, 16, 16)
        top = self.reduce(self.top(mid))
        fused = self.lateral(mid) + nn.functional.interpolate(top, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        out = self.head(self.squeeze(fused).flatten(1))
        pose, res = out[:, :5], out[:, 5:].view(-1, 4, 2)
        res = torch.tanh(res) * RESIDUAL
        quad = pose_to_quad(pose)
        short = pose[:, 2].exp()[:, None, None]
        d8 = self.dec8(self.lat8(s8) + nn.functional.interpolate(self.up16(fused), scale_factor=2, mode="bilinear", align_corners=False))
        heat = self.dec4(self.lat4(s4) + nn.functional.interpolate(d8, scale_factor=2, mode="bilinear", align_corners=False))
        return quad + res * short, res, pose, heat


HEAT_STRIDE = 4
HEAT_SIZE = DET_INPUT // HEAT_STRIDE


def heat_targets(quads: torch.Tensor, size: int = HEAT_SIZE) -> torch.Tensor:
    """(N, 4, 2) corners in [0, 1] window units -> (N, 1, size, size) Gaussian corner targets
    (max over the four corners). The Gaussian's sigma scales with the card so a small card's
    peak is still a point and a large card's is not needlessly sharp."""
    n = quads.shape[0]
    pts = quads * size  # heatmap px
    short = torch.minimum((pts[:, 1] - pts[:, 0]).norm(dim=1), (pts[:, 3] - pts[:, 0]).norm(dim=1))  # (N,)
    sigma = (0.05 * short).clamp(min=0.8)[:, None, None, None]
    grid = torch.arange(size, device=quads.device, dtype=torch.float32) + 0.5
    dy = grid[None, None, :, None] - pts[..., 1][:, :, None, None]  # (N, 4, size, 1)
    dx = grid[None, None, None, :] - pts[..., 0][:, :, None, None]  # (N, 4, 1, size)
    g = torch.exp(-(dx.pow(2) + dy.pow(2)) / (2 * sigma.pow(2)))  # (N, 4, size, size)
    # the corner sits between cell centres, so scale each Gaussian to peak at exactly 1 in
    # its nearest cell (the focal loss's positive) while keeping the sub-pixel shape around it
    g = g / g.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return g.max(dim=1, keepdim=True).values.view(n, 1, size, size)


def heat_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet's penalty-reduced pixelwise focal loss, normalised by the number of peaks."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target > 0.999).float()
    pos_loss = -(1 - p).pow(alpha) * torch.log(p) * pos
    neg_loss = -(1 - target).pow(beta) * p.pow(alpha) * torch.log(1 - p) * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def snap_corners(quads: np.ndarray, heat: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Move each corner of `quads` ((N, 4, 2), input px) to the strongest heatmap peak within
    ~12% of the card's short side, refined to sub-pixel by a soft-argmax over the peak's 3x3
    neighbourhood; corners with no peak above `threshold` in reach are left alone. `heat` is
    (N, HEAT_SIZE, HEAT_SIZE) of probabilities."""
    out = quads.copy()
    size = heat.shape[-1]
    for n, quad in enumerate(quads):
        short = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0])) / HEAT_STRIDE
        r = int(np.clip(round(0.12 * short), 1, 6))
        h = heat[n]
        for k, (x, y) in enumerate(quad / HEAT_STRIDE - 0.5):  # cell centres sit at i + 0.5
            cx, cy = int(np.clip(round(x), 0, size - 1)), int(np.clip(round(y), 0, size - 1))
            x0, x1, y0, y1 = max(cx - r, 0), min(cx + r + 1, size), max(cy - r, 0), min(cy + r + 1, size)
            patch = h[y0:y1, x0:x1]
            py, px = np.unravel_index(int(patch.argmax()), patch.shape)
            if patch[py, px] < threshold:
                continue
            py, px = py + y0, px + x0
            ny0, ny1, nx0, nx1 = max(py - 1, 0), min(py + 2, size), max(px - 1, 0), min(px + 2, size)
            w = h[ny0:ny1, nx0:nx1]
            ys, xs = np.mgrid[ny0:ny1, nx0:nx1]
            sx, sy = float((w * xs).sum() / w.sum()), float((w * ys).sum() / w.sum())
            out[n, k] = ((sx + 0.5) * HEAT_STRIDE, (sy + 0.5) * HEAT_STRIDE)
    return out


def corner_loss(pred: torch.Tensor, target: torch.Tensor, residual: torch.Tensor | None = None, residual_weight: float = 0.5) -> torch.Tensor:
    """Mean L1 corner error under the best of the 4 cyclic corner orderings (a rotated card has
    no privileged first corner), plus an L2 penalty keeping the residuals small."""
    losses = torch.stack([(pred - target.roll(k, dims=1)).abs().mean(dim=(1, 2)) for k in range(4)], dim=1)
    loss = losses.min(dim=1).values.mean()
    if residual is not None:
        loss = loss + residual_weight * residual.pow(2).mean()
    return loss


def quad_to_pose(quads: torch.Tensor) -> torch.Tensor:
    """Best-fit (N, 5) [cx, cy, log short, cos2t, sin2t] for target quads, same units as the corners."""
    rows = []
    for q in quads.detach().cpu().numpy():
        cx, cy, short, angle = fit_card_pose(q)
        t = np.radians(angle)
        rows.append((cx, cy, np.log(max(short, 1e-4)), np.cos(2 * t), np.sin(2 * t)))
    return torch.tensor(rows, dtype=torch.float32, device=quads.device)


def pose_loss(pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    """Direct supervision of the raw pose. The corner loss alone has a local minimum with the
    card turned 90 degrees (corners move only ~0.2 short sides, and the (cos2t, sin2t) output
    would have to pass through the origin to escape); the L2 distance to the target angle
    vector is convex in the raw outputs and has its maximum there instead. Centre and log
    size get L1 terms so early training does not have to discover them through the corners."""
    centre = (pose[:, :2] - target_pose[:, :2]).abs().mean()
    size = (pose[:, 2] - target_pose[:, 2]).abs().mean()
    angle = (pose[:, 3:] - target_pose[:, 3:]).pow(2).sum(dim=1).mean()
    return centre + 0.1 * size + 0.1 * angle


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


def load_checkpoint(model: CornerNet, path: Path, device: torch.device) -> None:
    """Load a state dict, tolerating checkpoints saved before the heatmap decoder existed
    (their decoder layers keep their fresh initialisation, so warm-starting still works)."""
    state = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"{path}: unexpected keys {sorted(unexpected)[:5]}")
    if missing:
        print(f"{path}: {len(missing)} keys not in checkpoint (pre-heatmap detector), left at init")


class Detector:
    """Loads a CornerNet checkpoint and locates the card under a click in a full frame."""

    def __init__(self, checkpoint: Path | None = None, device: torch.device | None = None, model: CornerNet | None = None):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.device = device or torch.device("cpu")
        if model is None:
            model = CornerNet(pretrained=False).to(self.device)
            load_checkpoint(model, self.checkpoint, self.device)
        self.model = model.eval()

    @torch.no_grad()
    def predict_window(self, img: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
        """Corners (4x2, image px) of the card in the `side`-px square centred on (cx, cy)."""
        win, M = window_around(img, cx, cy, side, DET_INPUT)
        x = scene_to_input(win)[None].to(self.device)
        quad, _, _, heat = self.model(x)
        quad = quad.cpu().numpy() * DET_INPUT
        quad = snap_corners(quad, torch.sigmoid(heat)[:, 0].cpu().numpy())[0]
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
