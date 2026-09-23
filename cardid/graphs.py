"""Inference-time wrappers that fold the numpy/OpenCV glue of the click-to-identify pipeline
into the exported ONNX graphs, so a runtime (onnxruntime-web in the camera owner's browser,
onnxruntime on a server) only resamples an affine window and does a handful of scalar ops.

Three graphs, all with the image handed over as the browser has it (uint8 RGBA, HWC):

    DetectorGraph  window (256, 256, 4)          -> quad (4, 2) window px in printed order,
                                                    up (2,), centre (2,), short ()
    EmbedGraph     scene (H, W, 4), quad (4, 2)  -> embeddings (F, 128), one per frame cut
    SearchGraph    embeddings (F, 128)           -> indices (k,), scores (k,)

`DetectorGraph` is `Detector.predict_window` plus `cyclic_order`/`orient_quad` for one window:
the four 90-degree rotations, CornerNet, the heatmap corner snap (`snap_corners`, written
over the whole 64x64 grid with masks instead of dynamic slices) and the up vote. `EmbedGraph`
is `detect.warp_card` + `detect.art_crops` + `Embedder`: a bilinear perspective warp of the
quad to the 250x350 card via the closed-form square-to-quad projective map, all frame
cuts resized to 128px and embedded in one batch. `SearchGraph` is `index.frame_similarities`
+ top-k with the gallery embeddings, per-art frames and the frame prior stored as constants.

`cardid.bundle.Bundle` runs the exported graphs from Python and is the reference for the
remaining glue (window affine, the two-pass refine, mapping back to image pixels).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import IMAGENET_MEAN, IMAGENET_STD
from .degrade import INPUT_SIZE
from .detect import CARD_H, CARD_W, FRAME_NAMES, FRAME_ROTATIONS, frame_box
from .detector import CARD_ASPECT, HEAT_SIZE, HEAT_STRIDE, CornerNet
from .model import Embedder
from .synth import DET_INPUT

SNAP_THRESHOLD = 0.3
ROTATIONS = 4


def _normalise(rgba_hwc: torch.Tensor) -> torch.Tensor:
    """uint8 HWC RGBA -> float CHW, ImageNet-normalised (what `data.to_tensor` does)."""
    x = rgba_hwc[..., :3].to(torch.float32) / 255.0
    mean = torch.as_tensor(IMAGENET_MEAN).view(1, 1, 3)
    std = torch.as_tensor(IMAGENET_STD).view(1, 1, 3)
    return ((x - mean) / std).permute(2, 0, 1)


def _rot90(chw: torch.Tensor) -> torch.Tensor:
    """`np.rot90` (counter-clockwise, first axis towards the second) on a CHW image."""
    return torch.flip(chw.transpose(1, 2), dims=(1,))


def _roll4(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """`np.roll(q, -k, axis=0)` for a (4, 2) tensor and a 0-d integer tensor `k`."""
    idx = (torch.arange(4) + k) % 4
    return q[idx]


def snap_corners_graph(quad: torch.Tensor, prob: torch.Tensor, threshold: float = SNAP_THRESHOLD) -> torch.Tensor:
    """`detector.snap_corners` for one quad ((4, 2) input px) and one heatmap ((64, 64)
    probabilities), expressed with masks over the whole grid so it exports without
    data-dependent slicing. Ties in the argmax resolve to the first cell in row-major order,
    like numpy."""
    size, stride = HEAT_SIZE, HEAT_STRIDE
    short = torch.minimum(torch.linalg.norm(quad[1] - quad[0]), torch.linalg.norm(quad[3] - quad[0])) / stride
    r = torch.clamp(torch.round(0.12 * short), 1, 6)
    cell = torch.clamp(torch.round(quad / stride - 0.5), 0, size - 1)  # (4, 2) cell centres sit at i + 0.5
    coords = torch.arange(size, dtype=torch.float32)
    ys, xs = coords[:, None].expand(size, size), coords[None, :].expand(size, size)
    in_window = ((xs[None] - cell[:, 0, None, None]).abs() <= r) & ((ys[None] - cell[:, 1, None, None]).abs() <= r)  # (4, 64, 64)
    masked = torch.where(in_window, prob[None], torch.full_like(prob, -1.0)[None]).flatten(1)
    peak = masked.argmax(dim=1)  # (4,)
    peak_val = masked.gather(1, peak[:, None])[:, 0]
    py, px = (peak // size).to(torch.float32), (peak % size).to(torch.float32)
    near = ((xs[None] - px[:, None, None]).abs() <= 1) & ((ys[None] - py[:, None, None]).abs() <= 1)
    w = torch.where(near, prob[None], torch.zeros_like(prob)[None])
    total = w.flatten(1).sum(dim=1)
    sx = (w * xs[None]).flatten(1).sum(dim=1) / total
    sy = (w * ys[None]).flatten(1).sum(dim=1) / total
    snapped = torch.stack([(sx + 0.5) * stride, (sy + 0.5) * stride], dim=1)
    return torch.where((peak_val >= threshold)[:, None], snapped, quad)


def cyclic_order_graph(quad: torch.Tensor) -> torch.Tensor:
    """`detector.cyclic_order`: clockwise (image coords) from the corner nearest the top-left."""
    c = quad.mean(dim=0)
    ang = torch.atan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[torch.argsort(ang)]
    return _roll4(q, torch.argmin(q.sum(dim=1)))


def orient_quad_graph(quad: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`detector.orient_quad`: roll a cyclic quad so corner 0 is the printed top-left."""
    edges = torch.roll(quad, -1, dims=0) - quad
    lengths = torch.linalg.norm(edges, dim=1)
    first = torch.where(lengths[0] + lengths[2] < lengths[1] + lengths[3], 0, 1)
    mids = (quad + torch.roll(quad, -1, dims=0)) / 2 - quad.mean(dim=0)
    votes = mids @ up  # (4,)
    k = torch.where(votes[first] >= votes[first + 2], first, first + 2)
    return _roll4(quad, k)


def card_pose_graph(quad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`detector.fit_card_pose` without the angle: (centre (2,), short side ()) of the 63x88
    rectangle fitted to a cyclically ordered quad."""
    edges = torch.roll(quad, -1, dims=0) - quad
    lengths = torch.linalg.norm(edges, dim=1)
    short_pair = lengths[0] + lengths[2] < lengths[1] + lengths[3]
    short = torch.where(short_pair, lengths[0] + lengths[2], lengths[1] + lengths[3]) / 2
    long = torch.where(short_pair, lengths[1] + lengths[3], lengths[0] + lengths[2]) / 2
    return quad.mean(dim=0), (short + long / CARD_ASPECT) / 2


class DetectorGraph(nn.Module):
    """One detector pass on a 256px window: `Detector.predict_window(..., rotations=4)`
    followed by `cyclic_order` and `orient_quad`. The quad and the pose come back in window
    pixels; the caller maps them through the inverse of its window affine (a scale and a
    translation, which preserve the corner order)."""

    def __init__(self, net: CornerNet):
        super().__init__()
        self.net = net.eval()
        # `unrotate_direction` for k = 0..3 as matrices acting on row vectors
        unrot = [torch.eye(2)]
        for _ in range(ROTATIONS - 1):
            unrot.append(unrot[-1] @ torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))
        self.register_buffer("unrotate", torch.stack(unrot))  # (4, 2, 2)

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = _normalise(window)
        views = [x]
        for _ in range(ROTATIONS - 1):
            views.append(_rot90(views[-1]))
        quad, _, _, heat, up = self.net(torch.stack(views))
        quad = quad[0] * DET_INPUT
        quad = snap_corners_graph(quad, torch.sigmoid(heat[0, 0]))
        up = (up[:, None, :] @ self.unrotate).sum(dim=(0, 1))  # (2,) vote sum over the rotations
        quad = orient_quad_graph(cyclic_order_graph(quad), up)
        centre, short = card_pose_graph(quad)
        return quad, up, centre, short


def square_to_quad(quad: torch.Tensor) -> torch.Tensor:
    """3x3 projective map from the unit square ((0,0), (1,0), (1,1), (0,1)) to the quad's
    corners in that order, in closed form (Heckbert, Fundamentals of Texture Mapping)."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
    dx1, dx2, dx3 = x1 - x2, x3 - x2, x0 - x1 + x2 - x3
    dy1, dy2, dy3 = y1 - y2, y3 - y2, y0 - y1 + y2 - y3
    det = dx1 * dy2 - dx2 * dy1
    g = (dx3 * dy2 - dx2 * dy3) / det
    h = (dx1 * dy3 - dx3 * dy1) / det
    a, b, c = x1 - x0 + g * x1, x3 - x0 + h * x3, x0
    d, e, f = y1 - y0 + g * y1, y3 - y0 + h * y3, y0
    one = torch.ones_like(g)
    return torch.stack([torch.stack([a, b, c]), torch.stack([d, e, f]), torch.stack([g, h, one])])


class EmbedGraph(nn.Module):
    """`warp_card` (bilinear, like OpenCV's warpPerspective) + `art_crops` + `Embedder` for
    a printed-order quad in the scene's pixel coordinates. Intermediate images are rounded to
    whole values like the uint8 images of the Python pipeline."""

    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder.eval()
        # the card's output pixel centres as unit-square coordinates, homogeneous (3, H*W)
        us = (torch.arange(CARD_W, dtype=torch.float32) / CARD_W)[None, :].expand(CARD_H, CARD_W)
        vs = (torch.arange(CARD_H, dtype=torch.float32) / CARD_H)[:, None].expand(CARD_H, CARD_W)
        self.register_buffer("uv1", torch.stack([us.flatten(), vs.flatten(), torch.ones(CARD_H * CARD_W)]))
        self.boxes = [
            tuple(int(v) for v in (x0 * CARD_W, y0 * CARD_H, x1 * CARD_W, y1 * CARD_H)) for x0, y0, x1, y1 in (frame_box(frame) for frame in FRAME_NAMES)
        ]

    def forward(self, scene: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
        img = scene[..., :3].to(torch.float32).permute(2, 0, 1)[None]  # (1, 3, H, W)
        h, w = scene.shape[0], scene.shape[1]
        xyw = square_to_quad(quad) @ self.uv1  # (3, H*W) scene coordinates of every card pixel
        xy = xyw[:2] / xyw[2:3]
        size = torch.stack([w, h]).to(torch.float32) - 1  # align_corners=True: -1 is pixel 0, +1 is pixel W-1
        grid = (2 * xy.T / size - 1).view(1, CARD_H, CARD_W, 2)
        card = nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        card = torch.round(card).clamp(0, 255)
        crops = []
        for frame, (x0, y0, x1, y1) in zip(FRAME_NAMES, self.boxes, strict=True):
            crop = card[:, :, y0:y1, x0:x1]
            # aten::rot90 is not exported at opset 17; transpose + flip is identical.
            for _ in range(FRAME_ROTATIONS.get(frame, 0)):
                crop = torch.flip(crop.transpose(2, 3), dims=(2,))
            crops.append(torch.round(nn.functional.interpolate(crop, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)))
        x = torch.cat(crops) / 255.0
        mean = torch.as_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.as_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return self.embedder((x - mean) / std)


class SearchGraph(nn.Module):
    """`index.frame_similarities` + top-k over a fixed gallery: every art is scored against the
    query cut for its own frame, minus the frame prior for rare frames."""

    def __init__(self, embeddings: np.ndarray, frames: np.ndarray, penalties: np.ndarray, k: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.register_buffer("gallery", torch.as_tensor(embeddings).to(dtype))
        self.register_buffer("frames", torch.as_tensor(frames, dtype=torch.int64)[:, None])
        self.register_buffer("penalties", torch.as_tensor(penalties, dtype=torch.float32))
        self.k = k

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sims = self.gallery.to(torch.float32) @ embeddings.T  # (N, F)
        scores = sims.gather(1, self.frames)[:, 0] - self.penalties
        top = torch.topk(scores, self.k)
        return top.indices, top.values
