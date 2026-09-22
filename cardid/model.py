"""Art embedding model: MobileNetV3-Small backbone -> 128-d L2-normalized vector.

MobileNetV3 is chosen because it quantizes to int8 cleanly and compiles fully to the Coral
Edge TPU; the same weights also export to ONNX for an in-browser (onnxruntime-web) backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

EMBED_DIM = 128


def pick_device(name: str = "auto") -> torch.device:
    """Resolve the --device flag. ROCm builds expose AMD GPUs through the `cuda` device type, so
    `cuda` is the right spelling on the RX 9070 XT box as well as on NVIDIA."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False (see ml/README.md, GPU training)")
    return device


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)}, torch {torch.__version__})"
    return f"{device} ({torch.get_num_threads()} threads, torch {torch.__version__})"


class Embedder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # (N, 576, 4, 4) at 128px input
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(576, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return F.normalize(self.head(x), dim=1)


class PretrainedBaseline(nn.Module):
    """Untrained reference: raw ImageNet pooled features, L2-normalized."""

    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.pool(self.features(x)).flatten(1), dim=1)


class ArcFaceHead(nn.Module):
    """Normalized-softmax head with an additive angular margin over every training art.

    In-batch InfoNCE only sees ~127 negatives per sample and saturates within a few epochs;
    this head compares each embedding against all N train classes, so look-alike arts
    (basic lands, same-artist cycles) keep supplying gradient. Training-only: retrieval still
    uses the embedding, so the head is discarded at export and new sets need no retraining.
    """

    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = emb @ F.normalize(self.weight, dim=1).T
        target = cos.gather(1, labels[:, None])
        # cos(theta + m) for the target class, plain cos elsewhere
        sin = torch.sqrt((1 - target**2).clamp_min(1e-6))
        cos_m = target * torch.cos(torch.tensor(self.margin)) - sin * torch.sin(torch.tensor(self.margin))
        logits = cos.scatter(1, labels[:, None], cos_m)
        return F.cross_entropy(self.scale * logits, labels)


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE: row i of `a` must match row i of `b` against all other rows."""
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
