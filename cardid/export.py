"""Export a trained embedder for the two runtime backends.

    uv run python -m cardid.export --checkpoint data/runs/m0/best.pt

ONNX (written here): loadable by onnxruntime-web in the browser and by Ortex/onnxruntime on the
server. Input: float32 NCHW 1x3x128x128 normalized with ImageNet mean/std. Output: 1x128 L2-normed.

TFLite int8 for the Coral Edge TPU is a separate step that needs `ai-edge-torch` (PyTorch ->
TFLite with post-training quantization) and then `edgetpu_compiler`; both are heavy, x86-only
toolchains, so they are documented in ml/README.md rather than pulled into this environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .degrade import INPUT_SIZE
from .model import Embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    model = Embedder(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    out = Path(args.out or Path(args.checkpoint).with_suffix(".onnx"))
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(model, x, str(out), input_names=["image"], output_names=["embedding"], opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = model(x).numpy()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); sample embedding norm {np.linalg.norm(ref):.3f}")


if __name__ == "__main__":
    main()
