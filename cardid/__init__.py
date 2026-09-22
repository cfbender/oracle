"""Card recognition spike for The Gathering's webcam table (M0)."""

import os
from pathlib import Path

# AMD's "Install PyTorch for ROCm" known issues: training workloads on the Radeon RX 9070
# series (gfx1201) "might experience GPU resets or application crashes"; the documented
# workaround is disabling hipBLASLt (at some matmul performance cost). torch reads this at
# import, and every `python -m cardid.*` entry point imports this package before torch.
# Export TORCH_BLAS_PREFER_HIPBLASLT=1 to override. No effect on CPU or NVIDIA builds.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data"
ART_DIR = DATA_DIR / "art"
CARD_DIR = DATA_DIR / "cards"  # full-card `normal` images, for rendering detector scenes
CACHE_DIR = DATA_DIR / "cache"  # decoded image banks for the scene renderer (rebuilt on demand)
RUNS_DIR = DATA_DIR / "runs"
