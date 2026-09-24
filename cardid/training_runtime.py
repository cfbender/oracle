"""Process setup shared by `train` and `train_detector`: device, worker/thread counts, seeding,
the training DataLoader and the run's metadata file.

Seeding covers Python's `random`, NumPy's global generator, torch (CPU and GPU) and the
DataLoader's shuffle/worker-seed generator. The datasets draw their augmentation from
per-sample generators keyed on (seed, epoch, index), so the trainers pass `--seed` into them
as well. Fixed evaluation sets (validation scenes, eval queries) keep their own constant
seeds so that runs with different `--seed` values stay comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import worker_init
from .model import describe_device, gpu, pick_device


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    workers: int
    threads: int
    seed: int

    @property
    def pin_memory(self) -> bool:
        return self.device.type == "cuda"


def add_runtime_args(parser: argparse.ArgumentParser, workers_help: str) -> None:
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cpu, or cuda (also AMD/ROCm)")
    parser.add_argument("--workers", type=int, help=workers_help)
    parser.add_argument("--threads", type=int, help="torch intra-op threads (default: the other half of the cores on CPU, 2 on GPU)")
    parser.add_argument("--seed", type=int, default=0, help="seed for Python, NumPy, torch, the DataLoader and the training augmentation (default 0)")


def default_counts(device: torch.device, cores: int | None = None) -> tuple[int, int]:
    """(workers, threads). Augmentation (worker processes) and the model's forward/backward
    (torch threads) run concurrently. On CPU they share the cores, so split them; on GPU the
    model needs almost no CPU and augmentation is the bottleneck, so it gets nearly everything."""
    cores = cores or os.cpu_count() or 8
    if gpu(device):
        return max(2, cores - 1), 2
    return max(2, cores // 2), max(2, cores - cores // 2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # also seeds every CUDA/ROCm device


def setup(args: argparse.Namespace, role: str) -> Runtime:
    """Resolve device, counts and seeds from the parsed runtime flags; `role` names the
    workers in the log line ("augmentation", "rendering")."""
    # The DataLoader forks its workers after the parent may have used OpenCV (loading real
    # captures). OpenCV's thread pool does not survive fork() and the children deadlock, so
    # keep the parent's OpenCV single-threaded; torch does the parent's heavy lifting anyway.
    cv2.setNumThreads(0)
    device = pick_device(args.device)
    workers, threads = default_counts(device)
    runtime = Runtime(device=device, workers=args.workers or workers, threads=args.threads or threads, seed=args.seed)
    torch.set_num_threads(runtime.threads)
    seed_everything(runtime.seed)
    print(f"device: {describe_device(device)}, {runtime.workers} {role} workers, seed {runtime.seed}")
    return runtime


def make_loader(dataset: Dataset, batch_size: int, runtime: Runtime, pin_memory: bool | None = None) -> DataLoader:
    """The shuffled training loader: persistent single-threaded workers, full batches, and a
    seeded generator so the shuffle order and per-worker seeds follow `--seed`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=runtime.workers,
        worker_init_fn=worker_init,
        drop_last=True,
        persistent_workers=runtime.workers > 0,
        pin_memory=runtime.pin_memory if pin_memory is None else pin_memory,
        generator=torch.Generator().manual_seed(runtime.seed),
    )


def run_metadata(args: argparse.Namespace, runtime: Runtime) -> dict:
    import torchvision

    return {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seed": runtime.seed,
        "device": str(runtime.device),
        "workers": runtime.workers,
        "threads": runtime.threads,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }


def write_run_metadata(run_dir: Path, args: argparse.Namespace, runtime: Runtime) -> Path:
    """`<run>/run.json`: what produced this run's checkpoints (flags, seed, library versions)."""
    path = run_dir / "run.json"
    path.write_text(json.dumps(run_metadata(args, runtime), indent=2, default=str) + "\n")
    return path
