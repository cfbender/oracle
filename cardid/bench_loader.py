"""Where does an epoch's time go? Times the three parts of `cardid.train` in isolation:
the augmentation DataLoader alone, the model's forward/backward on synthetic batches, and
the post-epoch `quick_eval` (which is included in the epoch `seconds` train.py reports).

    uv run python -m cardid.bench_loader --workers 15 8 --batch 256
    uv run python -m cardid.bench_loader --detector --workers 15 8 --batch 64   # cardid.train_detector's parts

Must be a module (not a stdin script): Python 3.14 starts DataLoader workers with the
`forkserver` method, which re-imports `__main__` from its file path.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from .data import (
    PairDataset,
    cached_eval_queries,
    gallery_images,
    load_arts,
    split,
    worker_init,
)
from .degrade import INPUT_SIZE
from .detector import (
    CornerNet,
    corner_loss,
    heat_loss,
    heat_targets,
    pose_loss,
    quad_to_pose,
    up_loss,
    up_targets,
)
from .evaluate import embed_images
from .model import Embedder, describe_device, gpu, info_nce, pick_device, sync
from .synth import DET_INPUT, SceneDataset


def bench_loader(dataset: Dataset, workers: int, batch: int, batches: int, pin: bool, single_ms: float | None = None) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=worker_init,
        drop_last=True,
        persistent_workers=True,
        pin_memory=pin,
    )
    it = iter(loader)
    # Start the workers, then drain what they prefetched (2 batches each) so the timed batches
    # measure steady-state production rather than the queue emptying.
    for _ in range(2 * workers + 1):
        next(it)
    t0 = time.perf_counter()
    for _ in range(batches):
        next(it)
    dt = time.perf_counter() - t0
    per_worker_ms = dt / (batches * batch) * workers * 1000
    note = f", {per_worker_ms:5.1f} ms/sample/worker ({single_ms / per_worker_ms:.0%} of single-thread speed)" if single_ms else ""
    print(f"loader, {workers:2d} workers: {batches / dt:5.2f} batch/s, {batches * batch / dt:6.0f} samples/s{note}")
    del it, loader


def bench_model(device: torch.device, batch: int, steps: int, temperature: float) -> None:
    model = Embedder(pretrained=False).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    clean = torch.randn(batch, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    degraded = torch.randn(batch, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    def step() -> None:
        loss = info_nce(model(clean), model(degraded), temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync(device)
    dt = time.perf_counter() - t0
    print(f"model fwd/bwd on {describe_device(device)}: {steps / dt:5.2f} batch/s, {steps * batch / dt:6.0f} samples/s")
    return model


def bench_detector_model(device: torch.device, batch: int, steps: int) -> None:
    model = CornerNet(pretrained=False).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, DET_INPUT, DET_INPUT, device=device)
    target = torch.rand(batch, 4, 2, device=device)

    def step() -> None:
        pred, res, pose, heat, up = model(x)
        loss = corner_loss(pred, target, res) + pose_loss(pose, quad_to_pose(target)) + 0.2 * heat_loss(heat, heat_targets(target)) + up_loss(up, up_targets(target))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync(device)
    dt = time.perf_counter() - t0
    print(f"CornerNet fwd/bwd on {describe_device(device)}: {steps / dt:5.2f} batch/s, {steps * batch / dt:6.0f} samples/s")


def _render_worker(args: tuple[int, int]) -> float:
    """Render `n` scenes in this process; returns the mean ms per scene."""
    seed, n = args
    worker_init(0)
    dataset = SceneDataset(n, seed=seed)
    dataset[0]
    t0 = time.perf_counter()
    for i in range(1, n):
        dataset[i]
    return (time.perf_counter() - t0) / (n - 1) * 1000


def bench_render_parallel(workers: int, n: int, single_ms: float) -> None:
    """`workers` plain processes rendering concurrently with no DataLoader and no IPC, so the
    per-process time shows what the CPU (clocks, shared cache, memory bandwidth) does under
    load, separately from what the loader path costs on top."""
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers) as pool:
        per_proc = pool.map(_render_worker, [(1000 + w, n) for w in range(workers)])
    in_proc = sum(per_proc) / len(per_proc)
    print(
        f"plain {workers:2d} processes: {workers * 1000 / in_proc:6.0f} samples/s, "
        f"{in_proc:5.1f} ms/scene inside each process ({single_ms / in_proc:.0%} of single-thread speed)"
    )


def bench_scene_render(n: int) -> float:
    """Single-process render cost in ms, the number the per-worker throughput should approach.
    When more workers give *less* throughput, the logical CPUs are SMT siblings of busy cores;
    pass --workers around the physical core count to train_detector."""
    dataset = SceneDataset(n, seed=123)
    dataset[0]
    t0 = time.perf_counter()
    for i in range(1, n):
        dataset[i]
    ms = (time.perf_counter() - t0) / (n - 1) * 1000
    print(f"render_scene single-thread: {ms:5.1f} ms/scene")
    return ms


def bench_eval(model: Embedder, arts: list[dict]) -> None:
    gallery = gallery_images(arts)
    queries, _, _ = cached_eval_queries(arts)
    model.eval()
    t0 = time.perf_counter()
    embed_images(model, gallery)
    t_gallery = time.perf_counter() - t0
    t0 = time.perf_counter()
    embed_images(model, queries)
    t_queries = time.perf_counter() - t0
    print(f"quick_eval: gallery {len(gallery)} imgs {t_gallery:.1f} s, queries {len(queries)} imgs {t_queries:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[max(2, (os.cpu_count() or 8) - 1)])
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--batches", type=int, default=30, help="timed batches per loader configuration")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--skip-loader", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--detector", action="store_true", help="time cardid.train_detector's scene renderer and CornerNet instead")
    parser.add_argument("--no-pin", action="store_true", help="skip pinned host memory staging, as train_detector --no-pin does")
    args = parser.parse_args()
    cv2.setNumThreads(0)
    device = pick_device(args.device)
    torch.set_num_threads(2 if gpu(device) else max(2, (os.cpu_count() or 8) // 2))

    if args.detector:
        if not args.skip_loader:
            single_ms = bench_scene_render(40)
            for workers in args.workers:
                bench_render_parallel(workers, 40, single_ms)
            for workers in args.workers:
                bench_loader(SceneDataset(10**6, seed=7), workers, args.batch, args.batches, pin=device.type == "cuda" and not args.no_pin, single_ms=single_ms)
        if not args.skip_model:
            bench_detector_model(device, args.batch, 20)
        return

    arts = load_arts()
    dataset = PairDataset(split(arts, "train"))
    print(f"{len(dataset)} train arts, batch {args.batch}, {len(dataset) // args.batch} batches/epoch")

    if not args.skip_loader:
        for workers in args.workers:
            bench_loader(dataset, workers, args.batch, args.batches, pin=device.type == "cuda")
    model = None
    if not args.skip_model:
        model = bench_model(device, args.batch, 20, args.temperature)
    if not args.skip_eval:
        bench_eval(model or Embedder(pretrained=False).to(device), arts)


if __name__ == "__main__":
    main()
