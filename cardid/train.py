"""Fine-tune the embedder with symmetric InfoNCE on (clean, degraded) pairs of train-split arts.

    uv run python -m cardid.train --epochs 12 --batch 128 --run m0
    uv run python -m cardid.train --resume data/runs/full/best.pt --real --epochs 4 --run full-real

`--real` mixes the train split of real webcam captures labeled with `cardid.capture` into
each epoch (oversampled `--real-repeat` times, lightly augmented), and the best checkpoint is
then chosen by top-1 on usable held-out real captures, or synthetic queries if none exist.

Checkpoints to data/runs/<run>/{last,best}.pt; "best" is by eval top-1 on a fixed query set
drawn from the eval split (unseen arts), which is also what evaluate.py reports.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from . import RUNS_DIR
from .data import PairDataset, art_frames, cached_eval_queries, gallery_images, load_arts, split, worker_init
from .evaluate import cosine_topk, embed_images, frame_topk
from .gallery import printing_index
from .model import ArcFaceHead, Embedder, describe_device, gpu, info_nce, pick_device
from .real import RealDataset, load_labels, real_eval_queries


def quick_eval(model: Embedder, gallery: np.ndarray, queries: np.ndarray, targets: np.ndarray, frames: np.ndarray | None = None) -> float:
    """Top-1 over the gallery. Synthetic queries are one modern cut each (N x H x W x 3); real
    captures carry every frame's cut (N x F x H x W x 3) and score each art against the cut
    for its frame (`frames`), exactly as `evaluate` and `ArtIndex.search` do."""
    g = embed_images(model, gallery)
    q = embed_images(model, queries.reshape(-1, *queries.shape[-3:]))
    if queries.ndim == 5:
        idx, _ = frame_topk(q.reshape(*queries.shape[:2], -1), g, frames, 5)
    else:
        idx, _ = cosine_topk(q, g, 5)
    model.train()
    return float((idx[:, 0] == targets).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="m0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cpu, or cuda (also AMD/ROCm)")
    parser.add_argument("--workers", type=int, help="augmentation worker processes (default: half the cores on CPU, all but one on GPU)")
    parser.add_argument("--threads", type=int, help="torch intra-op threads (default: the other half of the cores on CPU, 2 on GPU)")
    parser.add_argument("--arcface", type=float, default=0.0, help="weight of the ArcFace class loss (0 disables)")
    parser.add_argument("--resume")
    parser.add_argument("--real", action="store_true", help="mix in labeled real captures from data/real")
    parser.add_argument("--real-repeat", type=int, default=20, help="how many times each real capture appears per epoch")
    args = parser.parse_args()
    # The DataLoader forks its workers after the parent may have used OpenCV (loading real
    # captures). OpenCV's thread pool does not survive fork() and the children deadlock, so
    # keep the parent's OpenCV single-threaded; torch does the parent's heavy lifting anyway.
    cv2.setNumThreads(0)
    if args.real and args.arcface > 0:
        parser.error("--real cannot be combined with --arcface (real labels may fall outside the train split)")
    device = pick_device(args.device)
    # Augmentation (worker processes) and the model's forward/backward (torch threads) run
    # concurrently. On CPU they share the cores, so split them; on GPU the model needs almost
    # no CPU and augmentation is the bottleneck, so it gets nearly everything.
    cores = os.cpu_count() or 8
    if gpu(device):
        workers, threads = max(2, cores - 1), 2
    else:
        workers, threads = max(2, cores // 2), max(2, cores - cores // 2)
    args.workers = args.workers or workers
    args.threads = args.threads or threads
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    print(f"device: {describe_device(device)}, {args.workers} augmentation workers")

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    arts = load_arts()
    train_arts = split(arts, "train")
    dataset = PairDataset(train_arts)
    train_set = dataset
    real_sets = []
    if args.real:
        # Labels are only consumed by ArcFace, which --real excludes, so index over every art.
        real_train = RealDataset(load_labels("train"), arts, repeat=args.real_repeat)
        real_sets.append(real_train)
        print(f"real captures: {len(real_train.rows)} train x{args.real_repeat}, {len(load_labels('eval'))} eval")
        train_set = ConcatDataset([dataset, real_train])
    loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        drop_last=True,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
    )

    gallery = gallery_images(arts)
    queries, targets, _ = cached_eval_queries(arts)
    # Subsample queries during training so each epoch's eval is cheap.
    sel = np.random.default_rng(0).choice(len(queries), size=min(1000, len(queries)), replace=False)
    queries, targets = queries[sel], targets[sel]
    frames = None
    if args.real:
        gallery_index = printing_index(arts)
        real_eval = [r for r in load_labels("eval") if r["label"] in gallery_index]
        if real_eval:
            queries, targets, _ = real_eval_queries(real_eval, gallery_index)
            frames = art_frames(arts)
            print(f"selecting best checkpoint by top-1 on {len(queries)} held-out real captures")
        else:
            print("no usable held-out real captures; selecting best checkpoint by synthetic top-1")

    model = Embedder()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    model.to(device)
    arc = ArcFaceHead(len(train_arts)).to(device) if args.arcface > 0 else None
    params = [
        {"params": model.features.parameters(), "lr": args.backbone_lr},
        {"params": model.head.parameters(), "lr": args.lr},
    ]
    if arc is not None:
        params.append({"params": arc.parameters(), "lr": args.lr * 10})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], total_steps=steps, pct_start=0.1)

    best = quick_eval(model, gallery, queries, targets, frames)
    print(f"start: top1={best:.3f} (untrained head)")
    history = []
    for epoch in range(args.epochs):
        for ds in [dataset, *real_sets]:
            ds.set_epoch(epoch)
        model.train()
        t0, losses = time.time(), []
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)
        for clean, degraded, labels in bar:
            a = model(clean.to(device, non_blocking=True))
            b = model(degraded.to(device, non_blocking=True))
            loss = info_nce(a, b, args.temperature)
            if arc is not None:
                labels = labels.to(device, non_blocking=True)
                loss = loss + args.arcface * (arc(a, labels) + arc(b, labels)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.3f}")
        bar.close()
        top1 = quick_eval(model, gallery, queries, targets, frames)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "top1": top1, "seconds": time.time() - t0})
        print(json.dumps(history[-1]))
        # Checkpoints are consumed on CPU (capture, bench, export), so store CPU tensors.
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, run_dir / "last.pt")
        if top1 >= best:
            best = top1
            torch.save(state, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best top1={best:.3f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
