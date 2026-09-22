"""Fine-tune the embedder with symmetric InfoNCE on (clean, degraded) pairs of train-split arts.

    uv run python -m cardid.train --epochs 12 --batch 128 --run m0

Checkpoints to data/runs/<run>/{last,best}.pt; "best" is by eval top-1 on a fixed query set
drawn from the eval split (unseen arts), which is also what evaluate.py reports.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import RUNS_DIR
from .data import PairDataset, cached_eval_queries, gallery_images, load_arts, split
from .evaluate import cosine_topk, embed_images
from .model import ArcFaceHead, Embedder, info_nce


def quick_eval(model: Embedder, gallery: np.ndarray, queries: np.ndarray, targets: np.ndarray) -> float:
    g = embed_images(model, gallery)
    q = embed_images(model, queries)
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
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2), help="augmentation worker processes")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8, help="torch intra-op threads")
    parser.add_argument("--arcface", type=float, default=0.0, help="weight of the ArcFace class loss (0 disables)")
    parser.add_argument("--resume")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    arts = load_arts()
    train_arts = split(arts, "train")
    dataset = PairDataset(train_arts)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True)

    gallery = gallery_images(arts)
    queries, targets, _ = cached_eval_queries(arts)
    # Subsample queries during training so each epoch's eval is cheap.
    sel = np.random.default_rng(0).choice(len(queries), size=min(1000, len(queries)), replace=False)
    queries, targets = queries[sel], targets[sel]

    model = Embedder()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    arc = ArcFaceHead(len(train_arts)) if args.arcface > 0 else None
    params = [
        {"params": model.features.parameters(), "lr": args.backbone_lr},
        {"params": model.head.parameters(), "lr": args.lr},
    ]
    if arc is not None:
        params.append({"params": arc.parameters(), "lr": args.lr * 10})
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], total_steps=steps, pct_start=0.1)

    best = quick_eval(model, gallery, queries, targets)
    print(f"start: top1={best:.3f} (untrained head)")
    history = []
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        t0, losses = time.time(), []
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)
        for clean, degraded, labels in bar:
            a = model(clean)
            b = model(degraded)
            loss = info_nce(a, b, args.temperature)
            if arc is not None:
                loss = loss + args.arcface * (arc(a, labels) + arc(b, labels)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.3f}")
        bar.close()
        top1 = quick_eval(model, gallery, queries, targets)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "top1": top1, "seconds": time.time() - t0})
        print(json.dumps(history[-1]))
        torch.save(model.state_dict(), run_dir / "last.pt")
        if top1 > best:
            best = top1
            torch.save(model.state_dict(), run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best top1={best:.3f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
