"""Train the card-corner detector on rendered scenes, optionally mixed with real captures.

    uv run python -m cardid.train_detector --epochs 10 --samples 20000 --batch 64 --run det
    uv run python -m cardid.train_detector --resume data/runs/det/best.pt --real --epochs 4 --run det-real

Per epoch it reports the corner error on a fixed synthetic validation set and, when labeled
real captures exist, on the held-out real split (against the quads the identification used,
which are only as good as the classical finder or the user's box). Error is the mean corner
distance as a fraction of the card's short side; "hit" is error < 5%, inside the recogniser's
crop-jitter tolerance. `real_e2e` runs the actual two-stage `Detector.locate` on the stored
crops. Checkpoints to data/runs/<run>/{last,best}.pt.
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

from . import DATA_DIR, RUNS_DIR
from .data import worker_init
from .detector import (
    CornerNet,
    Detector,
    corner_error,
    corner_loss,
    heat_loss,
    heat_targets,
    load_checkpoint,
    pose_loss,
    quad_to_pose,
    snap_corners,
)
from .model import describe_device, gpu, pick_device
from .real import REAL_DIR, load_labels
from .synth import DET_INPUT, RealSceneDataset, SceneDataset, batch_to_input, quad_short

HIT = 0.05


def val_scenes(n: int, workers: int, seed: int = 999) -> tuple[np.ndarray, np.ndarray]:
    """Fixed synthetic validation scenes (uint8) and quads (px), cached on disk."""
    cache = DATA_DIR / f"det-val-{n}-{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["scenes"], z["quads"]
    loader = DataLoader(SceneDataset(n, seed=seed, raw=True), batch_size=32, num_workers=workers, worker_init_fn=worker_init)
    scenes, quads = [], []
    for s, q in tqdm(loader, desc="val scenes", leave=False):
        scenes.append(s.numpy())
        quads.append(q.numpy())
    scenes, quads = np.concatenate(scenes), np.concatenate(quads)
    np.savez(cache, scenes=scenes, quads=quads)
    return scenes, quads


def summarize(pred: np.ndarray, target: np.ndarray) -> dict:
    """Relative corner error stats; `pred`/`target` are (N, 4, 2) in the same pixel units."""
    short = np.array([quad_short(q) for q in target])
    rel = corner_error(pred, target) / short
    return {"err": round(float(np.median(rel)), 4), "err_mean": round(float(rel.mean()), 4), "hit": round(float((rel < HIT).mean()), 3)}


@torch.no_grad()
def predict_scenes(model: CornerNet, scenes: np.ndarray, device: torch.device, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """(snapped, raw pose) corner predictions in input px for a stack of 256px scenes."""
    model.eval()
    snapped, raw = [], []
    for i in range(0, len(scenes), batch):
        x = batch_to_input(torch.from_numpy(scenes[i : i + batch]).to(device))
        quad, _, _, heat = model(x)
        quad = quad.cpu().numpy() * DET_INPUT
        raw.append(quad)
        snapped.append(snap_corners(quad, torch.sigmoid(heat)[:, 0].cpu().numpy()))
    return np.concatenate(snapped), np.concatenate(raw)


def eval_scenes(model: CornerNet, scenes: np.ndarray, quads: np.ndarray, device: torch.device, batch: int = 64) -> dict:
    """{"synth": snapped error, "synth_pose": raw pose-head error} on the fixed validation set."""
    snapped, raw = predict_scenes(model, scenes, device, batch)
    return {"synth": summarize(snapped, quads), "synth_pose": summarize(raw, quads)}


@torch.no_grad()
def eval_real(model: CornerNet, dataset: RealSceneDataset, device: torch.device) -> dict:
    """Single-pass error on the click windows, plus the end-to-end two-stage locate on the
    stored crops (what capture.py would do)."""
    model.eval()
    scenes, quads = zip(*(dataset.sample(i, None) for i in range(len(dataset))))
    single = summarize(predict_scenes(model, np.stack(scenes), device)[0], np.stack(quads))
    det = Detector(model=model, device=device)
    preds, targets = [], []
    for r in dataset.rows:
        img = cv2.cvtColor(cv2.imread(str(REAL_DIR / r["capture_id"] / "crop.jpg")), cv2.COLOR_BGR2RGB)
        click = r.get("click") or (img.shape[1] / 2, img.shape[0] / 2)
        preds.append(det.locate(img, tuple(click)))
        targets.append(np.float32(r["quad"]))
    e2e = summarize(np.stack(preds), np.stack(targets))
    return {"real": single, "real_e2e": e2e}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="det")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20_000, help="rendered scenes per epoch")
    parser.add_argument("--val", type=int, default=1000, help="fixed synthetic validation scenes")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--residual-weight", type=float, default=0.5)
    parser.add_argument("--pose-weight", type=float, default=1.0, help="weight of the direct pose loss (centre, log size, angle vector) next to the corner loss")
    parser.add_argument("--heat-weight", type=float, default=0.2, help="weight of the corner heatmap focal loss")
    parser.add_argument("--device", default="auto", help="auto (GPU if available), cpu, or cuda (also AMD/ROCm)")
    parser.add_argument("--workers", type=int, help="scene-rendering worker processes (default: half the logical CPUs on CPU, all but one on GPU; on SMT machines one per physical core is usually faster, see bench_loader --detector)")
    parser.add_argument("--threads", type=int, help="torch intra-op threads (default: the other half of the cores on CPU, 2 on GPU)")
    parser.add_argument("--no-pin", action="store_true", help="do not stage batches in pinned host memory (try if bench_loader shows the loader capped regardless of workers)")
    parser.add_argument("--resume")
    parser.add_argument("--real", action="store_true", help="mix in the train split of labeled real captures from data/real")
    parser.add_argument("--real-repeat", type=int, default=20, help="how many times each real capture appears per epoch")
    args = parser.parse_args()
    cv2.setNumThreads(0)
    device = pick_device(args.device)
    cores = os.cpu_count() or 8
    if gpu(device):
        workers, threads = max(2, cores - 1), 2
    else:
        workers, threads = max(2, cores // 2), max(2, cores - cores // 2)
    args.workers = args.workers or workers
    args.threads = args.threads or threads
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    print(f"device: {describe_device(device)}, {args.workers} rendering workers")

    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    synth = SceneDataset(args.samples)
    train_set = synth
    real_eval = None
    real_sets = []
    eval_rows = [r for r in load_labels("eval") if r.get("quad")]
    if eval_rows:
        real_eval = RealSceneDataset(eval_rows, augment=False)
    if args.real:
        real_train = RealSceneDataset(load_labels("train"), repeat=args.real_repeat)
        if not len(real_train):
            raise SystemExit("--real: no labeled captures with quads in data/real")
        real_sets.append(real_train)
        print(f"real captures: {len(real_train.rows)} train x{args.real_repeat}, {len(eval_rows)} eval")
        train_set = ConcatDataset([synth, real_train])
    loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        drop_last=True,
        persistent_workers=True,
        pin_memory=device.type == "cuda" and not args.no_pin,
    )
    scenes, quads = val_scenes(args.val, args.workers)

    model = CornerNet(pretrained=args.resume is None).to(device)
    if args.resume:
        load_checkpoint(model, args.resume, device)
    opt = torch.optim.AdamW(
        [{"params": model.backbone_parameters(), "lr": args.backbone_lr}, {"params": model.head_parameters(), "lr": args.lr}],
        weight_decay=1e-4,
    )
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.backbone_lr, args.lr], total_steps=steps, pct_start=0.15)

    def evaluate() -> dict:
        out = eval_scenes(model, scenes, quads, device)
        if real_eval is not None:
            out.update(eval_real(model, real_eval, device))
        model.train()
        return out

    def score(m: dict) -> float:
        # choose checkpoints by the real end-to-end hit rate once real data drives training
        key = "real_e2e" if args.real and "real_e2e" in m else "synth"
        return m[key]["hit"] - m[key]["err"]

    metrics = evaluate()
    best = score(metrics)
    print(f"start: {json.dumps(metrics)}")
    history = []
    for epoch in range(args.epochs):
        for ds in [synth, *real_sets]:
            ds.set_epoch(epoch)
        model.train()
        t0, losses, res_mag = time.time(), [], []
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)
        for x, target in bar:
            target_pose = quad_to_pose(target).to(device, non_blocking=True)  # fitted on the CPU copy, before the transfer
            x, target = batch_to_input(x.to(device, non_blocking=True)), target.to(device, non_blocking=True)
            pred, residual, pose, heat = model(x)
            loss = (
                corner_loss(pred, target, residual, args.residual_weight)
                + args.pose_weight * pose_loss(pose, target_pose)
                + args.heat_weight * heat_loss(heat, heat_targets(target))
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            res_mag.append(residual.abs().mean().item())
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")
        bar.close()
        train_seconds = time.time() - t0
        metrics = evaluate()
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "residual": round(float(np.mean(res_mag)), 4), **metrics, "seconds": round(train_seconds, 1)}
        history.append(entry)
        print(json.dumps(entry))
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, run_dir / "last.pt")
        if score(metrics) >= best:
            best = score(metrics)
            torch.save(state, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
