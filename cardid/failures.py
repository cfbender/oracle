"""Contact sheet of top-1 misses: [degraded query | predicted art | true art] per row, plus
the names, so we can tell genuine confusions from near-duplicate illustrations.

    uv run python -m cardid.failures --checkpoint data/runs/m0/best.pt --n 24
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from . import DATA_DIR
from .data import cached_eval_queries, gallery_images, load_arts
from .degrade import INPUT_SIZE
from .evaluate import cosine_topk, embed_images
from .model import Embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--out", default=str(DATA_DIR / "failures.jpg"))
    args = parser.parse_args()
    torch.set_num_threads(4)

    arts = load_arts()
    gallery = gallery_images(arts)
    queries, targets, infos = cached_eval_queries(arts)
    model = Embedder(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    g = embed_images(model, gallery)
    q = embed_images(model, queries)
    idx, sims = cosine_topk(q, g, 5)
    miss = np.where(idx[:, 0] != targets)[0]
    rng = np.random.default_rng(0)
    picked = rng.choice(miss, size=min(args.n, len(miss)), replace=False)

    same_name = 0
    rows = []
    for i in picked:
        pred, true = arts[idx[i, 0]], arts[targets[i]]
        same_name += pred["name"] == true["name"]
        row = np.concatenate([queries[i], gallery[idx[i, 0]], gallery[targets[i]]], axis=1)
        row = cv2.copyMakeBorder(row, 0, 16, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        label = f"{infos[i]['width']}px  pred: {pred['name'][:22]} ({sims[i, 0]:.2f})  true: {true['name'][:22]}"
        cv2.putText(row, label, (2, INPUT_SIZE + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    Image.fromarray(sheet).save(args.out, quality=88)
    print(f"{len(miss)} misses of {len(targets)}; {same_name}/{len(picked)} sampled misses predicted a card with the same name")
    print(args.out)


if __name__ == "__main__":
    main()
