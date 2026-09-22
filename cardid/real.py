"""Real webcam captures labeled through `cardid.capture`, and how training/eval consume them.

Layout under data/real/:
    labels.jsonl                 one line per labeled click (see `save_label`)
    <capture_id>/crop.jpg        the browser's full-resolution crop around the click
    <capture_id>/card.png        the card warped to the canonical 250x350 portrait card

The card warp is stored rather than the 128 px art crop so training can jitter the art box
the way the synthetic pipeline does (simulating detector misalignment) and so ART_BOX can be
tuned later without recapturing.
"""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
from torch.utils.data import Dataset

from . import DATA_DIR
from .data import art_path, to_tensor
from .degrade import INPUT_SIZE, clean_view, load_rgb
from .detect import ART_BOX, CARD_H, CARD_W

REAL_DIR = DATA_DIR / "real"
LABELS = REAL_DIR / "labels.jsonl"


def split_for(capture_id: str) -> str:
    """Deterministic 80/20 split by capture id, so re-labeling never moves a sample."""
    return "eval" if int(hashlib.sha1(capture_id.encode()).hexdigest(), 16) % 5 == 0 else "train"


def load_labels(split: str | None = None) -> list[dict]:
    if not LABELS.exists():
        return []
    rows = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    # A capture may be re-labeled; the last line wins. Skips (label null) are kept out.
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["capture_id"]] = r
    out = [r for r in latest.values() if r["label"] and (REAL_DIR / r["capture_id"] / "card.png").exists()]
    if split:
        out = [r for r in out if r["split"] == split]
    return out


def save_label(capture_id: str, row: dict, crop_rgb: np.ndarray | None = None, card_rgb: np.ndarray | None = None) -> dict:
    d = REAL_DIR / capture_id
    d.mkdir(parents=True, exist_ok=True)
    if crop_rgb is not None:
        cv2.imwrite(str(d / "crop.jpg"), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if card_rgb is not None:
        cv2.imwrite(str(d / "card.png"), cv2.cvtColor(card_rgb, cv2.COLOR_RGB2BGR))
    row = {"capture_id": capture_id, "split": split_for(capture_id), **row}
    with LABELS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def art_from_card(card: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Cut the art box out of a canonical card. With `rng`, jitter the box like a slightly
    misaligned detector (same ranges as the synthetic pipeline's mild case) and vary the
    photometrics a little. No resolution loss: the capture already has the real camera's."""
    x0, y0, x1, y1 = ART_BOX
    box = np.float32([[x0 * CARD_W, y0 * CARD_H], [x1 * CARD_W, y0 * CARD_H], [x1 * CARD_W, y1 * CARD_H], [x0 * CARD_W, y1 * CARD_H]])
    if rng is not None:
        c = box.mean(axis=0)
        box = (box - c) * rng.uniform(0.92, 1.08) + c + rng.uniform(-0.05, 0.05, size=2) * np.float32([CARD_W, CARD_H])
        box += rng.uniform(-0.02, 0.02, size=(4, 2)) * np.float32([CARD_W, CARD_H])
        rot = cv2.getRotationMatrix2D(tuple(c), rng.uniform(-3, 3), 1.0)
        box = (rot @ np.c_[box, np.ones(4)].T).T.astype(np.float32)
    dst = np.float32([[0, 0], [INPUT_SIZE, 0], [INPUT_SIZE, INPUT_SIZE], [0, INPUT_SIZE]])
    M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    art = cv2.warpPerspective(card, M, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng is not None:
        art = art.astype(np.float32) * rng.uniform(0.9, 1.1, size=3).astype(np.float32)
        art = (art - 128) * rng.uniform(0.85, 1.15) + 128 + rng.uniform(-15, 15)
        art = np.clip(art, 0, 255).astype(np.uint8)
    return art


class RealDataset(Dataset):
    """(clean gallery view of the labeled art, lightly augmented real art crop, art index).

    `repeat` oversamples the (small) real set so it is a meaningful share of each epoch.
    """

    def __init__(self, rows: list[dict], art_index: dict[str, int], repeat: int = 1, seed: int = 1):
        self.rows = [r for r in rows if r["label"] in art_index]
        self.art_index = art_index
        self.repeat = repeat
        self.seed = seed
        self.epoch = 0
        self.cards = [load_rgb(REAL_DIR / r["capture_id"] / "card.png") for r in self.rows]
        self.cleans = {r["label"]: clean_view(load_rgb(art_path({"id": r["label"]}))) for r in self.rows}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows) * self.repeat

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        row = self.rows[i % len(self.rows)]
        real = art_from_card(self.cards[i % len(self.rows)], rng)
        return to_tensor(self.cleans[row["label"]]), to_tensor(real), self.art_index[row["label"]]


def real_detector_queries(rows: list[dict], gallery_index: dict[str, int], locate) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Like `real_eval_queries`, but re-locate the card in the stored click crop with
    `locate(crop_rgb, click) -> quad` instead of using the stored quad, so a detector can be
    scored by the identification accuracy it produces. Returns two queries per capture (the
    card and its 180-degree rotation, since the detector does not know which way is up);
    the caller keeps whichever the recogniser is more confident about."""
    from .detect import art_crop, card_orientations, warp_card

    images, targets, infos = [], [], []
    for r in rows:
        if r["label"] not in gallery_index:
            continue
        crop = load_rgb(REAL_DIR / r["capture_id"] / "crop.jpg")
        click = tuple(r.get("click") or (crop.shape[1] / 2, crop.shape[0] / 2))
        quad = locate(crop, click)
        for card in card_orientations(warp_card(crop, quad)):
            images.append(art_crop(card))  # the same cut capture.py makes
        targets.append(gallery_index[r["label"]])
        infos.append({"width": int(r.get("art_px", 0)), "strong_perspective": False})
    if not images:
        raise SystemExit(f"no labeled real captures with crops in {LABELS}")
    return np.stack(images), np.array(targets), infos


def real_eval_queries(rows: list[dict], gallery_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Held-out real captures as eval queries, with the same info keys `evaluate.report` uses."""
    images, targets, infos = [], [], []
    for r in rows:
        if r["label"] not in gallery_index:
            continue
        card = load_rgb(REAL_DIR / r["capture_id"] / "card.png")
        images.append(art_from_card(card))
        targets.append(gallery_index[r["label"]])
        infos.append({"width": int(r.get("art_px", 0)), "strong_perspective": False})
    if not images:
        raise SystemExit(f"no labeled real captures in {LABELS}")
    return np.stack(images), np.array(targets), infos
