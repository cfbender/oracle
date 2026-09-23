"""Pull human-labelled table clicks into data/real; create card.png from crop.jpg + quad.

    python -m cardid.corrections pull --server https://games.example.com
    python -m cardid.corrections pull --from-dir /mnt/gathering/cardid/corrections

HTTP uses CARDID_CORRECTIONS_TOKEN (read-only admin export capability). Never put it in argv.
The append cursor advances only after a whole page is imported; retrying is idempotent.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image

from . import DATA_DIR
from .detect import warp_card

ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
GALLERY_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?:-1)?\Z")
REAL = DATA_DIR / "real"


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n")
    tmp.replace(path)


def latest_labels(real: Path) -> dict[str, dict]:
    path = real / "labels.jsonl"
    if not path.exists():
        return {}
    return {row["capture_id"]: row for line in path.read_text().splitlines() if line.strip() for row in [json.loads(line)]}


def capture_id(row: dict) -> str:
    value = row["capture_id"]
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("invalid correction capture_id")
    label = row.get("label")
    if label is not None and (not isinstance(label, str) or not GALLERY_ID.fullmatch(label)):
        raise ValueError("invalid correction label")
    return value


def merge(row: dict, jpeg: bytes, real: Path) -> bool:
    """Last label wins, including skips. Unknown quads stay pending (no card.png).

    This confirms identity, not detector geometry/orientation. Do not invent up_correct.
    The ordered quad is used as supplied, with orientation=0 relative to that quad.
    """
    cid = capture_id(row)
    dest = real / cid
    source = dest / "correction.json"
    if source.exists() and json.loads(source.read_text()) == row:
        return False
    if len(jpeg) > 150_000:
        raise ValueError("oversized correction crop")
    with Image.open(io.BytesIO(jpeg)) as image:
        if image.format != "JPEG" or not (0 < image.width <= 640 and 0 < image.height <= 640):
            raise ValueError("correction must be a JPEG of at most 640 x 640")
        rgb = np.array(image.convert("RGB"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "crop.jpg").write_bytes(jpeg)
    split = "eval" if int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 5 == 0 else "train"
    label = {**row, "split": split, "source": "webcam-table", "quad_source": "detector", "orientation": 0}
    quad = np.asarray(row.get("quad"), dtype=np.float32)
    valid_quad = (
        quad.shape == (4, 2) and np.isfinite(quad).all() and np.abs(quad).max() <= 2048 and cv2.isContourConvex(quad) and abs(cv2.contourArea(quad)) > 16
    )
    card_path = dest / "card.png"
    if row.get("label") and valid_quad:
        card = warp_card(rgb, quad)
        if not cv2.imwrite(str(card_path), cv2.cvtColor(card, cv2.COLOR_RGB2BGR)):
            raise OSError(f"cannot write {card_path}")
        short = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
        label.update(card_px=round(float(short)), art_px=round(float(short) * 0.84))
    else:
        card_path.unlink(missing_ok=True)
        print(f"pending geometry or skipped: {cid}")
    with (real / "labels.jsonl").open("a") as f:
        f.write(json.dumps(label) + "\n")
    atomic_json(source, row)
    return True


def pull(real: Path = REAL, server: str | None = None, from_dir: Path | None = None) -> int:
    real.mkdir(parents=True, exist_ok=True)
    with (real / ".corrections.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if from_dir:
            return sum(merge(row, (from_dir / capture_id(row) / "crop.jpg").read_bytes(), real) for row in latest_labels(from_dir).values())
        if not server or not server.startswith("https://"):
            raise ValueError("--server / CARDID_SERVER must use HTTPS (or use --from-dir)")
        token = os.environ.get("CARDID_CORRECTIONS_TOKEN")
        if not token:
            raise ValueError("set CARDID_CORRECTIONS_TOKEN in the environment")
        server = server.rstrip("/")
        state_path = real / ".corrections-cursor.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        cursor = state.get(server, 0)
        count = 0
        with httpx.Client(headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=60) as client:
            while True:
                response = client.get(f"{server}/api/cardid/corrections", params={"cursor": cursor})
                response.raise_for_status()
                page = response.json()["data"]
                for row in page["corrections"]:
                    cid = capture_id(row)
                    response = client.get(f"{server}/api/cardid/corrections/{cid}/crop")
                    response.raise_for_status()
                    count += merge(row, response.content, real)
                cursor = page["cursor"]
                state[server] = cursor
                atomic_json(state_path, state)
                if not page["has_more"]:
                    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pull", "selftest"])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--server")
    source.add_argument("--from-dir", type=Path)
    parser.add_argument("--real-dir", type=Path, default=REAL)
    args = parser.parse_args()
    if args.server is None and args.from_dir is None:
        args.server = os.environ.get("CARDID_SERVER")
        directory = os.environ.get("CARDID_CORRECTIONS_DIR")
        args.from_dir = Path(directory) if directory else None
    if args.command == "selftest":
        import unittest

        from .test_corrections import CorrectionsTest

        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CorrectionsTest))
        raise SystemExit(not result.wasSuccessful())
    print(f"merged {pull(args.real_dir, args.server, args.from_dir)} new/relabelled corrections")


if __name__ == "__main__":
    main()
