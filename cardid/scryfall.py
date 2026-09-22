"""Fetch Scryfall unique-artwork metadata and download a deterministic sample of art crops.

Usage:
    uv run python -m cardid.scryfall --train 5000 --eval 1000   # 6k sample (~25 min at 10 req/s)
    uv run python -m cardid.scryfall --all                       # then everything else as train

Writes:
    data/unique-artwork.jsonl.gz   raw bulk file
    data/arts.json                 sampled entries: [{id, name, set, split, url}]
    data/art/<id>.jpg              art_crop images
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

from . import ART_DIR, DATA_DIR

BULK_URL = "https://api.scryfall.com/bulk-data"
HEADERS = {
    "User-Agent": "TheGathering-CardID-Spike/0.1 (github.com/cfbender/the-gathering)",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}
# Scryfall asks for 50-100ms between requests (<10 req/s). Each worker sleeps WORKERS * 0.1s
# after its request so the pool as a whole stays at ~10 req/s.
WORKERS = 8
REQUEST_GAP_S = 0.1 * WORKERS

# Layouts whose art_crop is a normal art box. Tokens are included: they end up on tables.
ART_LAYOUTS = {"normal", "leveler", "saga", "class", "case", "mutate", "prototype", "token", "adventure"}


def download_bulk(client: httpx.Client, dest: Path) -> Path:
    if dest.exists():
        return dest
    meta = client.get(BULK_URL).raise_for_status().json()
    entry = next(e for e in meta["data"] if e["type"] == "unique_artwork")
    with client.stream("GET", entry["jsonl_download_uri"]) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="bulk") as bar:
            for chunk in r.iter_bytes(1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def usable(card: dict) -> bool:
    return (
        card.get("lang") == "en"
        and card.get("layout") in ART_LAYOUTS
        and "paper" in card.get("games", [])
        and card.get("image_status") in ("highres_scan", "lowres")
        and not card.get("digital", False)
        and "art_crop" in (card.get("image_uris") or {})
    )


def usable_entries(bulk: Path) -> list[dict]:
    entries = []
    with gzip.open(bulk, "rt", encoding="utf-8") as f:
        for line in f:
            card = json.loads(line)
            if usable(card):
                entries.append(
                    {
                        "id": card["id"],
                        "oracle_id": card.get("oracle_id"),
                        "name": card["name"],
                        "set": card["set"],
                        "url": card["image_uris"]["art_crop"],
                    }
                )
    print(f"{len(entries)} usable unique artworks in bulk file")
    return entries


def sample_arts(entries: list[dict], n_train: int, n_eval: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    entries = list(entries)
    rng.shuffle(entries)
    picked = entries[: n_train + n_eval]
    for i, e in enumerate(picked):
        e["split"] = "train" if i < n_train else "eval"
    return picked


def extend_to_all(existing: list[dict], entries: list[dict]) -> list[dict]:
    """Keep an existing sample and its splits (so evaluations stay comparable) and add every
    other usable artwork as training data."""
    known = {e["id"] for e in existing}
    added = [dict(e, split="train") for e in entries if e["id"] not in known]
    print(f"keeping {len(existing)} sampled arts, adding {len(added)} as train")
    return existing + added


def fetch_image(client: httpx.Client, entry: dict) -> tuple[str, bool]:
    dest = ART_DIR / f"{entry['id']}.jpg"
    if dest.exists():
        return entry["id"], True
    try:
        r = client.get(entry["url"], timeout=30)
        time.sleep(REQUEST_GAP_S)
        if r.status_code != 200:
            return entry["id"], False
        dest.write_bytes(r.content)
        return entry["id"], True
    except httpx.HTTPError:
        return entry["id"], False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--all", action="store_true", help="after sampling, add every remaining usable artwork as train (~49k images, ~3 GB)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        bulk = download_bulk(client, DATA_DIR / "unique-artwork.jsonl.gz")
        arts_path = DATA_DIR / "arts.json"
        arts = json.loads(arts_path.read_text()) if arts_path.exists() else None
        if arts is None or args.all:
            entries = usable_entries(bulk)
            if arts is None:
                arts = sample_arts(entries, args.train, args.eval, args.seed)
            if args.all:
                arts = extend_to_all(arts, entries)
            arts_path.write_text(json.dumps(arts))

        failed = []
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(fetch_image, client, e) for e in arts]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="art_crop"):
                art_id, ok = fut.result()
                if not ok:
                    failed.append(art_id)
    if failed:
        print(f"{len(failed)} downloads failed; rerun to retry")
    print(f"{len(arts) - len(failed)} images in {ART_DIR}")


if __name__ == "__main__":
    main()
