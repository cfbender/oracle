"""Fetch Scryfall unique-artwork metadata and download a deterministic sample of art crops.

Usage:
    uv run python -m cardid.scryfall --train 5000 --eval 1000   # 6k sample (~25 min at 10 req/s)
    uv run python -m cardid.scryfall --all                       # then everything else as train
    uv run python -m cardid.scryfall --cards 3000                # full-card images for the detector
    uv run python -m cardid.scryfall --metadata                  # backfill layout/collector_number into an older arts.json

Writes:
    data/unique-artwork.jsonl.gz   raw bulk file
    data/arts.json                 entries: [{id, oracle_id, name, set, layout, collector_number, face, lang, split, url}]
    data/art/<id>.jpg              art_crop images
    data/cards/<id>.jpg            `normal` full-card images (488x680) of a random subset
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

from . import ART_DIR, CARD_DIR, DATA_DIR

BULK_URL = "https://api.scryfall.com/bulk-data"
HEADERS = {
    "User-Agent": "TheGathering-CardID-Spike/0.1 (github.com/cfbender/the-gathering)",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}
# Scryfall asks for 50-100ms between requests (<10 req/s). Each worker sleeps WORKERS * 0.1s
# after its request so the pool as a whole stays at ~10 req/s.
WORKERS = 8
REQUEST_GAP_S = 0.1 * WORKERS

# Layouts whose art_crop fits an existing frame. Tokens also end up on tables.
# Split and flip cards need new crop geometry; art_series are not playable cards.
ART_LAYOUTS = {"normal", "leveler", "saga", "class", "case", "mutate", "prototype", "token", "adventure", "prepare", "meld"}
FACE_LAYOUTS = {"transform", "modal_dfc", "reversible_card", "double_faced_token"}


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
        "paper" in card.get("games", []) and card.get("image_status") in ("highres_scan", "lowres") and not card.get("digital", False) and bool(art_faces(card))
    )


def art_faces(card: dict) -> list[tuple[int, dict]]:
    """Only separate printed sides get face IDs; adventure/prepare share one top art box."""
    layout = card.get("layout")
    faces = card.get("card_faces", []) if layout in FACE_LAYOUTS else [card] if layout in ART_LAYOUTS else []
    return [(i, face) for i, face in enumerate(faces) if (face.get("image_uris") or {}).get("art_crop")]


def usable_entries(bulk: Path) -> list[dict]:
    entries = []
    with gzip.open(bulk, "rt", encoding="utf-8") as f:
        for line in f:
            card = json.loads(line)
            if usable(card):
                for face_index, face in art_faces(card):
                    entries.append(
                        {
                            "id": card["id"] if face_index == 0 else f"{card['id']}-{face_index}",
                            "oracle_id": card.get("oracle_id") or face.get("oracle_id"),
                            "name": face["name"],
                            "set": card["set"],
                            "layout": card["layout"],
                            "collector_number": card["collector_number"],
                            "face": face_index,
                            "lang": card["lang"],
                            "url": face["image_uris"]["art_crop"],
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


METADATA_FIELDS = ("layout", "collector_number", "face", "lang")


def add_metadata(arts: list[dict], entries: list[dict]) -> int:
    """Backfill METADATA_FIELDS into an arts.json written before they were recorded; returns
    how many entries changed. The recogniser needs `layout` to tell a saga's right-half art
    from a class or case card's left-half art (`detect.frame_of`); the capture tool's search
    shows and matches `collector_number` to pick one of a set's many Forests."""
    by_id = {e["id"]: e for e in entries}
    changed = 0
    for a in arts:
        e = by_id.get(a["id"])
        missing = [f for f in METADATA_FIELDS if f not in a and e and f in e]
        for f in missing:
            a[f] = e[f]
        changed += bool(missing)
    return changed


def extend_to_all(existing: list[dict], entries: list[dict]) -> list[dict]:
    """Keep an existing sample and its splits (so evaluations stay comparable) and add every
    other usable artwork as training data."""
    known = {e["id"] for e in existing}
    added = [dict(e, split="train") for e in entries if e["id"] not in known]
    print(f"keeping {len(existing)} sampled arts, adding {len(added)} as train")
    return existing + added


def fetch_image(client: httpx.Client, entry: dict, dest_dir: Path = ART_DIR, url_key: str = "url") -> tuple[str, bool]:
    dest = dest_dir / f"{entry['id']}.jpg"
    if dest.exists():
        return entry["id"], True
    try:
        r = client.get(entry[url_key], timeout=30)
        time.sleep(REQUEST_GAP_S)
        if r.status_code != 200:
            return entry["id"], False
        dest.write_bytes(r.content)
        return entry["id"], True
    except httpx.HTTPError:
        return entry["id"], False


def card_image_url(art_url: str) -> str:
    """Scryfall serves every image version at the same path; only the version segment differs."""
    return art_url.replace("/art_crop/", "/normal/", 1)


def download_cards(client: httpx.Client, arts: list[dict], n: int, seed: int) -> None:
    """Full-card images of `n` random train-split arts (the detector renders whole cards, so
    the sample includes borderless, showcase, and old frames in whatever proportion Scryfall
    has them). Deterministic in `seed`; rerunning with a larger `n` only adds cards."""
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    picked = [a for a in arts if a["split"] == "train"]
    rng.shuffle(picked)
    entries = [{"id": a["id"], "card_url": card_image_url(a["url"])} for a in picked[:n]]
    failed = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(fetch_image, client, e, CARD_DIR, "card_url") for e in entries]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="normal"):
            failed += not fut.result()[1]
    if failed:
        print(f"{failed} card downloads failed; rerun to retry")
    print(f"{len(list(CARD_DIR.glob('*.jpg')))} full-card images in {CARD_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--all", action="store_true", help="after sampling, add every remaining usable artwork as train (~49k images, ~3 GB)")
    parser.add_argument("--cards", type=int, help="only download full-card images of this many random train arts into data/cards (~100 KB each)")
    parser.add_argument(
        "--metadata",
        "--layouts",
        action="store_true",
        help="only backfill Scryfall metadata (layout, collector_number, face, lang) into an existing data/arts.json (no image downloads)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="fetch a fresh bulk file and add every usable artwork it has that data/arts.json lacks (new sets) as train, then download their art",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    arts_path = DATA_DIR / "arts.json"
    if args.cards:
        if not arts_path.exists():
            raise SystemExit("run the art_crop download first so data/arts.json exists")
        with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
            download_cards(client, json.loads(arts_path.read_text()), args.cards, args.seed)
        return
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        bulk_path = DATA_DIR / "unique-artwork.jsonl.gz"
        if args.update and bulk_path.exists():
            bulk_path.replace(bulk_path.with_suffix(".gz.previous"))  # keep one for a diff or a rollback
        bulk = download_bulk(client, bulk_path)
        arts = json.loads(arts_path.read_text()) if arts_path.exists() else None
        if args.metadata:
            if arts is None:
                raise SystemExit("run the art_crop download first so data/arts.json exists")
            changed = add_metadata(arts, usable_entries(bulk))
            arts_path.write_text(json.dumps(arts))
            print(f"metadata ({', '.join(METADATA_FIELDS)}) added to {changed} of {len(arts)} arts")
            return
        if arts is None or args.all or args.update:
            entries = usable_entries(bulk)
            if arts is None:
                arts = sample_arts(entries, args.train, args.eval, args.seed)
            if args.all or args.update:
                arts = extend_to_all(arts, entries)
            add_metadata(arts, entries)
            arts_path.write_text(json.dumps(arts))
        if args.update:
            # only the new arts need fetching; `fetch_image` skips files that exist anyway, but
            # this keeps a routine refresh from walking 49k files
            arts = [a for a in arts if not (ART_DIR / f"{a['id']}.jpg").exists()]
            print(f"{len(arts)} new arts to download; then re-export the bundle (`python -m cardid.export`) to add them to the gallery")

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
