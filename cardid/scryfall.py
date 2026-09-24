"""Fetch all Scryfall printings, group by illustration, and download representative crops.

HTTP and image handling live here; `cardid.catalog` turns card records into `arts.json`
entries (its names are re-exported for existing callers).

Usage:
    uv run python -m cardid.scryfall --train 5000 --eval 1000   # 6k sample (~25 min at 10 req/s)
    uv run python -m cardid.scryfall --all                       # then everything else as train
    uv run python -m cardid.scryfall --cards 3000                # full-card images for the detector
    uv run python -m cardid.scryfall --two-part-cards            # all usable split/flip printing scans, no bulk required
    uv run python -m cardid.scryfall --metadata                  # refresh metadata/printing siblings without downloading art

Writes:
    data/all-cards.jsonl.gz        raw bulk file (all languages)
    data/arts.json                 artwork rows with illustration_id, printing siblings, split and crop URL
    data/art/<id>.jpg              art_crop images
    data/cards/<id>.jpg            `normal` full-card images (488x680) of a random subset
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import httpx
from tqdm import tqdm

from . import ART_DIR, CARD_DIR, DATA_DIR
from .catalog import (
    ART_LAYOUTS,
    FACE_LAYOUTS,
    METADATA_FIELDS,
    TWO_PART_LAYOUTS,
    add_metadata,
    art_faces,
    card_image_url,
    embeds,
    extend_to_all,
    face_image_url,
    hub_card,
    layout_group,
    printing_order,
    sample_arts,
    supported_faces,
    two_part_cards,
    usable,
    usable_entries,
)
from .detect import frame_crop, frame_of
from .downloads import (
    BULK_MAX_BYTES,
    GZIP_TYPES,
    IMAGE_MAX_BYTES,
    IMAGE_TYPES,
    DownloadError,
    decode_image,
    download_file,
    fetch_bytes,
    fetch_json,
    validate_gzip_jsonl,
    write_atomic,
)

BULK_URL = "https://api.scryfall.com/bulk-data"
HEADERS = {
    "User-Agent": "TheGathering-CardID-Spike/0.1 (github.com/cfbender/the-gathering)",
    "Accept": "application/json;q=0.9,*/*;q=0.8",
}
# Scryfall asks for 50-100ms between requests (<10 req/s). Each worker sleeps WORKERS * 0.1s
# after its request so the pool as a whole stays at ~10 req/s.
WORKERS = 8
REQUEST_GAP_S = 0.1 * WORKERS

__all__ = [
    "ART_LAYOUTS",
    "FACE_LAYOUTS",
    "METADATA_FIELDS",
    "TWO_PART_LAYOUTS",
    "add_metadata",
    "art_faces",
    "card_image_url",
    "download_bulk",
    "download_cards",
    "download_two_part_cards",
    "embeds",
    "extend_to_all",
    "face_image_url",
    "fetch_image",
    "hub_card",
    "layout_group",
    "printing_order",
    "sample_arts",
    "supported_faces",
    "two_part_cards",
    "usable",
    "usable_entries",
]


def download_bulk(client: httpx.Client, dest: Path) -> Path:
    """The all-cards bulk file. An existing file is reused only if it is a complete gzip
    stream (older versions could leave a truncated one behind); new downloads are capped,
    type-checked, verified and renamed into place, so an interrupted run leaves nothing."""
    if dest.exists():
        try:
            validate_gzip_jsonl(dest)
            return dest
        except DownloadError as error:
            print(f"discarding {dest.name}: {error}")
            dest.unlink()
    meta = fetch_json(client, BULK_URL)
    entry = next(e for e in meta["data"] if e["type"] == "all_cards")
    with tqdm(total=entry.get("compressed_size"), unit="B", unit_scale=True, desc="bulk") as bar:
        download_file(
            client,
            entry["jsonl_download_uri"],
            dest,
            max_bytes=BULK_MAX_BYTES,
            content_types=GZIP_TYPES,
            validate=validate_gzip_jsonl,
            progress=lambda n, _total: bar.update(n),
            timeout=120,
        )
    return dest


def fetch_image(client: httpx.Client, entry: dict, dest_dir: Path = ART_DIR, url_key: str = "url") -> tuple[str, bool]:
    """Download one image to `<dest_dir>/<id>.jpg`: capped, checked to decode at a plausible
    size (a portrait card for full scans), written atomically. Existing files are kept."""
    dest = dest_dir / f"{entry['id']}.jpg"
    if dest.exists():
        return entry["id"], True
    full_card = url_key == "card_url" or entry.get("layout") in TWO_PART_LAYOUTS
    try:
        try:
            data = fetch_bytes(client, entry[url_key], max_bytes=IMAGE_MAX_BYTES, content_types=IMAGE_TYPES, timeout=30)
        finally:
            time.sleep(REQUEST_GAP_S)
        image = decode_image(data, card=full_card)
        if url_key == "url" and entry.get("layout") in TWO_PART_LAYOUTS:
            frame = frame_of(1, entry["layout"], entry["face"], entry["layout_group"])
            ok, encoded = cv2.imencode(".jpg", frame_crop(image, frame), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                return entry["id"], False
            data = encoded.tobytes()
        write_atomic(dest, data)
        return entry["id"], True
    except (httpx.HTTPError, DownloadError):
        return entry["id"], False


def download_two_part_cards(client: httpx.Client) -> None:
    """Add every usable split/flip printing, including translations, without a bulk download.
    The manifest identifies successful scans for CardBank; reruns retry missing images and
    retain old entries. Detector geometry validation is synthetic, not held-out artwork."""
    url = "https://api.scryfall.com/cards/search"
    params = {"q": "(layout:split or layout:flip) game:paper include:multilingual", "unique": "prints"}
    cards = []
    while url:
        page = fetch_json(client, url, params=params)
        cards.extend(page["data"])
        url = page.get("next_page") if page.get("has_more") else None
        params = None
        time.sleep(0.1)
    entries = two_part_cards(cards)
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CARD_DIR / "two-part.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    groups = {e["id"]: e["layout_group"] for e in entries}
    failed = []
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(fetch_image, client, e, CARD_DIR, "card_url") for e in entries]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="two-part normal"):
            card_id, ok = fut.result()
            if ok:
                manifest[card_id] = groups[card_id]
            else:
                failed.append(card_id)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    tmp.replace(manifest_path)
    print(f"two-part scans: {len(entries) - len(failed)}/{len(entries)}; manifest: {manifest_path}")
    if failed:
        raise SystemExit(f"{len(failed)} two-part downloads failed; rerun --two-part-cards to retry")


def download_cards(client: httpx.Client, arts: list[dict], n: int, seed: int) -> None:
    """Full-card images of `n` random train-split arts (the detector renders whole cards, so
    the sample includes borderless, showcase, and old frames in whatever proportion Scryfall
    has them). Deterministic in `seed`; rerunning with a larger `n` only adds cards."""
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    picked = [a for a in arts if a["split"] == "train" and embeds(a)]
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
    parser.add_argument("--all", action="store_true", help="after sampling, add every remaining usable artwork as train (~52k images, ~3 GB)")
    parser.add_argument("--cards", type=int, help="only download full-card images of this many random train arts into data/cards (~100 KB each)")
    parser.add_argument(
        "--two-part-cards", action="store_true", help="add all usable split/flip normal scans and sampling manifest (search API; no bulk needed)"
    )
    parser.add_argument(
        "--metadata",
        "--layouts",
        action="store_true",
        help="backfill illustration/face metadata and refresh printing siblings in data/arts.json from the cached bulk (no image downloads)",
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
    if args.cards or args.two_part_cards:
        if args.cards and not arts_path.exists():
            raise SystemExit("run the art_crop download first so data/arts.json exists")
        with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
            if args.cards:
                download_cards(client, json.loads(arts_path.read_text()), args.cards, args.seed)
            if args.two_part_cards:
                download_two_part_cards(client)
        return
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        bulk_path = DATA_DIR / "all-cards.jsonl.gz"
        previous = bulk_path.with_suffix(".gz.previous")
        if args.update and bulk_path.exists():
            bulk_path.replace(previous)  # keep one for a diff or a rollback
        try:
            bulk = download_bulk(client, bulk_path)
        except (httpx.HTTPError, DownloadError) as error:
            if args.update and previous.exists() and not bulk_path.exists():
                previous.replace(bulk_path)  # the failed refresh must not cost the working copy
            raise SystemExit(f"bulk download failed: {error}") from None
        arts = json.loads(arts_path.read_text()) if arts_path.exists() else None
        if args.metadata:
            if arts is None:
                raise SystemExit("run the art_crop download first so data/arts.json exists")
            excluded = set()
            entries = usable_entries(bulk, excluded)
            changed = add_metadata(arts, entries, excluded)
            arts_path.write_text(json.dumps(arts))
            print(f"metadata ({', '.join(METADATA_FIELDS)}) added to {changed} of {len(arts)} arts")
            return
        if arts is None or args.all or args.update:
            excluded = set()
            entries = usable_entries(bulk, excluded)
            if arts is None:
                arts = sample_arts(entries, args.train, args.eval, args.seed)
            if args.all or args.update:
                arts = extend_to_all(arts, entries, excluded)
            add_metadata(arts, entries, excluded)
            arts_path.write_text(json.dumps(arts))
        arts = [a for a in arts if embeds(a)]
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
