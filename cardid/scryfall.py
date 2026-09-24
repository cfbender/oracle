"""Fetch all Scryfall printings, group by illustration, and download representative crops.

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
import gzip
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import httpx
from tqdm import tqdm

from . import ART_DIR, CARD_DIR, DATA_DIR
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

# Tokens also end up on tables. Art series and landscape battles remain excluded.
ART_LAYOUTS = {"normal", "leveler", "saga", "class", "case", "mutate", "prototype", "token", "adventure", "prepare", "meld"}
FACE_LAYOUTS = {"transform", "modal_dfc", "reversible_card", "double_faced_token", "split", "flip"}
TWO_PART_LAYOUTS = {"split", "flip"}


def layout_group(card: dict) -> str:
    """Room and aftermath are both Scryfall split layouts; never infer them from aspect."""
    if card["layout"] == "split":
        if any("Room" in face.get("type_line", "").split() for face in card.get("card_faces", [])):
            return "room"
        if "Aftermath" in card.get("keywords", []):
            return "aftermath"
    return card["layout"]


def face_image_url(card: dict, face: dict) -> str | None:
    # Same-surface faces have no individual image_uris in all_cards. Its art_crop is a
    # montage (or a shared illustration), so always cut the normal scan ourselves.
    if card.get("layout") in TWO_PART_LAYOUTS:
        return (card.get("image_uris") or {}).get("normal")
    return (face.get("image_uris") or {}).get("art_crop")


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


def usable(card: dict) -> bool:
    return (
        "paper" in card.get("games", [])
        and card.get("image_status") in ("highres_scan", "lowres")
        and not card.get("digital", False)
        and not hub_card(card)
        and bool(art_faces(card))
    )


def hub_card(card: dict) -> bool:
    """Paper "cards" whose face is mostly flat text or sketch lines, whatever their nominal
    `layout`. Their embeddings sit near everything and become hubs that soak up degraded
    queries: on one real-camera evaluation, playtest cards took nine of sixteen wrong top-1
    hits and World Championship decklist/ad cards most of the wrong other-orientation hits.

    - Mystery Booster / Playtest sketch cards: `promo_types` contains `playtest`. Keyed on that
      rather than `set_type: funny`, which would also drop Unfinity's legal cards.
    - Non-game inserts whose `type_line` is the bare word `Card`: World Championship decklists,
      bios and ads, minigame cards. Real tokens keep their own type lines and stay."""
    return "playtest" in card.get("promo_types", []) or card.get("type_line") == "Card"


def art_faces(card: dict) -> list[tuple[int, dict]]:
    """Separate arts get face IDs; adventure/prepare share one top art box."""
    return [(i, face) for i, face in enumerate(supported_faces(card)) if face_image_url(card, face)]


def supported_faces(card: dict) -> list[dict]:
    layout = card.get("layout")
    faces = card.get("card_faces", []) if layout in FACE_LAYOUTS else [card] if layout in ART_LAYOUTS else []
    # Un/playtest cards with three or five split parts have neither these boxes nor
    # valid gallery suffixes. Do not quietly download them as two-part cards.
    return [] if layout in TWO_PART_LAYOUTS and len(faces) != 2 else faces


def usable_entries(bulk: Path, excluded: set[str] | None = None) -> list[dict]:
    """Every paper artwork worth embedding, with its selectable printings. When `excluded` is
    given, it collects the face IDs of paper cards this version rejects (see `hub_card`) so an
    arts.json written before the rule can retire those rows without renumbering the rest."""
    groups = {}
    with gzip.open(bulk, "rt", encoding="utf-8") as f:
        for line in f:
            card = json.loads(line)
            if "paper" not in card.get("games", []) or card.get("digital", False):
                continue
            layout = card.get("layout")
            faces = supported_faces(card)
            if hub_card(card):
                if excluded is not None:
                    excluded.update(card["id"] if i == 0 else f"{card['id']}-{i}" for i in range(len(faces)))
                continue
            for face_index, face in enumerate(faces):
                printing = {
                    "id": card["id"] if face_index == 0 else f"{card['id']}-{face_index}",
                    "name": face["name"],
                    "set": card["set"],
                    "layout": layout,
                    "collector_number": card["collector_number"],
                    "face": face_index,
                    "lang": card["lang"],
                    "border_color": card.get("border_color"),
                    "scryfall_frame": card.get("frame"),
                    "frame_effects": card.get("frame_effects", []),
                    "promo": card.get("promo", False),
                }
                if layout in TWO_PART_LAYOUTS:
                    printing["layout_group"] = layout_group(card)
                illustration = face.get("illustration_id") or printing["id"]
                if layout in TWO_PART_LAYOUTS:
                    # Some printings omit face 1's ID; others repeat face 0's ID there.
                    # These are regions of one scan, so key both by shared illustration
                    # plus region. A translation cannot collapse or duplicate a half.
                    shared = card.get("illustration_id") or face.get("illustration_id") or card["id"]
                    illustration = f"{shared}:face:{face_index}"
                group = groups.setdefault(illustration, {"printings": [], "art": None})
                group["printings"].append(printing)
                url = face_image_url(card, face)
                if url and card.get("image_status") in ("highres_scan", "lowres"):
                    art = {**printing, "illustration_id": illustration, "oracle_id": card.get("oracle_id") or face.get("oracle_id"), "url": url}
                    if group["art"] is None or printing_order(art) < printing_order(group["art"]):
                        group["art"] = art
    entries = [dict(g["art"], printings=sorted(g["printings"], key=printing_order)) for g in groups.values() if g["art"] is not None]
    print(f"{len(entries)} distinct artworks; {sum(len(e['printings']) for e in entries)} selectable paper printing faces")
    return entries


def printing_order(printing: dict) -> tuple:
    """Prefer ordinary English non-promo scans for new arts; never replace an existing crop."""
    return (
        printing["lang"] != "en",
        bool({"extendedart", "showcase"}.intersection(printing.get("frame_effects", []))) or printing.get("border_color") == "borderless",
        printing.get("promo", False),
        printing["set"],
        printing["collector_number"],
        printing["id"],
    )


def sample_arts(entries: list[dict], n_train: int, n_eval: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    entries = list(entries)
    rng.shuffle(entries)
    picked = entries[: n_train + n_eval]
    for i, e in enumerate(picked):
        e["split"] = "train" if i < n_train else "eval"
    return picked


METADATA_FIELDS = ("layout", "collector_number", "face", "lang", "illustration_id", "layout_group")


def add_metadata(arts: list[dict], entries: list[dict], excluded: set[str] = frozenset()) -> int:
    """Backfill METADATA_FIELDS into an arts.json written before they were recorded; returns
    how many entries changed. The recogniser needs `layout` to tell a saga's right-half art
    from a class or case card's left-half art (`detect.frame_of`); the capture tool's search
    shows and matches `collector_number` to pick one of a set's many Forests.

    Rows whose ID is in `excluded` are kept (so indices and splits of the others never move)
    but flagged `excluded`, which `data.load_arts` and the downloaders skip."""
    by_id = {p["id"]: (e, p) for e in entries for p in e["printings"]}
    changed = set()
    for a in arts:
        if a["id"] in excluded and not a.get("excluded"):
            a["excluded"] = True
            changed.add(a["id"])
        match = by_id.get(a["id"])
        if not match:
            continue
        e, printing = match
        before = dict(a)
        metadata = {**printing, "illustration_id": e["illustration_id"]}
        for field in METADATA_FIELDS:
            if field not in a and field in metadata:
                a[field] = metadata[field]
        a["printings"] = e["printings"]
        if a != before:
            changed.add(a["id"])
    # Old unique_artwork exports can repeat a reverse illustration. Keep every persisted
    # ID/split, but embed only one row; prefer a held-out row to avoid train/eval leakage.
    representatives = {}
    for a in sorted(arts, key=lambda a: (bool(a.get("excluded")), a["split"] != "eval")):
        key = a.get("illustration_id", a["id"])
        representative = representatives.setdefault(key, a["id"])
        if a["id"] != representative:
            if a.get("alias_of") != representative:
                changed.add(a["id"])
            a["alias_of"] = representative
    return len(changed)


def embeds(art: dict) -> bool:
    """Whether a persisted row contributes a gallery embedding. Aliases of another row's
    illustration and cards a later `usable` rule retired (`excluded`) keep their index and
    split in arts.json but are neither downloaded nor embedded."""
    return not (art.get("alias_of") or art.get("excluded"))


def extend_to_all(existing: list[dict], entries: list[dict], excluded: set[str] = frozenset()) -> list[dict]:
    """Keep an existing sample and its splits (so evaluations stay comparable) and add every
    other usable artwork as training data."""
    add_metadata(existing, entries, excluded)
    known = {e.get("illustration_id", e["id"]) for e in existing}
    added = [dict(e, split="train") for e in entries if e["illustration_id"] not in known]
    print(f"keeping {len(existing)} sampled arts, adding {len(added)} as train")
    return existing + added


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


def card_image_url(art_url: str) -> str:
    """Scryfall serves every image version at the same path; only the version segment differs."""
    return art_url.replace("/art_crop/", "/normal/", 1)


def two_part_cards(cards) -> list[dict]:
    """One untouched portrait scan per usable physical printing, not per artwork/half."""
    entries = {
        card["id"]: {"id": card["id"], "layout_group": layout_group(card), "card_url": card["image_uris"]["normal"]}
        for card in cards
        if card.get("layout") in TWO_PART_LAYOUTS and usable(card)
    }
    return [entries[key] for key in sorted(entries)]


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
