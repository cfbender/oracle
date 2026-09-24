"""Scryfall card records -> the artwork catalog (`data/arts.json`): which printings are usable,
how faces and two-part halves become gallery entries, and how an existing catalog is migrated.

Pure data transformation; downloading lives in `cardid.scryfall`.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

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
