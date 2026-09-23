"""Printing identities share artwork targets; gallery order remains the embedding order."""

import hashlib
import json

ART_FIELDS = ("id", "name", "set", "collector_number", "layout", "face", "lang", "illustration_id", "url")


def runtime_metadata(arts: list[dict], frames: list[str]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Keep embedding order stable; selectable siblings are a separate, on-demand file."""
    compact = []
    printings = {}
    for art, frame in zip(arts, frames, strict=True):
        siblings = art.get("printings", [])
        compact.append({**{k: art[k] for k in ART_FIELDS if k in art}, "frame": frame, "printing_count": len(siblings)})
        if siblings:
            printings[art["id"]] = siblings
    return compact, printings


def printing_index(arts: list[dict]) -> dict[str, int]:
    return {printing_id: i for i, art in enumerate(arts) for printing_id in [art["id"], *(p["id"] for p in art.get("printings", []))]}


def gallery_fingerprint(arts: list[dict]) -> str:
    """Invalidate cached pixels/targets when rows move, disappear, or change split."""
    return hashlib.sha256(json.dumps([(a["id"], a.get("split")) for a in arts]).encode()).hexdigest()[:16]
