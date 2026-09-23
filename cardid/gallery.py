"""Printing identities share artwork targets; gallery order remains the embedding order."""

import hashlib
import json


def printing_index(arts: list[dict]) -> dict[str, int]:
    return {printing_id: i for i, art in enumerate(arts) for printing_id in [art["id"], *(p["id"] for p in art.get("printings", []))]}


def gallery_fingerprint(arts: list[dict]) -> str:
    """Invalidate cached pixels/targets when rows move, disappear, or change split."""
    return hashlib.sha256(json.dumps([(a["id"], a.get("split")) for a in arts]).encode()).hexdigest()[:16]
