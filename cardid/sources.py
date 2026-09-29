"""Which real captures a model line trains and evaluates on.

Real captures come from several places: The Gathering's webcam table (`webcam-table`),
ManaVault's phone scanner (`manavault-scanner`) and the local `cardid.capture` tool (rows
without a source, called `capture` here). All of them land in data/real/. Set
`CARDID_SOURCES` to a comma-separated list to train and evaluate a model line on only those,
for example `CARDID_SOURCES=manavault-scanner` for ManaVault's phone model. Unset uses all.
"""

from __future__ import annotations

import os

ENV = "CARDID_SOURCES"
LOCAL = "capture"


def row_source(row: dict) -> str:
    return row.get("source") or LOCAL


def selected() -> frozenset[str] | None:
    names = {name.strip() for name in os.environ.get(ENV, "").split(",") if name.strip()}
    return frozenset(names) or None


def keep(rows: list[dict]) -> list[dict]:
    """Rows from the selected sources (all rows when CARDID_SOURCES is unset)."""
    wanted = selected()
    return rows if wanted is None else [r for r in rows if row_source(r) in wanted]
