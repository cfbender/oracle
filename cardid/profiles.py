"""Model lines on one training machine: one env file per app, selected by CARDID_PROFILE.

Each app (The Gathering's webcam table, ManaVault's phone scanner) has its own env file in
~/.config/cardid/<profile>.env with its server, token, publish target and capture sources
(templates in profiles/). The gallery, images, real captures and training runs in data/ are
shared; only the per-line bookkeeping (last published checkpoints, seen corrections) is kept
apart, under data/nightly/<profile>/. Without a profile everything stays in data/nightly/.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV = "CARDID_PROFILE"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def profile() -> str | None:
    name = os.environ.get(ENV, "").strip()
    if name and not _NAME.fullmatch(name):
        raise SystemExit(f"invalid {ENV} {name!r}; use letters, digits, dots, dashes and underscores")
    return name or None


def state_dir(data: Path) -> Path:
    """Where this model line's state.json and nightly logs live."""
    name = profile()
    return data / "nightly" / name if name else data / "nightly"
