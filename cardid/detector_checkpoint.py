"""Loading CornerNet checkpoints across head-layout versions (weights only, never pickled code)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .detector import CornerNet


def load_checkpoint(model: CornerNet, path: Path, device: torch.device) -> None:
    """Load a state dict, tolerating checkpoints saved before the heatmap decoder or the up
    output existed: missing layers keep their fresh initialisation and a shorter final head
    row block is copied into the first rows, so warm-starting still works."""
    state = torch.load(path, map_location=device, weights_only=True)
    notes = []
    own = model.state_dict()
    for key in ("head.3.weight", "head.3.bias"):
        if key in state and state[key].shape != own[key].shape:
            merged = own[key].clone()
            merged[: state[key].shape[0]] = state[key]
            notes.append(f"{key} widened {state[key].shape[0]} -> {own[key].shape[0]} outputs (pre-up detector), new rows left at init")
            state[key] = merged
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"{path}: unexpected keys {sorted(unexpected)[:5]}")
    if missing:
        notes.append(f"{len(missing)} keys not in checkpoint (pre-heatmap detector), left at init")
    for note in notes:
        print(f"{path}: {note}")
