"""Shared desktop pipeline steps; importing this module does not load torch."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

from . import ML_DIR
from .gallery import printing_index


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def command(*args: str) -> None:
    print("+", shlex.join(args), flush=True)
    subprocess.run(args, cwd=ML_DIR, check=True)


def fingerprint(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(sorted(rows, key=lambda r: r["capture_id"]), sort_keys=True).encode()).hexdigest()


def publish_allowed(new_data: bool, baseline: dict, candidate: dict) -> bool:
    return (
        new_data
        and baseline["count"] > 0
        and baseline["count"] == candidate["count"]
        and baseline["captures"] == candidate["captures"]
        and 0 <= baseline["correct"] <= candidate["correct"] <= candidate["count"]
    )


def score(bundle_path: Path, rows: list[dict], real: Path) -> dict:
    from .bundle import Bundle
    from .degrade import load_rgb

    bundle = Bundle(bundle_path)
    gallery = printing_index(bundle.arts)
    missing = {r["label"] for r in rows} - gallery.keys()
    if missing:
        raise SystemExit(f"refusing incomparable evaluation: {len(missing)} held-out labels missing from {bundle_path}")
    correct = 0
    for row in rows:
        crop = load_rgb(real / row["capture_id"] / "crop.jpg")
        click = tuple(row.get("click") or (crop.shape[1] / 2, crop.shape[0] / 2))
        prediction = bundle.identify(crop, click)["results"][0]["id"]
        correct += gallery[prediction] == gallery[row["label"]]
    return {"correct": correct, "count": len(rows), "top1": correct / len(rows), "captures": fingerprint(rows)}


def snapshot_bundle(source: str, directory: Path, runner=command) -> tuple[Path, dict, str]:
    """Copy and validate an immutable baseline before training touches anything."""
    from .publish import check_bundle

    snapshot = directory / "snapshot"
    runner("rsync", "-aL", "--", source.rstrip("/") + "/", str(snapshot) + "/")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", manifest["version"]):
        raise SystemExit("invalid baseline version")
    baseline = snapshot.with_name(manifest["version"])
    snapshot.rename(baseline)
    check_bundle(baseline)
    return baseline, manifest, sha256(baseline / "manifest.json")


def find_manifest(target: str | None, bundles: Path, directory: Path, runner=command) -> tuple[Path | None, str | None]:
    """Prefer current on the server; local fallback is a hint, never proof of publication."""
    if target:
        source = target.rstrip("/") + "/current"
        path = directory / "manifest.json"
        try:
            runner("rsync", "-aL", "--", source + "/manifest.json", str(path))
            json.loads(path.read_text())
            return path, source
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            print(f"WARNING: published manifest unavailable ({error}); looking for a local bundle", flush=True)
    manifests = sorted(bundles.glob("*/manifest.json"), key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    if manifests:
        print(f"WARNING: using newest local manifest {manifests[0]}; it may not be published", flush=True)
        return manifests[0], str(manifests[0].parent)
    print("WARNING: no published or local manifest available", flush=True)
    return None, None


def checkpoint_kind(path: Path) -> str | None:
    """Runs have arbitrary names. Identify our two state-dict formats, not name prefixes."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if "head.weight" in state and any(k.startswith("features.") for k in state):
        return "recogniser"
    if "head.0.weight" in state and any(k.startswith("stem.") for k in state):
        return "detector"
    return None


def resolve_checkpoint(kind: str, runs: Path, manifest: dict | None, explicit: Path | None = None, hint: Path | None = None, strict: bool = False) -> Path:
    """Explicit flags win. Nightly requires a manifest match, never an mtime guess."""
    expected = (manifest or {}).get(kind, {}).get("sha256")
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"{kind}: explicit checkpoint does not exist: {explicit}")
        if strict and (not expected or sha256(explicit) != expected):
            raise SystemExit(f"{kind}: checkpoint does not match the published manifest")
        print(f"{kind}: {explicit} (explicit flag)", flush=True)
        return explicit
    candidates = [p for p in runs.glob("*/*.pt") if p.name in {"best.pt", "last.pt"}]
    if hint and hint.is_file() and hint not in candidates:
        candidates.append(hint)
    candidates.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    if expected:
        for path in candidates:
            if sha256(path) == expected:
                print(f"{kind}: {path} (manifest SHA256 match)", flush=True)
                return path
        print(f"WARNING: {kind} manifest SHA256 matches no local checkpoint", flush=True)
    if strict:
        raise SystemExit(f"{kind}: no local checkpoint matches the published manifest; refusing retraining")
    print(f"WARNING: {kind}: falling back to newest mtime by model type; this may be an unpublished experiment", flush=True)
    for path in candidates:
        try:
            if checkpoint_kind(path) == kind:
                print(f"{kind}: {path} (newest matching model type)", flush=True)
                return path
        except Exception as error:
            print(f"WARNING: cannot inspect checkpoint {path}: {error}", flush=True)
    raise SystemExit(f"{kind}: no usable best.pt/last.pt under {runs}; supply --checkpoint/--detector or restore training runs")


def trained_checkpoint(run: Path) -> Path:
    """Neither trainer guarantees best.pt when resuming a stronger model."""
    for name in ("best.pt", "last.pt"):
        if (run / name).is_file():
            return run / name
    raise SystemExit(f"training wrote neither best.pt nor last.pt in {run}")
