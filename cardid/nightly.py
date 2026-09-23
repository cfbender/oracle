"""Conservative desktop retraining, invoked by nightly.sh (lock, nice and wall-clock budget).

The gate evaluates both exported bundles end-to-end on exactly the same held-out captures,
not a historical metric or training accuracy. No new usable corrections means no publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from . import DATA_DIR, ML_DIR
from .corrections import REAL, atomic_json, latest_labels, pull
from .gallery import printing_index


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


def command(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ML_DIR, check=True)


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


def run(args: argparse.Namespace) -> None:
    state_path = args.state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    count = pull(args.real_dir, args.server, args.from_dir)
    rows = [r for r in latest_labels(args.real_dir).values() if r.get("label") and (args.real_dir / r["capture_id"] / "card.png").exists()]
    corrections = [r for r in rows if r.get("source") == "webcam-table"]
    digest = fingerprint(corrections)
    new_data = bool(corrections) and digest != state.get("corrections")
    print(f"pull/merge: {count} changed; {len(corrections)} usable corrections; new since last completed run: {new_data}", flush=True)
    if not new_data:
        print("REFUSED: no new usable corrections; no training or publication")
        return
    if args.dry_run:
        # Exercise all gate branches without pretending these are measured model scores.
        base = {"count": 5, "correct": 4, "captures": "dry-run-fixture"}
        for name, candidate in [("equal", base), ("better", {**base, "correct": 5}), ("regression", {**base, "correct": 3})]:
            print(f"DRY RUN gate fixture {name}: {publish_allowed(new_data, base, candidate)}")
        print(f"DRY RUN no-new-data fixture: {publish_allowed(False, base, base)}; no training, publish, or completed-run marker")
        return
    if args.real_dir.resolve() != REAL.resolve():
        raise SystemExit("training uses data/real; --real-dir is only for dry runs")
    eval_rows = [r for r in rows if r["split"] == "eval"]
    if not eval_rows or not any(r["split"] == "train" for r in rows):
        raise SystemExit("need both train and held-out eval captures")
    target = os.environ["CARDID_PUBLISH_TO"].rstrip("/")
    checkpoint = Path(state.get("checkpoint") or os.environ["CARDID_CHECKPOINT"])
    detector = Path(os.environ["CARDID_DETECTOR"])
    version = datetime.now(UTC).strftime("nightly-%Y%m%dT%H%M%S%fZ")
    log_path = args.state_dir / f"{version}.json"
    with tempfile.TemporaryDirectory(prefix="cardid-baseline-") as tmp:
        snapshot = Path(tmp) / "snapshot"
        command("rsync", "-aL", "--", f"{target}/current/", str(snapshot) + "/")
        manifest = json.loads((snapshot / "manifest.json").read_text())
        # Version is used as a local path, so validate before renaming.
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", manifest["version"]):
            raise SystemExit("invalid baseline version")
        baseline_path = snapshot.with_name(manifest["version"])
        snapshot.rename(baseline_path)
        from .publish import check_bundle, sha256

        check_bundle(baseline_path)
        baseline_hash = sha256(baseline_path / "manifest.json")
        if sha256(checkpoint) != manifest["recogniser"]["sha256"]:
            raise SystemExit("CARDID_CHECKPOINT/state checkpoint is not the currently published model; update it before retraining")
        if sha256(detector) != manifest["detector"]["sha256"]:
            raise SystemExit("CARDID_DETECTOR is not the published detector; refusing a mixed change")
        baseline = score(baseline_path, eval_rows, args.real_dir)
        print("baseline:", json.dumps(baseline), flush=True)
        candidate_checkpoint = DATA_DIR / "runs" / version / "best.pt"
        command(
            sys.executable,
            "-m",
            "cardid.train",
            "--resume",
            str(checkpoint),
            "--real",
            "--epochs",
            os.environ.get("CARDID_EPOCHS", "2"),
            "--workers",
            os.environ.get("CARDID_WORKERS", "2"),
            "--threads",
            "2",
            "--batch",
            "64",
            "--lr",
            "0.0001",
            "--backbone-lr",
            "0.00003",
            "--run",
            version,
        )
        command(sys.executable, "-m", "cardid.export", "--checkpoint", str(candidate_checkpoint), "--detector", str(detector), "--version", version)
        candidate_bundle = DATA_DIR / "bundles" / version
        candidate = score(candidate_bundle, eval_rows, args.real_dir)
        allowed = publish_allowed(new_data, baseline, candidate)
        report = {"baseline_version": manifest["version"], "candidate_version": version, "baseline": baseline, "candidate": candidate, "allowed": allowed}
        atomic_json(log_path, report)
        print(json.dumps(report), flush=True)
        # Refuse a publish if another importer/labeler changed the data while training.
        after = [r for r in latest_labels(args.real_dir).values() if r.get("label") and (args.real_dir / r["capture_id"] / "card.png").exists()]
        if fingerprint(after) != fingerprint(rows):
            raise SystemExit("real dataset changed during training; refusing publication")
        if allowed:
            command(sys.executable, "-m", "cardid.publish", str(candidate_bundle), "--to", target, "--expected-current", baseline_hash)
            checkpoint = candidate_checkpoint
        atomic_json(state_path, {"corrections": digest, "checkpoint": str(checkpoint), "report": str(log_path)})
        print("PUBLISHED (previous retained)" if allowed else "REFUSED: held-out top-1 regressed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--server", default=os.environ.get("CARDID_SERVER"))
    parser.add_argument("--from-dir", type=Path, default=os.environ.get("CARDID_CORRECTIONS_DIR"))
    parser.add_argument("--real-dir", type=Path, default=REAL)
    parser.add_argument("--state-dir", type=Path, default=DATA_DIR / "nightly")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
