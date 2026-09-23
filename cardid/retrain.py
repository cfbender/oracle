"""Pull corrections, refresh the gallery, fine-tune, evaluate and publish in one run."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from . import DATA_DIR, ML_DIR
from .corrections import atomic_json, latest_labels
from .gallery import printing_index
from .workflow import (
    check_destination,
    command,
    comparable_rows,
    find_manifest,
    fingerprint,
    publish_allowed,
    resolve_checkpoint,
    score,
    sha256,
    snapshot_bundle,
    trained_checkpoint,
)


def load_env(path: Path) -> None:
    """Read literal shell-style assignments, without executing a secret-bearing file."""
    if not path.exists():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        words = shlex.split(line, comments=True)
        if words and words[0] == "export":
            words = words[1:]
        if not words:
            continue
        if len(words) != 1 or "=" not in words[0]:
            raise SystemExit(f"{path}:{number}: expected KEY=value (quote spaces; no shell expansion)")
        key, value = words[0].split("=", 1)
        if key.startswith("CARDID_"):
            os.environ.setdefault(key, value)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path, default=Path(os.environ.get("CARDID_ENV_FILE", "~/.config/cardid.env")).expanduser())
    env_args, _ = env_parser.parse_known_args(argv)
    load_env(env_args.env_file)
    parser = argparse.ArgumentParser(description=__doc__, parents=[env_parser])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--server", help="HTTPS correction server (overrides configured filesystem source)")
    source.add_argument("--from-dir", type=Path, help="import a filesystem correction export instead of HTTP")
    parser.add_argument("--checkpoint", type=Path, help="explicit recogniser; overrides manifest resolution")
    parser.add_argument("--detector", type=Path, help="explicit detector; overrides manifest resolution")
    parser.add_argument("--to", default=os.environ.get("CARDID_PUBLISH_TO"), help="publish destination, local directory or user@host:/path")
    parser.add_argument("--epochs", type=positive, default=os.environ.get("CARDID_RETRAIN_EPOCHS", "4"))
    parser.add_argument("--detector-epochs", type=positive, help="also fine-tune the detector (off by default)")
    parser.add_argument("--workers", type=positive, default=os.environ.get("CARDID_WORKERS"))
    parser.add_argument("--update-gallery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--force", action="store_true", help="publish even if comparable held-out top-1 regresses; never bypass parity/concurrency checks")
    parser.add_argument("--dry-run", action="store_true", help="read manifests/checkpoints and print the plan; do not pull, train, export or publish")
    args = parser.parse_args(argv)
    if args.server is None and args.from_dir is None:
        args.server = os.environ.get("CARDID_SERVER")
        directory = os.environ.get("CARDID_CORRECTIONS_DIR")
        args.from_dir = Path(directory).expanduser() if directory else None
    if not args.from_dir and not args.server:
        parser.error("configure --server / CARDID_SERVER or --from-dir / CARDID_CORRECTIONS_DIR")
    if not args.no_publish and not args.to:
        parser.error("configure --to / CARDID_PUBLISH_TO or use --no-publish")
    return args


def usable_rows(data: Path) -> list[dict]:
    return [r for r in latest_labels(data / "real").values() if r.get("label") and (data / "real" / r["capture_id"] / "card.png").is_file()]


def train_rows(data: Path, rows: list[dict]) -> list[dict]:
    """Match RealDataset's gallery filtering so unknown labels don't enable --real."""
    path = data / "arts.json"
    arts = json.loads(path.read_text()) if path.exists() else []
    gallery = printing_index([a for a in arts if not a.get("alias_of") and (data / "art" / f"{a['id']}.jpg").is_file()])
    return [r for r in rows if r["split"] == "train" and r["label"] in gallery]


def run(args: argparse.Namespace, *, data: Path = DATA_DIR, runner=command, scorer=score, version: str | None = None) -> dict:
    version = version or datetime.now(UTC).strftime("retrain-%Y%m%dT%H%M%S%fZ")
    report_dir = data / "retrain"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{version}.json"
    report = {
        "version": version,
        "dry_run": args.dry_run,
        "commands": [],
        "status": "started",
        "epochs": args.epochs,
        "detector_epochs": args.detector_epochs,
        "update_gallery": args.update_gallery,
        "force": args.force,
        "publish_to": args.to,
        "no_publish": args.no_publish,
    }

    def execute(*cmd: str, read_only: bool = False) -> None:
        report["commands"].append(list(cmd))
        if args.dry_run and not read_only:
            # These two shell variables are assigned below, after the planned trainers.
            rendered = [f'"${arg}"' if arg in {"candidate_checkpoint", "candidate_detector"} else shlex.quote(arg) for arg in cmd]
            print("+", " ".join(rendered), flush=True)
        else:
            runner(*cmd)

    def model_output(run_name: str, variable: str) -> str:
        path = data / "runs" / run_name
        if not args.dry_run:
            return str(trained_checkpoint(path))
        best, last = shlex.quote(str(path / "best.pt")), shlex.quote(str(path / "last.pt"))
        print(f"{variable}=$(if test -f {best}; then printf %s {best}; else printf %s {last}; fi)", flush=True)
        return variable

    try:
        for run_name in (version, version + "-detector"):
            if (data / "runs" / run_name).exists():
                raise SystemExit(f"refusing to reuse existing run {run_name}")
        if not args.no_publish:
            check_destination(args.to, lambda *cmd: execute(*cmd, read_only=True))
        pull = ["python", "-m", "cardid.corrections", "pull"]
        pull += ["--from-dir", str(args.from_dir)] if args.from_dir else ["--server", args.server]
        execute(*pull)
        if args.update_gallery:
            execute("python", "-m", "cardid.scryfall", "--update")
        with tempfile.TemporaryDirectory(prefix="cardid-retrain-") as tmp:
            path, source = find_manifest(args.to, data / "bundles", Path(tmp), lambda *cmd: execute(*cmd, read_only=True), require=not args.no_publish)
            manifest = json.loads(path.read_text()) if path else None
            baseline_hash = sha256(path) if path else None
            # Only a manifest fetched from the server can guard against a concurrent publish.
            published = source is not None and args.to is not None and source == args.to.rstrip("/") + "/current"
            checkpoint = resolve_checkpoint(
                "recogniser",
                data / "runs",
                manifest,
                args.checkpoint,
                hint=Path(os.environ["CARDID_CHECKPOINT"]) if os.environ.get("CARDID_CHECKPOINT") else None,
            )
            detector = resolve_checkpoint(
                "detector",
                data / "runs",
                manifest,
                args.detector,
                hint=Path(os.environ["CARDID_DETECTOR"]) if os.environ.get("CARDID_DETECTOR") else None,
            )
            report.update(checkpoint=str(checkpoint), detector=str(detector), baseline_source=source, baseline_hash=baseline_hash, baseline_published=published)
            rows = usable_rows(data)
            real_train = train_rows(data, rows)
            eval_rows = [r for r in rows if r["split"] == "eval"]
            report.update(train_captures=len(real_train), eval_captures=len(eval_rows))
            print(
                f"training: {'real + synthetic' if real_train else 'synthetic-only'} ({len(real_train)} usable train, {len(eval_rows)} held-out captures)",
                flush=True,
            )
            if args.dry_run:
                print("DRY RUN: correction/gallery reads use existing local data; the real run rechecks after pulling/updating.", flush=True)
            baseline_path = None
            if eval_rows:
                if not source:
                    raise SystemExit("held-out captures exist but no baseline bundle is available; cannot compare safely")
                if args.dry_run:
                    execute("rsync", "-aL", "--", source.rstrip("/") + "/", str(Path(tmp) / "snapshot") + "/")
                    print(f"DRY RUN: snapshot/verify {source}; score baseline and candidate on the same {len(eval_rows)} held-out crops", flush=True)
                else:
                    baseline_path, _, snapshot_hash = snapshot_bundle(source, Path(tmp), execute)
                    if snapshot_hash != baseline_hash:
                        raise SystemExit("published manifest changed during resolution; retry")
            else:
                report["real_gate"] = "unavailable: no held-out real captures"
                print("WARNING: no held-out real captures; synthetic metrics are informational, publication has no real non-regression gate", flush=True)
            training_options = ["--workers", str(args.workers)] if args.workers else []
            if real_train:
                training_options += ["--real"]
            execute("python", "-m", "cardid.train", "--resume", str(checkpoint), "--epochs", str(args.epochs), "--run", version, *training_options)
            candidate_checkpoint = model_output(version, "candidate_checkpoint")
            candidate_detector = str(detector)
            if args.detector_epochs:
                detector_run = version + "-detector"
                execute(
                    "python",
                    "-m",
                    "cardid.train_detector",
                    "--resume",
                    str(detector),
                    "--epochs",
                    str(args.detector_epochs),
                    "--run",
                    detector_run,
                    *training_options,
                )
                candidate_detector = model_output(detector_run, "candidate_detector")
            bundle = data / "bundles" / version
            execute(
                "python", "-m", "cardid.export", "--checkpoint", candidate_checkpoint, "--detector", candidate_detector, "--version", version, "--verify", "64"
            )
            allowed = True
            if baseline_path:
                scored_rows, dropped = comparable_rows(eval_rows, baseline_path, bundle)
                if dropped:
                    labels = ", ".join(f"{r['capture_id']} -> {r['label']}" for r in dropped)
                    print(f"WARNING: {len(dropped)} held-out captures are unknown to one bundle and are not compared: {labels}", flush=True)
                    report["dropped_captures"] = [r["capture_id"] for r in dropped]
                if not scored_rows:
                    raise SystemExit("refusing incomparable held-out evaluation: no held-out label is known to both bundles")
                baseline = scorer(baseline_path, scored_rows, data / "real")
                candidate = scorer(bundle, scored_rows, data / "real")
                # --force overrides regression only, not incomparable or empty evaluations.
                comparable = publish_allowed(True, {**baseline, "correct": 0}, candidate)
                allowed = publish_allowed(True, baseline, candidate)
                report.update(baseline=baseline, candidate=candidate, allowed=allowed)
                print(f"held-out baseline: {json.dumps(baseline)}\nheld-out candidate: {json.dumps(candidate)}", flush=True)
                if not comparable:
                    raise SystemExit("refusing incomparable held-out evaluation")
            execute("python", "-m", "cardid.evaluate", "--method", "checkpoint", "--checkpoint", candidate_checkpoint, "--profile", "realistic")
            if fingerprint(usable_rows(data)) != fingerprint(rows):
                raise SystemExit("real dataset changed during training; refusing publication")
            report.update(candidate_checkpoint=candidate_checkpoint, candidate_detector=candidate_detector)
            if args.no_publish:
                report["status"] = "not-published"
            elif not allowed and not args.force:
                report["status"] = "regressed"
                raise SystemExit("REFUSED: held-out top-1 regressed; review the report or explicitly use --force")
            else:
                guard = ["--expected-current", baseline_hash] if published else []
                if not published:
                    print("WARNING: no published baseline manifest; publishing without the concurrent-publish guard", flush=True)
                if args.dry_run and eval_rows:
                    print("DRY RUN: publish only if held-out gate passes (or comparable regression with --force).", flush=True)
                execute("python", "-m", "cardid.publish", str(bundle), "--to", args.to, *guard)
                report["status"] = "published"
                if not args.dry_run:
                    state_path = data / "nightly" / "state.json"
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    state.update(checkpoint=candidate_checkpoint, detector=candidate_detector, report=str(report_path))
                    atomic_json(state_path, state)
            if args.dry_run:
                report["status"] = "dry-run"
            return report
    except (Exception, SystemExit) as error:
        report.update(status="failed" if report["status"] == "started" else report["status"], error=str(error))
        raise
    finally:
        atomic_json(report_path, report)
        print(f"report: {report_path}", flush=True)


def main() -> None:
    args = parse_args()
    os.chdir(ML_DIR)
    # Use the same lock as nightly.sh. Manual importers should not run during either loop.
    directory = DATA_DIR / "nightly"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another nightly/retrain run is active") from None
        run(args)


if __name__ == "__main__":
    main()
