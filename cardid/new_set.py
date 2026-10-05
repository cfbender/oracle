"""Run `cardid.retrain` for every app profile, after one shared gallery refresh.

    mise run retrain-all                          # new scans: pull, fine-tune, gate, publish each app
    mise run new-set                              # new set: re-export the published models, no training
    mise run new-set -- --profile manavault       # only some (repeatable)
    mise run new-set -- --dry-run                 # any other flag goes to each retrain run

With --train (`mise run retrain-all`) each app runs a full retrain: pull its own captures,
fine-tune from its own published checkpoints, pass its held-out gate, publish.

Without it (new sets): new cards need gallery embeddings, not training. For each app profile this runs
`cardid.retrain --gallery-only` with that app's env file: its own published checkpoints (exact
SHA256 match), its own held-out gate and its own publish target. The gallery is refreshed once,
first, so no app can publish a bundle without the new cards. A failing app does not stop the
rest; the exit status is non-zero if any failed.
"""

from __future__ import annotations

import argparse
import fcntl
import shlex
import subprocess
import sys
from pathlib import Path

from . import DATA_DIR, ML_DIR

CONFIG_DIR = Path("~/.config/cardid").expanduser()


def profile_files(config_dir: Path, names: list[str]) -> list[Path]:
    """The env files to run, in a stable order: the named profiles, or every profile."""
    if names:
        files = [config_dir / f"{name}.env" for name in names]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise SystemExit(f"no such profile: {', '.join(missing)} (see `mise run setup-profiles`)")
        return files
    files = sorted(config_dir.glob("*.env"))
    if not files:
        raise SystemExit(f"no profiles in {config_dir}; run `mise run setup-profiles` and fill them in")
    return files


def commands(files: list[Path], passthrough: list[str], train: bool = False) -> list[list[str]]:
    """One retrain per profile (gallery-only unless training), all on the gallery refreshed beforehand."""
    mode = [] if train else ["--gallery-only"]
    return [[sys.executable, "-m", "cardid.retrain", "--env-file", str(path), *mode, "--no-update-gallery", *passthrough] for path in files]


def update_gallery(runner) -> None:
    """`scryfall --update` under the lock nightly and retrain share, so no run sees it half done."""
    directory = DATA_DIR / "nightly"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another nightly/retrain run is active") from None
        cmd = [sys.executable, "-m", "cardid.scryfall", "--update"]
        print(f"== gallery: {shlex.join(cmd[1:])}", flush=True)
        if runner(cmd, cwd=ML_DIR).returncode != 0:
            raise SystemExit("gallery update failed; nothing was exported or published")


def run_all(files: list[Path], passthrough: list[str], runner=subprocess.run, train: bool = False) -> dict[str, bool]:
    # A dry run only plans; --no-update-gallery reuses a gallery refreshed by hand.
    if not {"--dry-run", "--no-update-gallery"} & set(passthrough):
        update_gallery(runner)
    passthrough = [arg for arg in passthrough if arg != "--no-update-gallery"]
    results = {}
    for path, cmd in zip(files, commands(files, passthrough, train), strict=True):
        print(f"== {path.stem}: {shlex.join(cmd[1:])}", flush=True)
        results[path.stem] = runner(cmd, cwd=ML_DIR).returncode == 0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", action="append", default=[], help="profile name under the config dir (repeatable; default: all)")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--train", action="store_true", help="full retrain on each app's new captures instead of a gallery-only re-export")
    args, passthrough = parser.parse_known_args()
    results = run_all(profile_files(args.config_dir, args.profile), passthrough, train=args.train)
    print("\n".join(f"{name}: {'ok' if ok else 'FAILED (see its report above)'}" for name, ok in results.items()), flush=True)
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
